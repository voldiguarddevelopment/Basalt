// Loop-invariant code motion: hoists pure, side-effect-free instructions that recompute the
// same value on every iteration of a natural loop out into a preheader block that runs exactly
// once before the loop is entered.
//
// # What counts as invariant
//
// An instruction inside a loop's block set (`dom::NaturalLoop::blocks`) is a hoist candidate
// only if it is one of `Bin`, `ICmp`, `FCmp`, `Cast`, `Select`, `ConstInt`, `ConstFloat` — the
// ops with no effect beyond producing their own result — or one of the twelve GPU index
// intrinsics (`TidX..GdimZ`). The index ops are eligible even though they look like they should
// depend on "which thread", because they don't depend on the loop: a single thread's own
// `tid.x`/`bid.x`/`bdim.x`/`gdim.x` never change during that thread's own execution, loop or no
// loop. The per-thread native loop a CPU backend synthesizes at emission time to sweep `tid.x`
// across `nthreads` is a backend-level construct BIR has no visibility into at all; it is not
// the loop this pass reasons about. `Load`, `Store`, `Phi`, `Barrier`, `Atomic`, `AtomicCas` are
// never hoisted: the first two may have effects or be order-dependent, and a `Phi` is the loop's
// own merge point, not a relocatable computation.
//
// A candidate is confirmed invariant once every operand is either a function parameter, a value
// defined outside the loop's block set, or another instruction already confirmed invariant.
// This is a fixed point over the loop's candidates (`find_invariant`), not a single pass, since
// an instruction can look non-invariant only because it depends on a candidate that hasn't been
// confirmed yet.
//
// # Scope: one loop level per run
//
// Nested loops are handled by processing only the innermost natural loops on any given call —
// a loop is skipped if some other detected loop's block set is a strict subset of its own. A
// value hoisted out of an inner loop becomes a candidate for an enclosing outer loop only on a
// *subsequent* run of this pass; multi-level hoisting in one shot is unnecessary complexity this
// pass does not take on.
//
// # Preheader synthesis
//
// For an innermost loop with header `H`, the predecessors of `H` that are not themselves part
// of the loop ("outside" predecessors) are where hoisted code must run. If there is exactly one
// such predecessor and its only successor is `H` (an already-existing, clean preheader), the
// hoisted instructions are appended to it directly — no new block. Otherwise a new block is
// synthesized, holding the hoisted instructions and a plain `Br(H)` terminator, and every
// outside predecessor's terminator is redirected from `H` to the new block (edges from inside
// the loop, i.e. the back edge, are left pointing at `H` unchanged). A loop with no outside
// predecessor at all (unreachable in practice, but not asserted against) is left alone rather
// than risk fabricating a nonexistent entry into the loop.
//
// Moving an instruction keeps its relative order: the hoisted set, which was already in valid
// dependency order in the original code, is appended to the preheader in the same relative order
// its members had in the loop, which is sufficient without recomputing a fresh topological sort.
// Rebuilding the function (inserting a block, relocating instructions) requires renumbering both
// arenas from scratch, in final block-then-instruction order, the same append-only technique
// `ssa.rs`/`dce.rs` use for the same reason — this keeps every instruction's `InstId` a plain
// arena position again once the shuffling is done, and keeps a hoisted preheader's own
// instructions numbered *before* the loop body that still refers to them.
//
// # What this pass does not do
//
// This never reduces the static instruction count — it only relocates code, so by itself it
// looks like a no-op to anything counting instructions (unlike constant folding or DCE). Its
// entire benefit is dynamic: a value recomputed on every iteration is instead computed once,
// which only shows up in how much work actually runs at execution time, not in the printed
// program's size.

use std::collections::{BTreeMap, BTreeSet};

use basalt_bir::{Block, BlockId, Function, Inst, InstId, Module, Op, Term, ValRef};

use crate::dom::{detect_loops, Dominators};

/// Hoists loop-invariant, side-effect-free instructions out of every innermost natural loop in
/// every function of `module`. See this module's header for exactly what qualifies and the
/// one-level-per-run scope limit.
pub fn licm(module: &Module) -> Module {
    Module {
        funcs: module.funcs.iter().map(licm_fn).collect(),
        launch_bounds: module.launch_bounds,
        shared_mem_bytes: module.shared_mem_bytes,
        target_dtypes: module.target_dtypes.clone(),
    }
}

/// Every direct successor block of a terminator, in the order `Term` names them. Independent
/// copy of the same helper in `dom.rs`/`dce.rs` — this module intentionally does not reach into
/// either's private internals.
fn successors(term: &Term) -> Vec<BlockId> {
    match term {
        Term::Br(b) => vec![*b],
        Term::CondBr(_, t, e) => vec![*t, *e],
        Term::Switch(_, default, cases) => {
            let mut out = vec![*default];
            out.extend(cases.iter().map(|(_, b)| *b));
            out
        }
        Term::Ret(_) => Vec::new(),
    }
}

/// Predecessor lists for every block in `f`, indexed by `BlockId`. An edge appears at most once
/// even if a terminator names the same successor twice.
fn predecessors(f: &Function) -> Vec<Vec<BlockId>> {
    let n = f.blocks.len();
    let mut preds = vec![Vec::new(); n];
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        for succ in successors(&block.term) {
            let into = &mut preds[succ.0 as usize];
            if !into.contains(&from) {
                into.push(from);
            }
        }
    }
    preds
}

/// The block that owns each instruction, indexed by `InstId`.
fn owner_blocks(f: &Function) -> Vec<Option<BlockId>> {
    let mut owner = vec![None; f.insts.len()];
    for (idx, block) in f.blocks.iter().enumerate() {
        for &id in &block.insts {
            owner[id.0 as usize] = Some(BlockId(idx as u32));
        }
    }
    owner
}

/// Whether `op` is a pure, side-effect-free op eligible for hoisting at all — the set from this
/// module's header, before operand invariance is even considered.
fn is_hoistable_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Bin(..)
            | Op::ICmp(..)
            | Op::FCmp(..)
            | Op::Cast(..)
            | Op::Select(..)
            | Op::ConstInt(_)
            | Op::ConstFloat(_)
            | Op::TidX
            | Op::TidY
            | Op::TidZ
            | Op::BidX
            | Op::BidY
            | Op::BidZ
            | Op::BdimX
            | Op::BdimY
            | Op::BdimZ
            | Op::GdimX
            | Op::GdimY
            | Op::GdimZ
    )
}

/// Operands of a hoistable op, in the same order `is_hoistable_op` would need to check. Only
/// ever called on an op `is_hoistable_op` already accepted.
fn hoistable_operands(op: &Op) -> Vec<ValRef> {
    match op {
        Op::Bin(_, a, b) => vec![*a, *b],
        Op::ICmp(_, _, a, b) => vec![*a, *b],
        Op::FCmp(_, _, a, b) => vec![*a, *b],
        Op::Select(c, a, b) => vec![*c, *a, *b],
        Op::Cast(_, _, v) => vec![*v],
        Op::ConstInt(_) | Op::ConstFloat(_) => Vec::new(),
        Op::TidX
        | Op::TidY
        | Op::TidZ
        | Op::BidX
        | Op::BidY
        | Op::BidZ
        | Op::BdimX
        | Op::BdimY
        | Op::BdimZ
        | Op::GdimX
        | Op::GdimY
        | Op::GdimZ => Vec::new(),
        _ => unreachable!("hoistable_operands called on a non-hoistable op"),
    }
}

/// Fixed-point invariance scan over `blocks` (one loop's block set): starts from every
/// hoistable-op candidate owned by a block in `blocks`, and repeatedly admits any candidate
/// whose operands are all either a parameter, defined outside `blocks` entirely, or already
/// admitted, until nothing new is admitted.
fn find_invariant(
    f: &Function,
    blocks: &BTreeSet<BlockId>,
    owner: &[Option<BlockId>],
) -> BTreeSet<InstId> {
    let mut candidates: Vec<InstId> = Vec::new();
    for &b in blocks {
        for &id in &f.blocks[b.0 as usize].insts {
            if is_hoistable_op(&f.insts[id.0 as usize].op) {
                candidates.push(id);
            }
        }
    }

    let mut invariant: BTreeSet<InstId> = BTreeSet::new();
    loop {
        let mut changed = false;
        for &id in &candidates {
            if invariant.contains(&id) {
                continue;
            }
            let op = &f.insts[id.0 as usize].op;
            let ok = hoistable_operands(op).into_iter().all(|v| match v {
                ValRef::Param(_) => true,
                ValRef::Val(oid) => match owner[oid.0 as usize] {
                    Some(ob) if blocks.contains(&ob) => invariant.contains(&oid),
                    _ => true,
                },
            });
            if ok {
                invariant.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    invariant
}

/// Where a loop's hoisted instructions land.
enum Target {
    /// Append to this already-existing block, whose only successor is the loop header.
    Reuse(BlockId),
    /// Synthesize a fresh block; every block in `HoistPlan::outside` gets its edge to the
    /// header redirected to it.
    Synthesize,
}

/// One loop's worth of hoisting: which instructions move, and where they land.
struct HoistPlan {
    header: BlockId,
    outside: Vec<BlockId>,
    target: Target,
    hoist_ids: Vec<InstId>,
}

fn licm_fn(f: &Function) -> Function {
    if f.blocks.is_empty() {
        return f.clone();
    }

    let doms = Dominators::compute(f);
    let raw_loops = detect_loops(f, &doms);
    if raw_loops.is_empty() {
        return f.clone();
    }

    // Merge loops that share a header (one per back edge from `detect_loops`) into one block
    // set per header, since a preheader is synthesized per header, not per back edge.
    let mut merged: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
    for l in &raw_loops {
        merged
            .entry(l.header)
            .or_default()
            .extend(l.blocks.iter().copied());
    }

    // Innermost only: drop any loop whose block set strictly contains another detected loop's.
    let headers: Vec<BlockId> = merged.keys().copied().collect();
    let innermost: Vec<(BlockId, BTreeSet<BlockId>)> = headers
        .iter()
        .copied()
        .filter(|&h| {
            !headers.iter().any(|&h2| {
                h2 != h
                    && merged[&h2].len() < merged[&h].len()
                    && merged[&h2].is_subset(&merged[&h])
            })
        })
        .map(|h| (h, merged[&h].clone()))
        .collect();

    if innermost.is_empty() {
        return f.clone();
    }

    let preds = predecessors(f);
    let owner = owner_blocks(f);

    let mut plans: Vec<HoistPlan> = Vec::new();
    for (header, blocks) in &innermost {
        let invariant = find_invariant(f, blocks, &owner);
        if invariant.is_empty() {
            continue;
        }

        let outside: Vec<BlockId> = preds[header.0 as usize]
            .iter()
            .copied()
            .filter(|p| !blocks.contains(p))
            .collect();
        if outside.is_empty() {
            continue;
        }

        let target =
            if outside.len() == 1 && successors(&f.blocks[outside[0].0 as usize].term).len() == 1 {
                Target::Reuse(outside[0])
            } else {
                Target::Synthesize
            };

        let mut hoist_ids: Vec<InstId> = Vec::new();
        for &b in blocks {
            for &id in &f.blocks[b.0 as usize].insts {
                if invariant.contains(&id) {
                    hoist_ids.push(id);
                }
            }
        }

        plans.push(HoistPlan {
            header: *header,
            outside,
            target,
            hoist_ids,
        });
    }

    if plans.is_empty() {
        return f.clone();
    }

    rebuild(f, &plans, &owner)
}

/// Resolves what old block `target` becomes when reached from old block `pred`: the new
/// preheader if `(pred, target)` is a redirected edge, otherwise `target`'s ordinary new id.
fn resolve_target(
    pred: BlockId,
    target: BlockId,
    redirect: &BTreeSet<(BlockId, BlockId)>,
    preheader_new_id: &BTreeMap<BlockId, BlockId>,
    block_map: &BTreeMap<BlockId, BlockId>,
) -> BlockId {
    if redirect.contains(&(pred, target)) {
        preheader_new_id[&target]
    } else {
        block_map[&target]
    }
}

/// Resolves what a phi's incoming-edge key `pred` becomes, given the phi's own (never-moved)
/// owning block `header`: this is the mirror image of `resolve_target` — a phi's incoming pair
/// names the PREDECESSOR the value arrives from, not a forward branch target, so redirecting the
/// edge `pred -> header` through a synthesized preheader means the value now arrives FROM that
/// preheader (its new id), not from `pred`'s own renumbered identity.
fn resolve_phi_pred(
    pred: BlockId,
    header: BlockId,
    redirect: &BTreeSet<(BlockId, BlockId)>,
    preheader_new_id: &BTreeMap<BlockId, BlockId>,
    block_map: &BTreeMap<BlockId, BlockId>,
) -> BlockId {
    if redirect.contains(&(pred, header)) {
        preheader_new_id[&header]
    } else {
        block_map[&pred]
    }
}

/// Rewrites `term`'s block targets (only), as seen from old block `pred`, to their final ids.
/// Any `ValRef` operand is left as-is — that gets remapped once the instruction arena itself is
/// renumbered.
fn remap_term_targets(
    term: &Term,
    pred: BlockId,
    redirect: &BTreeSet<(BlockId, BlockId)>,
    preheader_new_id: &BTreeMap<BlockId, BlockId>,
    block_map: &BTreeMap<BlockId, BlockId>,
) -> Term {
    let map = |t: BlockId| resolve_target(pred, t, redirect, preheader_new_id, block_map);
    match term {
        Term::Br(b) => Term::Br(map(*b)),
        Term::CondBr(c, t, e) => Term::CondBr(*c, map(*t), map(*e)),
        Term::Switch(v, default, cases) => Term::Switch(
            *v,
            map(*default),
            cases.iter().map(|(val, b)| (*val, map(*b))).collect(),
        ),
        Term::Ret(v) => Term::Ret(*v),
    }
}

fn remap_valref(v: ValRef, inst_map: &BTreeMap<InstId, InstId>) -> ValRef {
    match v {
        ValRef::Param(p) => ValRef::Param(p),
        ValRef::Val(id) => ValRef::Val(
            *inst_map
                .get(&id)
                .expect("every operand of a kept instruction must itself have been renumbered"),
        ),
    }
}

/// Rewrites `term`'s `ValRef` operand (only) via `inst_map`; block targets are already final by
/// this point (`remap_term_targets` ran during block assembly).
fn remap_term_values(term: &Term, inst_map: &BTreeMap<InstId, InstId>) -> Term {
    match term {
        Term::Br(b) => Term::Br(*b),
        Term::CondBr(c, t, e) => Term::CondBr(remap_valref(*c, inst_map), *t, *e),
        Term::Switch(v, default, cases) => {
            Term::Switch(remap_valref(*v, inst_map), *default, cases.clone())
        }
        Term::Ret(v) => Term::Ret(v.map(|v| remap_valref(v, inst_map))),
    }
}

/// Rebuilds `op` with every `ValRef` operand passed through `f`, in the same structural order
/// `hoistable_operands`/the printer walk. `Op::Phi` is handled separately by the caller (its
/// incoming pairs also need their predecessor block ids remapped), so it passes through here
/// with only its values remapped.
fn map_op(op: &Op, mut f: impl FnMut(ValRef) -> ValRef) -> Op {
    match op {
        Op::ConstInt(v) => Op::ConstInt(*v),
        Op::ConstFloat(v) => Op::ConstFloat(*v),
        Op::Bin(o, a, b) => Op::Bin(*o, f(*a), f(*b)),
        Op::ICmp(p, t, a, b) => Op::ICmp(*p, *t, f(*a), f(*b)),
        Op::FCmp(p, t, a, b) => Op::FCmp(*p, *t, f(*a), f(*b)),
        Op::Select(c, a, b) => Op::Select(f(*c), f(*a), f(*b)),
        Op::Cast(c, t, v) => Op::Cast(*c, *t, f(*v)),
        Op::Load {
            ptr,
            space,
            align,
            volatile,
        } => Op::Load {
            ptr: f(*ptr),
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Store {
            ptr,
            val,
            ty,
            space,
            align,
            volatile,
        } => Op::Store {
            ptr: f(*ptr),
            val: f(*val),
            ty: *ty,
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Phi(incoming) => Op::Phi(incoming.iter().map(|(bb, v)| (*bb, f(*v))).collect()),
        Op::TidX => Op::TidX,
        Op::TidY => Op::TidY,
        Op::TidZ => Op::TidZ,
        Op::BidX => Op::BidX,
        Op::BidY => Op::BidY,
        Op::BidZ => Op::BidZ,
        Op::BdimX => Op::BdimX,
        Op::BdimY => Op::BdimY,
        Op::BdimZ => Op::BdimZ,
        Op::GdimX => Op::GdimX,
        Op::GdimY => Op::GdimY,
        Op::GdimZ => Op::GdimZ,
        Op::Barrier => Op::Barrier,
        Op::Shuffle(k, a, b) => Op::Shuffle(*k, f(*a), f(*b)),
        Op::Ballot(a) => Op::Ballot(f(*a)),
        Op::VoteAny(a) => Op::VoteAny(f(*a)),
        Op::VoteAll(a) => Op::VoteAll(f(*a)),
        Op::Atomic(o, ptr, v, s) => Op::Atomic(*o, f(*ptr), f(*v), *s),
        Op::AtomicCas(ptr, cmp, new, s) => Op::AtomicCas(f(*ptr), f(*cmp), f(*new), *s),
        Op::Mma {
            a,
            b,
            c,
            d,
            m,
            n,
            k,
            in_dtype,
            acc_dtype,
            layout_a,
            layout_b,
        } => Op::Mma {
            a: f(*a),
            b: f(*b),
            c: f(*c),
            d: f(*d),
            m: *m,
            n: *n,
            k: *k,
            in_dtype: *in_dtype,
            acc_dtype: *acc_dtype,
            layout_a: *layout_a,
            layout_b: *layout_b,
        },
    }
}

/// Remaps one instruction's operands to their final ids. `Op::Phi`'s incoming predecessor block
/// ids need the same redirect-aware treatment a terminator's targets get — `owner_block` is
/// this phi's own (never-moved) original owning block, standing in for "what did this
/// predecessor used to branch to".
fn remap_op(
    op: &Op,
    inst_map: &BTreeMap<InstId, InstId>,
    owner_block: Option<BlockId>,
    redirect: &BTreeSet<(BlockId, BlockId)>,
    preheader_new_id: &BTreeMap<BlockId, BlockId>,
    block_map: &BTreeMap<BlockId, BlockId>,
) -> Op {
    match op {
        Op::Phi(incoming) => {
            let owner_block = owner_block.expect("a phi's original owning block must be known");
            Op::Phi(
                incoming
                    .iter()
                    .map(|(pred, v)| {
                        let new_pred = resolve_phi_pred(
                            *pred,
                            owner_block,
                            redirect,
                            preheader_new_id,
                            block_map,
                        );
                        (new_pred, remap_valref(*v, inst_map))
                    })
                    .collect(),
            )
        }
        _ => map_op(op, |v| remap_valref(v, inst_map)),
    }
}

/// Applies every plan in `plans` to `f`: relocates hoisted instructions, synthesizes preheaders
/// where needed, redirects the affected edges, and renumbers both arenas from scratch in final
/// block-then-instruction order (block ids are never removed here, only inserted, so this is
/// simpler than `dce.rs`'s equivalent rebuild, which also has to drop unreachable blocks).
fn rebuild(f: &Function, plans: &[HoistPlan], owner: &[Option<BlockId>]) -> Function {
    let old_len = f.blocks.len() as u32;

    let mut hoist_ids_by_header: BTreeMap<BlockId, Vec<InstId>> = BTreeMap::new();
    let mut appended_to_block: BTreeMap<BlockId, Vec<InstId>> = BTreeMap::new();
    let mut synth_headers: BTreeSet<BlockId> = BTreeSet::new();
    let mut redirect: BTreeSet<(BlockId, BlockId)> = BTreeSet::new();
    let mut removed: BTreeSet<InstId> = BTreeSet::new();

    for plan in plans {
        removed.extend(plan.hoist_ids.iter().copied());
        match plan.target {
            Target::Reuse(b) => {
                appended_to_block
                    .entry(b)
                    .or_default()
                    .extend(plan.hoist_ids.iter().copied());
            }
            Target::Synthesize => {
                synth_headers.insert(plan.header);
                for &p in &plan.outside {
                    redirect.insert((p, plan.header));
                }
            }
        }
        hoist_ids_by_header.insert(plan.header, plan.hoist_ids.clone());
    }

    // Pass 1: assign final block ids. A synthesized preheader lands immediately before its
    // header, both in the block vector and (since pass 3 walks blocks in this same order) in
    // the instruction arena — so anything it hoists still gets a lower id than the loop body
    // that uses it, exactly as if it had always been written that way.
    let mut block_map: BTreeMap<BlockId, BlockId> = BTreeMap::new();
    let mut preheader_new_id: BTreeMap<BlockId, BlockId> = BTreeMap::new();
    let mut next: u32 = 0;
    for i in 0..old_len {
        let b = BlockId(i);
        if synth_headers.contains(&b) {
            preheader_new_id.insert(b, BlockId(next));
            next += 1;
        }
        block_map.insert(b, BlockId(next));
        next += 1;
    }

    // Pass 2: stage the final block list (final ids and terminator targets, still-old
    // instruction ids).
    let mut staged: Vec<Block> = Vec::with_capacity(next as usize);
    for i in 0..old_len {
        let b = BlockId(i);
        if synth_headers.contains(&b) {
            staged.push(Block {
                insts: hoist_ids_by_header[&b].clone(),
                term: Term::Br(block_map[&b]),
            });
        }

        let old_block = &f.blocks[i as usize];
        let mut insts: Vec<InstId> = old_block
            .insts
            .iter()
            .copied()
            .filter(|id| !removed.contains(id))
            .collect();
        if let Some(extra) = appended_to_block.get(&b) {
            insts.extend(extra.iter().copied());
        }
        let term = remap_term_targets(&old_block.term, b, &redirect, &preheader_new_id, &block_map);
        staged.push(Block { insts, term });
    }

    // Pass 3: renumber the instruction arena to match this final layout. Two stages, mirroring
    // `ssa.rs`'s own two-stage build (see that module's header): a `phi` can reference a value
    // defined later in the arena across a back edge, so every old id's final `InstId` must be
    // known (stage one) before any operand gets remapped (stage two) — remapping operands
    // interleaved with numbering would panic on exactly that forward reference.
    let mut inst_map: BTreeMap<InstId, InstId> = BTreeMap::new();
    let mut block_ids: Vec<Vec<InstId>> = Vec::with_capacity(staged.len());
    for staged_block in &staged {
        let mut kept: Vec<InstId> = Vec::with_capacity(staged_block.insts.len());
        for &old_id in &staged_block.insts {
            let new_id = InstId(inst_map.len() as u32);
            inst_map.insert(old_id, new_id);
            kept.push(new_id);
        }
        block_ids.push(kept);
    }

    let mut new_insts: Vec<Inst> = Vec::with_capacity(inst_map.len());
    let mut final_blocks: Vec<Block> = Vec::with_capacity(staged.len());
    for (staged_block, kept) in staged.iter().zip(block_ids) {
        for &old_id in &staged_block.insts {
            let old_inst = &f.insts[old_id.0 as usize];
            let new_op = remap_op(
                &old_inst.op,
                &inst_map,
                owner[old_id.0 as usize],
                &redirect,
                &preheader_new_id,
                &block_map,
            );
            new_insts.push(Inst {
                ty: old_inst.ty,
                op: new_op,
            });
        }
        let term = remap_term_values(&staged_block.term, &inst_map);
        final_blocks.push(Block { insts: kept, term });
    }

    Function {
        name: f.name.clone(),
        is_kernel: f.is_kernel,
        params: f.params.clone(),
        ret: f.ret,
        blocks: final_blocks,
        insts: new_insts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_backend::Backend;
    use basalt_bir::{BinOp, ICmpPred, Scalar, Ty};

    fn i32ty() -> Ty {
        Ty::Scalar(Scalar::I32)
    }

    fn func(name: &str, params: Vec<Ty>, blocks: Vec<Block>, insts: Vec<Inst>) -> Function {
        Function {
            name: name.to_string(),
            is_kernel: true,
            params,
            ret: Ty::Void,
            blocks,
            insts,
        }
    }

    fn module_of(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    /// Which block owns instruction `id` in `f`.
    fn owning_block(f: &Function, id: InstId) -> BlockId {
        f.blocks
            .iter()
            .enumerate()
            .find_map(|(i, b)| b.insts.contains(&id).then_some(BlockId(i as u32)))
            .expect("instruction must live in exactly one block")
    }

    fn find_op(f: &Function, pred: impl Fn(&Op) -> bool) -> Option<InstId> {
        f.insts
            .iter()
            .enumerate()
            .find(|(_, inst)| pred(&inst.op))
            .map(|(i, _)| InstId(i as u32))
    }

    /// `bb0 -> bb1 (header) -> bb2 (body/latch, condbr back to bb1 or forward to bb3) -> bb3`.
    /// `bb0`'s only successor is `bb1`, so it is a clean, already-existing preheader.
    /// Params: (a, b, n). `bb2` computes an invariant `a + b`, a variant `inv + i` (uses the
    /// phi), the step constant, `i_next = i + 1`, and the loop condition `i_next < n`.
    fn single_pred_loop() -> Function {
        let params = vec![i32ty(), i32ty(), i32ty()];
        let mut insts = Vec::new();

        // bb0
        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(0),
        });
        let i_init = InstId(0);

        // bb1: phi i = [bb0 -> i_init, bb2 -> i_next]; i_next is InstId(5), assigned below.
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Phi(vec![
                (BlockId(0), ValRef::Val(i_init)),
                (BlockId(2), ValRef::Val(InstId(5))),
            ]),
        });
        let i_phi = InstId(1);

        // bb2
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        });
        let inv = InstId(2);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(inv), ValRef::Val(i_phi)),
        });
        let variant = InstId(3);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(1),
        });
        let step = InstId(4);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(i_phi), ValRef::Val(step)),
        });
        let i_next = InstId(5);
        assert_eq!(i_next, InstId(5));
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(
                ICmpPred::Slt,
                i32ty(),
                ValRef::Val(i_next),
                ValRef::Param(2),
            ),
        });
        let cond = InstId(6);
        let _ = variant;

        let blocks = vec![
            Block {
                insts: vec![i_init],
                term: Term::Br(BlockId(1)),
            },
            Block {
                insts: vec![i_phi],
                term: Term::Br(BlockId(2)),
            },
            Block {
                insts: vec![inv, InstId(3), step, i_next, cond],
                term: Term::CondBr(ValRef::Val(cond), BlockId(1), BlockId(3)),
            },
            Block {
                insts: Vec::new(),
                term: Term::Ret(None),
            },
        ];

        func("single_pred_loop", params, blocks, insts)
    }

    #[test]
    fn invariant_expression_is_hoisted_into_an_existing_clean_preheader() {
        let f = single_pred_loop();
        let out = licm_fn(&f);

        // Same block count: bb0 was already a clean preheader, so no new block is synthesized.
        assert_eq!(out.blocks.len(), f.blocks.len());

        let inv_id = find_op(&out, |op| {
            matches!(op, Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)))
        })
        .expect("the invariant a + b must still exist somewhere");
        assert_eq!(
            owning_block(&out, inv_id),
            BlockId(0),
            "a + b must have moved into bb0, the existing preheader"
        );

        // The loop body no longer contains a `Bin` whose operands are both raw parameters.
        let doms = Dominators::compute(&out);
        let loops = detect_loops(&out, &doms);
        assert_eq!(loops.len(), 1);
        assert!(!loops[0].blocks.contains(&owning_block(&out, inv_id)));

        // The variant `inv + i` must still be inside the loop, referencing the (moved) inv.
        let variant_id = find_op(
            &out,
            |op| matches!(op, Op::Bin(BinOp::Add, ValRef::Val(v), _) if *v == inv_id),
        )
        .expect("inv + i must still exist, now referencing the hoisted inv");
        assert!(loops[0].blocks.contains(&owning_block(&out, variant_id)));

        let text = basalt_bir::print(&module_of(out.clone()));
        let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
        assert_eq!(reparsed, module_of(out), "parse(print(licm(m))) != licm(m)");
    }

    /// `bb0` condbr's to `bb1a`/`bb1b`, both of which br unconditionally into `bb2` (the
    /// header) — two outside predecessors, so a preheader must be synthesized and both edges
    /// redirected to it.
    fn two_pred_loop() -> Function {
        let params = vec![i32ty(), i32ty(), i32ty()];
        let mut insts = Vec::new();

        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(0),
        });
        let split_cond = InstId(0);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(0),
        });
        let i_init = InstId(1);

        // bb3 (header): phi
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Phi(vec![
                (BlockId(2), ValRef::Val(i_init)),
                (BlockId(4), ValRef::Val(InstId(6))),
            ]),
        });
        let i_phi = InstId(2);

        // bb4 (body/latch)
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Mul, ValRef::Param(0), ValRef::Param(1)),
        });
        let inv = InstId(3);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(inv), ValRef::Val(i_phi)),
        });
        let variant = InstId(4);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(1),
        });
        let step = InstId(5);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(i_phi), ValRef::Val(step)),
        });
        let i_next = InstId(6);
        assert_eq!(i_next, InstId(6));
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(
                ICmpPred::Slt,
                i32ty(),
                ValRef::Val(i_next),
                ValRef::Param(2),
            ),
        });
        let cond = InstId(7);
        let _ = variant;

        let blocks = vec![
            Block {
                insts: vec![split_cond],
                term: Term::CondBr(ValRef::Val(split_cond), BlockId(1), BlockId(2)),
            },
            Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![i_init],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![i_phi],
                term: Term::Br(BlockId(4)),
            },
            Block {
                insts: vec![inv, InstId(4), step, i_next, cond],
                term: Term::CondBr(ValRef::Val(cond), BlockId(3), BlockId(5)),
            },
            Block {
                insts: Vec::new(),
                term: Term::Ret(None),
            },
        ];

        func("two_pred_loop", params, blocks, insts)
    }

    #[test]
    fn multiple_outside_predecessors_get_a_synthesized_preheader() {
        let f = two_pred_loop();
        let before_blocks = f.blocks.len();
        let out = licm_fn(&f);

        assert_eq!(
            out.blocks.len(),
            before_blocks + 1,
            "exactly one new preheader block must appear"
        );

        let inv_id = find_op(&out, |op| {
            matches!(op, Op::Bin(BinOp::Mul, ValRef::Param(0), ValRef::Param(1)))
        })
        .expect("the invariant a * b must still exist somewhere");

        let doms = Dominators::compute(&out);
        let loops = detect_loops(&out, &doms);
        assert_eq!(loops.len(), 1);
        let inv_block = owning_block(&out, inv_id);
        assert!(
            !loops[0].blocks.contains(&inv_block),
            "a * b must have been hoisted out of the loop"
        );

        // The header now has exactly one predecessor from *outside* the loop: the synthesized
        // preheader (the other predecessor is the back edge, from inside the loop, which is
        // left pointing straight at the header). Both of the original two outside predecessors
        // must have been redirected to it, and it must have no other successor.
        let header_new = loops[0].header;
        let preds = predecessors(&out);
        let outside_preds: Vec<BlockId> = preds[header_new.0 as usize]
            .iter()
            .copied()
            .filter(|p| !loops[0].blocks.contains(p))
            .collect();
        assert_eq!(
            outside_preds.len(),
            1,
            "the header must have exactly one predecessor from outside the loop: the \
             synthesized preheader"
        );
        let preheader = outside_preds[0];
        assert_eq!(
            successors(&out.blocks[preheader.0 as usize].term),
            vec![header_new],
            "the synthesized preheader's only successor must be the header"
        );
        assert!(
            out.blocks[preheader.0 as usize].insts.contains(&inv_id),
            "the hoisted a * b must live in the synthesized preheader"
        );

        let text = basalt_bir::print(&module_of(out.clone()));
        let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
        assert_eq!(reparsed, module_of(out), "parse(print(licm(m))) != licm(m)");
    }

    /// Same shape as `single_pred_loop`, but the invariant `a + b` is combined with `BdimX`
    /// (transitively invariant) in one instruction, and `BdimX` is combined with the induction
    /// variable in another (never invariant, since it depends on the phi).
    fn loop_with_gpu_index() -> Function {
        let params = vec![i32ty(), i32ty(), i32ty()];
        let mut insts = Vec::new();

        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(0),
        });
        let i_init = InstId(0);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Phi(vec![
                (BlockId(0), ValRef::Val(i_init)),
                (BlockId(2), ValRef::Val(InstId(7))),
            ]),
        });
        let i_phi = InstId(1);

        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        });
        let inv = InstId(2);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::BdimX,
        });
        let bdim = InstId(3);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Mul, ValRef::Val(bdim), ValRef::Val(inv)),
        });
        let invariant_via_index = InstId(4);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Mul, ValRef::Val(bdim), ValRef::Val(i_phi)),
        });
        let variant_via_index = InstId(5);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(1),
        });
        let step = InstId(6);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(i_phi), ValRef::Val(step)),
        });
        let i_next = InstId(7);
        assert_eq!(i_next, InstId(7));
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(
                ICmpPred::Slt,
                i32ty(),
                ValRef::Val(i_next),
                ValRef::Param(2),
            ),
        });
        let cond = InstId(8);

        let blocks = vec![
            Block {
                insts: vec![i_init],
                term: Term::Br(BlockId(1)),
            },
            Block {
                insts: vec![i_phi],
                term: Term::Br(BlockId(2)),
            },
            Block {
                insts: vec![
                    inv,
                    bdim,
                    invariant_via_index,
                    variant_via_index,
                    step,
                    i_next,
                    cond,
                ],
                term: Term::CondBr(ValRef::Val(cond), BlockId(1), BlockId(3)),
            },
            Block {
                insts: Vec::new(),
                term: Term::Ret(None),
            },
        ];

        func("loop_with_gpu_index", params, blocks, insts)
    }

    #[test]
    fn gpu_index_op_hoists_transitively_but_only_the_truly_invariant_expression_moves() {
        let f = loop_with_gpu_index();
        let out = licm_fn(&f);

        let doms = Dominators::compute(&out);
        let loops = detect_loops(&out, &doms);
        assert_eq!(loops.len(), 1);

        let invariant_via_index_id = find_op(&out, |op| matches!(op, Op::Bin(BinOp::Mul, ValRef::Val(_), ValRef::Val(inv)) if {
            matches!(out.insts[inv.0 as usize].op, Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)))
        }))
        .expect("bdim.x * (a + b) must still exist");
        assert!(
            !loops[0]
                .blocks
                .contains(&owning_block(&out, invariant_via_index_id)),
            "bdim.x * (a + b) is transitively invariant and must be hoisted"
        );

        let bdim_id = find_op(&out, |op| matches!(op, Op::BdimX)).expect("bdim.x must still exist");
        assert!(
            !loops[0].blocks.contains(&owning_block(&out, bdim_id)),
            "bdim.x itself is always invariant and must be hoisted"
        );

        // The variant `bdim.x * i` must remain in the loop.
        let phi_id = find_op(&out, |op| matches!(op, Op::Phi(_)))
            .expect("the induction phi must still exist");
        let variant_via_index_id = find_op(&out, |op| {
            matches!(op, Op::Bin(BinOp::Mul, ValRef::Val(a), ValRef::Val(b)) if (*a == bdim_id || *b == bdim_id) && (*a == phi_id || *b == phi_id))
        })
        .expect("bdim.x * i must still exist");
        assert!(
            loops[0]
                .blocks
                .contains(&owning_block(&out, variant_via_index_id)),
            "bdim.x * i depends on the induction variable and must stay in the loop"
        );

        let text = basalt_bir::print(&module_of(out.clone()));
        let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
        assert_eq!(reparsed, module_of(out), "parse(print(licm(m))) != licm(m)");
    }

    /// Every instruction in the loop body depends, directly or transitively, on the induction
    /// variable — there is nothing to hoist, so the function must come back unchanged. The
    /// step is a parameter, not a `const.i`, since a bare constant is trivially invariant by
    /// this pass's own rules (see this module's header) and would otherwise give this fixture
    /// a hoist candidate after all.
    fn loop_with_no_invariant() -> Function {
        let params = vec![i32ty(), i32ty()];
        let mut insts = Vec::new();

        insts.push(Inst {
            ty: i32ty(),
            op: Op::ConstInt(0),
        });
        let i_init = InstId(0);
        insts.push(Inst {
            ty: i32ty(),
            op: Op::Phi(vec![
                (BlockId(0), ValRef::Val(i_init)),
                (BlockId(2), ValRef::Val(InstId(2))),
            ]),
        });
        let i_phi = InstId(1);

        insts.push(Inst {
            ty: i32ty(),
            op: Op::Bin(BinOp::Add, ValRef::Val(i_phi), ValRef::Param(1)),
        });
        let i_next = InstId(2);
        assert_eq!(i_next, InstId(2));
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(
                ICmpPred::Slt,
                i32ty(),
                ValRef::Val(i_next),
                ValRef::Param(0),
            ),
        });
        let cond = InstId(3);

        let blocks = vec![
            Block {
                insts: vec![i_init],
                term: Term::Br(BlockId(1)),
            },
            Block {
                insts: vec![i_phi],
                term: Term::Br(BlockId(2)),
            },
            Block {
                insts: vec![i_next, cond],
                term: Term::CondBr(ValRef::Val(cond), BlockId(1), BlockId(3)),
            },
            Block {
                insts: Vec::new(),
                term: Term::Ret(None),
            },
        ];

        func("loop_with_no_invariant", params, blocks, insts)
    }

    #[test]
    fn loop_with_nothing_invariant_is_left_completely_unchanged() {
        let f = loop_with_no_invariant();
        let out = licm_fn(&f);
        assert_eq!(
            out, f,
            "no candidate is invariant, so no preheader may be synthesized"
        );
    }

    #[test]
    fn licm_over_a_module_preserves_function_metadata() {
        let f = single_pred_loop();
        let m = module_of(f);
        let out = licm(&m);
        assert_eq!(out.funcs.len(), 1);
        assert_eq!(out.funcs[0].name, "single_pred_loop");
        assert_eq!(out.funcs[0].params, m.funcs[0].params);
        assert_eq!(out.funcs[0].ret, m.funcs[0].ret);
    }

    // # Real-pipeline proof
    //
    // Everything above builds `basalt_bir` fixtures directly, to pin down exact block/operand
    // shapes. This section instead runs a real CUDA-C-subset kernel through the actual
    // lex/preprocess/parse -> sema check/lower -> SSA construction pipeline (mirroring how
    // this pass would really be reached: SSA construction, then LICM, since a raw lowering's
    // slot-based `load`/`store` traffic has nothing pure enough to hoist until it has been
    // promoted to real SSA values first), confirms the transform is structurally sound and
    // round-trips, and then proves it by execution: the pre-LICM and post-LICM modules are
    // both emitted through the real x86-64 oracle, linked against the same C shim via the
    // system compiler, and run — their observable output must match exactly.

    const LICM_PROBE_SRC: &str = r#"
    __global__ void licm_probe(int *out, int a, int b, int n) {
        for (int i = 0; i < n; i = i + 1) {
            int inv = a * b;
            out[i] = inv + i;
        }
    }
    "#;

    /// `licm_probe`'s params are `(ptr.global, i32, i32, i32)`: `out` in `rdi`, `a` in `esi`,
    /// `b` in `edx`, `n` in `ecx`, all integer-class, so the oracle's trailing `nthreads`
    /// lands in `r8` — see `basalt-x86`'s `link_and_run.rs` for the same convention spelled
    /// out in full. `nthreads` is 1: this kernel never reads `tid.x`/`blockIdx.x`, so a single
    /// logical thread running the whole `for` loop once is exactly what it means to execute.
    const LICM_PROBE_SHIM_C: &str = r#"
#include <stdint.h>
#include <stdio.h>

extern void licm_probe(int32_t *out, int32_t a, int32_t b, int32_t n, int64_t nthreads);

int main(void) {
    int32_t out[8];
    for (int i = 0; i < 8; i++) {
        out[i] = -1;
    }
    licm_probe(out, 6, 7, 8, 1);
    for (int i = 0; i < 8; i++) {
        int32_t expected = 6 * 7 + i;
        if (out[i] != expected) {
            fprintf(stderr, "FAIL at %d: expected %d got %d\n", i, expected, out[i]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
"#;

    fn cc_available() -> bool {
        std::process::Command::new("cc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn run_cc(args: &[&std::ffi::OsStr]) {
        let out = std::process::Command::new("cc")
            .args(args)
            .output()
            .expect("cc is present and spawns");
        assert!(
            out.status.success(),
            "cc {args:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn real_pipeline_hoists_invariant_and_execution_still_matches_before_licm() {
        let (tokens, pp_errors) =
            basalt_frontend_c::preprocess(LICM_PROBE_SRC, &basalt_frontend_c::PpOpts::default());
        assert!(pp_errors.is_empty(), "preprocess errors: {pp_errors:?}");
        let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
        assert!(parse_errors.is_empty(), "parse errors: {parse_errors:?}");
        let sema_diags = basalt_sema::check(&tu);
        assert!(sema_diags.is_empty(), "sema diagnostics: {sema_diags:?}");
        let (module, lower_diags) = basalt_sema::lower(&tu);
        assert!(
            lower_diags.is_empty(),
            "lowering diagnostics: {lower_diags:?}"
        );

        let ssa_module = crate::ssa::construct_ssa(&module);

        // `out` is param 0, so `a` is param 1 and `b` is param 2; the multiply may come out as
        // either operand order depending on how the frontend walks the expression.
        let is_a_times_b = |op: &Op| {
            matches!(
                op,
                Op::Bin(BinOp::Mul, ValRef::Param(1), ValRef::Param(2))
                    | Op::Bin(BinOp::Mul, ValRef::Param(2), ValRef::Param(1))
            )
        };

        let ssa_f = &ssa_module.funcs[0];
        let mul_before = find_op(ssa_f, is_a_times_b)
            .expect("lowering + SSA construction must produce a direct a * b multiply");
        let doms_before = Dominators::compute(ssa_f);
        let loops_before = detect_loops(ssa_f, &doms_before);
        assert!(
            loops_before
                .iter()
                .any(|l| l.blocks.contains(&owning_block(ssa_f, mul_before))),
            "fixture assumption: a * b must start out inside the for loop:\n{}",
            basalt_bir::print(&ssa_module)
        );

        let licm_module = licm(&ssa_module);
        let licm_f = &licm_module.funcs[0];
        let mul_after = find_op(licm_f, is_a_times_b).expect("a * b must still exist after licm");
        let doms_after = Dominators::compute(licm_f);
        let loops_after = detect_loops(licm_f, &doms_after);
        assert!(
            !loops_after
                .iter()
                .any(|l| l.blocks.contains(&owning_block(licm_f, mul_after))),
            "a * b must have been hoisted out of every loop:\n{}",
            basalt_bir::print(&licm_module)
        );

        let text = basalt_bir::print(&licm_module);
        let reparsed = basalt_bir::parse(&text).expect("parse(print(m)) must parse");
        assert_eq!(
            reparsed, licm_module,
            "parse(print(licm(m))) != licm(m) on a real lowering"
        );

        if !cc_available() {
            eprintln!(
                "skipping real_pipeline_hoists_invariant_and_execution_still_matches_before_licm's \
                 execution proof: `cc` not found"
            );
            return;
        }

        assert_eq!(
            basalt_x86::X86Oracle.supports(&ssa_module),
            basalt_backend::Support::Supported
        );
        assert_eq!(
            basalt_x86::X86Oracle.supports(&licm_module),
            basalt_backend::Support::Supported
        );

        let before_bytes = basalt_x86::X86Oracle
            .emit(&ssa_module, &basalt_backend::EmitOpts::default())
            .expect("oracle emit succeeds before licm")
            .as_bytes()
            .expect("oracle emits an object payload")
            .to_vec();
        let after_bytes = basalt_x86::X86Oracle
            .emit(&licm_module, &basalt_backend::EmitOpts::default())
            .expect("oracle emit succeeds after licm")
            .as_bytes()
            .expect("oracle emits an object payload")
            .to_vec();

        let pid = std::process::id();
        let scratch = std::env::temp_dir();
        let shim_c = scratch.join(format!("basalt_licm_shim_{pid}.c"));
        std::fs::write(&shim_c, LICM_PROBE_SHIM_C).expect("writing shim source");
        let shim_o = scratch.join(format!("basalt_licm_shim_{pid}.o"));
        run_cc(&[
            std::ffi::OsStr::new("-c"),
            shim_c.as_os_str(),
            std::ffi::OsStr::new("-o"),
            shim_o.as_os_str(),
        ]);

        let mut outputs = Vec::new();
        for (tag, bytes) in [("before", &before_bytes), ("after", &after_bytes)] {
            let obj = scratch.join(format!("basalt_licm_{tag}_{pid}.o"));
            std::fs::write(&obj, bytes).expect("writing payload object");
            let exe = scratch.join(format!("basalt_licm_{tag}_{pid}"));
            run_cc(&[
                shim_o.as_os_str(),
                obj.as_os_str(),
                std::ffi::OsStr::new("-o"),
                exe.as_os_str(),
            ]);
            let out = std::process::Command::new(&exe)
                .output()
                .expect("built executable runs");
            outputs.push(out);
            let _ = std::fs::remove_file(&obj);
            let _ = std::fs::remove_file(&exe);
        }
        let _ = std::fs::remove_file(&shim_o);
        let _ = std::fs::remove_file(&shim_c);

        assert!(
            outputs[0].status.success(),
            "pre-licm binary failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&outputs[0].stdout),
            String::from_utf8_lossy(&outputs[0].stderr)
        );
        assert!(
            outputs[1].status.success(),
            "post-licm binary failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&outputs[1].stdout),
            String::from_utf8_lossy(&outputs[1].stderr)
        );
        assert_eq!(
            outputs[0].stdout, outputs[1].stdout,
            "licm must not change the kernel's observable output"
        );
    }
}
