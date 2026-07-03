// Dead-code elimination: a BIR-to-BIR pass that drops instructions whose result is never
// observed and blocks that can never execute.
//
// Two independent sweeps, both anchored at block 0 (the entry block):
//
//   1. Block reachability. A block not reachable from the entry via `Term` edges can never
//      run, so every instruction inside it is moot regardless of side effects — it is
//      dropped wholesale, not run through instruction-liveness at all.
//   2. Instruction liveness, mark-and-sweep over the reachable blocks only. A handful of ops
//      are "necessarily live" roots because they have an effect beyond their own SSA result
//      (or, for `load`, because a `volatile` one's effect *is* the read itself, e.g. a
//      hardware register, which must never be assumed elidable just because nothing
//      consumes the loaded value): `store`, `barrier`, `atomic`/`atomic.cas`, `mma` (it
//      writes through its `d` operand and produces no SSA result of its own to be "unused"),
//      and any `volatile` `load`. A block's own `Term` operand (a `condbr` condition, a `switch`
//      scrutinee, a `ret` value) is live for the same reason — control flow and the
//      function's result are always observed. Liveness then propagates backward from every
//      root through its own operands, transitively, via a worklist, until nothing new is
//      marked. Anything never marked is dead and dropped.
//
// Since both sweeps remove entries from the block/instruction arenas, `BlockId`/`InstId`
// must be renumbered; this walks the old function's reachable blocks in their original
// order and, within each, its live instructions in their original order, assigning each a
// fresh id in a first pass, then remapping every operand/terminator in a second pass once
// every surviving id in the function is known — the same two-pass technique `ssa.rs`'s own
// `Ctx::build` uses for the same reason (see that module's header): a loop header's live
// `phi` can reference a value produced later in program order, in the loop's own latch
// block, which a single combined assign-and-remap pass cannot resolve. A `phi`'s own
// incoming-edge list gets its block ids remapped the same way; an incoming pair whose
// predecessor block was unreachable is dropped from the pair list rather than remapped,
// since that edge can never actually be taken.
//
// `basalt-sema`'s lowering pass documents (see its `lower.rs` header, "trailing dead block")
// that it always opens a fresh block after a terminator whether or not anything branches to
// it, and says a later cleanup pass is expected to fold these away. This pass is that
// cleanup: every such trailing block is unreachable by construction and falls out of the
// reachability sweep above.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use basalt_bir::{Block, BlockId, Function, Inst, InstId, Module, Op, Term, ValRef};

/// Removes dead instructions (unused and side-effect-free) and unreachable blocks from every
/// function in `module`. See this module's header for the exact liveness/reachability rules.
pub fn eliminate_dead_code(module: &Module) -> Module {
    Module {
        funcs: module.funcs.iter().map(eliminate_dead_code_fn).collect(),
        launch_bounds: module.launch_bounds,
        shared_mem_bytes: module.shared_mem_bytes,
        target_dtypes: module.target_dtypes.clone(),
    }
}

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

fn term_operand(term: &Term) -> Option<ValRef> {
    match term {
        Term::Br(_) => None,
        Term::CondBr(c, _, _) => Some(*c),
        Term::Switch(v, _, _) => Some(*v),
        Term::Ret(v) => *v,
    }
}

/// Every `ValRef` operand of `op`, in structural order (a `phi`'s full incoming-value list,
/// unfiltered by reachability — callers that care about reachability use `live_operand_refs`
/// instead).
fn operand_refs(op: &Op) -> Vec<ValRef> {
    match op {
        Op::ConstInt(_) | Op::ConstFloat(_) => Vec::new(),
        Op::Bin(_, a, b) => vec![*a, *b],
        Op::ICmp(_, _, a, b) => vec![*a, *b],
        Op::FCmp(_, _, a, b) => vec![*a, *b],
        Op::Select(c, a, b) => vec![*c, *a, *b],
        Op::Cast(_, _, v) => vec![*v],
        Op::Load { ptr, .. } => vec![*ptr],
        Op::Store { ptr, val, .. } => vec![*ptr, *val],
        Op::Phi(incoming) => incoming.iter().map(|(_, v)| *v).collect(),
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
        | Op::GdimZ
        | Op::Barrier => Vec::new(),
        Op::Shuffle(_, a, b) => vec![*a, *b],
        Op::Ballot(a) | Op::VoteAny(a) | Op::VoteAll(a) => vec![*a],
        Op::Atomic(_, ptr, v, _) => vec![*ptr, *v],
        Op::AtomicCas(ptr, cmp, new, _) => vec![*ptr, *cmp, *new],
        Op::Mma { a, b, c, d, .. } => vec![*a, *b, *c, *d],
    }
}

/// `operand_refs`, except a `phi`'s incoming pairs whose predecessor block is not in
/// `reachable` are excluded — that edge can never be taken, so the value flowing along it is
/// not a real use.
fn live_operand_refs(op: &Op, reachable: &BTreeSet<u32>) -> Vec<ValRef> {
    match op {
        Op::Phi(incoming) => incoming
            .iter()
            .filter(|(bb, _)| reachable.contains(&bb.0))
            .map(|(_, v)| *v)
            .collect(),
        _ => operand_refs(op),
    }
}

/// Rebuilds `op` with every `ValRef` operand passed through `f`, in the same structural order
/// `operand_refs` walks. `Op::Phi` is handled separately by the caller (its incoming pairs
/// also need block-id remapping and unreachable-edge dropping), so it passes through here
/// with only its values remapped and its blocks left alone.
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

/// Whether `op` is necessarily live regardless of whether its own SSA result is used: it has
/// an effect beyond producing a value (`store`, `barrier`, the atomics) or its result *is*
/// that effect (a `volatile` `load`'s read must never be assumed droppable).
fn is_root(op: &Op) -> bool {
    match op {
        Op::Store { .. } | Op::Barrier | Op::Atomic(..) | Op::AtomicCas(..) | Op::Mma { .. } => {
            true
        }
        Op::Load { volatile, .. } => *volatile,
        _ => false,
    }
}

fn mark(id: InstId, live: &mut [bool], queue: &mut VecDeque<InstId>) {
    if !live[id.0 as usize] {
        live[id.0 as usize] = true;
        queue.push_back(id);
    }
}

/// BFS over `Term` edges from block 0. Returns the set of reachable block indices.
fn reachable_blocks(f: &Function) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    if f.blocks.is_empty() {
        return seen;
    }
    let mut stack = vec![BlockId(0)];
    seen.insert(0);
    while let Some(b) = stack.pop() {
        for s in successors(&f.blocks[b.0 as usize].term) {
            if seen.insert(s.0) {
                stack.push(s);
            }
        }
    }
    seen
}

/// Mark-and-sweep instruction liveness over `f`'s reachable blocks: seeds a worklist with
/// every necessarily-live root (see `is_root`) and every block's live `Term` operand, then
/// propagates backward through operands to a fixed point.
fn mark_live(f: &Function, reachable: &BTreeSet<u32>) -> Vec<bool> {
    let mut live = vec![false; f.insts.len()];
    let mut queue: VecDeque<InstId> = VecDeque::new();

    for (bidx, block) in f.blocks.iter().enumerate() {
        if !reachable.contains(&(bidx as u32)) {
            continue;
        }
        for &id in &block.insts {
            if is_root(&f.insts[id.0 as usize].op) {
                mark(id, &mut live, &mut queue);
            }
        }
        if let Some(ValRef::Val(id)) = term_operand(&block.term) {
            mark(id, &mut live, &mut queue);
        }
    }

    while let Some(id) = queue.pop_front() {
        for v in live_operand_refs(&f.insts[id.0 as usize].op, reachable) {
            if let ValRef::Val(oid) = v {
                mark(oid, &mut live, &mut queue);
            }
        }
    }

    live
}

fn remap_valref(v: ValRef, inst_map: &BTreeMap<u32, InstId>) -> ValRef {
    match v {
        ValRef::Param(p) => ValRef::Param(p),
        ValRef::Val(id) => ValRef::Val(
            *inst_map
                .get(&id.0)
                .expect("operand of a live instruction must itself have been marked live"),
        ),
    }
}

fn remap_op(
    op: &Op,
    inst_map: &BTreeMap<u32, InstId>,
    block_map: &BTreeMap<u32, u32>,
    reachable: &BTreeSet<u32>,
) -> Op {
    match op {
        Op::Phi(incoming) => Op::Phi(
            incoming
                .iter()
                .filter(|(bb, _)| reachable.contains(&bb.0))
                .map(|(bb, v)| (BlockId(block_map[&bb.0]), remap_valref(*v, inst_map)))
                .collect(),
        ),
        _ => map_op(op, |v| remap_valref(v, inst_map)),
    }
}

fn remap_term(
    term: &Term,
    inst_map: &BTreeMap<u32, InstId>,
    block_map: &BTreeMap<u32, u32>,
) -> Term {
    match term {
        Term::Br(b) => Term::Br(BlockId(block_map[&b.0])),
        Term::CondBr(c, t, e) => Term::CondBr(
            remap_valref(*c, inst_map),
            BlockId(block_map[&t.0]),
            BlockId(block_map[&e.0]),
        ),
        Term::Switch(v, default, cases) => Term::Switch(
            remap_valref(*v, inst_map),
            BlockId(block_map[&default.0]),
            cases
                .iter()
                .map(|(val, b)| (*val, BlockId(block_map[&b.0])))
                .collect(),
        ),
        Term::Ret(v) => Term::Ret(v.map(|v| remap_valref(v, inst_map))),
    }
}

fn eliminate_dead_code_fn(f: &Function) -> Function {
    let reachable = reachable_blocks(f);
    let live = mark_live(f, &reachable);

    // New block ids in original `BlockId` order, filtered to reachable ones — block 0 (the
    // BFS root) is always reachable, so it always keeps its id of 0.
    let mut block_map: BTreeMap<u32, u32> = BTreeMap::new();
    for bidx in 0..f.blocks.len() as u32 {
        if reachable.contains(&bidx) {
            let new_id = block_map.len() as u32;
            block_map.insert(bidx, new_id);
        }
    }

    // Two passes, mirroring `ssa.rs`'s own `Ctx::build` (see that module's header): a loop
    // header's live phi can reference a value produced later in program order (the loop's
    // latch block), so operands cannot be remapped in the same pass that assigns final ids —
    // pass 1 assigns every surviving instruction its final id, in the exact order it will be
    // printed (blocks in order, each block's own live instructions in order); pass 2 then
    // remaps every operand/terminator now that every id in the function is already known.
    let mut inst_map: BTreeMap<u32, InstId> = BTreeMap::new();
    let mut kept_per_block: Vec<Vec<u32>> = Vec::with_capacity(block_map.len());
    for (bidx, block) in f.blocks.iter().enumerate() {
        if !reachable.contains(&(bidx as u32)) {
            continue;
        }
        let mut kept = Vec::new();
        for &old_id in &block.insts {
            if live[old_id.0 as usize] {
                let new_id = InstId(inst_map.len() as u32);
                inst_map.insert(old_id.0, new_id);
                kept.push(old_id.0);
            }
        }
        kept_per_block.push(kept);
    }

    let mut new_insts: Vec<Inst> = Vec::with_capacity(inst_map.len());
    let mut new_blocks: Vec<Block> = Vec::with_capacity(block_map.len());
    let mut kept_per_block = kept_per_block.into_iter();

    for (bidx, block) in f.blocks.iter().enumerate() {
        if !reachable.contains(&(bidx as u32)) {
            continue;
        }
        let kept = kept_per_block
            .next()
            .expect("one kept-instruction list per reachable block, in the same order as pass 1");
        let mut kept_ids = Vec::with_capacity(kept.len());
        for old_id in kept {
            let old_inst = &f.insts[old_id as usize];
            let new_op = remap_op(&old_inst.op, &inst_map, &block_map, &reachable);
            new_insts.push(Inst {
                ty: old_inst.ty,
                op: new_op,
            });
            kept_ids.push(inst_map[&old_id]);
        }
        let new_term = remap_term(&block.term, &inst_map, &block_map);
        new_blocks.push(Block {
            insts: kept_ids,
            term: new_term,
        });
    }

    Function {
        name: f.name.clone(),
        params: f.params.clone(),
        ret: f.ret,
        blocks: new_blocks,
        insts: new_insts,
    }
}
