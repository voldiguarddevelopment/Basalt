// Static divergence analysis (Sampaio, Souza, Collange, Pereira, "Divergence Analysis",
// 2013): classifies every SSA value in a function as `Uniform` (guaranteed identical across
// every thread in a warp/block) or `Divergent` (may differ between threads).
//
// This is a pure analysis pass, not a transform: nothing here rewrites BIR, and it has no
// consumer wired up yet. It exists to feed a divergence-aware register allocator on the GPU
// backends, a later phase of this project; this module only builds and tests the analysis
// itself.
//
// # Part 1 — data divergence
//
// Sources (base cases):
//   - `tid.x`/`tid.y`/`tid.z` are always Divergent: each thread's own lane index within its
//     warp/block is inherently unique.
//   - Function parameters, `bid.*`/`bdim.*`/`gdim.*`, and `const.i`/`const.f` literals are
//     always Uniform: a kernel's declared arguments are the same value (the pointer or scalar
//     handed in, not what a pointer points to) for every thread in the launch, and block
//     index/block dimensions/grid dimensions never vary within a block.
//
// Propagation: `Bin`, `ICmp`, `FCmp`, `Select`, `Cast`, and `Load` are Divergent iff any
// operand is Divergent, Uniform iff every operand is. `Load`'s divergence conservatively
// follows its ADDRESS operand only: a divergent pointer means different threads may read
// different memory locations, so the loaded value is treated as Divergent even if the
// underlying bytes happen to coincide; a uniform pointer lets the load be Uniform, since nothing
// upstream of this analysis models memory contents becoming divergent through some non-SSA
// side channel. `Phi` uses the same operand-driven rule as a baseline (Divergent if any
// incoming value is Divergent), refined by the control-divergence rule in Part 2 below.
//
// `Shuffle`/`Ballot`/`VoteAny`/`VoteAll`/`Atomic`/`AtomicCas` results are always Divergent,
// regardless of operand uniformity. This is a deliberate over-approximation: these are
// inherently cross-lane or racy operations (a shuffle's source lane, a ballot's per-lane bit
// pattern, an atomic's fetch-and-modify ordering against other threads) whose result
// meaningfully differs by thread or by execution interleaving even when every operand is
// Uniform.
//
// Computed by a forward fixed point over the function's values: non-phi instructions only
// ever reference earlier-or-equal values in program order (BIR's arena is always built in
// construction order), so one pass over the arena settles them; a `Phi` can reference a value
// defined later in the arena across a loop back edge, so the whole arena is re-scanned to a
// fixed point (classification only ever moves Uniform -> Divergent, never back, so this
// always terminates).
//
// # Part 2 — control divergence (documented approximation)
//
// A full reconvergence analysis tracks, per divergent branch, the precise set of blocks where
// the branch's forked threads can still disagree, using an immediate-post-dominator-style
// computation over the CFG. That machinery is not built here. Instead: for every block `B`
// whose terminator (`CondBr`/`Switch`) branches on a Divergent value, every `Phi` in every
// block reachable from `B` — EXCEPT a block exclusively dominated by one single immediate
// successor of `B` (i.e. a block only reachable by committing to one of the branch's arms
// first, and so not a point where threads that took different arms could have met back up) —
// is forced Divergent, even if Part 1's plain operand rule would have called it Uniform.
//
// This is a conservative over-approximation relative to the published algorithm: it can mark
// a phi Divergent that a precise reconvergence computation would prove Uniform (e.g. a merge
// point downstream of the branch's true reconvergence point still gets tainted, since
// "reachable from B" is never narrowed back down once threads would actually have
// reconverged), but it never misses a phi that genuinely needs tainting — the safe direction
// for a downstream register allocator to get wrong in.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use basalt_bir::{BlockId, Function, Op, Term, ValRef};

use crate::dom::Dominators;

/// Whether a value is guaranteed identical across every thread (`Uniform`) or may differ
/// between threads (`Divergent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Divergence {
    Uniform,
    Divergent,
}

/// Identity of one SSA value, uniform across function parameters and instruction results.
/// Structurally the same idea as `regalloc::ValueId`, but kept independent here on purpose —
/// this module has no dependency on `regalloc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueId {
    Param(u32),
    Val(u32),
}

impl From<ValRef> for ValueId {
    fn from(v: ValRef) -> Self {
        match v {
            ValRef::Param(p) => ValueId::Param(p),
            ValRef::Val(id) => ValueId::Val(id.0),
        }
    }
}

/// Per-value divergence classification for one function, as computed by
/// `analyze_divergence`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DivergenceInfo {
    values: BTreeMap<ValueId, Divergence>,
}

impl DivergenceInfo {
    /// `value`'s classification. A key with no entry (a `ValRef` from a different function,
    /// or a void-typed instruction that never produces a value) conservatively reports
    /// `Divergent` — the safe direction for a consumer to get wrong in.
    pub fn of(&self, value: impl Into<ValueId>) -> Divergence {
        self.values
            .get(&value.into())
            .copied()
            .unwrap_or(Divergence::Divergent)
    }
}

/// Every direct successor block of a terminator, in the order `Term` names them.
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

/// How one op's result relates to its operands' divergence, for Part 1's fixed point.
enum Rule {
    /// A classification independent of any operand (a source, or one of the always-
    /// divergent cross-lane ops).
    Fixed(Divergence),
    /// Divergent iff any of these operands is Divergent.
    FromOperands(Vec<ValRef>),
}

fn classify_rule(op: &Op) -> Rule {
    match op {
        Op::TidX | Op::TidY | Op::TidZ => Rule::Fixed(Divergence::Divergent),
        Op::BidX
        | Op::BidY
        | Op::BidZ
        | Op::BdimX
        | Op::BdimY
        | Op::BdimZ
        | Op::GdimX
        | Op::GdimY
        | Op::GdimZ => Rule::Fixed(Divergence::Uniform),
        Op::ConstInt(_) | Op::ConstFloat(_) => Rule::Fixed(Divergence::Uniform),
        Op::Shuffle(..) | Op::Ballot(_) | Op::VoteAny(_) | Op::VoteAll(_) => {
            Rule::Fixed(Divergence::Divergent)
        }
        Op::Atomic(..) | Op::AtomicCas(..) => Rule::Fixed(Divergence::Divergent),
        Op::Bin(_, a, b) => Rule::FromOperands(vec![*a, *b]),
        Op::ICmp(_, _, a, b) => Rule::FromOperands(vec![*a, *b]),
        Op::FCmp(_, _, a, b) => Rule::FromOperands(vec![*a, *b]),
        Op::Select(c, a, b) => Rule::FromOperands(vec![*c, *a, *b]),
        Op::Cast(_, _, a) => Rule::FromOperands(vec![*a]),
        Op::Load { ptr, .. } => Rule::FromOperands(vec![*ptr]),
        Op::Phi(incoming) => Rule::FromOperands(incoming.iter().map(|(_, v)| *v).collect()),
        // No result (`Ty::Void`): never looked up, but every `Op` needs a rule.
        Op::Store { .. } | Op::Barrier => Rule::Fixed(Divergence::Uniform),
    }
}

/// Every block transitively reachable by following CFG edges starting at any block in
/// `starts` (each start block counts as reachable too).
fn reachable_from(function: &Function, starts: &[BlockId]) -> BTreeSet<BlockId> {
    let mut seen: BTreeSet<BlockId> = starts.iter().copied().collect();
    let mut worklist: VecDeque<BlockId> = starts.iter().copied().collect();
    while let Some(b) = worklist.pop_front() {
        for s in successors(&function.blocks[b.0 as usize].term) {
            if seen.insert(s) {
                worklist.push_back(s);
            }
        }
    }
    seen
}

/// Runs the two-part divergence analysis described in this module's header over `function`.
pub fn analyze_divergence(function: &Function) -> DivergenceInfo {
    let mut values: BTreeMap<ValueId, Divergence> = BTreeMap::new();

    for i in 0..function.params.len() {
        values.insert(ValueId::Param(i as u32), Divergence::Uniform);
    }
    // Optimistic init: every valued instruction starts Uniform. Part 1's fixed point below
    // only ever raises an entry to Divergent, never lowers one, so starting from the bottom
    // of the lattice and iterating to a fixed point is sound.
    for (idx, inst) in function.insts.iter().enumerate() {
        if inst.has_result() {
            values.insert(ValueId::Val(idx as u32), Divergence::Uniform);
        }
    }

    // Part 1 — data divergence: re-scan the arena until nothing changes. Non-phi
    // instructions never need more than one pass (their operands are always earlier in the
    // arena), but a phi's operand can be defined later via a loop back edge, so the whole
    // arena is repeated until stable.
    loop {
        let mut changed = false;
        for (idx, inst) in function.insts.iter().enumerate() {
            if !inst.has_result() {
                continue;
            }
            let key = ValueId::Val(idx as u32);
            let new = match classify_rule(&inst.op) {
                Rule::Fixed(d) => d,
                Rule::FromOperands(operands) => {
                    let any_divergent = operands
                        .iter()
                        .any(|v| values[&ValueId::from(*v)] == Divergence::Divergent);
                    if any_divergent {
                        Divergence::Divergent
                    } else {
                        Divergence::Uniform
                    }
                }
            };
            if values[&key] != new {
                values.insert(key, new);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Part 2 — control divergence: taint phis at plausible reconvergence points of every
    // divergent branch. See this module's header for the exact rule and its documented
    // conservatism.
    let doms = Dominators::compute(function);
    for block in &function.blocks {
        let cond = match &block.term {
            Term::CondBr(c, _, _) => *c,
            Term::Switch(c, _, _) => *c,
            _ => continue,
        };
        if values[&ValueId::from(cond)] != Divergence::Divergent {
            continue;
        }
        let succs = successors(&block.term);
        if succs.len() < 2 {
            // No actual fork out of this block, so there is nothing to reconverge.
            continue;
        }
        for m in reachable_from(function, &succs) {
            let exclusively_one_arm = succs.iter().any(|&s| doms.dominates(s, m));
            if exclusively_one_arm {
                continue;
            }
            for &inst_id in &function.blocks[m.0 as usize].insts {
                if matches!(function.insts[inst_id.0 as usize].op, Op::Phi(_)) {
                    values.insert(ValueId::Val(inst_id.0), Divergence::Divergent);
                }
            }
        }
    }

    DivergenceInfo { values }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{
        AddrSpace, AtomicOp, Block, FCmpPred, ICmpPred, Inst, InstId, Scalar, ShuffleKind, Ty,
    };

    /// Accumulates instructions into one function's arena, handing back a `ValRef` for each
    /// pushed instruction so tests can wire up operands without tracking raw ids by hand.
    struct Builder {
        insts: Vec<Inst>,
    }

    impl Builder {
        fn new() -> Builder {
            Builder { insts: Vec::new() }
        }

        fn push(&mut self, ty: Ty, op: Op) -> ValRef {
            let id = InstId(self.insts.len() as u32);
            self.insts.push(Inst { ty, op });
            ValRef::Val(id)
        }
    }

    fn i32() -> Ty {
        Ty::Scalar(Scalar::I32)
    }

    fn func(params: Vec<Ty>, blocks: Vec<Block>, insts: Vec<Inst>) -> Function {
        Function {
            name: "f".to_string(),
            params,
            ret: Ty::Void,
            blocks,
            insts,
        }
    }

    /// A single block ending in `ret`, built from whatever `b` pushed.
    fn one_block_func(params: Vec<Ty>, b: Builder) -> Function {
        func(
            params,
            vec![Block {
                insts: (0..b.insts.len() as u32).map(InstId).collect(),
                term: Term::Ret(None),
            }],
            b.insts,
        )
    }

    #[test]
    fn tid_sources_are_always_divergent() {
        let mut b = Builder::new();
        let x = b.push(i32(), Op::TidX);
        let y = b.push(i32(), Op::TidY);
        let z = b.push(i32(), Op::TidZ);
        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(x), Divergence::Divergent);
        assert_eq!(info.of(y), Divergence::Divergent);
        assert_eq!(info.of(z), Divergence::Divergent);
    }

    #[test]
    fn block_geometry_params_and_constants_are_uniform() {
        let mut b = Builder::new();
        let bid = b.push(i32(), Op::BidX);
        let bdim = b.push(i32(), Op::BdimX);
        let gdim = b.push(i32(), Op::GdimX);
        let c = b.push(i32(), Op::ConstInt(7));
        let f = one_block_func(vec![i32()], b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(ValRef::Param(0)), Divergence::Uniform);
        assert_eq!(info.of(bid), Divergence::Uniform);
        assert_eq!(info.of(bdim), Divergence::Uniform);
        assert_eq!(info.of(gdim), Divergence::Uniform);
        assert_eq!(info.of(c), Divergence::Uniform);
    }

    #[test]
    fn bin_and_icmp_over_uniform_operands_stay_uniform() {
        let mut b = Builder::new();
        let c1 = b.push(i32(), Op::ConstInt(1));
        let c2 = b.push(i32(), Op::ConstInt(2));
        let sum = b.push(i32(), Op::Bin(basalt_bir::BinOp::Add, c1, c2));
        let cmp = b.push(
            Ty::Scalar(Scalar::I1),
            Op::ICmp(ICmpPred::Slt, i32(), sum, c2),
        );
        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(sum), Divergence::Uniform);
        assert_eq!(info.of(cmp), Divergence::Uniform);
    }

    #[test]
    fn divergence_propagates_several_levels_deep() {
        let mut b = Builder::new();
        let tid = b.push(i32(), Op::TidX);
        let c1 = b.push(i32(), Op::ConstInt(1));
        let step1 = b.push(i32(), Op::Bin(basalt_bir::BinOp::Add, tid, c1));
        let step2 = b.push(i32(), Op::Bin(basalt_bir::BinOp::Mul, step1, c1));
        let step3 = b.push(i32(), Op::Bin(basalt_bir::BinOp::Sub, step2, c1));
        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(step1), Divergence::Divergent);
        assert_eq!(info.of(step2), Divergence::Divergent);
        assert_eq!(info.of(step3), Divergence::Divergent);
    }

    #[test]
    fn load_divergence_follows_the_address_operand() {
        let mut b = Builder::new();
        let uniform_ptr = b.push(Ty::Ptr(AddrSpace::Global), Op::ConstInt(0x1000));
        let uniform_load = b.push(
            i32(),
            Op::Load {
                ptr: uniform_ptr,
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        );

        let tid = b.push(i32(), Op::TidX);
        let divergent_ptr = b.push(
            Ty::Ptr(AddrSpace::Global),
            Op::Bin(basalt_bir::BinOp::Add, uniform_ptr, tid),
        );
        let divergent_load = b.push(
            i32(),
            Op::Load {
                ptr: divergent_ptr,
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        );

        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(uniform_load), Divergence::Uniform);
        assert_eq!(info.of(divergent_load), Divergence::Divergent);
    }

    #[test]
    fn cross_lane_ops_are_divergent_even_with_uniform_operands() {
        let mut b = Builder::new();
        let c1 = b.push(i32(), Op::ConstInt(1));
        let c2 = b.push(i32(), Op::ConstInt(2));
        let ptr = b.push(Ty::Ptr(AddrSpace::Global), Op::ConstInt(0x2000));

        let shuffled = b.push(i32(), Op::Shuffle(ShuffleKind::Idx, c1, c2));
        let ballot = b.push(i32(), Op::Ballot(c1));
        let vote = b.push(i32(), Op::VoteAny(c1));
        let atomic = b.push(i32(), Op::Atomic(AtomicOp::Add, ptr, c1, AddrSpace::Global));
        let cas = b.push(i32(), Op::AtomicCas(ptr, c1, c2, AddrSpace::Global));

        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(shuffled), Divergence::Divergent);
        assert_eq!(info.of(ballot), Divergence::Divergent);
        assert_eq!(info.of(vote), Divergence::Divergent);
        assert_eq!(info.of(atomic), Divergence::Divergent);
        assert_eq!(info.of(cas), Divergence::Divergent);
    }

    #[test]
    fn fcmp_select_and_cast_follow_the_operand_rule_too() {
        let mut b = Builder::new();
        let tid = b.push(i32(), Op::TidX);
        let c1 = b.push(i32(), Op::ConstInt(1));
        let f1 = b.push(Ty::Scalar(Scalar::F32), Op::ConstFloat(1.0));
        let f2 = b.push(Ty::Scalar(Scalar::F32), Op::ConstFloat(2.0));

        let fcmp = b.push(
            Ty::Scalar(Scalar::I1),
            Op::FCmp(FCmpPred::Olt, Ty::Scalar(Scalar::F32), f1, f2),
        );
        let sel = b.push(i32(), Op::Select(tid, c1, c1));
        let cast = b.push(
            Ty::Scalar(Scalar::I64),
            Op::Cast(basalt_bir::CastOp::Sext, i32(), c1),
        );

        let f = one_block_func(Vec::new(), b);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(fcmp), Divergence::Uniform);
        assert_eq!(info.of(sel), Divergence::Divergent);
        assert_eq!(info.of(cast), Divergence::Uniform);
    }

    /// `if (tid.x < 5) { x = 1; } else { x = 1; }` — a phi whose two incoming values are the
    /// literal same uniform constant on both arms must still be marked Divergent, because the
    /// branch that fed the merge was itself divergent (threads can't be assumed to agree on
    /// which arm they arrived from).
    #[test]
    fn phi_after_divergent_branch_is_divergent_even_with_identical_uniform_incomings() {
        let mut entry = Builder::new();
        let tid = entry.push(i32(), Op::TidX);
        let five = entry.push(i32(), Op::ConstInt(5));
        let cond = entry.push(
            Ty::Scalar(Scalar::I1),
            Op::ICmp(ICmpPred::Slt, i32(), tid, five),
        );
        // entry: %0 tid.x, %1 const 5, %2 icmp
        let mut insts = entry.insts;

        // bb1 (true arm): %3 = const 1
        let bb1_const = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::ConstInt(1),
        });

        // bb2 (false arm): %4 = const 1
        let bb2_const = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::ConstInt(1),
        });

        // bb3 (merge): %5 = phi [bb1 -> %3, bb2 -> %4]
        let phi_id = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::Phi(vec![
                (BlockId(1), ValRef::Val(bb1_const)),
                (BlockId(2), ValRef::Val(bb2_const)),
            ]),
        });

        let blocks = vec![
            Block {
                insts: vec![InstId(0), InstId(1), InstId(2)],
                term: Term::CondBr(cond, BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![bb1_const],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![bb2_const],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![phi_id],
                term: Term::Ret(None),
            },
        ];

        let f = func(Vec::new(), blocks, insts);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(ValRef::Val(phi_id)), Divergence::Divergent);
    }

    /// Same diamond shape, but the branch condition is Uniform (derived only from a
    /// parameter and a constant): the phi must follow Part 1's ordinary data rule and stay
    /// Uniform — control-flow tainting must not fire off a uniform branch.
    #[test]
    fn phi_after_uniform_branch_is_not_spuriously_tainted() {
        let mut entry = Builder::new();
        let param = ValRef::Param(0);
        let five = entry.push(i32(), Op::ConstInt(5));
        let cond = entry.push(
            Ty::Scalar(Scalar::I1),
            Op::ICmp(ICmpPred::Slt, i32(), param, five),
        );
        let mut insts = entry.insts;

        let bb1_const = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::ConstInt(1),
        });

        let bb2_const = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::ConstInt(1),
        });

        let phi_id = InstId(insts.len() as u32);
        insts.push(Inst {
            ty: i32(),
            op: Op::Phi(vec![
                (BlockId(1), ValRef::Val(bb1_const)),
                (BlockId(2), ValRef::Val(bb2_const)),
            ]),
        });

        let blocks = vec![
            Block {
                insts: vec![InstId(0), InstId(1)],
                term: Term::CondBr(cond, BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![bb1_const],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![bb2_const],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![phi_id],
                term: Term::Ret(None),
            },
        ];

        let f = func(vec![i32()], blocks, insts);
        let info = analyze_divergence(&f);
        assert_eq!(info.of(cond), Divergence::Uniform);
        assert_eq!(info.of(ValRef::Val(phi_id)), Divergence::Uniform);
    }
}
