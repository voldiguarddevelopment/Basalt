// Dominator tree and natural-loop detection over a single `basalt_bir::Function`'s CFG.
//
// Dominance is computed with the Cooper-Harvey-Kennedy "A Simple, Fast Dominance Algorithm"
// approach: number reachable blocks in reverse postorder (RPO) from the entry (block 0), then
// iterate a fixed point where each block's immediate dominator is the "intersection" of its
// already-processed predecessors' dominators — walking up two candidates' dominator chains in
// lockstep by RPO number until they meet. No dominance-frontier bookkeeping is built here;
// this module only produces `idom`/`dominates` and, from those, natural loops.
//
// Unreachable blocks (nothing in the CFG, following `Term` edges from block 0, ever reaches
// them) are left out of the dominator relation entirely: `idom` returns `None` for them and
// `dominates` treats them as dominating nothing and being dominated by nothing, including
// themselves. This is a deliberate choice, not an oversight — a block with no path from entry
// has no dominance relation to establish, and the alternative (pretending it dominates itself)
// would make `detect_loops` and future consumers (LICM, divergence analysis) reason about
// blocks that can never execute.
//
// Natural-loop detection follows the standard backward-reachability construction: a back edge
// is any CFG edge `latch -> header` where `header` (as computed above) dominates `latch`. Each
// back edge gets its own `NaturalLoop` record, built by a worklist walk backward from `latch`
// that adds predecessors until it reaches `header`, which is seeded into the block set up front
// and never itself explored backward. A header with multiple back edges (e.g. two continues
// into the same loop) therefore produces multiple `NaturalLoop` records sharing a header rather
// than one merged record — this keeps each record's block set exactly the reachable set for its
// own back edge, and callers that want the textbook "merge all back edges to one header into a
// single loop" view can do that themselves by unioning records that share a header.

use std::collections::BTreeSet;

use basalt_bir::{BlockId, Function, Term};

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

/// Predecessor lists for every block in `function`, indexed by `BlockId`. An edge appears at
/// most once even if a terminator names the same successor twice (e.g. a `switch` whose
/// default and a case coincide).
fn predecessors(function: &Function) -> Vec<Vec<BlockId>> {
    let n = function.blocks.len();
    let mut preds = vec![Vec::new(); n];
    for (idx, block) in function.blocks.iter().enumerate() {
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

/// Reverse postorder over the blocks reachable from `entry`, following CFG successors.
/// Computed with an explicit stack (rather than recursion) so pathologically large CFGs can't
/// blow the stack. Unreachable blocks never appear in the result.
fn reverse_postorder(entry: BlockId, succs: &[Vec<BlockId>]) -> Vec<BlockId> {
    let n = succs.len();
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack: Vec<(BlockId, usize)> = Vec::new();

    visited[entry.0 as usize] = true;
    stack.push((entry, 0));
    while let Some(&mut (b, ref mut next)) = stack.last_mut() {
        let children = &succs[b.0 as usize];
        if *next < children.len() {
            let c = children[*next];
            *next += 1;
            if !visited[c.0 as usize] {
                visited[c.0 as usize] = true;
                stack.push((c, 0));
            }
        } else {
            postorder.push(b);
            stack.pop();
        }
    }

    postorder.reverse();
    postorder
}

/// A function's dominator tree, computed by `compute`. Blocks unreachable from the entry are
/// excluded from the relation entirely — see this module's header comment.
pub struct Dominators {
    /// Immediate dominator of each block, indexed by `BlockId`. `None` for the entry (it has
    /// no idom) and for any block unreachable from the entry.
    idom: Vec<Option<BlockId>>,
    /// Reverse-postorder index of each block, indexed by `BlockId`. `None` for unreachable
    /// blocks.
    rpo_index: Vec<Option<usize>>,
}

/// Walks `a` and `b`'s dominator chains in lockstep, by RPO index, until they meet. Both
/// inputs must already have a computed idom (the entry's sentinel self-idom counts).
fn intersect(
    idom: &[Option<BlockId>],
    rpo_index: &[Option<usize>],
    a: BlockId,
    b: BlockId,
) -> BlockId {
    let mut a = a;
    let mut b = b;
    while a != b {
        while rpo_index[a.0 as usize] > rpo_index[b.0 as usize] {
            a = idom[a.0 as usize].expect("dominator chain must reach the entry");
        }
        while rpo_index[b.0 as usize] > rpo_index[a.0 as usize] {
            b = idom[b.0 as usize].expect("dominator chain must reach the entry");
        }
    }
    a
}

impl Dominators {
    /// Computes the dominator tree of `function`'s CFG, treating block 0 as the entry.
    pub fn compute(function: &Function) -> Dominators {
        let n = function.blocks.len();
        if n == 0 {
            return Dominators {
                idom: Vec::new(),
                rpo_index: Vec::new(),
            };
        }

        let entry = BlockId(0);
        let mut succs = Vec::with_capacity(n);
        for block in &function.blocks {
            succs.push(successors(&block.term));
        }
        let preds = predecessors(function);
        let rpo = reverse_postorder(entry, &succs);

        let mut rpo_index: Vec<Option<usize>> = vec![None; n];
        for (i, b) in rpo.iter().enumerate() {
            rpo_index[b.0 as usize] = Some(i);
        }

        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        idom[entry.0 as usize] = Some(entry);

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let bidx = b.0 as usize;
                let mut new_idom: Option<BlockId> = None;
                for &p in &preds[bidx] {
                    if idom[p.0 as usize].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(&idom, &rpo_index, cur, p),
                    });
                }
                if new_idom.is_some() && idom[bidx] != new_idom {
                    idom[bidx] = new_idom;
                    changed = true;
                }
            }
        }

        // The entry dominates itself in the intersection math above (it needs a fixed point
        // to walk chains up to), but it has no *immediate* dominator by definition.
        idom[entry.0 as usize] = None;

        Dominators { idom, rpo_index }
    }

    fn is_reachable(&self, block: BlockId) -> bool {
        matches!(self.rpo_index.get(block.0 as usize), Some(Some(_)))
    }

    /// The immediate dominator of `block`, or `None` if `block` is the entry or is unreachable
    /// from it.
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(block.0 as usize).copied().flatten()
    }

    /// Whether `a` dominates `b`. Reflexive (`dominates(b, b)` holds for any reachable `b`).
    /// Always `false` when `b` is unreachable from the entry, including `a == b` — an
    /// unreachable block has no dominance relation to itself or anything else.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if !self.is_reachable(b) {
            return false;
        }
        let mut cur = b;
        loop {
            if cur == a {
                return true;
            }
            match self.idom(cur) {
                Some(parent) => cur = parent,
                None => return false,
            }
        }
    }
}

/// One natural loop: the back edge that induced it, its header, and every block backward-
/// reachable from the latch without passing beyond the header (see this module's header
/// comment for the exact worklist construction and the multi-back-edge-per-header decision).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub blocks: BTreeSet<BlockId>,
    /// `(latch, header)` — the edge whose target dominates its source.
    pub back_edge: (BlockId, BlockId),
}

/// Finds every natural loop in `function`, one `NaturalLoop` per back edge, in the
/// deterministic order the back edges are discovered: source block in ascending `BlockId`
/// order, successors in the order their terminator names them.
pub fn detect_loops(function: &Function, doms: &Dominators) -> Vec<NaturalLoop> {
    let preds = predecessors(function);
    let mut loops = Vec::new();

    for (idx, block) in function.blocks.iter().enumerate() {
        let latch = BlockId(idx as u32);
        for header in successors(&block.term) {
            if !doms.dominates(header, latch) {
                continue;
            }

            let mut blocks: BTreeSet<BlockId> = BTreeSet::new();
            blocks.insert(header);
            let mut worklist = Vec::new();
            if latch != header {
                blocks.insert(latch);
                worklist.push(latch);
            }
            while let Some(m) = worklist.pop() {
                for &p in &preds[m.0 as usize] {
                    if blocks.insert(p) {
                        worklist.push(p);
                    }
                }
            }

            loops.push(NaturalLoop {
                header,
                blocks,
                back_edge: (latch, header),
            });
        }
    }

    loops
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{Inst, Op, Scalar, Term, Ty, ValRef};

    fn ret_void() -> Term {
        Term::Ret(None)
    }

    fn i32_const_block(insts: &mut Vec<Inst>, block_insts: &mut Vec<basalt_bir::InstId>) {
        let id = basalt_bir::InstId(insts.len() as u32);
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::ConstInt(0),
        });
        block_insts.push(id);
    }

    fn func(blocks: Vec<basalt_bir::Block>, insts: Vec<Inst>) -> Function {
        Function {
            is_kernel: true,
            name: "f".to_string(),
            params: Vec::new(),
            ret: Ty::Void,
            blocks,
            insts,
        }
    }

    /// A single block, no branches: `bb0: ret`.
    fn straight_line_one_block() -> Function {
        func(
            vec![basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            }],
            Vec::new(),
        )
    }

    /// A linear chain `bb0 -> bb1 -> bb2 -> ret`.
    fn straight_line_chain() -> Function {
        let blocks = vec![
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(1)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(2)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
        ];
        func(blocks, Vec::new())
    }

    /// The textbook if/else diamond:
    /// bb0 (entry, condbr) -> bb1, bb2; bb1 -> bb3; bb2 -> bb3; bb3 -> ret.
    /// Neither bb1 nor bb2 alone dominates bb3 — only bb0 does.
    fn diamond() -> Function {
        let mut insts = Vec::new();
        let mut bb0_insts = Vec::new();
        i32_const_block(&mut insts, &mut bb0_insts);
        let cond = ValRef::Val(bb0_insts[0]);

        let blocks = vec![
            basalt_bir::Block {
                insts: bb0_insts,
                term: Term::CondBr(cond, BlockId(1), BlockId(2)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(3)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(3)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
        ];
        func(blocks, insts)
    }

    /// A single natural loop:
    /// bb0 (entry) -> bb1 (header); bb1 -> bb2 (body, condbr back to bb1 or forward to bb3);
    /// bb2 -> bb1 (back edge) or bb2 -> bb3; bb3 -> ret (after the loop).
    fn single_loop() -> Function {
        let mut insts = Vec::new();
        let mut bb2_insts = Vec::new();
        i32_const_block(&mut insts, &mut bb2_insts);
        let cond = ValRef::Val(bb2_insts[0]);

        let blocks = vec![
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(1)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(2)),
            },
            basalt_bir::Block {
                insts: bb2_insts,
                term: Term::CondBr(cond, BlockId(1), BlockId(3)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
        ];
        func(blocks, insts)
    }

    /// Nested loops: bb0 -> bb1 (outer header) -> bb2 (inner header) -> bb3 (inner body,
    /// condbr back to bb2 or forward to bb4) -> bb4 (condbr back to bb1 or forward to bb5) ->
    /// bb5 -> ret. The inner loop is {bb2, bb3}; the outer loop is {bb1, bb2, bb3, bb4}.
    fn nested_loops() -> Function {
        let mut insts = Vec::new();

        let mut bb3_insts = Vec::new();
        i32_const_block(&mut insts, &mut bb3_insts);
        let inner_cond = ValRef::Val(bb3_insts[0]);

        let mut bb4_insts = Vec::new();
        i32_const_block(&mut insts, &mut bb4_insts);
        let outer_cond = ValRef::Val(bb4_insts[0]);

        let blocks = vec![
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(1)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(2)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(3)),
            },
            basalt_bir::Block {
                insts: bb3_insts,
                term: Term::CondBr(inner_cond, BlockId(2), BlockId(4)),
            },
            basalt_bir::Block {
                insts: bb4_insts,
                term: Term::CondBr(outer_cond, BlockId(1), BlockId(5)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
        ];
        func(blocks, insts)
    }

    /// bb0 -> bb1 -> ret; bb2 sits in the arena but nothing branches to it.
    fn with_unreachable_block() -> Function {
        let blocks = vec![
            basalt_bir::Block {
                insts: Vec::new(),
                term: Term::Br(BlockId(1)),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
            basalt_bir::Block {
                insts: Vec::new(),
                term: ret_void(),
            },
        ];
        func(blocks, Vec::new())
    }

    #[test]
    fn straight_line_single_block_has_no_idom() {
        let f = straight_line_one_block();
        let doms = Dominators::compute(&f);
        assert_eq!(doms.idom(BlockId(0)), None);
        assert!(doms.dominates(BlockId(0), BlockId(0)));
    }

    #[test]
    fn straight_line_chain_idoms_are_the_unique_predecessor() {
        let f = straight_line_chain();
        let doms = Dominators::compute(&f);
        assert_eq!(doms.idom(BlockId(0)), None);
        assert_eq!(doms.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(doms.idom(BlockId(2)), Some(BlockId(1)));
    }

    #[test]
    fn diamond_merge_block_is_dominated_by_entry_not_either_arm() {
        let f = diamond();
        let doms = Dominators::compute(&f);
        assert_eq!(doms.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(doms.idom(BlockId(2)), Some(BlockId(0)));
        assert_eq!(doms.idom(BlockId(3)), Some(BlockId(0)));

        assert!(doms.dominates(BlockId(0), BlockId(3)));
        assert!(!doms.dominates(BlockId(1), BlockId(3)));
        assert!(!doms.dominates(BlockId(2), BlockId(3)));
    }

    #[test]
    fn single_loop_detected_with_correct_header_and_members() {
        let f = single_loop();
        let doms = Dominators::compute(&f);
        assert_eq!(doms.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(doms.idom(BlockId(2)), Some(BlockId(1)));
        // bb3's only predecessor is bb2 (the loop's own condbr), not bb1 directly.
        assert_eq!(doms.idom(BlockId(3)), Some(BlockId(2)));

        let loops = detect_loops(&f, &doms);
        assert_eq!(loops.len(), 1);
        let l = &loops[0];
        assert_eq!(l.header, BlockId(1));
        assert_eq!(l.back_edge, (BlockId(2), BlockId(1)));
        assert_eq!(l.blocks, BTreeSet::from([BlockId(1), BlockId(2)]));
        assert!(!l.blocks.contains(&BlockId(3)));
        assert!(!l.blocks.contains(&BlockId(0)));
    }

    #[test]
    fn nested_loops_are_distinct_with_subset_relationship() {
        let f = nested_loops();
        let doms = Dominators::compute(&f);

        let loops = detect_loops(&f, &doms);
        assert_eq!(loops.len(), 2);

        let inner = loops
            .iter()
            .find(|l| l.header == BlockId(2))
            .expect("inner loop");
        let outer = loops
            .iter()
            .find(|l| l.header == BlockId(1))
            .expect("outer loop");

        assert_eq!(inner.back_edge, (BlockId(3), BlockId(2)));
        assert_eq!(inner.blocks, BTreeSet::from([BlockId(2), BlockId(3)]));

        assert_eq!(outer.back_edge, (BlockId(4), BlockId(1)));
        assert_eq!(
            outer.blocks,
            BTreeSet::from([BlockId(1), BlockId(2), BlockId(3), BlockId(4)])
        );

        assert!(inner.blocks.is_subset(&outer.blocks));
        assert!(!outer.blocks.contains(&BlockId(0)));
        assert!(!outer.blocks.contains(&BlockId(5)));
    }

    #[test]
    fn dominates_is_reflexive_and_transitive_on_a_chain() {
        let f = straight_line_chain();
        let doms = Dominators::compute(&f);
        for b in [BlockId(0), BlockId(1), BlockId(2)] {
            assert!(doms.dominates(b, b));
        }
        assert!(doms.dominates(BlockId(0), BlockId(1)));
        assert!(doms.dominates(BlockId(1), BlockId(2)));
        assert!(doms.dominates(BlockId(0), BlockId(2)));
    }

    #[test]
    fn unreachable_block_excluded_without_panicking() {
        let f = with_unreachable_block();
        let doms = Dominators::compute(&f);

        assert_eq!(doms.idom(BlockId(0)), None);
        assert_eq!(doms.idom(BlockId(1)), Some(BlockId(0)));
        assert_eq!(doms.idom(BlockId(2)), None);

        assert!(!doms.dominates(BlockId(2), BlockId(2)));
        assert!(!doms.dominates(BlockId(0), BlockId(2)));
        assert!(!doms.dominates(BlockId(2), BlockId(0)));

        let loops = detect_loops(&f, &doms);
        assert!(loops.is_empty());
    }
}
