// Fixture-based tests for `eliminate_dead_code`. Fixtures are built by hand via the small
// `FnB` builder below (mirroring `basalt-passes`'s other fixture-based test files, and,
// further back, `basalt-bir`'s own round-trip fixtures) so each test can pin down an exact
// live/dead or reachable/unreachable shape.

use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, Block, BlockId, Function, ICmpPred, Inst, InstId, MmaLayout,
    Module, Op, Scalar, Term, Ty, ValRef,
};
use basalt_passes::eliminate_dead_code;

#[derive(Default)]
struct FnB {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    cur: Vec<InstId>,
}

impl FnB {
    fn new() -> Self {
        FnB::default()
    }

    fn push(&mut self, ty: Ty, op: Op) -> ValRef {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(Inst { ty, op });
        self.cur.push(id);
        ValRef::Val(id)
    }

    fn push_void(&mut self, op: Op) {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(Inst { ty: Ty::Void, op });
        self.cur.push(id);
    }

    fn end_block(&mut self, term: Term) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        let insts = std::mem::take(&mut self.cur);
        self.blocks.push(Block { insts, term });
        id
    }

    fn finish(self, name: &str, params: Vec<Ty>, ret: Ty) -> Function {
        assert!(
            self.cur.is_empty(),
            "fixture bug: last block never closed with a terminator"
        );
        Function {
            name: name.to_string(),
            params,
            ret,
            blocks: self.blocks,
            insts: self.insts,
        }
    }
}

fn module_of(funcs: Vec<Function>) -> Module {
    Module {
        funcs,
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: Vec::new(),
    }
}

fn assert_roundtrip(m: &Module) {
    let text = basalt_bir::print(m);
    let reparsed = match basalt_bir::parse(&text) {
        Ok(m) => m,
        Err(e) => panic!("parse(print(m)) failed: {e}\n--- printed BIR ---\n{text}"),
    };
    assert_eq!(
        &reparsed, m,
        "parse(print(m)) != m\n--- printed BIR ---\n{text}"
    );
}

const I32: Ty = Ty::Scalar(Scalar::I32);
const PTR_GLOBAL: Ty = Ty::Ptr(AddrSpace::Global);

#[test]
fn unused_pure_computation_is_removed() {
    let mut b = FnB::new();
    let p0 = ValRef::Param(0);
    let _dead = b.push(I32, Op::Bin(BinOp::Add, p0, p0));
    let ret_val = b.push(I32, Op::ConstInt(7));
    b.end_block(Term::Ret(Some(ret_val)));
    let f = b.finish("f", vec![I32], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert_eq!(
        out.funcs[0].insts.len(),
        1,
        "the unused add should be removed, leaving only the returned constant: {}",
        basalt_bir::print(&out)
    );
    assert!(matches!(out.funcs[0].insts[0].op, Op::ConstInt(7)));
    assert_roundtrip(&out);
}

#[test]
fn non_volatile_load_with_unused_result_is_removed() {
    let mut b = FnB::new();
    let ptr = ValRef::Param(0);
    let _dead_load = b.push(
        I32,
        Op::Load {
            ptr,
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    );
    let ret_val = b.push(I32, Op::ConstInt(0));
    b.end_block(Term::Ret(Some(ret_val)));
    let f = b.finish("f", vec![PTR_GLOBAL], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert!(
        !out.funcs[0]
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::Load { .. })),
        "unused non-volatile load must be removed: {}",
        basalt_bir::print(&out)
    );
    assert_roundtrip(&out);
}

#[test]
fn dead_chain_is_removed_transitively() {
    // a -> b -> c -> d, none of which are ever used; only the returned constant survives.
    // Proves the worklist actually chases operands transitively, not just single-level use.
    let mut b = FnB::new();
    let a = b.push(I32, Op::ConstInt(1));
    let bb = b.push(I32, Op::Bin(BinOp::Add, a, a));
    let c = b.push(I32, Op::Bin(BinOp::Add, bb, bb));
    let _d = b.push(I32, Op::Bin(BinOp::Add, c, c));
    let ret_val = b.push(I32, Op::ConstInt(0));
    b.end_block(Term::Ret(Some(ret_val)));
    let f = b.finish("f", vec![], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert_eq!(
        out.funcs[0].insts.len(),
        1,
        "the whole dead chain (a, b, c, d) should be swept away, leaving only the returned \
         constant: {}",
        basalt_bir::print(&out)
    );
    assert_roundtrip(&out);
}

#[test]
fn side_effecting_ops_survive_with_unused_results() {
    let mut b = FnB::new();
    let ptr = ValRef::Param(0);
    let val = b.push(I32, Op::ConstInt(1));
    b.push_void(Op::Store {
        ptr,
        val,
        ty: I32,
        space: AddrSpace::Global,
        align: 4,
        volatile: false,
    });
    b.push_void(Op::Barrier);
    let _atomic = b.push(I32, Op::Atomic(AtomicOp::Add, ptr, val, AddrSpace::Global));
    let _vload = b.push(
        I32,
        Op::Load {
            ptr,
            space: AddrSpace::Global,
            align: 4,
            volatile: true,
        },
    );
    b.end_block(Term::Ret(None));
    let f = b.finish("f", vec![PTR_GLOBAL], Ty::Void);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    let after = &out.funcs[0];
    assert_eq!(
        after.insts.len(),
        5,
        "store/barrier/atomic/volatile-load must all survive despite unused results: {}",
        basalt_bir::print(&out)
    );
    assert!(after.insts.iter().any(|i| matches!(i.op, Op::Store { .. })));
    assert!(after.insts.iter().any(|i| matches!(i.op, Op::Barrier)));
    assert!(after.insts.iter().any(|i| matches!(i.op, Op::Atomic(..))));
    assert!(after
        .insts
        .iter()
        .any(|i| matches!(i.op, Op::Load { volatile: true, .. })));
    assert_roundtrip(&out);
}

#[test]
fn mma_survives_dce_despite_unused_result() {
    // `mma` is `Ty::Void` (it has no SSA result at all, live or otherwise) but writes
    // through its `d` pointer, so it must survive as a root exactly like `store`.
    let mut b = FnB::new();
    let a = ValRef::Param(0);
    let bref = ValRef::Param(1);
    let c = ValRef::Param(2);
    let d = ValRef::Param(3);
    b.push_void(Op::Mma {
        a,
        b: bref,
        c,
        d,
        m: 2,
        n: 2,
        k: 2,
        in_dtype: Scalar::F32,
        acc_dtype: Scalar::F32,
        layout_a: MmaLayout::RowMajor,
        layout_b: MmaLayout::RowMajor,
    });
    b.end_block(Term::Ret(None));
    let f = b.finish(
        "f",
        vec![PTR_GLOBAL, PTR_GLOBAL, PTR_GLOBAL, PTR_GLOBAL],
        Ty::Void,
    );
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert_eq!(
        out.funcs[0].insts.len(),
        1,
        "mma must survive as a root even though nothing reads a result from it: {}",
        basalt_bir::print(&out)
    );
    assert!(matches!(out.funcs[0].insts[0].op, Op::Mma { .. }));
    assert_roundtrip(&out);
}

#[test]
fn unreachable_block_is_removed_entirely() {
    let mut b = FnB::new();
    let live_ret = b.push(I32, Op::ConstInt(5));
    b.end_block(Term::Ret(Some(live_ret))); // bb0: entry, nothing branches elsewhere

    // bb1: unreachable — no predecessor ever targets it. Its own instruction would be "live"
    // in isolation (it feeds its block's own ret), but the whole block must still vanish.
    let garbage = b.push(I32, Op::ConstInt(99));
    b.end_block(Term::Ret(Some(garbage)));

    let f = b.finish("f", vec![], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert_eq!(
        out.funcs[0].blocks.len(),
        1,
        "the unreachable block must be dropped entirely: {}",
        basalt_bir::print(&out)
    );
    assert_eq!(out.funcs[0].insts.len(), 1);
    assert!(matches!(out.funcs[0].insts[0].op, Op::ConstInt(5)));
    assert_roundtrip(&out);
}

#[test]
fn phi_drops_only_the_unreachable_predecessor_pair() {
    // bb0 (entry) branches straight to bb2; bb1 also branches to bb2 but nothing ever
    // branches into bb1, so bb1 is unreachable. bb2's phi names both bb0 and bb1 as incoming
    // edges; only the bb1 pair should be dropped, not the whole phi.
    let mut b = FnB::new();
    let v0 = b.push(I32, Op::ConstInt(1));
    b.end_block(Term::Br(BlockId(2))); // bb0

    let v1 = b.push(I32, Op::ConstInt(2));
    b.end_block(Term::Br(BlockId(2))); // bb1, unreachable

    let phi = b.push(I32, Op::Phi(vec![(BlockId(0), v0), (BlockId(1), v1)]));
    b.end_block(Term::Ret(Some(phi))); // bb2

    let f = b.finish("f", vec![], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    let after = &out.funcs[0];
    assert_eq!(
        after.blocks.len(),
        2,
        "bb1 must be dropped, leaving bb0 and bb2 (renumbered): {}",
        basalt_bir::print(&out)
    );

    let phi_inst = after
        .insts
        .iter()
        .find(|i| matches!(i.op, Op::Phi(_)))
        .expect("phi must survive (it feeds the return)");
    let Op::Phi(incoming) = &phi_inst.op else {
        unreachable!()
    };
    assert_eq!(
        incoming.len(),
        1,
        "only the reachable bb0 edge should remain on the phi: {}",
        basalt_bir::print(&out)
    );
    assert_eq!(
        incoming[0].0,
        BlockId(0),
        "bb0 keeps its id (it's the entry block)"
    );
    assert_roundtrip(&out);
}

#[test]
fn loop_header_phi_referencing_its_own_latch_survives_dce() {
    // bb0 (entry) -> bb1 (header, a live phi merging bb0's initial value and bb2's own
    // incremented one) -> bb2 (body/latch, back-edges to bb1) or bb3 (exit, returns the phi).
    // The phi's back-edge operand is defined in bb2, whose block index (2) is *higher* than
    // the header's own (1) — a real forward reference in block-array order, exactly the shape
    // any loop produces once `construct_ssa` promotes its counter to a real phi. This must not
    // panic (a prior version of this pass assumed every operand was already assigned a new id
    // by the time its user was remapped, which only holds for straight-line code) and the
    // back-edge must still resolve to the right value once renumbered.
    let mut b = FnB::new();
    let zero = b.push(I32, Op::ConstInt(0));
    b.end_block(Term::Br(BlockId(1))); // bb0

    // The back-edge operand isn't known until `next` (bb2) is built, so the phi is pushed with
    // a placeholder incoming pair for bb2 and patched below once `next`'s real id exists —
    // mirroring how a real SSA-construction pass wires up a loop header incrementally.
    let ValRef::Val(phi_id) = b.push(I32, Op::Phi(vec![(BlockId(0), zero), (BlockId(2), zero)]))
    else {
        unreachable!()
    };
    let phi = ValRef::Val(phi_id);
    let one = b.push(I32, Op::ConstInt(1));
    let cond = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Slt, I32, phi, one),
    );
    b.end_block(Term::CondBr(cond, BlockId(2), BlockId(3))); // bb1

    let next = b.push(I32, Op::Bin(BinOp::Add, phi, one));
    b.end_block(Term::Br(BlockId(1))); // bb2, the latch — back-edges to the header

    b.end_block(Term::Ret(Some(phi))); // bb3, exit

    let mut f = b.finish("f", vec![], I32);
    let Op::Phi(incoming) = &mut f.insts[phi_id.0 as usize].op else {
        unreachable!()
    };
    incoming[1] = (BlockId(2), next);

    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    let after = &out.funcs[0];
    assert_eq!(
        after.blocks.len(),
        4,
        "every block here is reachable and live: {}",
        basalt_bir::print(&out)
    );

    let phi_inst = after
        .insts
        .iter()
        .find(|i| matches!(i.op, Op::Phi(_)))
        .unwrap_or_else(|| panic!("phi must survive: {}", basalt_bir::print(&out)));
    let Op::Phi(incoming) = &phi_inst.op else {
        unreachable!()
    };
    assert_eq!(
        incoming.len(),
        2,
        "both edges are reachable, neither is dropped"
    );
    let back_edge = incoming
        .iter()
        .find(|(bb, _)| bb.0 == 2)
        .expect("the bb2 (latch) edge must survive");
    assert!(
        matches!(back_edge.1, ValRef::Val(_)),
        "the back-edge operand must resolve to a real (renumbered) instruction, not dangle"
    );

    assert_roundtrip(&out);
}

#[test]
fn nothing_to_remove_is_a_no_op_shape() {
    let mut b = FnB::new();
    let ret_val = b.push(I32, Op::ConstInt(3));
    b.end_block(Term::Ret(Some(ret_val)));
    let f = b.finish("f", vec![], I32);
    let m = module_of(vec![f]);

    let out = eliminate_dead_code(&m);
    assert_eq!(out, m, "a function with nothing dead must be unchanged");
    assert_roundtrip(&out);
}
