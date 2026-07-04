// Fixture-based tests for `regalloc::allocate`. Fixtures are built by hand via the small
// `FnB` builder below (mirroring `basalt-passes`'s own `ssa.rs` fixtures and, further back,
// `basalt-bir`'s round-trip fixtures), directly in valid SSA form — no frontend, no need to
// route everything through `construct_ssa` first (one test does, deliberately, to exercise
// the two passes together).

use basalt_bir::{
    BinOp, Block, BlockId, Function, ICmpPred, Inst, InstId, Module, Op, Scalar, Term, Ty, ValRef,
};
use basalt_passes::{allocate, construct_ssa, Location, RegClass, ValueId};

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
            is_kernel: true,
            name: name.to_string(),
            params,
            ret,
            blocks: self.blocks,
            insts: self.insts,
        }
    }
}

const I32: Ty = Ty::Scalar(Scalar::I32);
const F32: Ty = Ty::Scalar(Scalar::F32);
const I1: Ty = Ty::Scalar(Scalar::I1);

fn value_id_of(v: ValRef) -> ValueId {
    ValueId::from(v)
}

/// Asserts that if both `x` and `y` ended up in a register, they got distinct ones. Used at
/// sites where `x` and `y` are, by construction, operands of the same instruction — i.e.
/// genuinely alive at the same program point — so a shared register there would be a real
/// allocator bug, not just an overly conservative interval.
fn assert_no_reg_collision(alloc: &basalt_passes::Allocation, x: ValRef, y: ValRef) {
    let lx = alloc.locations[&value_id_of(x)];
    let ly = alloc.locations[&value_id_of(y)];
    if let (Location::Reg(cx, rx), Location::Reg(cy, ry)) = (lx, ly) {
        assert!(
            cx != cy || rx != ry,
            "expected distinct registers for simultaneously-live values, got {lx:?} and {ly:?}"
        );
    }
}

#[test]
fn ample_budget_gives_every_value_a_distinct_register_and_no_spills() {
    let p0 = ValRef::Param(0);
    let p1 = ValRef::Param(1);

    let mut b = FnB::new();
    let sum = b.push(I32, Op::Bin(BinOp::Add, p0, p1));
    b.end_block(Term::Ret(Some(sum)));
    let f = b.finish("small", vec![I32, I32], I32);

    let alloc = allocate(&f, 4, 0);

    assert_eq!(alloc.num_int_spills, 0);
    assert_eq!(alloc.num_float_spills, 0);
    assert_eq!(alloc.locations.len(), 3, "params + sum = 3 values");
    for loc in alloc.locations.values() {
        assert!(matches!(loc, Location::Reg(RegClass::Int, _)));
    }
    assert_no_reg_collision(&alloc, p0, p1);
}

#[test]
fn tight_budget_forces_a_spill_without_register_collisions() {
    let mut b = FnB::new();
    let a = b.push(I32, Op::ConstInt(1));
    let bb = b.push(I32, Op::ConstInt(2));
    let c = b.push(I32, Op::ConstInt(3));
    let d = b.push(I32, Op::ConstInt(4));
    let s1 = b.push(I32, Op::Bin(BinOp::Add, a, bb));
    let s2 = b.push(I32, Op::Bin(BinOp::Add, c, d));
    let s3 = b.push(I32, Op::Bin(BinOp::Add, s1, s2));
    b.end_block(Term::Ret(Some(s3)));
    let f = b.finish("spill_case", vec![], I32);

    // Four independent values (a, b, c, d) all concurrently alive by the time s1/s2 combine
    // them, on a two-register budget: at least one of them (or their sums) must spill.
    let alloc = allocate(&f, 2, 0);

    assert_eq!(alloc.num_float_spills, 0);
    assert!(alloc.num_int_spills >= 1, "expected at least one spill");
    assert_eq!(alloc.locations.len(), 7);

    // a/b are simultaneously live where s1 combines them; likewise c/d at s2, and s1/s2 at s3.
    assert_no_reg_collision(&alloc, a, bb);
    assert_no_reg_collision(&alloc, c, d);
    assert_no_reg_collision(&alloc, s1, s2);

    for loc in alloc.locations.values() {
        match loc {
            Location::Reg(RegClass::Int, r) => assert!(*r < 2),
            Location::Spill(RegClass::Int, _) => {}
            other => panic!("unexpected location in an all-int function: {other:?}"),
        }
    }
}

#[test]
fn loop_carried_value_survives_the_back_edge_without_collision() {
    // Same shape as ssa.rs's own loop fixture (a local-slot induction variable across a
    // header/body back edge), run through `construct_ssa` first so the allocator sees a real
    // `phi` at the header, then allocated directly — exercising both passes together and
    // specifically the liveness dataflow's handling of a back edge, not just straight-line
    // code.
    let n = ValRef::Param(0);
    let ptr_local = Ty::Ptr(basalt_bir::AddrSpace::Local);

    let mut b = FnB::new();
    let i_addr = b.push(ptr_local, Op::ConstInt(0));
    let zero = b.push(I32, Op::ConstInt(0));
    b.push_void(Op::Store {
        ptr: i_addr,
        val: zero,
        ty: I32,
        space: basalt_bir::AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(1)));

    let load_i_hdr = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: basalt_bir::AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    let cond = b.push(I1, Op::ICmp(ICmpPred::Slt, I32, load_i_hdr, n));
    b.end_block(Term::CondBr(cond, BlockId(2), BlockId(3)));

    let load_i_body = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: basalt_bir::AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    let one = b.push(I32, Op::ConstInt(1));
    let inc = b.push(I32, Op::Bin(BinOp::Add, load_i_body, one));
    b.push_void(Op::Store {
        ptr: i_addr,
        val: inc,
        ty: I32,
        space: basalt_bir::AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(1)));

    let load_i_exit = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: basalt_bir::AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_i_exit)));

    let f = b.finish("loop", vec![I32], I32);
    let module = Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: Vec::new(),
    };

    let ssa_module = construct_ssa(&module);
    let ssa_f = &ssa_module.funcs[0];

    let phi_id = ssa_f
        .insts
        .iter()
        .enumerate()
        .find_map(|(idx, inst)| matches!(inst.op, Op::Phi(_)).then_some(ValueId::Val(idx as u32)))
        .expect("construct_ssa must have inserted the header phi");

    // n (the loop bound) and the induction phi are both read every time the header condition
    // is checked; a budget of 4 easily covers everything alive across the whole function, so
    // this is purely a correctness check, not a spill-pressure one.
    let alloc = allocate(ssa_f, 4, 0);

    assert!(alloc.locations.contains_key(&phi_id));
    assert_eq!(alloc.num_int_spills, 0);

    // The phi and `n` must never collide: they are both live across the whole loop body,
    // reachable simultaneously at the header's compare on every iteration.
    let n_id = ValueId::Param(0);
    let phi_loc = alloc.locations[&phi_id];
    let n_loc = alloc.locations[&n_id];
    if let (Location::Reg(cp, rp), Location::Reg(cn, rn)) = (phi_loc, n_loc) {
        assert!(cp != cn || rp != rn, "phi and n must not share a register");
    }

    // Run again: the phi's location must be identical (one fixed location for its whole
    // lifetime, never reassigned mid-loop).
    let alloc2 = allocate(ssa_f, 4, 0);
    assert_eq!(alloc.locations[&phi_id], alloc2.locations[&phi_id]);
}

#[test]
fn register_classes_never_cross_and_pressure_does_not_leak_across_classes() {
    let pi0 = ValRef::Param(0);
    let pi1 = ValRef::Param(1);
    let pf0 = ValRef::Param(2);
    let pf1 = ValRef::Param(3);

    let mut b = FnB::new();
    let isum = b.push(I32, Op::Bin(BinOp::Add, pi0, pi1));
    let fsum = b.push(F32, Op::Bin(BinOp::FAdd, pf0, pf1));
    b.end_block(Term::Ret(None));
    let f = b.finish("classes", vec![I32, I32, F32, F32], Ty::Void);

    // One int register (tight, forces int spilling) but three float registers (ample for the
    // two float params plus their sum) — pressure in the int class must never spill a float
    // value or vice versa.
    let alloc = allocate(&f, 1, 3);

    assert!(alloc.num_int_spills >= 1);
    assert_eq!(
        alloc.num_float_spills, 0,
        "float class must be unaffected by int register pressure"
    );

    let expect_class = |v: ValRef, want: RegClass| match alloc.locations[&value_id_of(v)] {
        Location::Reg(c, _) | Location::Spill(c, _) => {
            assert_eq!(c, want, "wrong register class for {v:?}")
        }
    };
    expect_class(pi0, RegClass::Int);
    expect_class(pi1, RegClass::Int);
    expect_class(isum, RegClass::Int);
    expect_class(pf0, RegClass::Float);
    expect_class(pf1, RegClass::Float);
    expect_class(fsum, RegClass::Float);
}

#[test]
fn phi_result_gets_a_location_like_any_other_value() {
    let arg0 = ValRef::Param(0);

    let mut b = FnB::new();
    let c0 = b.push(I32, Op::ConstInt(0));
    let cond = b.push(I1, Op::ICmp(ICmpPred::Sgt, I32, arg0, c0));
    b.end_block(Term::CondBr(cond, BlockId(1), BlockId(2)));

    let ten = b.push(I32, Op::ConstInt(10));
    let bb1 = b.end_block(Term::Br(BlockId(3)));

    let twenty = b.push(I32, Op::ConstInt(20));
    let bb2 = b.end_block(Term::Br(BlockId(3)));

    let phi = b.push(I32, Op::Phi(vec![(bb1, ten), (bb2, twenty)]));
    b.end_block(Term::Ret(Some(phi)));

    let f = b.finish("phi_test", vec![I32], I32);
    let alloc = allocate(&f, 4, 0);

    assert!(alloc.locations.contains_key(&value_id_of(phi)));
    assert_eq!(
        alloc.locations.len(),
        6,
        "arg0, c0, cond, ten, twenty, and phi"
    );
}

#[test]
fn allocation_is_deterministic_across_runs() {
    let mut b = FnB::new();
    let a = b.push(I32, Op::ConstInt(1));
    let bb = b.push(I32, Op::ConstInt(2));
    let c = b.push(I32, Op::ConstInt(3));
    let d = b.push(I32, Op::ConstInt(4));
    let s1 = b.push(I32, Op::Bin(BinOp::Add, a, bb));
    let s2 = b.push(I32, Op::Bin(BinOp::Add, c, d));
    let s3 = b.push(I32, Op::Bin(BinOp::Add, s1, s2));
    b.end_block(Term::Ret(Some(s3)));
    let f = b.finish("determinism", vec![], I32);

    let alloc1 = allocate(&f, 2, 0);
    let alloc2 = allocate(&f, 2, 0);
    assert_eq!(alloc1, alloc2);
}
