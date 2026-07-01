// Fixture-based tests for `construct_ssa`. Fixtures are built by hand via the small `FnB`
// builder below (mirroring `basalt-bir`'s own round-trip test fixtures) rather than through a
// frontend, so each test can pin down an exact promotable/escaping shape.

use basalt_bir::{
    AddrSpace, Block, BlockId, Function, ICmpPred, Inst, InstId, Module, Op, Scalar, Term, Ty,
    ValRef,
};
use basalt_passes::construct_ssa;

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

fn has_load_or_store(f: &Function) -> bool {
    f.insts
        .iter()
        .any(|i| matches!(i.op, Op::Load { .. } | Op::Store { .. }))
}

fn phi_count(f: &Function) -> usize {
    f.insts
        .iter()
        .filter(|i| matches!(i.op, Op::Phi(_)))
        .count()
}

const I32: Ty = Ty::Scalar(Scalar::I32);
const PTR_LOCAL: Ty = Ty::Ptr(AddrSpace::Local);

#[test]
fn straight_line_locals_promote_away() {
    let arg0 = ValRef::Param(0);
    let arg1 = ValRef::Param(1);

    let mut b = FnB::new();
    let a_addr = b.push(PTR_LOCAL, Op::ConstInt(0));
    b.push_void(Op::Store {
        ptr: a_addr,
        val: arg0,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    let load_a = b.push(
        I32,
        Op::Load {
            ptr: a_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    let b_addr = b.push(PTR_LOCAL, Op::ConstInt(65536));
    let sum = b.push(I32, Op::Bin(basalt_bir::BinOp::Add, load_a, arg1));
    b.push_void(Op::Store {
        ptr: b_addr,
        val: sum,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    let load_b = b.push(
        I32,
        Op::Load {
            ptr: b_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_b)));

    let f = b.finish("straight_line", vec![I32, I32], I32);
    let m = module_of(vec![f]);

    let out = construct_ssa(&m);
    let out_f = &out.funcs[0];

    assert!(
        !has_load_or_store(out_f),
        "expected every local load/store to be promoted away, got: {}",
        basalt_bir::print(&out)
    );
    // The chain should now be `ret (add %arg0, %arg1)` directly.
    match out_f.blocks.last().unwrap().term {
        Term::Ret(Some(ValRef::Val(id))) => {
            assert_eq!(
                out_f.insts[id.0 as usize].op,
                Op::Bin(basalt_bir::BinOp::Add, arg0, arg1)
            );
        }
        ref other => panic!("expected a direct ret of the add, got {other:?}"),
    }

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}

#[test]
fn if_else_merge_inserts_real_phi() {
    let arg0 = ValRef::Param(0);

    let mut b = FnB::new();
    let x_addr = b.push(PTR_LOCAL, Op::ConstInt(0));
    let c0 = b.push(I32, Op::ConstInt(0));
    let cond = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Sgt, I32, arg0, c0),
    );
    b.end_block(Term::CondBr(cond, BlockId(1), BlockId(2)));

    let ten = b.push(I32, Op::ConstInt(10));
    b.push_void(Op::Store {
        ptr: x_addr,
        val: ten,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    let bb1 = b.end_block(Term::Br(BlockId(3)));

    let twenty = b.push(I32, Op::ConstInt(20));
    b.push_void(Op::Store {
        ptr: x_addr,
        val: twenty,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    let bb2 = b.end_block(Term::Br(BlockId(3)));

    let load_x = b.push(
        I32,
        Op::Load {
            ptr: x_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_x)));

    let f = b.finish("if_else", vec![I32], I32);
    let m = module_of(vec![f]);

    let out = construct_ssa(&m);
    let out_f = &out.funcs[0];

    assert!(!has_load_or_store(out_f));
    assert_eq!(
        phi_count(out_f),
        1,
        "expected exactly one phi at the merge block"
    );

    let phi = out_f
        .insts
        .iter()
        .find_map(|i| match &i.op {
            Op::Phi(incoming) => Some(incoming.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(phi.len(), 2);
    let preds: Vec<BlockId> = phi.iter().map(|(bb, _)| *bb).collect();
    assert!(preds.contains(&bb1));
    assert!(preds.contains(&bb2));
    // The two incoming values must be the (renumbered) `10` and `20` constants, distinct.
    let vals: Vec<ValRef> = phi.iter().map(|(_, v)| *v).collect();
    assert_ne!(vals[0], vals[1]);

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}

#[test]
fn loop_induction_variable_gets_header_phi_without_hanging() {
    let n = ValRef::Param(0);

    let mut b = FnB::new();
    // bb0: entry — i = 0; br header
    let i_addr = b.push(PTR_LOCAL, Op::ConstInt(0));
    let zero = b.push(I32, Op::ConstInt(0));
    b.push_void(Op::Store {
        ptr: i_addr,
        val: zero,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(1)));

    // bb1: header — load i; condbr (i < n), body, exit
    let load_i_hdr = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    let cond = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Slt, I32, load_i_hdr, n),
    );
    b.end_block(Term::CondBr(cond, BlockId(2), BlockId(3)));

    // bb2: body/latch — i = i + 1; br header
    let load_i_body = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    let one = b.push(I32, Op::ConstInt(1));
    let inc = b.push(I32, Op::Bin(basalt_bir::BinOp::Add, load_i_body, one));
    b.push_void(Op::Store {
        ptr: i_addr,
        val: inc,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(1)));

    // bb3: exit — ret (load i)
    let load_i_exit = b.push(
        I32,
        Op::Load {
            ptr: i_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_i_exit)));

    let f = b.finish("loop", vec![I32], I32);
    let m = module_of(vec![f]);

    // The point of this test: constructing SSA over a loop must terminate at all (no infinite
    // recursion chasing the back edge) and must produce exactly one phi, at the header.
    let out = construct_ssa(&m);
    let out_f = &out.funcs[0];

    assert!(!has_load_or_store(out_f));
    assert_eq!(
        phi_count(out_f),
        1,
        "expected exactly one phi, at the loop header"
    );

    let phi_block = out_f
        .blocks
        .iter()
        .position(|blk| {
            blk.insts
                .iter()
                .any(|&id| matches!(out_f.insts[id.0 as usize].op, Op::Phi(_)))
        })
        .unwrap();
    assert_eq!(phi_block, 1, "the phi must live in the header block (bb1)");

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}

#[test]
fn trivial_phi_elimination_fires_when_both_arms_agree() {
    let arg0 = ValRef::Param(0);

    let mut b = FnB::new();
    let x_addr = b.push(PTR_LOCAL, Op::ConstInt(0));
    // Computed once, before the branch, so both arms store the exact same SSA value.
    let v = b.push(I32, Op::Bin(basalt_bir::BinOp::Add, arg0, arg0));
    let c0 = b.push(I32, Op::ConstInt(0));
    let cond = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Sgt, I32, arg0, c0),
    );
    b.end_block(Term::CondBr(cond, BlockId(1), BlockId(2)));

    b.push_void(Op::Store {
        ptr: x_addr,
        val: v,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(3)));

    b.push_void(Op::Store {
        ptr: x_addr,
        val: v,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    b.end_block(Term::Br(BlockId(3)));

    let load_x = b.push(
        I32,
        Op::Load {
            ptr: x_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_x)));

    let f = b.finish("agreeing_arms", vec![I32], I32);
    let m = module_of(vec![f]);

    let out = construct_ssa(&m);
    let out_f = &out.funcs[0];

    assert!(!has_load_or_store(out_f));
    assert_eq!(
        phi_count(out_f),
        0,
        "the phi should have been trivially eliminated: {}",
        basalt_bir::print(&out)
    );
    match out_f.blocks.last().unwrap().term {
        Term::Ret(Some(ValRef::Val(id))) => {
            assert_eq!(
                out_f.insts[id.0 as usize].op,
                Op::Bin(basalt_bir::BinOp::Add, arg0, arg0)
            );
        }
        ref other => panic!("expected a direct ret of the shared value, got {other:?}"),
    }

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}

#[test]
fn escaping_slot_address_is_never_promoted() {
    let arg0 = ValRef::Param(0);

    let mut b = FnB::new();
    let a_addr = b.push(PTR_LOCAL, Op::ConstInt(0));
    b.push_void(Op::Store {
        ptr: a_addr,
        val: arg0,
        ty: I32,
        space: AddrSpace::Local,
        align: 4,
        volatile: false,
    });
    let load_a = b.push(
        I32,
        Op::Load {
            ptr: a_addr,
            space: AddrSpace::Local,
            align: 4,
            volatile: false,
        },
    );
    // Escape: the slot's own address used as an ordinary operand, not a load/store `ptr`.
    let _escaped = b.push(PTR_LOCAL, Op::Bin(basalt_bir::BinOp::Add, a_addr, a_addr));
    b.end_block(Term::Ret(Some(load_a)));

    let f = b.finish("escaping", vec![I32], I32);
    let m = module_of(vec![f]);

    let out = construct_ssa(&m);
    assert_eq!(
        out, m,
        "an escaping slot must be left completely untouched (real load/store surviving)"
    );
    assert!(has_load_or_store(&out.funcs[0]));

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}

#[test]
fn global_space_memory_is_never_touched() {
    let arg0 = ValRef::Param(0);
    let ptr_global = Ty::Ptr(AddrSpace::Global);

    let mut b = FnB::new();
    // Same shape as a promotable local slot, but in `global` space — must never be promoted.
    let g_addr = b.push(ptr_global, Op::ConstInt(0));
    b.push_void(Op::Store {
        ptr: g_addr,
        val: arg0,
        ty: I32,
        space: AddrSpace::Global,
        align: 4,
        volatile: false,
    });
    let load_g = b.push(
        I32,
        Op::Load {
            ptr: g_addr,
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(Some(load_g)));

    let f = b.finish("global_untouched", vec![I32], I32);
    let m = module_of(vec![f]);

    let out = construct_ssa(&m);
    assert_eq!(out, m, "global-space memory ops must never be promoted");

    assert_roundtrip(&m);
    assert_roundtrip(&out);
}
