// BIR round-trip test: `parse(print(m)) == m`. Enforces the BIR round-trip invariant
// (`cargo test -p basalt-bir roundtrip`).
//
// Fixtures are built by hand rather than derived from any frontend (there isn't one yet).
// Each fixture's instruction arena is populated strictly in block order — the
// discipline `lib.rs` documents as required for the textual form to reconstruct it — via the
// small `FnB` builder below, which assigns each pushed instruction its arena index directly.

use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, Block, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId,
    LaunchBounds, MmaLayout, Module, Op, Scalar, ShuffleKind, Term, Ty, ValRef,
};

/// Appends instructions/blocks to one function's arenas in construction order.
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

    /// Appends a result-producing instruction, returning a reference to its value.
    fn push(&mut self, ty: Ty, op: Op) -> ValRef {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(Inst { ty, op });
        self.cur.push(id);
        ValRef::Val(id)
    }

    /// Appends a void instruction (`store`, `barrier`).
    fn push_void(&mut self, op: Op) {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(Inst { ty: Ty::Void, op });
        self.cur.push(id);
    }

    /// Closes the current block with `term` and starts the next one. Returns the closed
    /// block's id (== its index) so callers can wire up `phi`/`switch` targets.
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

#[test]
fn roundtrip_empty_module() {
    let m = module_of(vec![]);
    assert_roundtrip(&m);
}

#[test]
fn roundtrip_arith_bitwise_compare_select_casts() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);

    let mut b = FnB::new();
    let arg0 = ValRef::Param(0);
    let arg1 = ValRef::Param(1);
    let arg2 = ValRef::Param(2);
    let arg3 = ValRef::Param(3);

    let c42 = b.push(i32t, Op::ConstInt(42));
    let add = b.push(i32t, Op::Bin(BinOp::Add, arg0, arg1));
    let sub = b.push(i32t, Op::Bin(BinOp::Sub, add, c42));
    let mul = b.push(i32t, Op::Bin(BinOp::Mul, sub, arg0));
    let div = b.push(i32t, Op::Bin(BinOp::Div, mul, arg1));
    let rem = b.push(i32t, Op::Bin(BinOp::Rem, div, arg0));

    let c1_5 = b.push(f32t, Op::ConstFloat(1.5));
    let fadd = b.push(f32t, Op::Bin(BinOp::FAdd, arg2, arg3));
    let fsub = b.push(f32t, Op::Bin(BinOp::FSub, fadd, c1_5));
    let fmul = b.push(f32t, Op::Bin(BinOp::FMul, fsub, arg2));
    let fdiv = b.push(f32t, Op::Bin(BinOp::FDiv, fmul, arg3));
    let frem = b.push(f32t, Op::Bin(BinOp::FRem, fdiv, arg2));

    let and = b.push(i32t, Op::Bin(BinOp::And, rem, arg0));
    let or = b.push(i32t, Op::Bin(BinOp::Or, and, arg1));
    let xor = b.push(i32t, Op::Bin(BinOp::Xor, or, arg0));
    let shl = b.push(i32t, Op::Bin(BinOp::Shl, xor, arg1));
    let lshr = b.push(i32t, Op::Bin(BinOp::Lshr, shl, arg0));
    let ashr = b.push(i32t, Op::Bin(BinOp::Ashr, lshr, arg1));

    let icmp = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Slt, i32t, ashr, arg0),
    );
    let _fcmp = b.push(
        Ty::Scalar(Scalar::I1),
        Op::FCmp(FCmpPred::Olt, f32t, frem, arg2),
    );
    let sel = b.push(i32t, Op::Select(icmp, ashr, arg0));

    let tr = b.push(Ty::Scalar(Scalar::I8), Op::Cast(CastOp::Trunc, i32t, sel));
    let ze = b.push(i32t, Op::Cast(CastOp::Zext, Ty::Scalar(Scalar::I8), tr));
    let _se = b.push(Ty::Scalar(Scalar::I64), Op::Cast(CastOp::Sext, i32t, ze));
    let fpt = b.push(
        Ty::Scalar(Scalar::F16),
        Op::Cast(CastOp::FpTrunc, f32t, frem),
    );
    let fpe = b.push(
        Ty::Scalar(Scalar::F64),
        Op::Cast(CastOp::FpExt, Ty::Scalar(Scalar::F16), fpt),
    );
    let f2si = b.push(i32t, Op::Cast(CastOp::FpToSi, Ty::Scalar(Scalar::F64), fpe));
    let f2ui = b.push(i32t, Op::Cast(CastOp::FpToUi, Ty::Scalar(Scalar::F64), fpe));
    let s2f = b.push(f32t, Op::Cast(CastOp::SiToFp, i32t, f2si));
    let _u2f = b.push(
        Ty::Scalar(Scalar::F64),
        Op::Cast(CastOp::UiToFp, i32t, f2ui),
    );
    let bc = b.push(i32t, Op::Cast(CastOp::Bitcast, f32t, s2f));

    b.end_block(Term::Ret(Some(bc)));

    let f = b.finish("arith", vec![i32t, i32t, f32t, f32t], i32t);
    assert_roundtrip(&module_of(vec![f]));
}

#[test]
fn roundtrip_memory_ops_across_address_spaces() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let i64t = Ty::Scalar(Scalar::I64);
    let v4f32 = Ty::Vec(Scalar::F32, 4);

    let ptr_global = ValRef::Param(0);
    let ptr_shared = ValRef::Param(1);
    let ptr_const = ValRef::Param(2);
    let ptr_local = ValRef::Param(3);
    let ptr_param = ValRef::Param(4);

    let mut b = FnB::new();
    let v0 = b.push(
        i32t,
        Op::Load {
            ptr: ptr_global,
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    );
    b.push_void(Op::Store {
        ptr: ptr_global,
        val: v0,
        ty: i32t,
        space: AddrSpace::Global,
        align: 4,
        volatile: true,
    });
    let v1 = b.push(
        f32t,
        Op::Load {
            ptr: ptr_shared,
            space: AddrSpace::Shared,
            align: 4,
            volatile: true,
        },
    );
    b.push_void(Op::Store {
        ptr: ptr_shared,
        val: v1,
        ty: f32t,
        space: AddrSpace::Shared,
        align: 4,
        volatile: false,
    });
    let v2 = b.push(
        i64t,
        Op::Load {
            ptr: ptr_const,
            space: AddrSpace::Constant,
            align: 8,
            volatile: false,
        },
    );
    b.push_void(Op::Store {
        ptr: ptr_local,
        val: v2,
        ty: i64t,
        space: AddrSpace::Local,
        align: 8,
        volatile: false,
    });
    let _v3 = b.push(
        v4f32,
        Op::Load {
            ptr: ptr_param,
            space: AddrSpace::Param,
            align: 16,
            volatile: false,
        },
    );
    b.end_block(Term::Ret(None));

    let params = vec![
        Ty::Ptr(AddrSpace::Global),
        Ty::Ptr(AddrSpace::Shared),
        Ty::Ptr(AddrSpace::Constant),
        Ty::Ptr(AddrSpace::Local),
        Ty::Ptr(AddrSpace::Param),
    ];
    let f = b.finish("mem", params, Ty::Void);
    assert_roundtrip(&module_of(vec![f]));
}

#[test]
fn roundtrip_gpu_intrinsics_and_atomics() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let ptr = ValRef::Param(0);

    let mut b = FnB::new();
    let tid_x = b.push(i32t, Op::TidX);
    let _tid_y = b.push(i32t, Op::TidY);
    let _tid_z = b.push(i32t, Op::TidZ);
    let bid_x = b.push(i32t, Op::BidX);
    let _bid_y = b.push(i32t, Op::BidY);
    let _bid_z = b.push(i32t, Op::BidZ);
    let _bdim_x = b.push(i32t, Op::BdimX);
    let _bdim_y = b.push(i32t, Op::BdimY);
    let _bdim_z = b.push(i32t, Op::BdimZ);
    let _gdim_x = b.push(i32t, Op::GdimX);
    let _gdim_y = b.push(i32t, Op::GdimY);
    let _gdim_z = b.push(i32t, Op::GdimZ);

    b.push_void(Op::Barrier);

    let _s_idx = b.push(i32t, Op::Shuffle(ShuffleKind::Idx, tid_x, bid_x));
    let _s_up = b.push(i32t, Op::Shuffle(ShuffleKind::Up, tid_x, bid_x));
    let _s_down = b.push(i32t, Op::Shuffle(ShuffleKind::Down, tid_x, bid_x));
    let _s_xor = b.push(i32t, Op::Shuffle(ShuffleKind::Xor, tid_x, bid_x));

    let pred = b.push(i1t, Op::ICmp(ICmpPred::Slt, i32t, tid_x, bid_x));
    let _ballot = b.push(i32t, Op::Ballot(pred));
    let _any = b.push(i1t, Op::VoteAny(pred));
    let _all = b.push(i1t, Op::VoteAll(pred));

    let a_add = b.push(
        i32t,
        Op::Atomic(AtomicOp::Add, ptr, tid_x, AddrSpace::Global),
    );
    let _a_sub = b.push(
        i32t,
        Op::Atomic(AtomicOp::Sub, ptr, tid_x, AddrSpace::Global),
    );
    let _a_exch = b.push(
        i32t,
        Op::Atomic(AtomicOp::Exch, ptr, tid_x, AddrSpace::Global),
    );
    let _a_min = b.push(
        i32t,
        Op::Atomic(AtomicOp::Min, ptr, tid_x, AddrSpace::Global),
    );
    let _a_max = b.push(
        i32t,
        Op::Atomic(AtomicOp::Max, ptr, tid_x, AddrSpace::Global),
    );
    let _a_and = b.push(
        i32t,
        Op::Atomic(AtomicOp::And, ptr, tid_x, AddrSpace::Global),
    );
    let _a_or = b.push(
        i32t,
        Op::Atomic(AtomicOp::Or, ptr, tid_x, AddrSpace::Global),
    );
    let _a_xor = b.push(
        i32t,
        Op::Atomic(AtomicOp::Xor, ptr, tid_x, AddrSpace::Global),
    );
    let cas = b.push(i32t, Op::AtomicCas(ptr, tid_x, a_add, AddrSpace::Global));

    b.end_block(Term::Ret(Some(cas)));

    let f = b.finish("gpu", vec![Ty::Ptr(AddrSpace::Global)], i32t);
    assert_roundtrip(&module_of(vec![f]));
}

/// A non-trivial multi-block function: a diamond (`condbr` / `phi`) feeding a `switch`
/// with two cases and a default, each landing on its own `ret`.
#[test]
fn roundtrip_multi_block_phi_and_switch() {
    let i32t = Ty::Scalar(Scalar::I32);
    let arg0 = ValRef::Param(0);

    let mut b = FnB::new();
    let zero = b.push(i32t, Op::ConstInt(0));
    let cond = b.push(
        Ty::Scalar(Scalar::I1),
        Op::ICmp(ICmpPred::Sgt, i32t, arg0, zero),
    );
    // bb0: condbr into the diamond's two arms (bb1, bb2).
    let _bb0 = b.end_block(Term::CondBr(cond, BlockId(1), BlockId(2)));

    let one = b.push(i32t, Op::ConstInt(1));
    let bb1 = b.end_block(Term::Br(BlockId(3)));

    let two = b.push(i32t, Op::ConstInt(2));
    let bb2 = b.end_block(Term::Br(BlockId(3)));

    let merged = b.push(i32t, Op::Phi(vec![(bb1, one), (bb2, two)]));
    b.end_block(Term::Switch(
        merged,
        BlockId(4),
        vec![(1, BlockId(5)), (2, BlockId(6))],
    ));

    // bb4: default arm, returns the merged phi value directly.
    b.end_block(Term::Ret(Some(merged)));

    let hundred = b.push(i32t, Op::ConstInt(100));
    b.end_block(Term::Ret(Some(hundred)));

    let two_hundred = b.push(i32t, Op::ConstInt(200));
    b.end_block(Term::Ret(Some(two_hundred)));

    let f = b.finish("branchy", vec![i32t], i32t);
    assert_eq!(f.blocks.len(), 7);
    assert_roundtrip(&module_of(vec![f]));
}

#[test]
fn roundtrip_module_metadata_and_multiple_functions() {
    let i32t = Ty::Scalar(Scalar::I32);

    let mut b1 = FnB::new();
    let v = b1.push(i32t, Op::ConstInt(7));
    b1.end_block(Term::Ret(Some(v)));
    let f1 = b1.finish("first", vec![], i32t);

    let mut b2 = FnB::new();
    let a = ValRef::Param(0);
    let doubled = b2.push(i32t, Op::Bin(BinOp::Add, a, a));
    b2.end_block(Term::Ret(Some(doubled)));
    let f2 = b2.finish("second", vec![i32t], i32t);

    let m = Module {
        funcs: vec![f1, f2],
        launch_bounds: Some(LaunchBounds {
            max_threads: 256,
            min_blocks: 4,
        }),
        shared_mem_bytes: 8192,
        target_dtypes: vec![Scalar::I32, Scalar::F32, Scalar::F64],
    };
    assert_roundtrip(&m);
}

#[test]
fn roundtrip_mma() {
    let ptr_global = ValRef::Param(0);
    let ptr_shared = ValRef::Param(1);

    let mut b = FnB::new();
    b.push_void(Op::Mma {
        a: ptr_shared,
        b: ptr_shared,
        c: ptr_global,
        d: ptr_global,
        m: 16,
        n: 16,
        k: 16,
        in_dtype: Scalar::F16,
        acc_dtype: Scalar::F32,
        layout_a: MmaLayout::RowMajor,
        layout_b: MmaLayout::ColMajor,
    });
    b.end_block(Term::Ret(None));

    let params = vec![Ty::Ptr(AddrSpace::Global), Ty::Ptr(AddrSpace::Shared)];
    let f = b.finish("mma_fn", params, Ty::Void);
    assert_roundtrip(&module_of(vec![f]));
}

#[test]
fn roundtrip_vector_types_and_bitcast() {
    let v2i32 = Ty::Vec(Scalar::I32, 2);
    let v3f32 = Ty::Vec(Scalar::F32, 3);

    let mut b = FnB::new();
    let arg0 = ValRef::Param(0);
    let bc = b.push(v3f32, Op::Cast(CastOp::Bitcast, v2i32, arg0));
    b.end_block(Term::Ret(Some(bc)));

    let f = b.finish("vecs", vec![v2i32], v3f32);
    assert_roundtrip(&module_of(vec![f]));
}
