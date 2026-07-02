// Coverage: one representative case per lowered construct (hand-built modules, mirroring
// `basalt-ptx`'s own test style), an if/else with a real merge-block phi, every `CastOp`
// variant, load/store, and the explicit refusal path for an out-of-scope op. Every lowered
// module is checked against LLVM's own verifier, the strongest correctness signal available
// without target-machine object emission (out of scope for this file).

use super::*;
use basalt_bir::{Block, InstId, LaunchBounds, MmaLayout};
use inkwell::context::Context;

fn wrap(f: Function) -> Module {
    Module {
        funcs: vec![f],
        launch_bounds: None::<LaunchBounds>,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

fn ids(n: usize) -> Vec<InstId> {
    (0..n as u32).map(InstId).collect()
}

#[test]
fn trivial_function_returns_a_constant() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "answer".into(),
        params: vec![],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::ConstInt(42),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("define i32 @answer()"));
    assert!(text.contains("ret i32 42"));
}

#[test]
fn arithmetic_and_return() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "add".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("add i32"));
}

#[test]
fn signed_div_and_rem_use_the_signed_convention() {
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Div, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Rem, ValRef::Param(0), ValRef::Param(1)),
        },
    ];
    let f = Function {
        name: "divrem".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts,
        blocks: vec![Block {
            insts: ids(2),
            term: Term::Ret(Some(ValRef::Val(InstId(1)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("sdiv i32"));
    assert!(text.contains("srem i32"));
}

#[test]
fn if_else_with_merge_block_phi() {
    let i1t = Ty::Scalar(Scalar::I1);
    let i32t = Ty::Scalar(Scalar::I32);

    // bb0: %0 = icmp slt %arg0, %arg1; condbr %0, bb1, bb2
    // bb1: %1 = const.i 1; %2 = add %arg0, %1; br bb3
    // bb2: %3 = const.i 2; %4 = add %arg1, %3; br bb3
    // bb3: %5 = phi [bb1 -> %2, bb2 -> %4]; ret %5
    let insts = vec![
        Inst {
            ty: i1t,
            op: Op::ICmp(ICmpPred::Slt, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(1),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(1))),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(2),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(1), ValRef::Val(InstId(3))),
        },
        Inst {
            ty: i32t,
            op: Op::Phi(vec![
                (BlockId(1), ValRef::Val(InstId(2))),
                (BlockId(2), ValRef::Val(InstId(4))),
            ]),
        },
    ];
    let blocks = vec![
        Block {
            insts: vec![InstId(0)],
            term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
        },
        Block {
            insts: vec![InstId(1), InstId(2)],
            term: Term::Br(BlockId(3)),
        },
        Block {
            insts: vec![InstId(3), InstId(4)],
            term: Term::Br(BlockId(3)),
        },
        Block {
            insts: vec![InstId(5)],
            term: Term::Ret(Some(ValRef::Val(InstId(5)))),
        },
    ];
    let f = Function {
        name: "branchy".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts,
        blocks,
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("phi i32"));
    assert!(text.contains("icmp slt"));
    assert!(text.contains("br i1"));
}

#[test]
fn every_cast_op_variant_lowers_and_verifies() {
    let i8t = Ty::Scalar(Scalar::I8);
    let i32t = Ty::Scalar(Scalar::I32);
    let i64t = Ty::Scalar(Scalar::I64);
    let f32t = Ty::Scalar(Scalar::F32);
    let f64t = Ty::Scalar(Scalar::F64);

    let params = vec![i64t, i8t, i8t, f64t, f32t, f32t, f32t, i32t, i32t, i32t];
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Trunc, i64t, ValRef::Param(0)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Zext, i8t, ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Sext, i8t, ValRef::Param(2)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::FpTrunc, f64t, ValRef::Param(3)),
        },
        Inst {
            ty: f64t,
            op: Op::Cast(CastOp::FpExt, f32t, ValRef::Param(4)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::FpToSi, f32t, ValRef::Param(5)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::FpToUi, f32t, ValRef::Param(6)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::SiToFp, i32t, ValRef::Param(7)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::UiToFp, i32t, ValRef::Param(8)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::Bitcast, i32t, ValRef::Param(9)),
        },
    ];
    let n = insts.len();
    let f = Function {
        name: "casts".into(),
        params,
        ret: Ty::Void,
        insts,
        blocks: vec![Block {
            insts: ids(n),
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    for mnemonic in [
        "trunc", "zext", "sext", "fptrunc", "fpext", "fptosi", "fptoui", "sitofp", "uitofp",
        "bitcast",
    ] {
        assert!(text.contains(mnemonic), "missing `{mnemonic}` in:\n{text}");
    }
}

#[test]
fn bitcast_between_identical_llvm_types_is_a_value_passthrough() {
    // Two BIR pointer types in different address spaces both map to LLVM's single opaque
    // `ptr`, so bitcasting between them must not try to emit an invalid same-type `bitcast`.
    let global_ptr = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let shared_ptr = Ty::Ptr(basalt_bir::AddrSpace::Shared);
    let f = Function {
        name: "ptrcast".into(),
        params: vec![global_ptr],
        ret: Ty::Void,
        insts: vec![Inst {
            ty: shared_ptr,
            op: Op::Cast(CastOp::Bitcast, global_ptr, ValRef::Param(0)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
}

#[test]
fn load_and_store_round_trip() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let f = Function {
        name: "memcopy".into(),
        params: vec![ptrt],
        ret: Ty::Void,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::Load {
                    ptr: ValRef::Param(0),
                    space: basalt_bir::AddrSpace::Global,
                    align: 4,
                    volatile: false,
                },
            },
            Inst {
                ty: Ty::Void,
                op: Op::Store {
                    ptr: ValRef::Param(0),
                    val: ValRef::Val(InstId(0)),
                    ty: i32t,
                    space: basalt_bir::AddrSpace::Global,
                    align: 4,
                    volatile: false,
                },
            },
        ],
        blocks: vec![Block {
            insts: ids(2),
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("load i32"));
    assert!(text.contains("store i32"));
}

#[test]
fn out_of_scope_vector_type_is_a_clean_refusal_not_a_panic() {
    let vecty = Ty::Vec(Scalar::F32, 4);
    let f = Function {
        name: "usesvec".into(),
        params: vec![vecty],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![Block {
            insts: vec![],
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx)
        .expect_err("vector types are out of scope for this lane");
    assert_eq!(err.code, ECode::UnsupportedType);
}

fn tid_x_fn(i32t: Ty) -> Function {
    Function {
        name: "usesthreadidx".into(),
        params: vec![],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::TidX,
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    }
}

#[test]
fn nvptx_tid_x_lowers_to_the_nvvm_sreg_intrinsic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ctx = Context::create();
    let llvm_mod =
        lower_module(&wrap(tid_x_fn(i32t)), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call i32 @llvm.nvvm.read.ptx.sreg.tid.x()"),
        "{text}"
    );
}

#[test]
fn amdgpu_tid_x_lowers_to_the_amdgcn_workitem_intrinsic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ctx = Context::create();
    let llvm_mod =
        lower_module(&wrap(tid_x_fn(i32t)), &ctx, GpuDialect::Amdgpu).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call i32 @llvm.amdgcn.workitem.id.x()"),
        "{text}"
    );
}

#[test]
fn amdgpu_block_dim_is_a_clean_refusal_not_a_panic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "usesblockdim".into(),
        params: vec![],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::BdimX,
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Amdgpu)
        .expect_err("bdim.x has no confident amdgpu mapping in this lane");
    assert_eq!(err.code, ECode::UnsupportedOp);
}

/// Builds a single-instruction `mma` kernel over four global pointer params (`a`,`b`,`c`,`d`
/// in that order), letting each case pick its own shape/dtype/layout.
#[allow(clippy::too_many_arguments)]
fn mma_fn(
    m: u32,
    n: u32,
    k: u32,
    in_dtype: Scalar,
    acc_dtype: Scalar,
    layout_a: MmaLayout,
    layout_b: MmaLayout,
) -> Function {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    Function {
        name: "usesmma".into(),
        params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
        ret: Ty::Void,
        insts: vec![Inst {
            ty: Ty::Void,
            op: Op::Mma {
                a: ValRef::Param(0),
                b: ValRef::Param(1),
                c: ValRef::Param(2),
                d: ValRef::Param(3),
                m,
                n,
                k,
                in_dtype,
                acc_dtype,
                layout_a,
                layout_b,
            },
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(None),
        }],
    }
}

#[test]
fn nvptx_mma_unsupported_shape_is_a_clean_refusal_not_a_panic() {
    let f = mma_fn(
        2,
        2,
        2,
        Scalar::F32,
        Scalar::F32,
        MmaLayout::RowMajor,
        MmaLayout::RowMajor,
    );

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx)
        .expect_err("only the canonical m16n16k16 f16-input tile is lowered in this lane");
    assert_eq!(err.code, ECode::UnsupportedType);
}

#[test]
fn nvptx_mma_unsupported_accumulator_is_a_clean_refusal_not_a_panic() {
    let f = mma_fn(
        16,
        16,
        16,
        Scalar::F16,
        Scalar::I32,
        MmaLayout::RowMajor,
        MmaLayout::RowMajor,
    );

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx)
        .expect_err("only an f16 or f32 accumulator is lowered in this lane");
    assert_eq!(err.code, ECode::UnsupportedType);
}

#[test]
fn amdgpu_mma_is_a_clean_refusal_not_a_panic_even_for_the_canonical_shape() {
    // Amdgcn's tensor-core path (MFMA) belongs to a separate, hand-rolled backend. This must
    // stay a clean refusal no matter the shape/dtype, including the exact tile this lane
    // lowers for real on Nvptx.
    let f = mma_fn(
        16,
        16,
        16,
        Scalar::F16,
        Scalar::F32,
        MmaLayout::RowMajor,
        MmaLayout::RowMajor,
    );

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Amdgpu)
        .expect_err("mma has no LLVM IR lowering in this lane yet");
    assert_eq!(err.code, ECode::UnsupportedOp);
}

#[test]
fn nvptx_canonical_wmma_f32_accumulator_lowers_and_verifies() {
    let f = mma_fn(
        16,
        16,
        16,
        Scalar::F16,
        Scalar::F32,
        MmaLayout::RowMajor,
        MmaLayout::RowMajor,
    );

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16"),
        "{text}"
    );
    assert!(
        text.contains("call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32"),
        "{text}"
    );
    assert!(
        text.contains("call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32"),
        "{text}"
    );
}

#[test]
fn nvptx_canonical_wmma_f16_accumulator_lowers_and_verifies() {
    let f = mma_fn(
        16,
        16,
        16,
        Scalar::F16,
        Scalar::F16,
        MmaLayout::ColMajor,
        MmaLayout::RowMajor,
    );

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("@llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16"),
        "{text}"
    );
    assert!(
        text.contains("@llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16"),
        "{text}"
    );
    assert!(
        text.contains("@llvm.nvvm.wmma.m16n16k16.mma.col.row.f16.f16"),
        "{text}"
    );
    assert!(
        text.contains("@llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16"),
        "{text}"
    );
}

#[test]
fn nvptx_barrier_lowers_to_barrier0() {
    let f = Function {
        name: "syncs".into(),
        params: vec![],
        ret: Ty::Void,
        insts: vec![Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("call void @llvm.nvvm.barrier0()"), "{text}");
}

#[test]
fn amdgpu_barrier_lowers_to_s_barrier() {
    let f = Function {
        name: "syncs".into(),
        params: vec![],
        ret: Ty::Void,
        insts: vec![Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(None),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Amdgpu).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call void @llvm.amdgcn.s.barrier()"),
        "{text}"
    );
}

#[test]
fn nvptx_shuffle_idx_lowers_to_shfl_sync() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "shuf".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Idx, ValRef::Param(0), ValRef::Param(1)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call i32 @llvm.nvvm.shfl.sync.idx.i32("),
        "{text}"
    );
}

#[test]
fn amdgpu_shuffle_is_a_clean_refusal_not_a_panic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "shuf".into(),
        params: vec![i32t, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Xor, ValRef::Param(0), ValRef::Param(1)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Amdgpu)
        .expect_err("shuffle has no settled amdgpu mapping in this lane");
    assert_eq!(err.code, ECode::UnsupportedOp);
}

fn ballot_fn(i32t: Ty) -> Function {
    Function {
        name: "votes".into(),
        params: vec![i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Ballot(ValRef::Param(0)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    }
}

#[test]
fn nvptx_ballot_lowers_to_vote_ballot_sync() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ctx = Context::create();
    let llvm_mod =
        lower_module(&wrap(ballot_fn(i32t)), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(
        text.contains("call i32 @llvm.nvvm.vote.ballot.sync("),
        "{text}"
    );
    // The truthy i32 predicate operand must be turned into a real i1 before the call.
    assert!(text.contains("icmp ne i32"), "{text}");
}

#[test]
fn amdgpu_ballot_lowers_to_amdgcn_ballot_at_the_requested_width() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ctx = Context::create();
    let llvm_mod =
        lower_module(&wrap(ballot_fn(i32t)), &ctx, GpuDialect::Amdgpu).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("call i32 @llvm.amdgcn.ballot.i32("), "{text}");
}

#[test]
fn nvptx_vote_any_and_all_lower_and_verify() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "votes".into(),
        params: vec![i32t],
        ret: i32t,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::VoteAny(ValRef::Param(0)),
            },
            Inst {
                ty: i32t,
                op: Op::VoteAll(ValRef::Param(0)),
            },
            Inst {
                ty: i32t,
                op: Op::Bin(BinOp::Add, ValRef::Val(InstId(0)), ValRef::Val(InstId(1))),
            },
        ],
        blocks: vec![Block {
            insts: ids(3),
            term: Term::Ret(Some(ValRef::Val(InstId(2)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("call i1 @llvm.nvvm.vote.any.sync("), "{text}");
    assert!(text.contains("call i1 @llvm.nvvm.vote.all.sync("), "{text}");
}

#[test]
fn amdgpu_vote_any_is_a_clean_refusal_not_a_panic() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = Function {
        name: "votes".into(),
        params: vec![i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::VoteAny(ValRef::Param(0)),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Amdgpu)
        .expect_err("vote.any has no settled amdgpu mapping in this lane");
    assert_eq!(err.code, ECode::UnsupportedOp);
}

/// Atomics lower through target-agnostic `atomicrmw`/`cmpxchg` IR, not intrinsics, so the
/// same code path must produce valid IR under either `GpuDialect`.
#[test]
fn atomic_add_lowers_identically_regardless_of_dialect() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let f = Function {
        name: "bump".into(),
        params: vec![ptrt],
        ret: i32t,
        insts: vec![
            Inst {
                ty: i32t,
                op: Op::ConstInt(1),
            },
            Inst {
                ty: i32t,
                op: Op::Atomic(
                    AtomicOp::Add,
                    ValRef::Param(0),
                    ValRef::Val(InstId(0)),
                    basalt_bir::AddrSpace::Global,
                ),
            },
        ],
        blocks: vec![Block {
            insts: ids(2),
            term: Term::Ret(Some(ValRef::Val(InstId(1)))),
        }],
    };

    for dialect in [GpuDialect::Nvptx, GpuDialect::Amdgpu] {
        let ctx = Context::create();
        let llvm_mod = lower_module(&wrap(f.clone()), &ctx, dialect).expect("lowering succeeds");
        llvm_mod.verify().expect("module verifies");
        let text = llvm_mod.print_to_string().to_string();
        assert!(text.contains("atomicrmw add ptr"), "{text}");
    }
}

#[test]
fn atomic_min_uses_the_signed_atomicrmw_variant() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let f = Function {
        name: "clampmin".into(),
        params: vec![ptrt, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::Atomic(
                AtomicOp::Min,
                ValRef::Param(0),
                ValRef::Param(1),
                basalt_bir::AddrSpace::Global,
            ),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let llvm_mod = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx).expect("lowering succeeds");
    llvm_mod.verify().expect("module verifies");
    let text = llvm_mod.print_to_string().to_string();
    assert!(text.contains("atomicrmw min ptr"), "{text}");
}

#[test]
fn atomic_rmw_on_a_float_type_is_a_clean_refusal_not_a_panic() {
    let f32t = Ty::Scalar(Scalar::F32);
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let f = Function {
        name: "faddatomic".into(),
        params: vec![ptrt, f32t],
        ret: f32t,
        insts: vec![Inst {
            ty: f32t,
            op: Op::Atomic(
                AtomicOp::Add,
                ValRef::Param(0),
                ValRef::Param(1),
                basalt_bir::AddrSpace::Global,
            ),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    let ctx = Context::create();
    let err = lower_module(&wrap(f), &ctx, GpuDialect::Nvptx)
        .expect_err("float atomicrmw is out of reach of inkwell 0.9's typed wrapper");
    assert_eq!(err.code, ECode::UnsupportedType);
}

#[test]
fn atomic_cas_lowers_to_cmpxchg_and_extracts_the_old_value() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ptrt = Ty::Ptr(basalt_bir::AddrSpace::Global);
    let f = Function {
        name: "cas".into(),
        params: vec![ptrt, i32t, i32t],
        ret: i32t,
        insts: vec![Inst {
            ty: i32t,
            op: Op::AtomicCas(
                ValRef::Param(0),
                ValRef::Param(1),
                ValRef::Param(2),
                basalt_bir::AddrSpace::Global,
            ),
        }],
        blocks: vec![Block {
            insts: ids(1),
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };

    for dialect in [GpuDialect::Nvptx, GpuDialect::Amdgpu] {
        let ctx = Context::create();
        let llvm_mod = lower_module(&wrap(f.clone()), &ctx, dialect).expect("lowering succeeds");
        llvm_mod.verify().expect("module verifies");
        let text = llvm_mod.print_to_string().to_string();
        assert!(text.contains("cmpxchg ptr"), "{text}");
        assert!(text.contains("extractvalue"), "{text}");
    }
}
