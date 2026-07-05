// Coverage: one representative case per BIR op category this backend claims to support (hand-
// built modules, mirroring `basalt-x86`'s own test style), a phi/control-flow test, the real
// frontend/sema/passes pipeline over `tests/kernels/vector_add.cu`, determinism, and the
// refusals this backend actually takes (`f16`, `mma`).

use super::*;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{Block, Inst, MmaLayout};

fn wrap(f: Function) -> Module {
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

/// A single-block function: `insts` in order, terminated by `term`.
fn simple_fn(name: &str, params: Vec<Ty>, insts: Vec<Inst>, term: Term) -> Function {
    let ids = (0..insts.len() as u32).map(InstId).collect();
    Function {
        is_kernel: true,
        name: name.into(),
        params,
        ret: Ty::Void,
        insts,
        blocks: vec![Block { insts: ids, term }],
    }
}

fn emit_text(module: &Module) -> String {
    assert_eq!(Ptx.supports(module), Support::Supported);
    let artifact = Ptx
        .emit(module, &EmitOpts::default())
        .expect("emit succeeds for a supported module");
    artifact
        .as_text()
        .expect("PTX artifact is a text payload")
        .to_string()
}

#[test]
fn preamble_declares_version_target_and_address_size() {
    let f = simple_fn("empty", vec![], vec![], Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains(".version 8.0\n"));
    assert!(text.contains(".target sm_70\n"));
    assert!(text.contains(".address_size 64\n"));
    assert!(text.contains(".visible .entry empty()"));
}

#[test]
fn bin_arithmetic_emits_expected_mnemonics() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Sub, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Mul, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Div, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Rem, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::And, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Shl, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Ashr, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Lshr, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FAdd, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FDiv, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FRem, ValRef::Param(2), ValRef::Param(3)),
        },
    ];
    let f = simple_fn(
        "arith",
        vec![i32t, i32t, f32t, f32t],
        insts,
        Term::Ret(None),
    );
    let text = emit_text(&wrap(f));
    for expect in [
        "add.s32",
        "sub.s32",
        "mul.lo.s32",
        "div.s32",
        "rem.s32",
        "and.b32",
        "shl.b32",
        "shr.s32",
        "shr.u32",
        "add.f32",
        "div.rn.f32",
    ] {
        assert!(text.contains(expect), "missing `{expect}` in:\n{text}");
    }
    // frem has no native PTX instruction; the emulation must still divide/truncate/multiply.
    assert!(text.contains("cvt.rzi.s32.f32"));
}

#[test]
fn i64_and_pointer_arithmetic_share_the_b64_class() {
    let i64t = Ty::Scalar(Scalar::I64);
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let insts = vec![
        Inst {
            ty: i64t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        },
        // Pointer + i64 offset, the exact shape `basalt-sema`'s `lower_ptr_offset` produces.
        Inst {
            ty: ptrt,
            op: Op::Bin(BinOp::Add, ValRef::Param(2), ValRef::Param(0)),
        },
    ];
    let f = simple_fn("ptr_arith", vec![i64t, i64t, ptrt], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("add.s64"));
}

#[test]
fn compares_emit_setp_with_the_right_type_suffix() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(ICmpPred::Slt, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::ICmp(ICmpPred::Ult, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::FCmp(FCmpPred::Olt, f32t, ValRef::Param(2), ValRef::Param(3)),
        },
        Inst {
            ty: Ty::Scalar(Scalar::I1),
            op: Op::FCmp(FCmpPred::Uno, f32t, ValRef::Param(2), ValRef::Param(3)),
        },
    ];
    let f = simple_fn("cmp", vec![i32t, i32t, f32t, f32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("setp.lt.s32"));
    assert!(text.contains("setp.lt.u32"));
    assert!(text.contains("setp.lt.f32"));
    assert!(text.contains("setp.nan.f32"));
}

#[test]
fn casts_emit_cvt() {
    let i8t = Ty::Scalar(Scalar::I8);
    let i32t = Ty::Scalar(Scalar::I32);
    let i64t = Ty::Scalar(Scalar::I64);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: i8t,
            op: Op::Cast(CastOp::Trunc, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Zext, i8t, ValRef::Val(InstId(0))),
        },
        Inst {
            ty: i64t,
            op: Op::Cast(CastOp::Sext, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: f32t,
            op: Op::Cast(CastOp::SiToFp, i32t, ValRef::Param(0)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::FpToSi, f32t, ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Cast(CastOp::Bitcast, f32t, ValRef::Param(1)),
        },
    ];
    let f = simple_fn("casts", vec![i32t, f32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("cvt.u32.u8"));
    assert!(text.contains("cvt.s64.s32"));
    assert!(text.contains("cvt.rn.f32.s32"));
    assert!(text.contains("cvt.rzi.s32.f32"));
    assert!(text.contains("mov.b32")); // the bitcast
}

#[test]
fn load_store_cover_every_address_space() {
    let spaces = [
        AddrSpace::Global,
        AddrSpace::Shared,
        AddrSpace::Constant,
        AddrSpace::Local,
        AddrSpace::Param,
    ];
    let params: Vec<Ty> = spaces.iter().map(|&s| Ty::Ptr(s)).collect();
    let mut insts = Vec::new();
    for (i, &space) in spaces.iter().enumerate() {
        insts.push(Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Load {
                ptr: ValRef::Param(i as u32),
                space,
                align: 4,
                volatile: false,
            },
        });
    }
    for (i, &space) in spaces.iter().enumerate() {
        insts.push(Inst {
            ty: Ty::Void,
            op: Op::Store {
                ptr: ValRef::Param(i as u32),
                val: ValRef::Val(InstId(i as u32)),
                ty: Ty::Scalar(Scalar::I32),
                space,
                align: 4,
                volatile: false,
            },
        });
    }
    let f = simple_fn("loadstore", params, insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    for space_word in ["global", "shared", "const", "local"] {
        assert!(
            text.contains(&format!("ld.{space_word}.u32")),
            "missing ld.{space_word}.u32 in:\n{text}"
        );
        assert!(
            text.contains(&format!("st.{space_word}.u32")),
            "missing st.{space_word}.u32 in:\n{text}"
        );
    }
}

#[test]
fn all_twelve_gpu_index_ops_read_special_registers() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ops = [
        Op::TidX,
        Op::TidY,
        Op::TidZ,
        Op::BidX,
        Op::BidY,
        Op::BidZ,
        Op::BdimX,
        Op::BdimY,
        Op::BdimZ,
        Op::GdimX,
        Op::GdimY,
        Op::GdimZ,
    ];
    let insts = ops.into_iter().map(|op| Inst { ty: i32t, op }).collect();
    let f = simple_fn("indices", vec![], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    for special in [
        "%tid.x",
        "%tid.y",
        "%tid.z",
        "%ctaid.x",
        "%ctaid.y",
        "%ctaid.z",
        "%ntid.x",
        "%ntid.y",
        "%ntid.z",
        "%nctaid.x",
        "%nctaid.y",
        "%nctaid.z",
    ] {
        assert!(text.contains(special), "missing {special} in:\n{text}");
    }
}

#[test]
fn i64_typed_index_op_widens_via_a_32_bit_scratch() {
    // Every `.sreg` special register PTX exposes is natively `.u32`; CUDA-C's own lowering
    // always types these ops `i32` (see `basalt-sema`'s `lower.rs`), but nothing in BIR itself
    // requires that — a frontend is free to type an index op `i64` (Triton's own lowering does
    // exactly this uniformly, see `triton_lower.rs`'s module header). `ptxas` genuinely rejects
    // `mov.u32` into a `.b64` register (`Arguments mismatch for instruction 'mov'`), so this
    // must widen through a 32-bit scratch instead of assuming the destination always matches
    // the special register's own width.
    let i64t = Ty::Scalar(Scalar::I64);
    let f = simple_fn(
        "wide_index",
        vec![],
        vec![Inst {
            ty: i64t,
            op: Op::BidX,
        }],
        Term::Ret(None),
    );
    let text = emit_text(&wrap(f));
    assert!(
        text.contains("mov.u32 %rs0, %ctaid.x;"),
        "expected a 32-bit read of the special register into a scratch: {text}"
    );
    assert!(
        text.contains("cvt.u64.u32 %rd0, %rs0;"),
        "expected the scratch to widen into the real (b64) destination: {text}"
    );
    assert!(
        !text.contains("mov.u32 %rd0, %ctaid.x;"),
        "must never mix a b32 mov into a b64 destination register: {text}"
    );
}

#[test]
fn barrier_emits_real_sync() {
    let f = simple_fn(
        "barrier",
        vec![],
        vec![Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        }],
        Term::Ret(None),
    );
    let text = emit_text(&wrap(f));
    assert!(text.contains("bar.sync 0;"));
}

#[test]
fn all_four_shuffle_kinds_and_vote_ops_emit_sync_forms_with_full_warp_mask() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i1t,
            op: Op::ICmp(ICmpPred::Eq, i32t, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Idx, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Up, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Down, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Shuffle(ShuffleKind::Xor, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Ballot(ValRef::Val(InstId(0))),
        },
        Inst {
            ty: i1t,
            op: Op::VoteAny(ValRef::Val(InstId(0))),
        },
        Inst {
            ty: i1t,
            op: Op::VoteAll(ValRef::Val(InstId(0))),
        },
    ];
    let f = simple_fn("warp", vec![i32t, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("shfl.sync.idx.b32"));
    assert!(text.contains("shfl.sync.up.b32"));
    assert!(text.contains("shfl.sync.down.b32"));
    assert!(text.contains("shfl.sync.bfly.b32"));
    assert!(text.contains("vote.sync.ballot.b32"));
    assert!(text.contains("vote.sync.any.pred"));
    assert!(text.contains("vote.sync.all.pred"));
    assert!(text.contains("0xffffffff"));
}

#[test]
fn shuffle_of_a_64_bit_value_splits_into_two_32_bit_halves() {
    let i64t = Ty::Scalar(Scalar::I64);
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![Inst {
        ty: i64t,
        op: Op::Shuffle(ShuffleKind::Idx, ValRef::Param(0), ValRef::Param(1)),
    }];
    let f = simple_fn("shuffle64", vec![i64t, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("mov.b64"));
    assert!(text.contains("shfl.sync.idx.b32"));
}

#[test]
fn all_eight_atomics_and_cas_emit_atom_instructions() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let atomic_ops = [
        AtomicOp::Add,
        AtomicOp::Sub,
        AtomicOp::Exch,
        AtomicOp::Min,
        AtomicOp::Max,
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
    ];
    let mut insts: Vec<Inst> = atomic_ops
        .into_iter()
        .map(|op| Inst {
            ty: i32t,
            op: Op::Atomic(op, ValRef::Param(0), ValRef::Param(1), AddrSpace::Global),
        })
        .collect();
    insts.push(Inst {
        ty: i32t,
        op: Op::AtomicCas(
            ValRef::Param(0),
            ValRef::Param(1),
            ValRef::Param(1),
            AddrSpace::Global,
        ),
    });
    let f = simple_fn("atomics", vec![ptrt, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("atom.global.add.s32"));
    assert!(text.contains("neg.s32")); // Sub, no native atom.sub
    assert!(text.contains("atom.global.exch.b32"));
    assert!(text.contains("atom.global.min.s32"));
    assert!(text.contains("atom.global.max.s32"));
    assert!(text.contains("atom.global.and.b32"));
    assert!(text.contains("atom.global.or.b32"));
    assert!(text.contains("atom.global.xor.b32"));
    assert!(text.contains("atom.global.cas.b32"));
}

#[test]
fn float_atomic_min_max_lowers_via_a_cas_retry_loop() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![Inst {
        ty: f32t,
        op: Op::Atomic(
            AtomicOp::Min,
            ValRef::Param(0),
            ValRef::Param(1),
            AddrSpace::Global,
        ),
    }];
    let f = simple_fn("atomic_fmin", vec![ptrt, f32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("min.f32"));
    assert!(text.contains("atom.global.cas.b32"));
    assert!(text.contains("bra $atomic_fminmax_loop"));
}

#[test]
fn vector_load_store_uses_native_v4_form() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let vecty = Ty::Vec(Scalar::F32, 4);
    let insts = vec![
        Inst {
            ty: vecty,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 16,
                volatile: false,
            },
        },
        Inst {
            ty: Ty::Void,
            op: Op::Store {
                ptr: ValRef::Param(0),
                val: ValRef::Val(InstId(0)),
                ty: vecty,
                space: AddrSpace::Global,
                align: 16,
                volatile: false,
            },
        },
    ];
    let f = simple_fn("vecload", vec![ptrt], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("ld.global.v4.f32 {%f0, %f1, %f2, %f3}, ["));
    assert!(text.contains("st.global.v4.f32 ["));
}

#[test]
fn vector_load_store_falls_back_to_per_lane_form_for_non_native_lane_counts() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let vecty = Ty::Vec(Scalar::F32, 3);
    let insts = vec![Inst {
        ty: vecty,
        op: Op::Load {
            ptr: ValRef::Param(0),
            space: AddrSpace::Global,
            align: 4,
            volatile: false,
        },
    }];
    let f = simple_fn("vec3load", vec![ptrt], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("ld.global.f32 %f0, ["));
    assert!(text.contains("+4]"));
    assert!(text.contains("+8]"));
    assert!(!text.contains(".v3."));
}

#[test]
fn vector_bin_decomposes_per_lane() {
    let vecty = Ty::Vec(Scalar::F32, 2);
    let insts = vec![Inst {
        ty: vecty,
        op: Op::Bin(BinOp::FAdd, ValRef::Param(0), ValRef::Param(1)),
    }];
    let f = simple_fn("vecadd", vec![vecty, vecty], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("add.f32 %f2, %f0, %f2;") || text.contains("add.f32"));
    // Two independent lane-wise adds, one per destination register.
    assert_eq!(text.matches("add.f32").count(), 2);
}

#[test]
fn phi_resolves_via_a_mov_per_predecessor_edge() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::ConstInt(0),
        },
        Inst {
            ty: i1t,
            op: Op::ICmp(
                ICmpPred::Sgt,
                i32t,
                ValRef::Param(0),
                ValRef::Val(InstId(0)),
            ),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(10),
        },
        Inst {
            ty: i32t,
            op: Op::ConstInt(20),
        },
        Inst {
            ty: i32t,
            op: Op::Phi(vec![
                (basalt_bir::BlockId(1), ValRef::Val(InstId(2))),
                (basalt_bir::BlockId(2), ValRef::Val(InstId(3))),
            ]),
        },
    ];
    let f = Function {
        is_kernel: true,
        name: "phi_fn".into(),
        params: vec![i32t],
        ret: Ty::Void,
        insts,
        blocks: vec![
            Block {
                insts: vec![InstId(0), InstId(1)],
                term: Term::CondBr(
                    ValRef::Val(InstId(1)),
                    basalt_bir::BlockId(1),
                    basalt_bir::BlockId(2),
                ),
            },
            Block {
                insts: vec![InstId(2)],
                term: Term::Br(basalt_bir::BlockId(3)),
            },
            Block {
                insts: vec![InstId(3)],
                term: Term::Br(basalt_bir::BlockId(3)),
            },
            Block {
                insts: vec![InstId(4)],
                term: Term::Ret(None),
            },
        ],
    };
    let text = emit_text(&wrap(f));
    // param0 -> %r0 (B32), inst0 (ConstInt) -> %r1, inst1 (ICmp) -> %p0, inst2 -> %r2,
    // inst3 -> %r3, inst4 (Phi) -> %r4: each predecessor copies its own value into %r4.
    assert!(
        text.contains("mov.b32 %r4, %r2;"),
        "bb1 -> phi copy missing:\n{text}"
    );
    assert!(
        text.contains("mov.b32 %r4, %r3;"),
        "bb2 -> phi copy missing:\n{text}"
    );
    assert!(text.contains("@%p0 bra $L1;"));
    assert!(text.contains("bra $L2;"));
}

#[test]
fn f16_arithmetic_is_refused_not_guessed() {
    let f16t = Ty::Scalar(Scalar::F16);
    let f = simple_fn(
        "f16_add",
        vec![f16t, f16t],
        vec![Inst {
            ty: f16t,
            op: Op::Bin(BinOp::FAdd, ValRef::Param(0), ValRef::Param(1)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Ptx.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedType)
    );
    assert!(Ptx.emit(&module, &EmitOpts::default()).is_err());
}

#[test]
fn mma_is_refused_not_guessed() {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let f = simple_fn(
        "usesmma",
        vec![ptr_global, ptr_global, ptr_global, ptr_global],
        vec![Inst {
            ty: Ty::Void,
            op: Op::Mma {
                a: ValRef::Param(0),
                b: ValRef::Param(1),
                c: ValRef::Param(2),
                d: ValRef::Param(3),
                m: 2,
                n: 2,
                k: 2,
                in_dtype: Scalar::F32,
                acc_dtype: Scalar::F32,
                layout_a: MmaLayout::RowMajor,
                layout_b: MmaLayout::RowMajor,
            },
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Ptx.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Ptx
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

/// P13-T1b's kernel-launch/CUDA-Runtime-API ops are sema-only today (see
/// `basalt_bir::Op::KernelLaunch`'s own doc comment) — every backend refuses them cleanly.
#[test]
fn kernel_launch_and_cuda_runtime_api_ops_are_refused_not_guessed() {
    let f = simple_fn(
        "launch_stub",
        vec![],
        vec![Inst {
            ty: Ty::Void,
            op: Op::CudaDeviceSynchronize,
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    assert_eq!(
        Ptx.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Ptx
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

/// `Op::Call` (P13-T-calls-i) has no lowering in this backend yet — refuse cleanly rather
/// than falling through to the scalar per-op emitters, which have no case for it.
#[test]
fn function_call_is_refused_not_guessed() {
    let f = simple_fn(
        "caller",
        vec![Ty::Scalar(Scalar::I32)],
        vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Call {
                func: "callee".to_string(),
                args: vec![ValRef::Param(0)],
            },
        }],
        Term::Ret(Some(ValRef::Val(InstId(0)))),
    );
    let module = wrap(f);
    assert_eq!(
        Ptx.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedOp)
    );
    let err = Ptx
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedOp);
}

#[test]
fn non_kernel_function_is_refused_not_silently_emitted_as_a_kernel() {
    // The live gap this test guards: every function in a module is emitted as its own
    // `.visible .entry` kernel (see `emit_module`), so a non-kernel function must be refused
    // rather than silently miscompiled as a launchable one.
    let mut f = simple_fn("host_fn", vec![], vec![], Term::Ret(None));
    f.is_kernel = false;
    let module = wrap(f);
    assert_eq!(
        Ptx.supports(&module),
        Support::Unsupported(basalt_diag::ECode::UnsupportedFeature)
    );
    let err = Ptx
        .emit(&module, &EmitOpts::default())
        .expect_err("emit must refuse what supports() refuses, not guess");
    assert_eq!(err.code, basalt_diag::ECode::UnsupportedFeature);
}

#[test]
fn emit_is_deterministic() {
    let f = simple_fn(
        "det",
        vec![Ty::Scalar(Scalar::I32)],
        vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(0)),
        }],
        Term::Ret(None),
    );
    let module = wrap(f);
    let a = emit_text(&module);
    let b = emit_text(&module);
    assert_eq!(a, b);
}

#[test]
fn ret_with_a_value_at_kernel_scope_drops_it_and_emits_a_plain_ret() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn(
        "retval",
        vec![i32t],
        vec![],
        Term::Ret(Some(ValRef::Param(0))),
    );
    let text = emit_text(&wrap(f));
    assert!(text.contains("\tret;\n"));
}

// ---- real pipeline: tests/kernels/vector_add.cu -------------------------------------------

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

fn lower_vector_add() -> Module {
    let (tokens, pp_errors) =
        basalt_frontend_c::preprocess(VECTOR_ADD_SRC, &basalt_frontend_c::PpOpts::default());
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
    module
}

#[test]
fn vector_add_emits_a_visible_entry_kernel_via_the_real_pipeline() {
    let module = lower_vector_add();
    assert_eq!(Ptx.supports(&module), Support::Supported);
    let text = emit_text(&module);

    assert!(text.contains(".version 8.0\n"));
    assert!(text.contains(".target sm_70\n"));
    assert!(text.contains(".visible .entry vector_add("));
    assert!(text.contains("%tid.x"));
    assert!(text.contains("%ctaid.x"));
    assert!(text.contains("%ntid.x"));
    assert!(text.contains("ld.global.f32"));
    assert!(text.contains("st.global.f32"));
}

#[test]
fn vector_add_emit_is_deterministic_through_the_real_pipeline() {
    let module = lower_vector_add();
    let a = emit_text(&module);
    let b = emit_text(&module);
    assert_eq!(a, b);
}
