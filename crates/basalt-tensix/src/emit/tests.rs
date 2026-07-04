// Coverage: one representative case per BIR op category this backend claims to support (hand-
// built modules, mirroring `basalt-ptx`'s own test style), a phi/control-flow test, the real
// frontend/sema/passes pipeline over `tests/kernels/vector_add.cu`, determinism, and the
// refusals this backend actually takes (`f16`, `mma`, non-global address spaces, atomics,
// warp-collectives, vector types, multi-function modules, inconsistent pointer element types).

use super::*;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{Block, BlockId, Inst};

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
    assert_eq!(Tensix.supports(module), Support::Supported);
    let artifact = Tensix
        .emit(module, &EmitOpts::default())
        .expect("emit succeeds for a supported module");
    artifact
        .as_text()
        .expect("Tensix artifact is a text payload")
        .to_string()
}

fn unsupported_code(module: &Module) -> ECode {
    match Tensix.supports(module) {
        Support::Unsupported(code) => code,
        Support::Supported => panic!("expected this module to be refused"),
    }
}

#[test]
fn preamble_declares_kernel_main_and_runtime_args() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f = simple_fn("scalar_only", vec![i32t], vec![], Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.starts_with("void kernel_main() {\n"));
    assert!(text.contains("int32_t p0 = (int32_t)get_arg_val<uint32_t>(0);"));
    assert!(text.contains("uint32_t nthreads = get_arg_val<uint32_t>(1);"));
    assert!(text.contains("for (uint32_t __tid = 0; __tid < nthreads; ++__tid) {"));
    assert!(text.contains("continue;"));
    assert!(text.ends_with("}\n"));
    // A single-block function's entry is reached by falling out of the `for` line, never by a
    // `goto` naming it — printing an unreferenced label there would be a real
    // `-Werror=unused-label` failure against tt_metal's own build flags (see `goto_targets`).
    assert!(!text.contains("L0:"));
}

#[test]
fn only_actual_goto_targets_get_a_label() {
    let i1t = Ty::Scalar(Scalar::I1);
    let f = Function {
        is_kernel: true,
        name: "branch".into(),
        params: vec![i1t],
        ret: Ty::Void,
        insts: vec![],
        blocks: vec![
            Block {
                insts: vec![],
                term: Term::CondBr(ValRef::Param(0), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
            Block {
                insts: vec![],
                term: Term::Ret(None),
            },
        ],
    };
    let text = emit_text(&wrap(f));
    // bb0 is never a goto target (it is the loop's own entry); bb1/bb2 are, via the condbr.
    assert!(!text.contains("L0:"));
    assert!(text.contains("L1:"));
    assert!(text.contains("L2:"));
}

#[test]
fn bin_arithmetic_emits_expected_c_operators() {
    // f32/f64 scalar kernel *parameters* are refused (see the module header), so the float
    // operands here are computed via `ConstFloat` rather than taken in as params.
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Div, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: i32t,
            op: Op::Bin(BinOp::Lshr, ValRef::Param(0), ValRef::Param(1)),
        },
        Inst {
            ty: f32t,
            op: Op::ConstFloat(2.5),
        },
        Inst {
            ty: f32t,
            op: Op::ConstFloat(1.5),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FAdd, ValRef::Val(InstId(3)), ValRef::Val(InstId(4))),
        },
        Inst {
            ty: f32t,
            op: Op::Bin(BinOp::FRem, ValRef::Val(InstId(3)), ValRef::Val(InstId(4))),
        },
    ];
    let f = simple_fn("arith", vec![i32t, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("v0 = (p0 + p1);"));
    assert!(text.contains("v1 = (p0 / p1);"));
    assert!(text.contains("v2 = (int32_t)((uint32_t)p0 >> p1);"));
    assert!(text.contains("v5 = (v3 + v4);"));
    // frem has no native C++ operator on floats; the emulation still truncates via a cast.
    assert!(text.contains("(int32_t)(v3 / v4)"));
}

#[test]
fn unsigned_icmp_casts_both_operands() {
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![Inst {
        ty: Ty::Scalar(Scalar::I1),
        op: Op::ICmp(ICmpPred::Ult, i32t, ValRef::Param(0), ValRef::Param(1)),
    }];
    let f = simple_fn("ucmp", vec![i32t, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("v0 = ((uint32_t)p0 < (uint32_t)p1);"));
}

#[test]
fn select_lowers_to_a_ternary() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![Inst {
        ty: i32t,
        op: Op::Select(ValRef::Param(0), ValRef::Param(1), ValRef::Param(2)),
    }];
    let f = simple_fn("sel", vec![i1t, i32t, i32t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("v0 = p0 ? p1 : p2;"));
}

#[test]
fn zext_routes_through_the_unsigned_source_width() {
    let i1t = Ty::Scalar(Scalar::I1);
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![Inst {
        ty: i32t,
        op: Op::Cast(CastOp::Zext, i1t, ValRef::Param(0)),
    }];
    let f = simple_fn("zext", vec![i1t], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("v0 = (int32_t)(bool)p0;"));
}

#[test]
fn gpu_index_ops_match_the_oracles_single_block_table() {
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::TidX,
        },
        Inst {
            ty: i32t,
            op: Op::BdimX,
        },
        Inst {
            ty: i32t,
            op: Op::BidX,
        },
        Inst {
            ty: i32t,
            op: Op::GdimX,
        },
        Inst {
            ty: Ty::Void,
            op: Op::Barrier,
        },
    ];
    let f = simple_fn("idx", vec![], insts, Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("v0 = (int32_t)__tid;"));
    assert!(text.contains("v1 = (int32_t)nthreads;"));
    assert!(text.contains("v2 = (int32_t)0;"));
    assert!(text.contains("v3 = (int32_t)1;"));
    assert!(text.contains("barrier: no-op"));
}

#[test]
fn condbr_with_phi_emits_copies_on_both_arms() {
    let i32t = Ty::Scalar(Scalar::I32);
    let i1t = Ty::Scalar(Scalar::I1);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Phi(vec![
                (BlockId(1), ValRef::Param(1)),
                (BlockId(2), ValRef::Param(2)),
            ]),
        }, // %0 in bb3
    ];
    let f = Function {
        is_kernel: true,
        name: "diamond".into(),
        params: vec![i1t, i32t, i32t],
        ret: Ty::Void,
        insts,
        blocks: vec![
            Block {
                insts: vec![],
                term: Term::CondBr(ValRef::Param(0), BlockId(1), BlockId(2)),
            },
            Block {
                insts: vec![],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![],
                term: Term::Br(BlockId(3)),
            },
            Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            },
        ],
    };
    let text = emit_text(&wrap(f));
    assert!(text.contains("if (p0) {"));
    assert!(text.contains("v0 = p1;"));
    assert!(text.contains("v0 = p2;"));
    assert!(text.contains("goto L1;"));
    assert!(text.contains("goto L2;"));
}

#[test]
fn ret_advances_the_loop_instead_of_returning() {
    let f = simple_fn("void_ret", vec![], vec![], Term::Ret(None));
    let text = emit_text(&wrap(f));
    assert!(text.contains("continue;"));
    assert!(!text.contains("return;"));
}

// ---- refusals -----------------------------------------------------------------------------

#[test]
fn f16_refuses() {
    let f16t = Ty::Scalar(Scalar::F16);
    let f = simple_fn("half", vec![f16t], vec![], Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedType);
}

#[test]
fn vector_types_refuse() {
    let vt = Ty::Vec(Scalar::F32, 4);
    let f = simple_fn("vecty", vec![vt], vec![], Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedType);
}

#[test]
fn non_global_address_space_refuses() {
    let ptrt = Ty::Ptr(AddrSpace::Shared);
    let f = simple_fn("sharedptr", vec![ptrt], vec![], Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedAddressSpace);
}

#[test]
fn mma_refuses() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let insts = vec![Inst {
        ty: Ty::Void,
        op: Op::Mma {
            a: ValRef::Param(0),
            b: ValRef::Param(0),
            c: ValRef::Param(0),
            d: ValRef::Param(0),
            m: 16,
            n: 16,
            k: 16,
            in_dtype: Scalar::F32,
            acc_dtype: Scalar::F32,
            layout_a: basalt_bir::MmaLayout::RowMajor,
            layout_b: basalt_bir::MmaLayout::RowMajor,
        },
    }];
    let f = simple_fn("mma", vec![ptrt], insts, Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

/// P13-T1b's kernel-launch/CUDA-Runtime-API ops are sema-only today (see
/// `basalt_bir::Op::KernelLaunch`'s own doc comment) — every backend refuses them cleanly.
#[test]
fn kernel_launch_and_cuda_runtime_api_ops_refuse() {
    let insts = vec![Inst {
        ty: Ty::Void,
        op: Op::CudaDeviceSynchronize,
    }];
    let f = simple_fn("launch_stub", vec![], insts, Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

#[test]
fn warp_collective_ops_refuse() {
    let i32t = Ty::Scalar(Scalar::I32);
    let insts = vec![Inst {
        ty: i32t,
        op: Op::Ballot(ValRef::Param(0)),
    }];
    let f = simple_fn(
        "ballot",
        vec![Ty::Scalar(Scalar::I1)],
        insts,
        Term::Ret(None),
    );
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

#[test]
fn atomics_refuse() {
    let i32t = Ty::Scalar(Scalar::I32);
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let insts = vec![Inst {
        ty: i32t,
        op: Op::Atomic(
            basalt_bir::AtomicOp::Add,
            ValRef::Param(0),
            ValRef::Param(1),
            AddrSpace::Global,
        ),
    }];
    let f = simple_fn("atomic", vec![ptrt, i32t], insts, Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

#[test]
fn bitcast_refuses() {
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![Inst {
        ty: f32t,
        op: Op::Cast(CastOp::Bitcast, i32t, ValRef::Param(0)),
    }];
    let f = simple_fn("bitcast", vec![i32t], insts, Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

#[test]
fn multi_function_module_refuses() {
    let f1 = simple_fn("a", vec![], vec![], Term::Ret(None));
    let f2 = simple_fn("b", vec![], vec![], Term::Ret(None));
    let module = Module {
        funcs: vec![f1, f2],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    };
    assert_eq!(unsupported_code(&module), ECode::UnsupportedOp);
}

#[test]
fn non_kernel_function_refuses() {
    let mut f = simple_fn("host_only", vec![], vec![], Term::Ret(None));
    f.is_kernel = false;
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedOp);
}

#[test]
fn inconsistent_pointer_element_type_refuses() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let i32t = Ty::Scalar(Scalar::I32);
    let f32t = Ty::Scalar(Scalar::F32);
    let insts = vec![
        Inst {
            ty: i32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
        Inst {
            ty: f32t,
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        },
    ];
    let f = simple_fn("mixedwidth", vec![ptrt], insts, Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedType);
}

#[test]
fn never_dereferenced_pointer_param_refuses() {
    let ptrt = Ty::Ptr(AddrSpace::Global);
    let f = simple_fn("deadptr", vec![ptrt], vec![], Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedType);
}

#[test]
fn i64_scalar_param_refuses() {
    let i64t = Ty::Scalar(Scalar::I64);
    let f = simple_fn("wide", vec![i64t], vec![], Term::Ret(None));
    assert_eq!(unsupported_code(&wrap(f)), ECode::UnsupportedType);
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
    basalt_passes::optimize(&module)
}

#[test]
fn vector_add_emits_a_real_kernel_main_via_the_real_pipeline() {
    let module = lower_vector_add();
    assert_eq!(Tensix.supports(&module), Support::Supported);
    let text = emit_text(&module);

    assert!(text.starts_with("void kernel_main() {\n"));
    // a (param 0) and b (param 1) are read-only; c (param 2) is write-only; n (param 3) is a
    // plain scalar runtime arg.
    assert!(text.contains("noc_async_read(p0_gen.get_noc_addr(0)"));
    assert!(text.contains("noc_async_read(p1_gen.get_noc_addr(0)"));
    assert!(!text.contains("noc_async_read(p2_gen"));
    assert!(text.contains("noc_async_read_barrier();"));
    assert!(text.contains("noc_async_write(p2_l1, p2_gen.get_noc_addr(0)"));
    assert!(text.contains("noc_async_write_barrier();"));
    assert!(text.contains("int32_t p3 = (int32_t)get_arg_val<uint32_t>(6);"));
    assert!(text.contains("uint32_t nthreads = get_arg_val<uint32_t>(7);"));
    assert!(text.contains("for (uint32_t __tid = 0; __tid < nthreads; ++__tid) {"));
    assert!(text.contains("continue;"));
}

#[test]
fn vector_add_emit_is_deterministic_through_the_real_pipeline() {
    let module = lower_vector_add();
    let a = emit_text(&module);
    let b = emit_text(&module);
    assert_eq!(a, b);
}
