// Coverage: hand-built refusal cases (one per E-code this backend actually returns, mirroring
// `basalt-llvm`/`basalt-spirv`'s own per-refusal test style), and the real
// frontend/sema/passes pipeline over `tests/kernels/vector_add.cu` — the smallest real kernel
// every other backend in this tree bootstrapped against first.
//
// The `vector_add.cu` tests are this task's load-bearing verification: they shell out to a
// real `mlir-opt` (present wherever this crate's own `melior`/`mlir-sys` build succeeded,
// since both come from the same LLVM/MLIR toolchain install — unlike `basalt-spirv`, which
// avoids a `spirv-val` runtime dependency because it is part of the always-built default
// lane, this crate is entirely `feature = "mlir"`-gated, so a hard dependency on the matching
// `mlir-opt` binary here is safe) and assert it parses and verifies with **no** diagnostics,
// not merely that this crate's own Rust code did not panic. A defensive skip (mirroring
// `basalt-llvm::tests::link_and_run`'s own `cc`-not-found handling) keeps the suite from
// hard-failing in the one hypothetical case `mlir-opt` is missing despite the feature having
// built.

use std::process::Command;

use basalt_bir::{Block as BirBlock, Function, Inst, InstId, Module as BirModule, Term, Ty};
use basalt_diag::ECode;

use super::lower_to_text;

fn wrap(f: Function) -> BirModule {
    BirModule {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

fn one_block_fn(name: &str, params: Vec<Ty>, ret: Ty, insts: Vec<Inst>, term: Term) -> Function {
    let inst_ids = (0..insts.len() as u32).map(InstId).collect();
    Function {
        name: name.to_string(),
        params,
        ret,
        blocks: vec![BirBlock {
            insts: inst_ids,
            term,
        }],
        insts,
    }
}

// ---- hand-built refusals -------------------------------------------------------------------

#[test]
fn refuses_vector_typed_parameter() {
    use basalt_bir::Scalar;

    let f = one_block_fn(
        "k",
        vec![Ty::Vec(Scalar::F32, 4)],
        Ty::Void,
        vec![],
        Term::Ret(None),
    );
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedType);
}

/// A scalar-typed phi lowers to a real MLIR block argument (P11-T3), not a refusal — see this
/// module's own "Op::Phi" section above. `bb1` here has a single predecessor, so this is the
/// degenerate one-incoming-edge case (a real multi-predecessor merge is exercised by
/// `vector_add.cu`'s own masked-store diamond, already covered by
/// `vector_add_lowers_to_a_well_formed_module_via_the_real_pipeline`); it is still real,
/// non-hypothetical BIR shape and worth its own direct, minimal proof.
#[test]
fn phi_lowers_to_a_block_argument() {
    use basalt_bir::{BlockId, Op, ValRef};

    let f = Function {
        name: "k".to_string(),
        params: vec![Ty::Scalar(basalt_bir::Scalar::I32)],
        ret: Ty::Void,
        blocks: vec![
            BirBlock {
                insts: vec![],
                term: Term::Br(BlockId(1)),
            },
            BirBlock {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            },
        ],
        insts: vec![Inst {
            ty: Ty::Scalar(basalt_bir::Scalar::I32),
            op: Op::Phi(vec![(BlockId(0), ValRef::Param(0))]),
        }],
    };
    let text = lower_to_text(&wrap(f)).expect("a scalar-typed phi lowers to a block argument");

    let context = super_test_context();
    let reparsed =
        melior::ir::Module::parse(&context, &text).expect("emitted text re-parses as valid MLIR");
    assert!(
        melior::ir::operation::OperationLike::verify(&reparsed.as_operation()),
        "re-parsed module fails melior's own verifier"
    );

    // bb1 takes the phi as a real block argument, fed by bb0's own branch operand.
    assert!(text.contains("cf.br ^bb1(%arg0"));
    assert!(text.contains("^bb1(%0: i32):"));
}

#[test]
fn refuses_mma() {
    use basalt_bir::{MmaLayout, Op, Scalar, ValRef};

    let f = one_block_fn(
        "k",
        vec![Ty::Ptr(basalt_bir::AddrSpace::Global); 3],
        Ty::Void,
        vec![Inst {
            ty: Ty::Void,
            op: Op::Mma {
                a: ValRef::Param(0),
                b: ValRef::Param(1),
                c: ValRef::Param(2),
                d: ValRef::Param(2),
                m: 16,
                n: 16,
                k: 16,
                in_dtype: Scalar::F16,
                acc_dtype: Scalar::F32,
                layout_a: MmaLayout::RowMajor,
                layout_b: MmaLayout::RowMajor,
            },
        }],
        Term::Ret(None),
    );
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedOp);
}

#[test]
fn refuses_switch_terminator() {
    use basalt_bir::{BlockId, ValRef};

    let f = Function {
        name: "k".to_string(),
        params: vec![Ty::Scalar(basalt_bir::Scalar::I32)],
        ret: Ty::Void,
        blocks: vec![
            BirBlock {
                insts: vec![],
                term: Term::Switch(ValRef::Param(0), BlockId(1), vec![(0, BlockId(1))]),
            },
            BirBlock {
                insts: vec![],
                term: Term::Ret(None),
            },
        ],
        insts: vec![],
    };
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedOp);
}

#[test]
fn refuses_non_void_return() {
    use basalt_bir::{Op, Scalar};

    let f = one_block_fn(
        "k",
        vec![],
        Ty::Scalar(Scalar::I32),
        vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::ConstInt(0),
        }],
        Term::Ret(Some(basalt_bir::ValRef::Val(InstId(0)))),
    );
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedOp);
}

#[test]
fn refuses_bare_pointer_parameter_used_with_no_offset_arithmetic() {
    use basalt_bir::{AddrSpace, Op, Scalar, ValRef};

    let f = one_block_fn(
        "k",
        vec![Ty::Ptr(AddrSpace::Global)],
        Ty::Void,
        vec![Inst {
            ty: Ty::Scalar(Scalar::F32),
            op: Op::Load {
                ptr: ValRef::Param(0),
                space: AddrSpace::Global,
                align: 4,
                volatile: false,
            },
        }],
        Term::Ret(None),
    );
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedAddressSpace);
}

#[test]
fn refuses_shuffle() {
    use basalt_bir::{Op, Scalar, ShuffleKind, ValRef};

    let f = one_block_fn(
        "k",
        vec![Ty::Scalar(Scalar::I32)],
        Ty::Void,
        vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Shuffle(ShuffleKind::Idx, ValRef::Param(0), ValRef::Param(0)),
        }],
        Term::Ret(None),
    );
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedFeature);
}

// ---- real pipeline: tests/kernels/vector_add.cu ---------------------------------------------

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

fn lower_vector_add() -> BirModule {
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

/// Runs a real `mlir-opt` over `text`, asserting it parses and verifies with no diagnostics.
/// Returns `None` (skipping the caller's assertions on its output) if `mlir-opt` is not on
/// `PATH`, mirroring `basalt-llvm::tests::link_and_run`'s own `cc`-not-found handling.
fn run_mlir_opt(text: &str) -> Option<String> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("basalt-mlir-test-{}.mlir", std::process::id()));
    std::fs::write(&path, text).expect("write scratch .mlir file");

    let result = Command::new("mlir-opt").arg(&path).output();
    let _ = std::fs::remove_file(&path);

    let output = match result {
        Ok(output) => output,
        Err(_) => {
            eprintln!("skipping mlir-opt round-trip: `mlir-opt` not found on PATH");
            return None;
        }
    };

    assert!(
        output.status.success(),
        "mlir-opt rejected the emitted module:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "mlir-opt reported diagnostics on an emitted module it otherwise accepted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[test]
fn vector_add_lowers_to_a_well_formed_module_via_the_real_pipeline() {
    let module = lower_vector_add();
    let text = lower_to_text(&module).expect("vector_add lowers cleanly");

    // In-process confirmation: melior's own C-API parser/verifier (not just this crate's
    // own construction path) accepts the printed text.
    let context = super_test_context();
    let reparsed =
        melior::ir::Module::parse(&context, &text).expect("emitted text re-parses as valid MLIR");
    assert!(
        melior::ir::operation::OperationLike::verify(&reparsed.as_operation()),
        "re-parsed module fails melior's own verifier"
    );

    assert!(text.contains("gpu.module"));
    assert!(text.contains("gpu.func"));
    assert!(text.contains("gpu.thread_id"));
    assert!(text.contains("gpu.block_id"));
    assert!(text.contains("gpu.block_dim"));
    assert!(text.contains("memref.load"));
    assert!(text.contains("memref.store"));
    assert!(text.contains("arith.addf"));
    assert!(text.contains("cf.cond_br"));

    // Out-of-process confirmation: a real `mlir-opt` (LLVM/MLIR 22.1.6 on the one machine
    // this feature lane builds on) parses and verifies it with no diagnostics.
    run_mlir_opt(&text);
}

#[test]
fn vector_add_emit_is_deterministic_through_the_real_pipeline() {
    let module = lower_vector_add();
    let a = lower_to_text(&module).unwrap();
    let b = lower_to_text(&module).unwrap();
    assert_eq!(a, b);
}

fn super_test_context() -> melior::Context {
    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}

// ---- real pipeline: tests/kernels/tri_vadd.py (masked Triton vector-add), P11-T3 ------------
//
// See this module's own "Local/shared/constant/param storage", "Op::Phi", and especially
// "Known gap" sections above: this is the real, `mlir-opt`-checked proof that P11-T3's actual
// blocker for `tri_vadd.py` is neither local-slot storage nor `Op::Phi` (both are implemented
// and exercised by this test's own BIR) but `basalt-sema::triton_lower`'s tile-scratch reuse
// of the kernel's own last pointer parameter (`c_ptr`) at more than one element type, which
// this lowering's `memref`-per-parameter model has no representation for. This test asserts
// the honest current outcome — a stable, precise refusal — rather than a guessed-at success.

const TRI_VADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/tri_vadd.py"
));

/// Runs the real `parse -> check_triton -> lower_triton -> basalt_passes::optimize` pipeline
/// over `src`, asserting no diagnostics at any stage, and returns the optimized BIR module —
/// mirrors `crates/basalt-x86/tests/triton_link_and_run.rs`'s own `compile_triton` helper.
fn compile_triton(src: &str) -> BirModule {
    let (module, parse_diags) = basalt_frontend_triton::parse(src);
    assert!(
        parse_diags.is_empty(),
        "parsing produced diagnostics: {:?}",
        parse_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (shapes, check_diags) = basalt_sema::check_triton(&module);
    assert!(
        check_diags.is_empty(),
        "check_triton produced diagnostics: {:?}",
        check_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (bir, lower_diags) = basalt_sema::lower_triton(&module, &shapes);
    assert!(
        lower_diags.is_empty(),
        "lower_triton produced diagnostics: {:?}",
        lower_diags
            .iter()
            .map(|d| (d.code, d.args.clone()))
            .collect::<Vec<_>>()
    );
    basalt_passes::optimize(&bir)
}

#[test]
fn tri_vadd_refuses_on_a_stable_e_code_pending_a_byte_addressed_memref_layer() {
    let module = compile_triton(TRI_VADD_SRC);
    let err = lower_to_text(&module).expect_err(
        "tri_vadd.py's scratch-sharing parameter (c_ptr, read/written at i64/i1/f32 through \
         basalt-sema's tile-scratch reuse) is not yet representable by this lowering's \
         one-memref-type-per-parameter model — see this module's own \"Known gap\" section",
    );
    assert_eq!(err.code, ECode::UnsupportedAddressSpace);
}

#[test]
fn tri_vadd_refusal_is_deterministic() {
    let module = compile_triton(TRI_VADD_SRC);
    let a = lower_to_text(&module).unwrap_err();
    let b = lower_to_text(&module).unwrap_err();
    assert_eq!(a.code, b.code);
}
