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

#[test]
fn refuses_phi() {
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
    let err = lower_to_text(&wrap(f)).unwrap_err();
    assert_eq!(err.code, ECode::UnsupportedOp);
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
