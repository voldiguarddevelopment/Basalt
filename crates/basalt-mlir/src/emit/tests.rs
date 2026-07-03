// Real-pipeline coverage for `emit_ptx_text`: the same `vector_add.cu` bring-up kernel
// `lower::tests` already round-trips through `mlir-opt` for dialect verification, now taken
// one step further to genuine PTX text. A defensive skip (mirroring
// `basalt-llvm::tests::link_and_run`'s `cc`-not-found handling and `lower::tests`'s own
// `mlir-opt`-not-found handling) keeps this suite from hard-failing in the one hypothetical
// case the toolchain is missing despite the feature having built.

use super::emit_ptx_text;

const VECTOR_ADD_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/vector_add.cu"
));

fn lower_vector_add() -> basalt_bir::Module {
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

/// `true` unless the failure looks like "the toolchain isn't on this machine" (the one
/// hypothetical this suite is allowed to skip rather than fail on).
fn is_missing_toolchain(diag: &basalt_diag::Diag) -> bool {
    diag.args
        .iter()
        .any(|a| a.contains("could not run mlir-opt"))
}

#[test]
fn vector_add_emits_real_nvptx_text() {
    let module = lower_vector_add();
    let ptx = match emit_ptx_text(&module) {
        Ok(ptx) => ptx,
        Err(diag) if is_missing_toolchain(&diag) => {
            eprintln!("skipping vector_add_emits_real_nvptx_text: mlir-opt not found on PATH");
            return;
        }
        Err(diag) => panic!("emit_ptx_text failed: {diag} ({:?})", diag.args),
    };

    assert!(
        ptx.contains(".version"),
        "missing .version directive:\n{ptx}"
    );
    assert!(
        ptx.contains(".target sm_70"),
        "missing .target sm_70 directive:\n{ptx}"
    );
    assert!(
        ptx.contains(".visible .entry vector_add"),
        "missing vector_add entry point:\n{ptx}"
    );
    // The unpacked memref descriptor ABI (see this module's header): three `float*`
    // parameters plus one `int n` explode to sixteen PTX parameters, not four.
    assert!(
        ptx.contains("vector_add_param_15"),
        "expected the exploded 16-parameter memref-descriptor ABI:\n{ptx}"
    );
    assert!(
        !ptx.contains("vector_add_param_16"),
        "expected exactly 16 parameters (0..15), found a 17th:\n{ptx}"
    );
}

#[test]
fn vector_add_ptx_emission_is_deterministic() {
    let module = lower_vector_add();
    let a = match emit_ptx_text(&module) {
        Ok(ptx) => ptx,
        Err(diag) if is_missing_toolchain(&diag) => {
            eprintln!(
                "skipping vector_add_ptx_emission_is_deterministic: mlir-opt not found on PATH"
            );
            return;
        }
        Err(diag) => panic!("emit_ptx_text failed: {diag} ({:?})", diag.args),
    };
    let b = emit_ptx_text(&module).expect("second emission succeeds since the first did");
    assert_eq!(
        a, b,
        "two invocations of emit_ptx_text over the identical BIR module produced different PTX text"
    );
}
