// Integration tests for the pipeline-backed modes (`--ast`, `--sema`, `--ir` on real C/CUDA
// source rather than a `.bir` file). Runs the built `basalt` binary as a subprocess, same
// approach as `ir_flag.rs`, since the CLI contract (exit codes, stdout/stderr) is what's
// under test.

use std::path::PathBuf;
use std::process::{Command, Output};

fn kernel_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/kernels")).join(name)
}

fn basalt() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basalt"))
}

fn run(args: &[&str]) -> Output {
    basalt().args(args).output().expect("failed to run basalt")
}

#[test]
fn ir_on_vector_add_source_exits_0_and_emits_bir() {
    let out = run(&["--ir", kernel_path("vector_add.cu").to_str().unwrap()]);
    assert!(
        out.status.success(),
        "basalt --ir vector_add.cu did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    assert!(
        stdout.contains("func @vector_add"),
        "expected a vector_add function in BIR output, got: {stdout}"
    );
    assert!(stdout.contains("module {"), "{stdout}");
    assert!(
        out.stderr.is_empty(),
        "expected no diagnostics on a clean kernel, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn ast_on_vector_add_source_exits_0_and_emits_a_dump() {
    let out = run(&["--ast", kernel_path("vector_add.cu").to_str().unwrap()]);
    assert!(
        out.status.success(),
        "basalt --ast vector_add.cu did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    assert!(!stdout.is_empty(), "expected a non-empty AST dump");
    assert!(stdout.contains("TranslationUnit"), "{stdout}");
    assert!(stdout.contains("vector_add"), "{stdout}");
}

#[test]
fn sema_on_vector_add_source_exits_0_with_no_diagnostics_message() {
    let out = run(&["--sema", kernel_path("vector_add.cu").to_str().unwrap()]);
    assert!(
        out.status.success(),
        "basalt --sema vector_add.cu did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    assert!(
        stdout.contains("no diagnostics"),
        "expected a success message, got: {stdout}"
    );
}

#[test]
fn sema_on_deliberately_broken_kernel_reports_multiple_e_codes_and_exits_nonzero() {
    let out = run(&[
        "--sema",
        kernel_path("deliberate_errors.cu").to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "basalt --sema deliberate_errors.cu unexpectedly exited 0"
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    let stderr = String::from_utf8(out.stderr).expect("stderr must be valid UTF-8");
    let combined = format!("{stdout}{stderr}");

    let distinct_e_codes: std::collections::HashSet<&str> = combined
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|tok| {
            tok.len() == 4 && tok.starts_with('E') && tok[1..].chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    assert!(
        distinct_e_codes.len() >= 2,
        "expected at least 2 distinct E-codes, got {distinct_e_codes:?} in: {combined}"
    );
}

#[test]
fn ast_on_a_bir_file_is_a_mismatched_combination_and_reports_e102() {
    let out = run(&["--ast", kernel_path("hand_written.bir").to_str().unwrap()]);
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).expect("stderr must be valid UTF-8");
    assert!(
        stderr.contains("E102"),
        "expected E102 in stderr, got: {stderr}"
    );
}
