// Integration test for `--nvidia-ptx`: proves the CLI wiring itself (flag parsing, pipeline
// reuse, stdout/`-o` output convention), mirroring `cpu_flag.rs`'s own `--cpu` test. The PTX
// backend's own codegen correctness is proven at the library level by `basalt-ptx`'s tests;
// this only needs to show the CLI reaches the backend and prints real PTX text via the real
// `basalt` binary.

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
fn nvidia_ptx_flag_on_vector_add_prints_ptx_to_stdout() {
    let out = run(&[
        "--nvidia-ptx",
        kernel_path("vector_add.cu").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "basalt --nvidia-ptx vector_add.cu did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "expected no diagnostics on a clean kernel, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    assert!(
        stdout.contains(".visible .entry vector_add("),
        "expected a vector_add entry point in PTX output, got: {stdout}"
    );
    assert!(stdout.contains(".version"), "{stdout}");
    assert!(stdout.contains(".target"), "{stdout}");
}

#[test]
fn nvidia_ptx_flag_writes_to_output_file_when_given() {
    let pid = std::process::id();
    let ptx_path = std::env::temp_dir().join(format!("basalt_cli_nvidia_ptx_vadd_{pid}.ptx"));
    let _ = std::fs::remove_file(&ptx_path);

    let out = run(&[
        "--nvidia-ptx",
        kernel_path("vector_add.cu").to_str().unwrap(),
        "-o",
        ptx_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "basalt --nvidia-ptx vector_add.cu -o ... did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "expected nothing on stdout when -o is given, got: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let written = std::fs::read_to_string(&ptx_path).expect("expected a PTX file at -o");
    assert!(written.contains(".visible .entry vector_add("));

    let _ = std::fs::remove_file(&ptx_path);
}

#[test]
fn nvidia_ptx_flag_on_deliberately_broken_kernel_exits_nonzero() {
    let out = run(&[
        "--nvidia-ptx",
        kernel_path("deliberate_errors.cu").to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "basalt --nvidia-ptx deliberate_errors.cu unexpectedly exited 0"
    );
}
