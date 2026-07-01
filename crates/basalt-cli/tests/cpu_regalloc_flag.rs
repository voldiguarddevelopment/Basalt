// Integration test for `--cpu-regalloc`: proves the CLI wiring itself (flag parsing, `-o`
// handling, pipeline reuse), mirroring `cpu_flag.rs`'s own `--cpu` test exactly. The codegen
// itself (real registers, real spills) is proven at the library level by
// `basalt-x86/tests/link_and_run_regalloc.rs`; this only needs to show the CLI reaches the
// regalloc backend and produces a working, linkable object via the real `basalt` binary.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

fn kernel_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/kernels")).join(name)
}

fn basalt() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basalt"))
}

fn run(args: &[&str]) -> Output {
    basalt().args(args).output().expect("failed to run basalt")
}

fn cc_available() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_cc(args: &[&OsStr]) {
    let out = Command::new("cc")
        .args(args)
        .output()
        .expect("cc is present and spawns");
    assert!(
        out.status.success(),
        "cc {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cpu_regalloc_flag_on_vector_add_links_and_runs_via_the_cli_binary() {
    if !cc_available() {
        eprintln!(
            "skipping cpu_regalloc_flag_on_vector_add_links_and_runs_via_the_cli_binary: `cc` not found"
        );
        return;
    }

    let root = workspace_root();
    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let obj = scratch.join(format!("basalt_cli_cpuregalloc_vadd_{pid}.o"));
    let shim_o = scratch.join(format!("basalt_cli_cpuregalloc_vadd_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_cli_cpuregalloc_vadd_exe_{pid}"));

    let out = run(&[
        "--cpu-regalloc",
        kernel_path("vector_add.cu").to_str().unwrap(),
        "-o",
        obj.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "basalt --cpu-regalloc vector_add.cu -o ... did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "expected no diagnostics on a clean kernel, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(obj.exists(), "expected an object file at {}", obj.display());

    let shim_path = root.join("examples/cpu_launch_vadd.c");
    run_cc(&[
        OsStr::new("-c"),
        shim_path.as_os_str(),
        OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);
    run_cc(&[
        shim_o.as_os_str(),
        obj.as_os_str(),
        OsStr::new("-o"),
        exe.as_os_str(),
    ]);

    let run_out = Command::new(&exe).output().expect("built executable runs");
    assert!(
        run_out.status.success(),
        "{} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
        exe.display(),
        run_out.status.code(),
        String::from_utf8_lossy(&run_out.stdout),
        String::from_utf8_lossy(&run_out.stderr)
    );
    let stdout = String::from_utf8_lossy(&run_out.stdout);
    assert!(
        stdout.contains("PASS"),
        "expected a PASS line, got: {stdout}"
    );

    let _ = std::fs::remove_file(&obj);
    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
}

#[test]
fn cpu_regalloc_flag_without_output_reports_e101() {
    let out = run(&[
        "--cpu-regalloc",
        kernel_path("vector_add.cu").to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).expect("stderr must be valid UTF-8");
    assert!(
        stderr.contains("E101"),
        "expected E101 in stderr, got: {stderr}"
    );
}

#[test]
fn cpu_regalloc_flag_on_deliberately_broken_kernel_exits_nonzero_without_writing_output() {
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_cli_cpuregalloc_broken_{pid}.o"));
    let _ = std::fs::remove_file(&obj);

    let out = run(&[
        "--cpu-regalloc",
        kernel_path("deliberate_errors.cu").to_str().unwrap(),
        "-o",
        obj.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "basalt --cpu-regalloc deliberate_errors.cu unexpectedly exited 0"
    );
    assert!(
        !obj.exists(),
        "a module with sema problems must never produce an object file"
    );
}
