// Proves the hand-written HSACO `hsaco::write_hsaco` produces is not just structurally
// plausible but genuinely loadable and dispatchable by a real, compiler-agnostic AMDGPU object
// loader — the actual bar `hsaco.rs`'s module header sets, not just "the `object` crate can
// parse it". Same "skip cleanly on a machine without the tooling" pattern as
// `basalt-runtime/tests/cuda_driver.rs`: the harness this drives
// (`tests/diff/rdna3_sim/run_kernel.py`) already reports "skip" (exit 77) when no tinygrad
// checkout with `test/mockgpu` is importable, and `tests/diff/run_diff.sh`'s own rdna3-sim lane
// (which drives the same harness against the LLVM backend's HSACO output) honors the same
// `RDNA3_SIM_PYTHON` override for which interpreter to use.

use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_amdgpu::enc;
use basalt_amdgpu::hsaco::{write_hsaco, GfxArch, HsacoSpec};

const SKIP: i32 = 77;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

fn rdna3_python() -> String {
    std::env::var("RDNA3_SIM_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn python_available(python: &str) -> bool {
    Command::new(python)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Writes a bare `s_endpgm`-only kernel (no arguments, no register usage beyond the wave32
/// default) through `write_hsaco`, then drives it through tinygrad's real
/// `elf_loader`/`AMDProgram` path via the shared harness, asking for one workgroup of one
/// thread — enough to prove the container loads *and* the wave actually launches and
/// terminates cleanly, not just that the bytes parse as ELF.
#[test]
fn s_endpgm_kernel_loads_and_dispatches_on_the_rdna3_emulator() {
    let python = rdna3_python();
    if !python_available(&python) {
        eprintln!("skipping: no {python} interpreter on this machine");
        return;
    }

    let spec = HsacoSpec::new(GfxArch::Gfx1100, "endpgm_only", enc::s_endpgm());
    let bytes = write_hsaco(&spec).expect("write_hsaco succeeds for a bare s_endpgm kernel");

    let hsaco_path =
        std::env::temp_dir().join(format!("basalt_amdgpu_endpgm_{}.hsaco", std::process::id()));
    std::fs::write(&hsaco_path, &bytes).expect("writing the HSACO to a scratch file");

    let harness = workspace_root().join("tests/diff/rdna3_sim/run_kernel.py");
    let out = Command::new(&python)
        .arg(&harness)
        .args(["--hsaco"])
        .arg(&hsaco_path)
        .args([
            "--kernel",
            "endpgm_only",
            "--global",
            "1,1,1",
            "--local",
            "1,1,1",
        ])
        .output()
        .expect("spawning the rdna3-sim harness");

    let _ = std::fs::remove_file(&hsaco_path);

    match out.status.code() {
        Some(SKIP) => eprintln!(
            "skipping: rdna3-sim unavailable ({})",
            String::from_utf8_lossy(&out.stderr).trim()
        ),
        Some(0) => {}
        _ => panic!(
            "rdna3-sim harness did not exit 0 loading a hand-written HSACO:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}
