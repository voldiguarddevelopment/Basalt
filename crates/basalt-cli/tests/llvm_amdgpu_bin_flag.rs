// Integration test for `--llvm --amdgpu-bin`: the only way `--amdgpu-bin` currently produces a
// real artifact, since there is no hand-rolled `basalt-amdgpu` backend yet. Drives the built
// `basalt` binary as a subprocess, same approach as `cpu_flag.rs`/`ir_flag.rs`. There is no
// AMDGCN hardware or simulator to run the result on yet (later work); the bar here is a clean
// exit, a real HSACO/ELF file at `-o`, and structural ELF validity via the `object` crate's
// read side, matching the oracle-validated/sim-validated/silicon-validated tier discipline —
// this backend is neither of the latter two yet.
//
// Uses `stress.cu`, not `vector_add.cu`: the AMDGCN dialect in `basalt-llvm`'s own lowering
// (from a prior task, unmodified here) deliberately refuses `bdim.x`/`gdim.*` reads — no
// no-argument LLVM 18 AMDGCN intrinsic exists for block/grid dimensions, a documented and
// tested scope limit, not a bug — and `vector_add.cu` reads `blockDim.x`. `stress.cu` indexes
// with `threadIdx.x` alone, which the AMDGCN dialect fully supports, so it is the existing
// fixture that actually exercises this path end to end without hitting that gap.
#![cfg(feature = "llvm")]

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
fn llvm_amdgpu_bin_on_stress_produces_a_valid_elf_object() {
    let pid = std::process::id();
    let out_path = std::env::temp_dir().join(format!("basalt_cli_llvm_amdgpu_{pid}.hsaco"));
    let _ = std::fs::remove_file(&out_path);

    let out = run(&[
        "--llvm",
        "--amdgpu-bin",
        kernel_path("stress.cu").to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "basalt --llvm --amdgpu-bin stress.cu -o ... did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "expected no diagnostics on a clean kernel, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out_path.exists(),
        "expected an object file at {}",
        out_path.display()
    );

    let bytes = std::fs::read(&out_path).expect("reading the emitted hsaco");
    let file = object::read::File::parse(&*bytes).expect("parses as a real object file");
    assert_eq!(file.format(), object::BinaryFormat::Elf);

    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn amdgpu_bin_without_llvm_is_an_unimplemented_refusal() {
    let pid = std::process::id();
    let out_path = std::env::temp_dir().join(format!("basalt_cli_amdgpu_nollvm_{pid}.hsaco"));
    let _ = std::fs::remove_file(&out_path);

    let out = run(&[
        "--amdgpu-bin",
        kernel_path("vector_add.cu").to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "--amdgpu-bin without --llvm unexpectedly exited 0"
    );
    assert!(!out_path.exists());
}

#[test]
fn llvm_with_a_non_amdgpu_bin_mode_is_an_unimplemented_refusal() {
    let out = run(&[
        "--llvm",
        "--cpu",
        kernel_path("vector_add.cu").to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "--llvm combined with --cpu unexpectedly exited 0"
    );
}
