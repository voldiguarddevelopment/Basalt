// Real NVPTX target lowering: takes the dialect text `lower_to_text` prints and drives it
// through the rest of a real MLIR toolchain, out to genuine PTX text. `lower.rs` stops at
// `gpu`/`arith`/`memref`/`cf` — connective-tissue dialects with no target-specific meaning of
// their own; this is the layer that actually reaches a concrete target, mirroring
// `basalt-llvm`'s own `lower.rs`/`emit.rs` split (see that crate's `emit.rs` header).
//
// # Why this shells out rather than links a pass-manager API
//
// `melior` exposes a real `PassManager` that can run registered conversion passes
// in-process, but every one of the conversions this lane needs
// (`convert-gpu-to-nvvm`, `gpu-module-to-binary`, ...) lives in upstream MLIR's own
// `Conversion`/`GPU` libraries, none of which `melior` wraps as a Rust API — reaching them
// in-process would mean linking `libMLIRGPUTransforms`/`libMLIRNVGPUTransforms`/... directly
// (a real, separate Rust-crate-boundary decision `basalt-mlir` has never made and that this
// task's scope does not ask for) rather than the one `mlir-sys`/`melior` pair this crate
// already depends on. Shelling out to `mlir-opt` — the same real, installed-toolchain binary
// `lower/tests.rs` already shells out to for verification — reaches the identical compiled
// passes with no new Rust-level dependency, exactly the discipline this task's brief called
// for.
//
// # The pipeline, empirically determined against a real LLVM/MLIR 22.1.6 build
//
// The task brief that set up this lane assumed a `mlir-translate --mlir-to-llvmir` /
// `llc -march=nvptx64` finish, mirroring `basalt-llvm::emit`'s own `TargetMachine`-based
// finish. That assumption does not survive contact with the real 22.1.6 build: this crate's
// `lower_module` wraps every kernel in a `gpu.module` (so a later, real launch/host-outlining
// pass has something to outline `gpu.launch_func` calls against), and `mlir-translate
// --mlir-to-llvmir` has no translation rule for `gpu.module` itself — confirmed empirically:
// translating a plain top-level `llvm.func` succeeds, but the identical function nested one
// level inside a `gpu.module` translates to an empty module (no diagnostic, no error, just
// nothing). Manually string-splicing the `gpu.module` body back out to the top level before
// calling `mlir-translate` would work, but MLIR already ships the real, first-class answer to
// exactly this problem: `-gpu-module-to-binary`, the same pass real MLIR-based GPU pipelines
// (`gpu-lower-to-nvvm-pipeline` and friends) use to turn a fully-converted `gpu.module` into a
// `gpu.binary` op holding a compiled object per requested target. Asked for `format=isa` (an
// option confirmed only by trying it — the pass's own `--help` text does not enumerate its
// legal `format` strings), it runs the identical LLVM NVPTX backend `llc` itself uses, but
// linked directly into `mlir-opt`'s own process rather than shelled out to a second binary —
// a strictly more direct path to the same target-specific codegen the brief asked for, not a
// different one. The settled pipeline, run as ordinary legacy pass-name flags (verified
// against `mlir-opt --help`'s real registered pass list on the 22.1.6 build, not recalled from
// training data):
//
//   mlir-opt <file>
//     -convert-gpu-to-nvvm -convert-arith-to-llvm -convert-cf-to-llvm
//     -finalize-memref-to-llvm -reconcile-unrealized-casts
//     -nvvm-attach-target="chip=sm_70" -gpu-module-to-binary="format=isa"
//
// `sm_70` matches `basalt-llvm::emit::LlvmTarget::Nvptx`'s own documented floor (see that
// crate's `emit.rs` header) so both independently-built NVPTX lanes agree on a baseline part.
// `llc -march=nvptx64 -mcpu=help` on this same build lists `sm_70` as a real, registered CPU
// and `llc --version` lists `nvptx`/`nvptx64` as registered targets, so the NVPTX backend
// itself is present in this build either way; `-gpu-module-to-binary` simply reaches it
// through `mlir-opt`'s own process instead of a second `llc` invocation.
//
// # The kernel ABI this pipeline actually produces
//
// `-convert-gpu-to-nvvm`'s bare-pointer calling convention
// (`use-bare-ptr-memref-call-conv=1`) requires every `memref` parameter to have a *static*
// shape; `lower_module`'s own `memref<?xf32>` parameters (BIR carries no compile-time buffer
// length — `n` is always a separate runtime parameter) are dynamically shaped, so that
// convention is not legal here (confirmed empirically: `mlir-opt` reports `gpu.func`
// "explicitly marked illegal" under that flag). Without it, the *default* unpacked
// convention explodes every `memref<?xf32>` kernel parameter into five scalar PTX
// parameters — allocated pointer, aligned pointer, offset, size, stride, in that order, the
// ordinary MLIR memref descriptor fields — rather than the one flat pointer
// `basalt-ptx`/`basalt-llvm` both emit for the same source-level parameter. `vector_add`'s
// three `float*` parameters plus one `int n` therefore reach real PTX as sixteen parameters,
// not four: params 0..5 (a's allocated ptr, aligned ptr, offset, size, stride), 5..10 (b's),
// 10..15 (c's), then a plain `i32` for `n` at slot 15. A caller driving this PTX from
// `basalt-runtime` (see `crates/basalt-mlir/tests/nvptx_gpu_proof.rs`) must build a
// `cuLaunchKernel` parameter array against this exact descriptor ABI, not the flat-pointer one
// the other two NVPTX-producing lanes share.
//
// # Determinism
//
// `-gpu-module-to-binary` embeds an `LLVMIRToISATimeInMs` wall-clock timing attribute
// alongside the compiled object (confirmed empirically: running the identical input through
// this exact pipeline twice reproduces byte-identical PTX text but a different
// `LLVMIRToISATimeInMs` value each time). This lane only ever extracts the `assembly` string
// field out of the surrounding module text — it never serializes or hashes the module as a
// whole — so that timing attribute never reaches this function's return value; the PTX text
// itself carries no timestamp, no temp-file path, and no other run-to-run variation (see
// `emit::tests` for the byte-identical check across two independent invocations of this
// function).

use std::io::Write;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use basalt_bir::Module as BirModule;
use basalt_diag::{Diag, ECode};

use crate::lower::lower_to_text;

/// Matches `basalt-llvm::emit::LlvmTarget::Nvptx`'s own documented floor.
const NVPTX_CHIP: &str = "sm_70";

fn tool_failure(detail: impl Into<String>) -> Diag {
    Diag::new(ECode::UnsupportedFeature).with_arg(detail.into())
}

/// Lowers `module` to MLIR dialect text (via `lower_to_text`) and drives it through a real,
/// installed `mlir-opt` to genuine NVPTX PTX text — see the module header for the exact
/// pipeline and why it differs from a naive `mlir-translate`/`llc` guess. Returns a clean
/// `Diag` (never panics) if `mlir-opt` is missing, exits non-zero, or its output does not
/// contain the `gpu.binary` object this pipeline is known to always produce for a
/// successfully-converted module — an external-tool failure is diagnosed, not crashed on,
/// the same convention `basalt-llvm::emit`'s own `TargetMachine` failures already follow.
pub fn emit_ptx_text(module: &BirModule) -> Result<String, Diag> {
    let mlir_text = lower_to_text(module)?;
    let opt_stdout = run_gpu_to_nvptx_pipeline(&mlir_text)?;
    extract_assembly(&opt_stdout)
}

fn scratch_path(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "basalt-mlir-emit-{tag}-{}-{nanos}.mlir",
        std::process::id()
    ))
}

/// Runs the pipeline documented in this file's header over `text` and returns `mlir-opt`'s raw
/// stdout (a `builtin.module` containing one `gpu.binary` op with an embedded `assembly`
/// string attribute holding real PTX text).
fn run_gpu_to_nvptx_pipeline(text: &str) -> Result<String, Diag> {
    let path = scratch_path("in");
    let mut file = std::fs::File::create(&path).map_err(|e| {
        tool_failure(format!(
            "could not create scratch file {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(text.as_bytes()).map_err(|e| {
        tool_failure(format!(
            "could not write scratch file {}: {e}",
            path.display()
        ))
    })?;
    drop(file);

    let output = Command::new("mlir-opt")
        .arg(&path)
        .arg("-convert-gpu-to-nvvm")
        .arg("-convert-arith-to-llvm")
        .arg("-convert-cf-to-llvm")
        .arg("-finalize-memref-to-llvm")
        .arg("-reconcile-unrealized-casts")
        .arg(format!("-nvvm-attach-target=chip={NVPTX_CHIP}"))
        .arg("-gpu-module-to-binary=format=isa")
        .output();
    let _ = std::fs::remove_file(&path);

    let output = output.map_err(|e| tool_failure(format!("could not run mlir-opt: {e}")))?;
    if !output.status.success() {
        return Err(tool_failure(format!(
            "mlir-opt exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Pulls the `assembly = "..."` field out of `mlir-opt`'s printed `gpu.binary` op and
/// unescapes it. MLIR prints an opaque string attribute using LLVM's own `printEscapedString`
/// convention: any byte outside the printable-ASCII range, plus `"` and `\` themselves, is
/// escaped as `\XY` (two uppercase hex digits); every other byte is copied verbatim. This is
/// the one place this lane needs to un-print that convention, rather than teaching `melior` a
/// new attribute accessor for a custom (non-hand-wrapped) `gpu.object` attribute.
fn extract_assembly(opt_stdout: &str) -> Result<String, Diag> {
    const MARKER: &str = "assembly = \"";
    let start = opt_stdout.find(MARKER).ok_or_else(|| {
        tool_failure(
            "mlir-opt's output has no `assembly = \"...\"` field; the gpu-module-to-binary \
             pass did not produce the object this pipeline expects",
        )
    })?;
    let bytes = opt_stdout.as_bytes();
    let mut i = start + MARKER.len();
    let mut out = Vec::new();
    loop {
        match bytes.get(i) {
            None => {
                return Err(tool_failure(
                    "mlir-opt's `assembly` string attribute was never closed",
                ))
            }
            Some(b'"') => break,
            Some(b'\\') => {
                let hex = bytes.get(i + 1..i + 3).ok_or_else(|| {
                    tool_failure("mlir-opt's `assembly` string has a truncated \\XY escape")
                })?;
                let hex_str = std::str::from_utf8(hex).map_err(|_| {
                    tool_failure("mlir-opt's `assembly` string has a non-ASCII \\XY escape")
                })?;
                let byte = u8::from_str_radix(hex_str, 16).map_err(|_| {
                    tool_failure(format!(
                        "mlir-opt's `assembly` string has an invalid \\{hex_str} escape"
                    ))
                })?;
                out.push(byte);
                i += 3;
            }
            Some(&b) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|e| {
        tool_failure(format!(
            "mlir-opt's `assembly` string is not valid UTF-8: {e}"
        ))
    })
}

#[cfg(test)]
mod tests;
