// The Triton pipeline's moment of truth, mirroring `link_and_run.rs`'s own rationale for the
// CUDA-C side exactly: does `basalt_sema::lower_triton`'s output actually link, via the real
// system C compiler, and run to the correct answer? Two proofs, run through the genuine
// parse -> `check_triton` -> `lower_triton` -> `X86Oracle` pipeline (`basalt-cli`'s own path,
// not a hand-built BIR fixture):
//
//   - `masked_triton_vector_add_links_and_runs`: a real `@triton.jit vector_add`, launched with
//     a block size (1024) that is *not* a multiple of the real array length (1000) — the mask
//     genuinely has to guard real out-of-bounds accesses (`a`/`b` are only 1000 floats wide)
//     for this to produce a correct result, not just happen to work.
//   - `triton_matmul_links_and_runs`: a real `@triton.jit` matmul kernel, `tl.dot` at a
//     non-square M/N/K, checked against a host triple-loop reference (mirrors
//     `tiled_sgemm.rs`'s own reference style).
//
// See `crates/basalt-sema/src/triton_lower.rs`'s module header for the scoping decision this
// proves out: `tl.dot` lowers to a real scalar triple loop, never `Op::Mma`.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_x86::X86Oracle;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
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

fn run_and_check(exe: &Path) {
    let out = Command::new(exe).output().expect("built executable runs");
    assert!(
        out.status.success(),
        "{} exited with {:?}\nstdout:\n{}\nstderr:\n{}",
        exe.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    print!("{}", String::from_utf8_lossy(&out.stdout));
}

fn write_object(bytes: &[u8], path: &Path) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

/// Runs the real `parse -> check_triton -> lower_triton` pipeline over `src`, asserting no
/// diagnostics at any stage, and returns the lowered BIR module.
fn compile_triton(src: &str) -> basalt_bir::Module {
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
    bir
}

/// Emits `bir` through the real x86-64 oracle, links it against `shim_c` via the host's own
/// `cc`, runs the result, and asserts a zero exit status. The common tail shared by both proofs
/// below, mirroring `link_and_run.rs`'s own `compile_link_and_run`.
fn emit_link_and_run(bir: &basalt_bir::Module, shim_c: &str, tag: &str) {
    assert_eq!(X86Oracle.supports(bir), Support::Supported);
    let artifact = X86Oracle
        .emit(bir, &EmitOpts::default())
        .unwrap_or_else(|e| panic!("oracle emit failed for {tag}: {e}"));
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let obj = scratch.join(format!("basalt_{tag}_{pid}.o"));
    let shim_o = scratch.join(format!("basalt_{tag}_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_{tag}_exe_{pid}"));
    write_object(bytes, &obj);

    let shim_path = root.join(shim_c);
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

    run_and_check(&exe);

    let _ = std::fs::remove_file(&obj);
    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
}

const MASKED_VECTOR_ADD: &str = r#"
import triton
import triton.language as tl


@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
"#;

#[test]
fn masked_triton_vector_add_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping masked_triton_vector_add_links_and_runs: `cc` not found");
        return;
    }
    let bir = compile_triton(MASKED_VECTOR_ADD);
    emit_link_and_run(&bir, "examples/cpu_launch_triton_vadd.c", "triton_vadd");
}

const MATMUL: &str = r#"
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, out_ptr, K: tl.constexpr, M: tl.constexpr = 4, N: tl.constexpr = 3):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.load(c_ptrs)
    acc = tl.zeros((M, N), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    acc = acc + c
    out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, acc)
"#;

#[test]
fn triton_matmul_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping triton_matmul_links_and_runs: `cc` not found");
        return;
    }
    let bir = compile_triton(MATMUL);
    emit_link_and_run(&bir, "examples/cpu_launch_triton_matmul.c", "triton_matmul");
}
