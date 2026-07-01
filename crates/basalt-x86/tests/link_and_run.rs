// The oracle's moment of truth: does the machine code `X86Oracle::emit` actually produces
// link, via the real system C compiler, and run to the correct answer? Everything before this
// point (unit tests, `--ir` dumps, ELF-shape assertions in `oracle.rs`) only checks structure;
// nothing has been executed. These two tests shell out to `cc` and a real subprocess, no
// in-process JIT.
//
// Two proofs, each covering a different slice:
//   - `vector_add_links_and_runs_via_full_pipeline` runs the actual lex/preprocess/parse/
//     check/lower pipeline over `tests/kernels/vector_add.cu`, exactly like `basalt-cli`'s own
//     `--ir` path, then links the oracle's output against a real C caller.
//   - `hand_built_add_i32_links_and_runs` builds BIR directly, skipping the frontend/sema
//     stages entirely, to isolate the oracle's basic scalar calling-convention/return-value
//     path from everything upstream of it. If the first test ever breaks, this one narrows
//     down whether the fault is in the oracle or further up the pipeline.
//
// Both read the exact calling convention off `oracle.rs`'s own module header and its
// `INT_ARG_REGS`/`SSE_ARG_REGS` classification, not off any assumption: every param BIR sees is
// integer-class here (pointers and `i32`), so they consume the SysV integer registers in
// order, and the trailing `nthreads` argument always takes the next integer register after the
// function's own params — always read back a full 8 bytes on the oracle side, hence the C
// shims declare it `int64_t`, not `int`.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{BinOp, Block, Function, Inst, InstId, Module, Op, Scalar, Term, Ty, ValRef};
use basalt_frontend_c::PpOpts;
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

/// Runs `cc`, failing the test with its stderr if it doesn't exit 0 — a compile/link failure
/// must be diagnosable, not a silent test failure.
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

/// Runs `exe`, asserting a zero exit status; the process's own stdout/stderr are folded into
/// the panic message so a wrong-answer failure shows exactly what mismatched.
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

/// Compiles `shim_c` and links it with `payload_o` into `exe`, then runs and checks it. The
/// common tail shared by both tests below.
fn compile_link_and_run(root: &Path, shim_c: &str, payload_o: &Path, tag: &str) {
    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let shim_o = scratch.join(format!("basalt_{tag}_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_{tag}_exe_{pid}"));

    let shim_path = root.join(shim_c);
    run_cc(&[
        OsStr::new("-c"),
        shim_path.as_os_str(),
        OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);
    run_cc(&[
        shim_o.as_os_str(),
        payload_o.as_os_str(),
        OsStr::new("-o"),
        exe.as_os_str(),
    ]);

    run_and_check(&exe);

    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
}

#[test]
fn vector_add_links_and_runs_via_full_pipeline() {
    if !cc_available() {
        eprintln!("skipping vector_add_links_and_runs_via_full_pipeline: `cc` not found");
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/vector_add.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = PpOpts {
        include_dirs: vec![],
        defines: vec![],
        base_dir: src_path.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &opts);
    assert!(
        pp_errors.is_empty(),
        "preprocessing vector_add.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing vector_add.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking vector_add.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering vector_add.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for vector_add");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_vadd_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_vadd.c", &obj, "vadd");

    let _ = std::fs::remove_file(&obj);
}

/// `add_i32(i32, i32) -> i32`, built directly from `basalt_bir` types (the same shape as
/// `oracle.rs`'s own private `func_add_i32` fixture, reconstructed here since that fixture
/// lives in a `#[cfg(test)]` module private to that crate). Both params are integer-class, so
/// `nthreads` is the third integer register — this exercises a non-void scalar return, which
/// `vector_add` (a `void` kernel) never does.
fn hand_built_add_i32() -> Module {
    let f = Function {
        name: "add_i32".into(),
        params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
        ret: Ty::Scalar(Scalar::I32),
        insts: vec![Inst {
            ty: Ty::Scalar(Scalar::I32),
            op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
        }],
        blocks: vec![Block {
            insts: vec![InstId(0)],
            term: Term::Ret(Some(ValRef::Val(InstId(0)))),
        }],
    };
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

#[test]
fn hand_built_add_i32_links_and_runs() {
    if !cc_available() {
        eprintln!("skipping hand_built_add_i32_links_and_runs: `cc` not found");
        return;
    }

    let module = hand_built_add_i32();
    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    let artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for hand-built add_i32");
    let bytes = artifact.as_bytes().expect("oracle emits an object payload");

    let root = workspace_root();
    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_add_i32_{pid}.o"));
    write_object(bytes, &obj);

    compile_link_and_run(&root, "examples/cpu_launch_add_i32.c", &obj, "add_i32");

    let _ = std::fs::remove_file(&obj);
}
