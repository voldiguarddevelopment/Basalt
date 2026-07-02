// The cross-check this crate exists to deliver: does the object `emit_object(..., LlvmTarget::
// X86)` produces actually link, via the real system C compiler, and run to the same correct
// answer the hand-rolled x86-64 oracle already proved (`basalt-x86/tests/link_and_run.rs`)?
// Structure and helpers mirror that file closely on purpose — same fixture
// (`tests/kernels/vector_add.cu`), same host shim (`examples/cpu_launch_vadd.c`), same
// `cc`-then-run mechanics — because the whole point is an independent confirmation that
// LLVM's own x86 codegen agrees with the hand-rolled oracle on BIR's semantics, not a new
// technique to validate.
#![cfg(feature = "llvm")]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_llvm::{emit_object, LlvmTarget};

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

#[test]
fn vector_add_via_llvm_x86_links_and_runs_and_matches_the_oracle() {
    if !cc_available() {
        eprintln!(
            "skipping vector_add_via_llvm_x86_links_and_runs_and_matches_the_oracle: `cc` not found"
        );
        return;
    }

    let root = workspace_root();
    let src_path = root.join("tests/kernels/vector_add.cu");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    let opts = basalt_frontend_c::PpOpts {
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

    let llvm_ctx = inkwell::context::Context::create();
    let bytes = emit_object(&module, &llvm_ctx, LlvmTarget::X86)
        .expect("llvm x86 object emission succeeds for vector_add");

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("basalt_llvm_vadd_{pid}.o"));
    write_object(&bytes, &obj);

    let scratch = std::env::temp_dir();
    let shim_o = scratch.join(format!("basalt_llvm_vadd_shim_{pid}.o"));
    let exe = scratch.join(format!("basalt_llvm_vadd_exe_{pid}"));

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

    run_and_check(&exe);

    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_file(&obj);
}
