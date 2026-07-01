// Proves the register allocator's payoff directly: the regalloc backend measurably lowers
// pressure relative to the oracle on a kernel built to force real spilling in both, and the
// two backends still compute byte-identical results when linked and run. Reuses
// link_and_run.rs's full-pipeline technique (lex -> preprocess -> parse -> check -> lower,
// zero diagnostics asserted at every stage) rather than reinventing it; see that file's
// header for the general rationale.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::Module;
use basalt_frontend_c::PpOpts;
use basalt_x86::{X86Oracle, X86Regalloc};
use object::read::{Object as ReadObject, ObjectSection};

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

/// Runs `exe`, returning its exit code and stdout without panicking on a nonzero exit — the
/// oracle-vs-regalloc comparison below wants to see both outcomes so a divergence is reported
/// with both sides' output, rather than panicking on whichever backend happens to fail first.
fn run_capture(exe: &Path) -> (i32, String) {
    let out = Command::new(exe).output().expect("built executable runs");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn write_object(bytes: &[u8], path: &Path) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}

/// Runs the real lex/preprocess/parse/check/lower pipeline over `tests/kernels/stress.cu`,
/// asserting zero diagnostics at every stage — mirrors `link_and_run.rs`'s own
/// `vector_add_links_and_runs_via_full_pipeline` exactly, just against a different fixture.
fn lower_stress_module(root: &Path) -> Module {
    let src_path = root.join("tests/kernels/stress.cu");
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
        "preprocessing stress.cu produced problems: {:?}",
        pp_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(
        parse_errors.is_empty(),
        "parsing stress.cu produced problems: {:?}",
        parse_errors
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );

    let sema_diags = basalt_sema::check(&tu);
    assert!(
        sema_diags.is_empty(),
        "type-checking stress.cu produced diagnostics: {:?}",
        sema_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering stress.cu produced diagnostics: {:?}",
        lower_diags.iter().map(|d| d.code).collect::<Vec<_>>()
    );

    module
}

/// Counts occurrences of a `[rbp + disp32]` memory operand in `text` — the only stack-slot
/// addressing form either backend's encoder ever emits (`enc.rs`'s `Rm::RbpDisp`, used for
/// every spill load/store, parameter/phi staging slot, and the return-value home). Per
/// `enc.rs::push_modrm`, that operand always lowers to a ModRM byte with `mod=10`, `rm=101`
/// (rbp) and an arbitrary 3-bit reg field — i.e. a byte matching `0b10rrr101` for any `rrr`
/// (`0x85, 0x8D, 0x95, 0x9D, 0xA5, 0xAD, 0xB5, 0xBD`) — immediately followed by 4
/// little-endian displacement bytes. The scan walks the buffer once: a byte matching that
/// mask counts one memory operand and skips the 4 following displacement bytes, so a
/// displacement that happens to contain the same bit pattern is never mistaken for another
/// operand or double-counted. This can still overcount if some unrelated immediate byte
/// elsewhere in the stream coincides with the mask at an unlucky offset, but both objects
/// here come from the same encoder along the same instruction shapes, so that bias lands
/// equally on both sides and does not affect the relative comparison this test makes.
fn count_rbp_disp_mem_ops(text: &[u8]) -> usize {
    const MASK: u8 = 0b1100_0111;
    const PATTERN: u8 = 0b1000_0101;
    let mut count = 0usize;
    let mut i = 0usize;
    while i < text.len() {
        if text[i] & MASK == PATTERN {
            count += 1;
            i += 5;
        } else {
            i += 1;
        }
    }
    count
}

fn text_section(bytes: &[u8]) -> &[u8] {
    let file = object::read::File::parse(bytes).expect("emitted bytes parse as an object file");
    let text = file
        .section_by_name(".text")
        .expect(".text section present");
    text.data().expect(".text data readable")
}

#[test]
fn regalloc_has_lower_register_pressure_than_oracle_on_stress() {
    let root = workspace_root();
    let module = lower_stress_module(&root);

    assert_eq!(X86Oracle.supports(&module), Support::Supported);
    assert_eq!(X86Regalloc.supports(&module), Support::Supported);

    let oracle_artifact = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for stress");
    let oracle_bytes = oracle_artifact
        .as_bytes()
        .expect("oracle emits an object payload");

    let regalloc_artifact = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for stress");
    let regalloc_bytes = regalloc_artifact
        .as_bytes()
        .expect("regalloc emits an object payload");

    let oracle_text = text_section(oracle_bytes);
    let regalloc_text = text_section(regalloc_bytes);

    let oracle_size = oracle_text.len();
    let regalloc_size = regalloc_text.len();
    let oracle_mem_ops = count_rbp_disp_mem_ops(oracle_text);
    let regalloc_mem_ops = count_rbp_disp_mem_ops(regalloc_text);

    eprintln!("stress.cu .text size (bytes): oracle={oracle_size} regalloc={regalloc_size}");
    eprintln!(
        "stress.cu rbp-relative memory operands: oracle={oracle_mem_ops} regalloc={regalloc_mem_ops}"
    );

    assert!(
        regalloc_size < oracle_size,
        "expected regalloc .text ({regalloc_size}B) smaller than oracle's ({oracle_size}B)"
    );
    assert!(
        regalloc_mem_ops < oracle_mem_ops,
        "expected regalloc to touch memory less than the oracle: regalloc={regalloc_mem_ops} oracle={oracle_mem_ops}"
    );
    assert!(
        regalloc_mem_ops * 2 < oracle_mem_ops,
        "expected regalloc to at least halve the oracle's memory traffic on stress.cu: \
         regalloc={regalloc_mem_ops} oracle={oracle_mem_ops}"
    );
}

#[test]
fn stress_executes_identically_through_oracle_and_regalloc() {
    if !cc_available() {
        eprintln!(
            "skipping stress_executes_identically_through_oracle_and_regalloc: `cc` not found"
        );
        return;
    }

    let root = workspace_root();
    let module = lower_stress_module(&root);

    let oracle_bytes = X86Oracle
        .emit(&module, &EmitOpts::default())
        .expect("oracle emit succeeds for stress")
        .as_bytes()
        .expect("oracle emits an object payload")
        .to_vec();
    let regalloc_bytes = X86Regalloc
        .emit(&module, &EmitOpts::default())
        .expect("regalloc emit succeeds for stress")
        .as_bytes()
        .expect("regalloc emits an object payload")
        .to_vec();

    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let oracle_o = scratch.join(format!("basalt_pressure_oracle_{pid}.o"));
    let regalloc_o = scratch.join(format!("basalt_pressure_regalloc_{pid}.o"));
    write_object(&oracle_bytes, &oracle_o);
    write_object(&regalloc_bytes, &regalloc_o);

    let shim_path = root.join("examples/cpu_launch_stress.c");
    let shim_o = scratch.join(format!("basalt_pressure_shim_{pid}.o"));
    run_cc(&[
        OsStr::new("-c"),
        shim_path.as_os_str(),
        OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);

    let oracle_exe = scratch.join(format!("basalt_pressure_oracle_exe_{pid}"));
    let regalloc_exe = scratch.join(format!("basalt_pressure_regalloc_exe_{pid}"));
    run_cc(&[
        shim_o.as_os_str(),
        oracle_o.as_os_str(),
        OsStr::new("-o"),
        oracle_exe.as_os_str(),
    ]);
    run_cc(&[
        shim_o.as_os_str(),
        regalloc_o.as_os_str(),
        OsStr::new("-o"),
        regalloc_exe.as_os_str(),
    ]);

    let (oracle_code, oracle_stdout) = run_capture(&oracle_exe);
    let (regalloc_code, regalloc_stdout) = run_capture(&regalloc_exe);

    assert_eq!(
        oracle_code, 0,
        "oracle exe exited {oracle_code}, stdout:\n{oracle_stdout}"
    );
    assert_eq!(
        regalloc_code, 0,
        "regalloc exe exited {regalloc_code}, stdout:\n{regalloc_stdout}"
    );
    assert_eq!(
        oracle_stdout, regalloc_stdout,
        "oracle and regalloc produced different output for stress.cu"
    );
    print!("{oracle_stdout}");

    let _ = std::fs::remove_file(&oracle_o);
    let _ = std::fs::remove_file(&regalloc_o);
    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&oracle_exe);
    let _ = std::fs::remove_file(&regalloc_exe);
}
