// Proves `optimize`'s instruction-count claim concretely, on `tests/kernels/mymathhomework.cu`
// run through the real lex/preprocess/parse -> sema check/lower pipeline (not a hand-built BIR
// fixture), and proves execution is preserved: the optimized module's emitted object still
// produces the same hand-verified output as the unoptimized one, linked against the same C
// shim via the system compiler.

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_frontend_c::PpOpts;
use basalt_passes::optimize;

const MYMATHHOMEWORK_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/kernels/mymathhomework.cu"
));

const MYMATHHOMEWORK_SHIM_C: &str = r#"
#include <stdint.h>
#include <stdio.h>

extern void mymathhomework(int32_t *out, int64_t nthreads);

int main(void) {
    int32_t out[1] = {-1};
    mymathhomework(out, 1);
    int32_t expected = 20;
    if (out[0] != expected) {
        fprintf(stderr, "FAIL: expected %d, got %d\n", expected, out[0]);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
"#;

fn lower_mymathhomework() -> basalt_bir::Module {
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(MYMATHHOMEWORK_SRC, &PpOpts::default());
    assert!(pp_errors.is_empty(), "preprocess errors: {pp_errors:?}");
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(parse_errors.is_empty(), "parse errors: {parse_errors:?}");
    let sema_diags = basalt_sema::check(&tu);
    assert!(sema_diags.is_empty(), "sema diagnostics: {sema_diags:?}");
    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(
        lower_diags.is_empty(),
        "lowering diagnostics: {lower_diags:?}"
    );
    module
}

fn total_insts(module: &basalt_bir::Module) -> usize {
    module.funcs.iter().map(|f| f.insts.len()).sum()
}

fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_cc(args: &[&std::ffi::OsStr]) {
    let out = std::process::Command::new("cc")
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
fn optimize_shrinks_mymathhomework_and_preserves_its_output() {
    let before = lower_mymathhomework();
    let before_count = total_insts(&before);

    let after = optimize(&before);
    let after_count = total_insts(&after);

    println!("mymathhomework.cu instruction count: before = {before_count}, after = {after_count}");
    assert!(
        after_count < before_count,
        "optimize must strictly shrink the instruction count: before = {before_count}, \
         after = {after_count}\n--- before ---\n{}\n--- after ---\n{}",
        basalt_bir::print(&before),
        basalt_bir::print(&after)
    );

    let text = basalt_bir::print(&after);
    let reparsed = basalt_bir::parse(&text).expect("parse(print(optimize(m))) must parse");
    assert_eq!(
        reparsed, after,
        "parse(print(m)) != m on optimize's output for the real mymathhomework.cu lowering"
    );

    if !cc_available() {
        eprintln!(
            "skipping optimize_shrinks_mymathhomework_and_preserves_its_output's execution \
             proof: `cc` not found"
        );
        return;
    }

    assert_eq!(
        basalt_x86::X86Oracle.supports(&after),
        Support::Supported,
        "the oracle must accept optimize's output"
    );

    let bytes = basalt_x86::X86Oracle
        .emit(&after, &EmitOpts::default())
        .expect("oracle emit succeeds on optimize's output")
        .as_bytes()
        .expect("oracle emits an object payload")
        .to_vec();

    let pid = std::process::id();
    let scratch = std::env::temp_dir();
    let shim_c = scratch.join(format!("basalt_pipeline_shim_{pid}.c"));
    std::fs::write(&shim_c, MYMATHHOMEWORK_SHIM_C).expect("writing shim source");
    let shim_o = scratch.join(format!("basalt_pipeline_shim_{pid}.o"));
    run_cc(&[
        std::ffi::OsStr::new("-c"),
        shim_c.as_os_str(),
        std::ffi::OsStr::new("-o"),
        shim_o.as_os_str(),
    ]);

    let obj = scratch.join(format!("basalt_pipeline_optimized_{pid}.o"));
    std::fs::write(&obj, &bytes).expect("writing payload object");
    let exe = scratch.join(format!("basalt_pipeline_optimized_{pid}"));
    run_cc(&[
        shim_o.as_os_str(),
        obj.as_os_str(),
        std::ffi::OsStr::new("-o"),
        exe.as_os_str(),
    ]);

    let out = std::process::Command::new(&exe)
        .output()
        .expect("built executable runs");

    let _ = std::fs::remove_file(&obj);
    let _ = std::fs::remove_file(&exe);
    let _ = std::fs::remove_file(&shim_o);
    let _ = std::fs::remove_file(&shim_c);

    assert!(
        out.status.success(),
        "optimized binary failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "PASS",
        "optimize must not change mymathhomework's observable output"
    );
}
