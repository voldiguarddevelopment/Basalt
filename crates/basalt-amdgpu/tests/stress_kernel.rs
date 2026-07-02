// The real proof `lower.rs`'s own module header promises: compiles `tests/kernels/stress.cu`
// all the way through the real frontend/sema/optimize pipeline and this crate's own `Amdgcn`
// backend into a genuine HSACO object, then drives it through
// `tests/diff/rdna3_sim/run_kernel.py` (tinygrad's real, compiler-agnostic RDNA3 emulator) with
// the exact same input data `examples/cpu_launch_stress.c` uses, and checks the observed output
// against `tests/diff/golden/stress.txt`'s already-established correct value (`191.0`). Same
// "skip cleanly on a machine without the tooling" pattern as `rdna3_dispatch.rs`.
//
// `stress` is driven with one workgroup of one thread (`--global 1,1,1 --local 1,1,1`),
// matching the driver's own `stress(a, out, 1, 1)` call (`n=1`, one thread total): the kernel's
// `if (i < n)` becomes a data-dependent branch taken by every active lane in the (single-lane)
// wave, squarely inside this backend's documented control-flow scope (see `lower.rs`'s own
// header on why a genuinely divergent multi-lane launch is out of scope for now).

use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_amdgpu::Amdgcn;
use basalt_backend::{Backend, EmitOpts, Support};

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

/// Runs `stress.cu` through the real frontend/sema/`basalt_passes::optimize` pipeline (the same
/// one `basalt-cli`'s `--cpu`/`--nvidia-ptx` modes use) and hands the result to `Amdgcn`.
fn build_stress_hsaco() -> Vec<u8> {
    let root = workspace_root();
    let src_path = root.join("tests/kernels/stress.cu");
    let src = std::fs::read_to_string(&src_path).expect("stress.cu is readable");

    let pp = basalt_frontend_c::PpOpts::default();
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(&src, &pp);
    assert!(pp_errors.is_empty(), "stress.cu preprocesses cleanly");
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    assert!(parse_errors.is_empty(), "stress.cu parses cleanly");

    let sema_diags = basalt_sema::check(&tu);
    assert!(sema_diags.is_empty(), "stress.cu passes sema cleanly");
    let (module, lower_diags) = basalt_sema::lower(&tu);
    assert!(lower_diags.is_empty(), "stress.cu lowers to BIR cleanly");

    let module = basalt_passes::optimize(&module);

    let backend = Amdgcn;
    assert_eq!(
        backend.supports(&module),
        Support::Supported,
        "Amdgcn::supports must accept stress.cu's BIR"
    );
    let artifact = backend
        .emit(&module, &EmitOpts::default())
        .expect("Amdgcn::emit succeeds for stress.cu");
    artifact
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact")
        .to_vec()
}

#[test]
fn stress_hsaco_is_deterministic() {
    let a = build_stress_hsaco();
    let b = build_stress_hsaco();
    assert_eq!(
        a, b,
        "same BIR module in must produce byte-identical HSACO out"
    );
}

#[test]
fn stress_kernel_computes_the_oracle_fold_on_the_rdna3_emulator() {
    let python = rdna3_python();
    if !python_available(&python) {
        eprintln!("skipping: no {python} interpreter on this machine");
        return;
    }

    let bytes = build_stress_hsaco();
    let hsaco_path =
        std::env::temp_dir().join(format!("basalt_amdgpu_stress_{}.hsaco", std::process::id()));
    std::fs::write(&hsaco_path, &bytes).expect("writing the HSACO to a scratch file");

    // a[i] = (i+1)*0.5 - 3.0, exactly matching examples/cpu_launch_stress.c's own input.
    let a: Vec<f32> = (0..20).map(|i| (i as f32 + 1.0) * 0.5 - 3.0).collect();
    let a_csv = a
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let harness = workspace_root().join("tests/diff/rdna3_sim/run_kernel.py");
    let out = Command::new(&python)
        .arg(&harness)
        .args(["--hsaco"])
        .arg(&hsaco_path)
        .args([
            "--kernel",
            "stress",
            "--buf",
            &format!("in:f32:{a_csv}"),
            "--buf",
            "out:f32:1",
            "--scalar",
            "i32:1",
            "--global",
            "1,1,1",
            "--local",
            "1,1,1",
        ])
        .output()
        .expect("spawning the rdna3-sim harness");

    let _ = std::fs::remove_file(&hsaco_path);

    match out.status.code() {
        Some(SKIP) => {
            eprintln!(
                "skipping: rdna3-sim unavailable ({})",
                String::from_utf8_lossy(&out.stderr).trim()
            );
            return;
        }
        Some(0) => {}
        _ => panic!(
            "rdna3-sim harness did not exit 0 running stress.cu's real HSACO:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let observed: f32 = stdout
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("harness printed a parseable f32, got {stdout:?}"));

    // tests/diff/golden/stress.txt records the oracle's own already-established correct value.
    let expected = 191.0f32;
    assert!(
        (observed - expected).abs() < 1e-3,
        "stress kernel produced {observed}, expected {expected} (the CPU oracle's own golden value)"
    );
}
