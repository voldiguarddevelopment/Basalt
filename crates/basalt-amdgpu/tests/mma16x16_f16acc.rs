// A narrower, independent proof for `Op::Mma`'s `f16`-accumulator path (the second half of
// `check_mma_shape`'s declared scope, alongside `tiled_sgemm_wmma.rs`'s `f32`-accumulator
// proof): a single `m16n16k16` tile, no scratch staging or K-step chaining needed since every
// operand is already exactly the canonical tile shape — `a`/`b`/`c`/`d` are four independent
// 16x16 buffers, one `Op::Mma` call computes `D = A@B + C` directly. Small position-varying
// integers keep every partial sum exact in `f16` (values stay far under 2048, `f16`'s exact
// integer range), so this is a bit-exact check against an independently computed reference, the
// same discipline `tiled_sgemm_wmma.rs` uses for its own `f32` case.

use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_amdgpu::Amdgcn;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, Block, Function, Inst, InstId, MmaLayout, Module, Op, Scalar, Term, Ty, ValRef,
};

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

fn build_single_mma_f16acc(name: &str) -> Module {
    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let f = Function {
        name: name.into(),
        params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
        ret: Ty::Void,
        insts: vec![Inst {
            ty: Ty::Void,
            op: Op::Mma {
                a: ValRef::Param(0),
                b: ValRef::Param(1),
                c: ValRef::Param(2),
                d: ValRef::Param(3),
                m: 16,
                n: 16,
                k: 16,
                in_dtype: Scalar::F16,
                acc_dtype: Scalar::F16,
                layout_a: MmaLayout::RowMajor,
                layout_b: MmaLayout::RowMajor,
            },
        }],
        blocks: vec![Block {
            insts: vec![InstId(0)],
            term: Term::Ret(None),
        }],
    };
    Module {
        funcs: vec![f],
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: vec![],
    }
}

const F16_SMALL_INT: [u16; 10] = [
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880,
];

fn f16_bits(v: u32) -> u16 {
    F16_SMALL_INT[v as usize]
}

fn f16_bits_to_decimal(bits: u16) -> f64 {
    F16_SMALL_INT
        .iter()
        .position(|&b| b == bits)
        .expect("every value in this fixture is one of the small ints in F16_SMALL_INT") as f64
}

#[test]
fn single_mma_f16_acc_matches_the_reference_on_the_rdna3_emulator() {
    let python = rdna3_python();
    if !python_available(&python) {
        eprintln!("skipping: no {python} interpreter on this machine");
        return;
    }

    const N: usize = 16;
    let mut a = [0u16; N * N];
    let mut b = [0u16; N * N];
    let mut c = [0u16; N * N];
    let mut expected = [0u32; N * N];
    for i in 0..N {
        for k in 0..N {
            a[i * N + k] = f16_bits(((i + k) % 5 + 1) as u32);
        }
    }
    for k in 0..N {
        for j in 0..N {
            b[k * N + j] = f16_bits(((k + 2 * j) % 5 + 1) as u32);
        }
    }
    for i in 0..N {
        for j in 0..N {
            c[i * N + j] = f16_bits(((i + j) % 3) as u32);
            let mut sum = ((i + j) % 3) as u32;
            for k in 0..N {
                let av = (i + k) % 5 + 1;
                let bv = (k + 2 * j) % 5 + 1;
                sum += (av * bv) as u32;
            }
            expected[i * N + j] = sum;
        }
    }

    let module = build_single_mma_f16acc("mma16x16_f16acc");
    let backend = Amdgcn;
    assert_eq!(backend.supports(&module), Support::Supported);
    let bytes = backend
        .emit(&module, &EmitOpts::default())
        .expect("Amdgcn::emit succeeds for the canonical f16-in/f16-acc single-tile fixture")
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact")
        .to_vec();

    let hsaco_path = std::env::temp_dir().join(format!(
        "basalt_amdgpu_mma16x16_f16acc_{}.hsaco",
        std::process::id()
    ));
    std::fs::write(&hsaco_path, &bytes).expect("writing the HSACO to a scratch file");

    let a_csv = a
        .iter()
        .map(|&v| f16_bits_to_decimal(v).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let b_csv = b
        .iter()
        .map(|&v| f16_bits_to_decimal(v).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let c_csv = c
        .iter()
        .map(|&v| f16_bits_to_decimal(v).to_string())
        .collect::<Vec<_>>()
        .join(",");

    let harness = workspace_root().join("tests/diff/rdna3_sim/run_kernel.py");
    let out = Command::new(&python)
        .arg(&harness)
        .args(["--hsaco"])
        .arg(&hsaco_path)
        .args([
            "--kernel",
            "mma16x16_f16acc",
            "--buf",
            &format!("in:f16:{a_csv}"),
            "--buf",
            &format!("in:f16:{b_csv}"),
            "--buf",
            &format!("in:f16:{c_csv}"),
            "--buf",
            &format!("out:f16:{}", N * N),
            "--global",
            "32,1,1",
            "--local",
            "32,1,1",
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
            "rdna3-sim harness did not exit 0 running mma16x16_f16acc's real HSACO:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let got: Vec<f32> = stdout
        .split_whitespace()
        .map(|w| {
            w.parse()
                .unwrap_or_else(|_| panic!("harness printed a parseable f32, got {w:?}"))
        })
        .collect();
    assert_eq!(got.len(), N * N, "expected one printed value per D element");

    for i in 0..N {
        for j in 0..N {
            assert_eq!(
                got[i * N + j],
                expected[i * N + j] as f32,
                "mismatch at ({i},{j}): expected {}, got {}",
                expected[i * N + j],
                got[i * N + j]
            );
        }
    }

    println!(
        "PASS: mma16x16_f16acc on the real RDNA3 emulator, single 16x16 tile, f16 accumulator, \
         exact match against the reference"
    );
}
