// `lower.rs`'s own real proof for `Op::Mma`: the same 32x32 tiled-matmul BIR fixture
// `basalt-x86/tests/tiled_sgemm.rs` and `basalt-llvm/tests/tiled_sgemm_wmma.rs` already use (2x2
// output tiles, each accumulated over two 16-deep K-steps, eight `Op::Mma` calls with real
// accumulator chaining), built by hand the same way those two files do, this time lowered
// through this crate's own hand-rolled `v_wmma_f32_16x16x16_f16` encoding and driven through
// tinygrad's real, instruction-level RDNA3 emulator (the same harness/skip convention
// `stress_kernel.rs` already uses). The oracle cannot run this module directly (its own
// `Op::Mma` lowering refuses `f16`), so the expected result is computed independently here, the
// same way `basalt-llvm`'s own WMMA test does.

use std::path::{Path, PathBuf};
use std::process::Command;

use basalt_amdgpu::Amdgcn;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, BinOp, Block, Function, Inst, InstId, MmaLayout, Module, Op, Scalar, Term, Ty,
    ValRef,
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

fn push(insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValRef {
    insts.push(Inst { ty, op });
    ValRef::Val(InstId((insts.len() - 1) as u32))
}

/// `base + offset` as a real `ptr.global` value — `Op::Bin(Add, ..)` on a pointer operand,
/// BIR's own address-arithmetic idiom, matching `basalt-x86`'s and `basalt-llvm`'s own copies of
/// this same fixture.
fn addr(insts: &mut Vec<Inst>, base: ValRef, offset: i64) -> ValRef {
    let off = push(insts, Ty::Scalar(Scalar::I64), Op::ConstInt(offset));
    push(
        insts,
        Ty::Ptr(AddrSpace::Global),
        Op::Bin(BinOp::Add, base, off),
    )
}

fn load_at(insts: &mut Vec<Inst>, base: ValRef, offset: i64, ty: Ty, align: u32) -> ValRef {
    let p = addr(insts, base, offset);
    push(
        insts,
        ty,
        Op::Load {
            ptr: p,
            space: AddrSpace::Global,
            align,
            volatile: false,
        },
    )
}

fn store_at(insts: &mut Vec<Inst>, base: ValRef, offset: i64, ty: Ty, align: u32, val: ValRef) {
    let p = addr(insts, base, offset);
    push(
        insts,
        Ty::Void,
        Op::Store {
            ptr: p,
            val,
            ty,
            space: AddrSpace::Global,
            align,
            volatile: false,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn copy_tile(
    insts: &mut Vec<Inst>,
    src_param: ValRef,
    src_base_off: i64,
    src_row_stride: i64,
    src_row0: i64,
    src_col0: i64,
    dst_param: ValRef,
    dst_base_off: i64,
    dst_row_stride: i64,
    dst_row0: i64,
    dst_col0: i64,
    elem_bytes: i64,
    ty: Ty,
) {
    let align = elem_bytes as u32;
    for i in 0..16i64 {
        for j in 0..16i64 {
            let src_off =
                src_base_off + ((src_row0 + i) * src_row_stride + (src_col0 + j)) * elem_bytes;
            let v = load_at(insts, src_param, src_off, ty, align);
            let dst_off =
                dst_base_off + ((dst_row0 + i) * dst_row_stride + (dst_col0 + j)) * elem_bytes;
            store_at(insts, dst_param, dst_off, ty, align, v);
        }
    }
}

/// Same construction as `basalt-x86/tests/tiled_sgemm.rs`'s `build_tiled_sgemm` and
/// `basalt-llvm/tests/tiled_sgemm_wmma.rs`'s own copy of it — see either file's doc comment for
/// the full rationale (no stride field on `Op::Mma`, scratch tiles staged past `d`'s real
/// payload, one scratch buffer of each kind reused across all eight `Op::Mma` calls). Kept as an
/// independent copy rather than a shared helper crate, matching how both of those files already
/// do the same.
fn build_tiled_sgemm(name: &str, in_dtype: Scalar, in_elem_bytes: i64) -> Module {
    const FULL: i64 = 32;
    const TILE: i64 = 16;
    let d_real_bytes = FULL * FULL * 4;
    let a_scratch_off = d_real_bytes;
    let b_scratch_off = a_scratch_off + TILE * TILE * in_elem_bytes;
    let c_scratch_off = b_scratch_off + TILE * TILE * in_elem_bytes;

    let ptr_global = Ty::Ptr(AddrSpace::Global);
    let in_ty = Ty::Scalar(in_dtype);
    let acc_ty = Ty::Scalar(Scalar::F32);

    let mut insts: Vec<Inst> = Vec::new();
    let a_p = ValRef::Param(0);
    let b_p = ValRef::Param(1);
    let c_p = ValRef::Param(2);
    let d_p = ValRef::Param(3);

    for tr in 0..2i64 {
        for tc in 0..2i64 {
            for ks in 0..2i64 {
                copy_tile(
                    &mut insts,
                    a_p,
                    0,
                    FULL,
                    tr * TILE,
                    ks * TILE,
                    d_p,
                    a_scratch_off,
                    TILE,
                    0,
                    0,
                    in_elem_bytes,
                    in_ty,
                );
                copy_tile(
                    &mut insts,
                    b_p,
                    0,
                    FULL,
                    ks * TILE,
                    tc * TILE,
                    d_p,
                    b_scratch_off,
                    TILE,
                    0,
                    0,
                    in_elem_bytes,
                    in_ty,
                );
                if ks == 0 {
                    copy_tile(
                        &mut insts,
                        c_p,
                        0,
                        FULL,
                        tr * TILE,
                        tc * TILE,
                        d_p,
                        c_scratch_off,
                        TILE,
                        0,
                        0,
                        4,
                        acc_ty,
                    );
                }
                push(&mut insts, Ty::Void, Op::Barrier);
                let a_scratch_ptr = addr(&mut insts, d_p, a_scratch_off);
                let b_scratch_ptr = addr(&mut insts, d_p, b_scratch_off);
                let c_scratch_ptr = addr(&mut insts, d_p, c_scratch_off);
                push(
                    &mut insts,
                    Ty::Void,
                    Op::Mma {
                        a: a_scratch_ptr,
                        b: b_scratch_ptr,
                        c: c_scratch_ptr,
                        d: c_scratch_ptr,
                        m: 16,
                        n: 16,
                        k: 16,
                        in_dtype,
                        acc_dtype: Scalar::F32,
                        layout_a: MmaLayout::RowMajor,
                        layout_b: MmaLayout::RowMajor,
                    },
                );
                push(&mut insts, Ty::Void, Op::Barrier);
            }
            copy_tile(
                &mut insts,
                d_p,
                c_scratch_off,
                TILE,
                0,
                0,
                d_p,
                0,
                FULL,
                tr * TILE,
                tc * TILE,
                4,
                acc_ty,
            );
        }
    }

    let n_insts = insts.len();
    let f = Function {
        is_kernel: true,
        name: name.into(),
        params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
        ret: Ty::Void,
        insts,
        blocks: vec![Block {
            insts: (0..n_insts as u32).map(InstId).collect(),
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

/// IEEE754 half-precision bit pattern for the exact small integers this test's input data is
/// built from — hand-derived rather than pulled from a crate, same table
/// `basalt-llvm/tests/tiled_sgemm_wmma.rs` uses (every value here is a small exact integer, so
/// general float-to-half rounding is unnecessary).
const F16_SMALL_INT: [u16; 10] = [
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880,
];

fn f16_bits(v: u32) -> u16 {
    F16_SMALL_INT[v as usize]
}

/// A `struct`-`e`-format-compatible decimal string for an f16 bit pattern that is always one of
/// the small exact integers `F16_SMALL_INT` encodes — round-trips exactly through
/// `run_kernel.py`'s own `f16` buffer parsing (`float(x)` then packed with `struct`'s `e` code).
fn f16_bits_to_decimal(bits: u16) -> f64 {
    F16_SMALL_INT
        .iter()
        .position(|&b| b == bits)
        .expect("every A/B value in this fixture is one of the small ints in F16_SMALL_INT")
        as f64
}

#[test]
fn tiled_sgemm_wmma_hsaco_is_deterministic() {
    let module = build_tiled_sgemm("tiled_sgemm_wmma", Scalar::F16, 2);
    let backend = Amdgcn;
    assert_eq!(backend.supports(&module), Support::Supported);
    let a = backend
        .emit(&module, &EmitOpts::default())
        .expect("Amdgcn::emit succeeds for the canonical f16-in/f32-acc tiled fixture")
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact")
        .to_vec();
    let b = backend
        .emit(&module, &EmitOpts::default())
        .expect("Amdgcn::emit succeeds for the canonical f16-in/f32-acc tiled fixture")
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact")
        .to_vec();
    assert_eq!(
        a, b,
        "same BIR module in must produce byte-identical HSACO out"
    );
}

#[test]
fn tiled_sgemm_wmma_matches_the_reference_on_the_rdna3_emulator() {
    let python = rdna3_python();
    if !python_available(&python) {
        eprintln!("skipping: no {python} interpreter on this machine");
        return;
    }

    const N: usize = 32;
    const D_WORDS: usize = 1536; // 1024 real output floats + 512 words of tile scratch (see
                                 // build_tiled_sgemm's own a/b/c_scratch_off derivation: 512 +
                                 // 512 + 1024 bytes of scratch = 512 f32 words).

    let mut a = [0u16; N * N];
    let mut b = [0u16; N * N];
    let mut c = [0f32; N * N];
    let mut expected = [0f32; N * N];
    let mut a_val = [0f32; N * N];
    let mut b_val = [0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let av = ((3 * i + j) % 7 + 1) as u32;
            let bv = ((i + 5 * j) % 7 + 1) as u32;
            a[i * N + j] = f16_bits(av);
            b[i * N + j] = f16_bits(bv);
            a_val[i * N + j] = av as f32;
            b_val[i * N + j] = bv as f32;
            c[i * N + j] = ((i + j) % 4) as f32;
        }
    }
    for i in 0..N {
        for j in 0..N {
            let mut sum = c[i * N + j];
            for k in 0..N {
                sum += a_val[i * N + k] * b_val[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    let module = build_tiled_sgemm("tiled_sgemm_wmma", Scalar::F16, 2);
    let backend = Amdgcn;
    assert_eq!(backend.supports(&module), Support::Supported);
    let bytes = backend
        .emit(&module, &EmitOpts::default())
        .expect("Amdgcn::emit succeeds for the canonical f16-in/f32-acc tiled fixture")
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact")
        .to_vec();

    let hsaco_path = std::env::temp_dir().join(format!(
        "basalt_amdgpu_tiled_sgemm_wmma_{}.hsaco",
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
            "tiled_sgemm_wmma",
            "--buf",
            &format!("in:f16:{a_csv}"),
            "--buf",
            &format!("in:f16:{b_csv}"),
            "--buf",
            &format!("in:f32:{c_csv}"),
            "--buf",
            &format!("out:f32:{D_WORDS}"),
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
            "rdna3-sim harness did not exit 0 running tiled_sgemm_wmma's real HSACO:\nstdout:\n{}\nstderr:\n{}",
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
    assert_eq!(got.len(), D_WORDS, "expected one printed value per D word");

    for i in 0..N {
        for j in 0..N {
            assert_eq!(
                got[i * N + j],
                expected[i * N + j],
                "mismatch at ({i},{j}): expected {}, got {}",
                expected[i * N + j],
                got[i * N + j]
            );
        }
    }

    println!(
        "PASS: tiled_sgemm_wmma on the real RDNA3 emulator, 32x32 (2x2 tiles, K-accumulated \
         over 2 steps), exact match against the reference"
    );
}
