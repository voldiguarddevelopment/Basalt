// The WMMA lane's real proof: the same tiled-matmul shape `basalt-x86/tests/tiled_sgemm.rs`
// proves on the CPU oracle, this time with `in_dtype = f16` so every one of its eight
// `Op::Mma` calls hits the canonical m16n16k16 tile `lower_wmma` actually lowers to a real
// NVVM tensor-core intrinsic, compiled to PTX text and run on real hardware through
// `basalt-runtime`'s CUDA Driver API loader. The oracle cannot run this same module (its own
// `Op::Mma` lowering refuses `f16` — see `oracle.rs`'s module header), so the two variants
// share a construction function parameterized by `in_dtype`/`in_elem_bytes` rather than
// literally sharing one `Module` value; both compute the identical `D = A@B + C` problem over
// the identical input values, so the GPU's result is checked against the same reference this
// file computes independently, matching what the oracle side already proved correct.
#![cfg(feature = "llvm")]

use std::ffi::c_void;

use basalt_bir::{
    AddrSpace, BinOp, Block, Function, Inst, InstId, MmaLayout, Module, Op, Scalar, Term, Ty,
    ValRef,
};
use basalt_llvm::{emit_assembly, LlvmTarget};
use basalt_runtime::CudaDriver;
use inkwell::context::Context;

fn push(insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValRef {
    insts.push(Inst { ty, op });
    ValRef::Val(InstId((insts.len() - 1) as u32))
}

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

/// Same construction as `basalt-x86/tests/tiled_sgemm.rs`'s `build_tiled_sgemm` — see that
/// file's doc comment for the full rationale (no stride field on `Op::Mma`, scratch tiles
/// staged past `d`'s real payload, one scratch buffer of each kind reused across all eight
/// `Op::Mma` calls). Kept as an independent copy rather than a shared helper crate, matching
/// how `link_and_run.rs` in both `basalt-x86` and this crate already each carry their own copy
/// of `hand_built_add_i32`/`add_i32_module`.
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

                // A real launch runs this kernel with a full warp of redundant threads (see
                // this fixture's caller below): every lane stages the identical tile data via
                // its own scalar stores above, and `Op::Mma` itself is warp-collective, but
                // cross-lane visibility of one lane's global-memory stores to another lane's
                // subsequent load is not guaranteed without an explicit barrier — plain SIMT
                // "same instruction, same cycle" execution does not imply that. `barrier` is a
                // genuine no-op under the CPU oracle's one-thread-at-a-time execution, so this
                // costs that side nothing; on real hardware it is the fence that makes the
                // scratch tile's data actually visible before `Op::Mma` reads it.
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
                // Same reasoning in reverse: the next K-step's re-staging of `a`/`b` scratch,
                // or (after the last K-step) the D write-back below, must not race the wmma
                // store this call just issued.
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

/// IEEE754 half-precision bit pattern for the exact integers this test's input data is built
/// from (0..=9, more than the 0..=7 the data actually needs). Hand-derived rather than pulled
/// from a crate: every value here is a small exact integer, so the general float-to-half
/// rounding machinery a real conversion routine needs is unnecessary — each entry is just
/// `sign=0, exponent = n's own power-of-two + 15 bias, mantissa = the fractional part scaled
/// to 10 bits`.
const F16_SMALL_INT: [u16; 10] = [
    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800, 0x4880,
];

fn f16_bits(v: u32) -> u16 {
    F16_SMALL_INT[v as usize]
}

fn open_driver_or_skip(test_name: &str) -> Option<CudaDriver> {
    match CudaDriver::load() {
        Ok(driver) => Some(driver),
        Err(err) => {
            eprintln!("skipping {test_name}: CUDA driver unavailable ({err})");
            None
        }
    }
}

#[test]
fn hand_built_tiled_sgemm_wmma_runs_on_real_hardware_and_matches_reference() {
    const N: usize = 32;
    const D_WORDS: usize = 1024 + 384; // 1024 real output floats + 384 words of tile scratch

    // Same input formulas as `basalt-x86/tests/tiled_sgemm.rs`'s C shim: small,
    // position-varying integers, so every product and every partial sum along a 32-deep dot
    // product stays far below 2^24 — exact in f32 regardless of summation order, so this is an
    // exact-match check against the reference below, not a tolerance-bounded one.
    let mut a = [0u16; N * N];
    let mut b = [0u16; N * N];
    let mut c = [0f32; N * N];
    let mut expected = [0f32; N * N];
    // Plain f32 mirrors of `a`/`b`'s real values (pre-`f16`-encoding), kept only so the
    // reference loop below can read the same values the kernel actually multiplies straight
    // out of an array — indexed exactly like `a[i * N + k]`/`b[k * N + j]` below, the same
    // shape the oracle-side C shim's reference loop uses — rather than re-deriving each
    // formula a second time by hand, which is exactly the kind of index-transposition mistake
    // (row/column swapped between the two formulas) a hand-rewritten reference invites.
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
    let ctx = Context::create();
    let ptx_text = emit_assembly(&module, &ctx, LlvmTarget::Nvptx)
        .expect("nvptx assembly emission succeeds for the canonical m16n16k16 tiled fixture");
    assert!(
        ptx_text.contains(".visible .entry tiled_sgemm_wmma"),
        "expected a dispatchable kernel entry in the emitted PTX:\n{ptx_text}"
    );
    assert!(
        ptx_text.contains("wmma.mma"),
        "expected a real wmma.mma instruction in the emitted PTX:\n{ptx_text}"
    );

    let Some(driver) = open_driver_or_skip(
        "hand_built_tiled_sgemm_wmma_runs_on_real_hardware_and_matches_reference",
    ) else {
        return;
    };
    let count = match driver.device_count() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("skipping: cuDeviceGetCount failed ({err})");
            return;
        }
    };
    if count == 0 {
        eprintln!("skipping: driver loaded but reports zero devices");
        return;
    }

    let cuda_ctx = driver
        .create_context(0)
        .expect("creating a context on device 0 of a driver that reports >=1 device");
    let cuda_module = cuda_ctx
        .load_module(&ptx_text)
        .expect("JIT-loading the emitted PTX for tiled_sgemm_wmma");
    let function = cuda_module
        .get_function("tiled_sgemm_wmma")
        .expect("looking up the tiled_sgemm_wmma entry point declared in the emitted PTX");

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let c_bytes: Vec<u8> = c.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let a_buf = cuda_ctx.alloc(a_bytes.len()).expect("allocating a");
    let b_buf = cuda_ctx.alloc(b_bytes.len()).expect("allocating b");
    let c_buf = cuda_ctx.alloc(c_bytes.len()).expect("allocating c");
    let d_buf = cuda_ctx
        .alloc(D_WORDS * std::mem::size_of::<f32>())
        .expect("allocating d (real output plus tile scratch)");

    a_buf.copy_from_host(&a_bytes).expect("cuMemcpyHtoD a");
    b_buf.copy_from_host(&b_bytes).expect("cuMemcpyHtoD b");
    c_buf.copy_from_host(&c_bytes).expect("cuMemcpyHtoD c");

    // `cuLaunchKernel`'s `kernelParams` is an array of pointers, one per kernel argument, each
    // pointing AT that argument's own storage (see `ptx_gpu_proof.rs` in `basalt-runtime` for
    // the same convention spelled out in full). Real tensor-core `wmma` instructions are
    // warp-collective: all 32 lanes of a warp must execute each `wmma.load`/`wmma.mma`/
    // `wmma.store` together, cooperatively holding one tile's fragments across the warp's
    // registers — a 1-thread launch leaves 31 lanes of the warp not just idle but never
    // started, so the tensor core never sees a complete warp and the load/mma/store produce
    // nothing usable (confirmed empirically: a 1x1x1 launch runs without a driver error but
    // reads back all zeros). This kernel has no `tid.x`/`bdim.x` use at all — every one of a
    // full warp's 32 threads redundantly executes the identical straight-line program and
    // writes the identical values to the identical addresses, which is wasteful but exactly
    // as correct as one thread doing it once, and satisfies the tensor core's real warp
    // requirement without needing this hand-built kernel to branch on its own thread index.
    let mut a_dptr: u64 = a_buf.device_ptr();
    let mut b_dptr: u64 = b_buf.device_ptr();
    let mut c_dptr: u64 = c_buf.device_ptr();
    let mut d_dptr: u64 = d_buf.device_ptr();
    let mut params: [*mut c_void; 4] = [
        &mut a_dptr as *mut u64 as *mut c_void,
        &mut b_dptr as *mut u64 as *mut c_void,
        &mut c_dptr as *mut u64 as *mut c_void,
        &mut d_dptr as *mut u64 as *mut c_void,
    ];

    function
        .launch((1, 1, 1), (32, 1, 1), 0, &mut params)
        .expect("launching tiled_sgemm_wmma");

    let mut d_bytes = vec![0u8; N * N * std::mem::size_of::<f32>()];
    d_buf.copy_to_host(&mut d_bytes).expect("cuMemcpyDtoH d");
    let got: Vec<f32> = d_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().expect("4-byte chunk")))
        .collect();

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
        "PASS: tiled_sgemm_wmma on real GPU hardware, 32x32 (2x2 tiles, K-accumulated over 2 \
         steps), exact match against the reference"
    );
}
