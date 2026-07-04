#![cfg(feature = "mlir")]
// The MLIR-lane counterpart to `basalt-runtime::tests::triton_ptx_gpu_proof`'s
// `masked_triton_vector_add_runs_on_real_gpu_hardware`: same real kernel
// (`tests/kernels/tri_vadd.py`), same real CUDA Driver API dispatch, same host-computed
// reference and poison-then-check convention, but the PTX text comes from this crate's
// `emit_ptx_text` (MLIR dialects -> `mlir-opt`'s own NVPTX backend) instead of `basalt-ptx`'s
// hand-rolled encoder — the Triton counterpart to `nvptx_gpu_proof.rs`'s own `vector_add.cu`
// proof, this task's own exit criterion for Phase 11 (P11-T4).
//
// `emit.rs`'s own module header documents the exploded memref-descriptor kernel ABI
// `-convert-gpu-to-nvvm`'s default calling convention produces; `lower.rs`'s own module
// header (see its "A `Global` parameter accessed at more than one element type" section)
// documents why `tri_vadd.py`'s `c_ptr` parameter is uniquely `memref<?xi8>` rather than
// `memref<?xf32>` like `a_ptr`/`b_ptr` — `basalt-sema::triton_lower` reuses `c_ptr` as both
// the kernel's real output buffer and scratch space for the `offsets`/`mask` tiles it
// materializes, read/written at `i64`/`i1`/`f32`. The real, `mlir-opt`-verified signature this
// produces is `gpu.func @vector_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2:
// memref<?xi8>, %arg3: i64, %arg4: i64)`, which explodes to 5 (a) + 5 (b) + 5 (c) + 1 (n) + 1
// (BLOCK_SIZE) = 17 real PTX parameters — mirroring `nvptx_gpu_proof.rs`'s own `MemrefDesc`/
// `desc()` pattern for the first two (`f32`-element) descriptors, with `c_ptr`'s own
// descriptor's `size`/`stride` fields in bytes (its memref is `i8`-element) rather than `f32`
// elements.
//
// "Matches the hand-rolled path" (the other half of this task's own exit criterion) is not a
// byte-level PTX diff: `basalt-ptx`'s hand-rolled PTX takes three flat pointers plus two plain
// scalars (5 real parameters total, see `triton_ptx_gpu_proof.rs`), while this lane's PTX takes
// the 17-parameter exploded-descriptor ABI above — two genuinely different real ABIs for the
// same source kernel, with no meaningful byte-level correspondence between their `params`
// arrays or PTX text. What "matches" means here is that both, run for real on the same
// hardware with the same `N`/`BLOCK`/poison convention, produce the identical correct numeric
// result — this test and `triton_ptx_gpu_proof.rs`'s own
// `masked_triton_vector_add_runs_on_real_gpu_hardware`, run back to back, demonstrate exactly
// that.
//
// Whether `mlir-opt` (this crate's own toolchain dependency) or a CUDA driver is missing, this
// test self-skips rather than failing the default `--features mlir` run on a machine without
// real hardware, mirroring `nvptx_gpu_proof.rs`'s own `open_driver_or_skip` convention.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use basalt_mlir::emit_ptx_text;
use basalt_runtime::CudaDriver;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

/// Runs the real `parse -> check_triton -> lower_triton -> basalt_passes::optimize` pipeline
/// over `src`, asserting no diagnostics at any stage, and returns the optimized BIR module —
/// mirrors `crates/basalt-mlir/src/lower/tests.rs`'s own `compile_triton` helper (and
/// `crates/basalt-x86/tests/triton_link_and_run.rs`'s identically-shaped one), duplicated here
/// rather than shared, matching this project's own per-test-binary convention (`lower/tests.rs`
/// and `nvptx_gpu_proof.rs` each already keep their own copy of the equivalent CUDA-C helper).
fn compile_triton_to_bir(src: &str) -> basalt_bir::Module {
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

    basalt_passes::optimize(&bir)
}

fn read_kernel(rel_path: &str) -> String {
    let path = workspace_root().join(rel_path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
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
fn masked_triton_vector_add_via_mlir_nvptx_runs_on_real_gpu_hardware() {
    // Compiling all the way to PTX text is real compiler work, not hardware-gated, and runs
    // unconditionally on every machine — only what follows (touching an actual CUDA driver) is
    // allowed to self-skip.
    let src = read_kernel("tests/kernels/tri_vadd.py");
    let module = compile_triton_to_bir(&src);

    let ptx_text = match emit_ptx_text(&module) {
        Ok(text) => text,
        Err(diag)
            if diag
                .args
                .iter()
                .any(|a| a.contains("could not run mlir-opt")) =>
        {
            eprintln!(
                "skipping masked_triton_vector_add_via_mlir_nvptx_runs_on_real_gpu_hardware: \
                 mlir-opt not found on PATH"
            );
            return;
        }
        Err(diag) => panic!("emit_ptx_text failed: {diag} ({:?})", diag.args),
    };

    let Some(driver) =
        open_driver_or_skip("masked_triton_vector_add_via_mlir_nvptx_runs_on_real_gpu_hardware")
    else {
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

    let ctx = driver
        .create_context(0)
        .expect("creating a context on device 0 of a driver that reports >=1 device");

    let cuda_module = ctx
        .load_module(&ptx_text)
        .unwrap_or_else(|e| panic!("JIT-loading emit_ptx_text's PTX for tri_vadd.py: {e}"));

    let function = cuda_module
        .get_function("vector_add")
        .expect("looking up the vector_add entry point declared in the emitted PTX");

    const N: usize = 1000;
    const BLOCK: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let ab_byte_len = N * std::mem::size_of::<f32>();

    // `c` is this kernel's scratch pointer too (see `basalt-sema::triton_lower`'s own module
    // header, `Storage::Scratch`): every tile it materializes is carved out of real bytes past
    // `c`'s own `BLOCK`-float payload, so the device buffer must be far wider than `BLOCK`
    // floats — mirrors `triton_ptx_gpu_proof.rs`'s own `C_WORDS` convention exactly, so both
    // lanes give the kernel identical scratch headroom.
    const C_WORDS: usize = 32768;
    let c_byte_len = C_WORDS * std::mem::size_of::<f32>();

    let a_buf = ctx.alloc(ab_byte_len).expect("allocating device buffer a");
    let b_buf = ctx.alloc(ab_byte_len).expect("allocating device buffer b");
    let c_buf = ctx.alloc(c_byte_len).expect("allocating device buffer c");

    a_buf.copy_from_host(&a_bytes).expect("cuMemcpyHtoD a");
    b_buf.copy_from_host(&b_bytes).expect("cuMemcpyHtoD b");
    // Poison every c word so an untouched (masked-out) lane is caught if the mask fails to
    // guard it, exactly like the hand-rolled lane's own proof.
    let poison: Vec<u8> = vec![0xffu8; c_byte_len];
    c_buf
        .copy_from_host(&poison)
        .expect("cuMemcpyHtoD poison c");

    // `-convert-gpu-to-nvvm`'s default (non-bare-pointer) calling convention explodes each
    // `memref<?xT>` kernel parameter into five scalar PTX parameters — allocated pointer,
    // aligned pointer, offset, size, stride — rather than the one flat pointer
    // `basalt-ptx`/`basalt-llvm` both emit (see `emit.rs`'s module header). `a_ptr`/`b_ptr` are
    // `memref<?xf32>` (offset/size/stride in `f32` elements); `c_ptr` is `memref<?xi8>` (P11-T3c's
    // byte-addressed fallback — `basalt-sema::triton_lower` visits it at `i64`/`i1`/`f32`, which
    // no single typed `memref` can represent), so its own descriptor's offset/size/stride are in
    // bytes instead. `allocated`/`aligned` are conventionally the same device address for this
    // crate's own allocations; `offset=0`, `stride=1` (contiguous), `size` = the buffer's own
    // element count in whichever unit that parameter's memref is typed at.
    struct MemrefDesc {
        allocated: u64,
        aligned: u64,
        offset: i64,
        size: i64,
        stride: i64,
    }
    fn desc(dptr: u64, len: usize) -> MemrefDesc {
        MemrefDesc {
            allocated: dptr,
            aligned: dptr,
            offset: 0,
            size: len as i64,
            stride: 1,
        }
    }

    let mut a_desc = desc(a_buf.device_ptr(), N);
    let mut b_desc = desc(b_buf.device_ptr(), N);
    let mut c_desc = desc(c_buf.device_ptr(), c_byte_len);
    let mut n: i64 = N as i64;
    let mut block_size: i64 = BLOCK as i64;

    let mut params: [*mut c_void; 17] = [
        &mut a_desc.allocated as *mut u64 as *mut c_void,
        &mut a_desc.aligned as *mut u64 as *mut c_void,
        &mut a_desc.offset as *mut i64 as *mut c_void,
        &mut a_desc.size as *mut i64 as *mut c_void,
        &mut a_desc.stride as *mut i64 as *mut c_void,
        &mut b_desc.allocated as *mut u64 as *mut c_void,
        &mut b_desc.aligned as *mut u64 as *mut c_void,
        &mut b_desc.offset as *mut i64 as *mut c_void,
        &mut b_desc.size as *mut i64 as *mut c_void,
        &mut b_desc.stride as *mut i64 as *mut c_void,
        &mut c_desc.allocated as *mut u64 as *mut c_void,
        &mut c_desc.aligned as *mut u64 as *mut c_void,
        &mut c_desc.offset as *mut i64 as *mut c_void,
        &mut c_desc.size as *mut i64 as *mut c_void,
        &mut c_desc.stride as *mut i64 as *mut c_void,
        &mut n as *mut i64 as *mut c_void,
        &mut block_size as *mut i64 as *mut c_void,
    ];

    // One program, exactly like the hand-rolled lane's own single-block scope (see
    // `triton_lower.rs`'s module header): `tl.program_id` always reads 0.
    function
        .launch((1, 1, 1), (1, 1, 1), 0, &mut params)
        .expect("launching vector_add (triton/mlir/nvptx lane)");

    let mut c_bytes = vec![0u8; c_byte_len];
    c_buf.copy_to_host(&mut c_bytes).expect("cuMemcpyDtoH c");
    let c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect();

    for i in 0..BLOCK {
        let want = if i < N {
            a[i] + b[i]
        } else {
            f32::from_bits(0xffff_ffff)
        };
        assert_eq!(
            c[i].to_bits(),
            want.to_bits(),
            "mismatch at index {i}: got {}, want {}",
            c[i],
            want
        );
    }

    println!(
        "PASS: masked triton vector_add via basalt-mlir's NVPTX lane on real GPU hardware, \
         computed c[i]=a[i]+b[i] for i<{N} and left the poisoned c[i] untouched for {N}<=i<{BLOCK}"
    );
}
