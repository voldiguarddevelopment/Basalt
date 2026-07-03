#![cfg(feature = "mlir")]
// The MLIR-lane counterpart to `basalt-runtime::tests::ptx_gpu_proof` — same real kernel,
// same real CUDA Driver API dispatch, same host-computed reference, but the PTX text comes
// from this crate's `emit_ptx_text` (MLIR dialects -> `mlir-opt`'s own NVPTX backend) instead
// of `basalt-ptx`'s hand-rolled encoder. See `crates/basalt-mlir/src/emit.rs`'s module header
// for the exploded memref-descriptor kernel ABI this pipeline produces — the reason this test
// cannot simply reuse `ptx_gpu_proof.rs`'s four-pointer `params` array verbatim.
//
// Whether `mlir-opt` (this crate's own toolchain dependency) or a CUDA driver is missing, this
// test self-skips rather than failing the default `--features mlir` run on a machine without
// real hardware, mirroring `ptx_gpu_proof.rs`'s own `open_driver_or_skip` convention.

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

fn compile_vector_add_to_bir() -> basalt_bir::Module {
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
    basalt_passes::optimize(&module)
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
fn vector_add_via_mlir_nvptx_runs_on_real_hardware() {
    let module = compile_vector_add_to_bir();

    let ptx_text = match emit_ptx_text(&module) {
        Ok(text) => text,
        Err(diag)
            if diag
                .args
                .iter()
                .any(|a| a.contains("could not run mlir-opt")) =>
        {
            eprintln!(
                "skipping vector_add_via_mlir_nvptx_runs_on_real_hardware: mlir-opt not found on PATH"
            );
            return;
        }
        Err(diag) => panic!("emit_ptx_text failed: {diag} ({:?})", diag.args),
    };

    let Some(driver) = open_driver_or_skip("vector_add_via_mlir_nvptx_runs_on_real_hardware")
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

    let module = ctx
        .load_module(&ptx_text)
        .unwrap_or_else(|e| panic!("JIT-loading emit_ptx_text's PTX for vector_add: {e}"));

    let function = module
        .get_function("vector_add")
        .expect("looking up the vector_add entry point declared in the emitted PTX");

    const N: usize = 1024;
    // Same generator as `ptx_gpu_proof.rs`: both operands are small non-negative integers
    // exactly representable in f32, so the comparison below needs no ULP tolerance — one
    // `fadd` per element, no reassociation, no FMA anywhere in this kernel or either of its
    // two independent lowerings (basalt-ptx's hand-rolled encoder and this crate's
    // gpu/arith/memref/cf-through-mlir-opt path).
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let byte_len = N * std::mem::size_of::<f32>();

    let a_buf = ctx.alloc(byte_len).expect("allocating device buffer a");
    let b_buf = ctx.alloc(byte_len).expect("allocating device buffer b");
    let c_buf = ctx.alloc(byte_len).expect("allocating device buffer c");

    a_buf.copy_from_host(&a_bytes).expect("cuMemcpyHtoD a");
    b_buf.copy_from_host(&b_bytes).expect("cuMemcpyHtoD b");

    // `-convert-gpu-to-nvvm`'s default (non-bare-pointer) calling convention explodes each
    // `memref<?xf32>` kernel parameter into five scalar PTX parameters — allocated pointer,
    // aligned pointer, offset, size, stride — rather than the one flat pointer
    // `basalt-ptx`/`basalt-llvm` both emit (see `emit.rs`'s module header). This kernel's
    // compiled body only ever reads the aligned-pointer field (offset/size/stride are dead:
    // `vector_add`'s access pattern has a statically-known identity layout, so the compiled
    // PTX computes each address as `aligned_ptr + index * 4` directly rather than consulting
    // the descriptor's own stride/offset fields at runtime) but the launch must still supply
    // all five words per buffer to match the kernel's real declared parameter list — a driver
    // that trusted the caller to omit "unused" parameters would have no way to tell "unused
    // by this kernel" from "unused because I got the ABI wrong". `allocated` and `aligned` are
    // conventionally the same device address for this crate's own allocations; offset=0,
    // size=N (elements, not bytes), stride=1 (elements) matches the contiguous layout every
    // buffer here actually has.
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
    let mut c_desc = desc(c_buf.device_ptr(), N);
    let mut n: i32 = N as i32;

    let mut params: [*mut c_void; 16] = [
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
        &mut n as *mut i32 as *mut c_void,
    ];

    let block = (256u32, 1u32, 1u32);
    let grid = ((N as u32).div_ceil(block.0), 1u32, 1u32);

    function
        .launch(grid, block, 0, &mut params)
        .expect("launching vector_add (mlir/nvptx lane)");

    let mut c_bytes = vec![0u8; byte_len];
    c_buf.copy_to_host(&mut c_bytes).expect("cuMemcpyDtoH c");

    let c: Vec<f32> = c_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect();

    for i in 0..N {
        assert_eq!(
            c[i],
            a[i] + b[i],
            "mismatch at index {i}: {} + {} != {}",
            a[i],
            b[i],
            c[i]
        );
    }

    println!("PASS: vector_add via basalt-mlir's NVPTX lane on real GPU hardware, {N} elements bit-exact");
}
