// The Phase 5 proof: a real CUDA-C kernel, compiled through this project's own frontend/sema/
// passes/PTX backend, JIT-loaded and launched on real hardware via the real CUDA Driver API,
// with the result checked byte-for-byte against a host-computed answer. This is the GPU
// counterpart to `basalt-x86`'s `link_and_run.rs`: that test proves the oracle's object code
// links and runs through a real C toolchain; this one proves `basalt-ptx`'s emitted text loads
// and runs through the real driver. See `cuda_driver.rs` in this same directory for the
// "compile the pipeline unconditionally, gate everything hardware-touching on driver presence"
// pattern this test follows.
//
// PTX kernels carry none of the CPU backends' synthesized thread-loop machinery: `vector_add`'s
// BIR parameter list is exactly `(a, b, c, n)`, nothing appended. The number of threads that
// actually run is a launch-time choice (grid/block dimensions passed to `CudaFunction::launch`),
// not part of the kernel's own signature — real CUDA's actual launch model, unlike the
// synthesized `nthreads` trailing parameter the CPU oracle needs to fake per-thread iteration.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_frontend_c::PpOpts;
use basalt_ptx::Ptx;
use basalt_runtime::CudaDriver;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

/// Runs the real lex/preprocess/parse/check/lower/optimize/emit pipeline over
/// `tests/kernels/vector_add.cu` — the same sequence `basalt-cli`'s own `--ir`/emit paths use —
/// and returns the PTX text `basalt-ptx` emits for it.
fn compile_vector_add_to_ptx() -> String {
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

    let module = basalt_passes::optimize(&module);

    assert_eq!(Ptx.supports(&module), Support::Supported);
    let artifact = Ptx
        .emit(&module, &EmitOpts::default())
        .expect("PTX emit succeeds for vector_add");
    artifact
        .as_text()
        .expect("the PTX backend emits a text payload, never bytes")
        .to_string()
}

/// Opens the driver, or reports why it can't and lets the caller skip the rest of the test —
/// the exact pattern `cuda_driver.rs` uses for every hardware-touching test in this crate.
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
fn vector_add_compiles_loads_and_runs_on_real_hardware() {
    // The frontend-through-PTX-emission half of this test is real compiler work, not
    // hardware-gated, and runs unconditionally on every machine — only what follows (touching
    // an actual CUDA driver) is allowed to self-skip.
    let ptx_text = compile_vector_add_to_ptx();

    let Some(driver) = open_driver_or_skip("vector_add_compiles_loads_and_runs_on_real_hardware")
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
        .expect("JIT-loading basalt-ptx's emitted PTX for vector_add");

    let function = module
        .get_function("vector_add")
        .expect("looking up the vector_add entry point declared in the emitted PTX");

    const N: usize = 1024;
    // `i` and `i * 2` are both small non-negative integers, exactly representable in f32 (well
    // under the 2^24 exact-integer threshold), so the host- and device-computed sums are
    // bit-identical with no ULP tolerance needed: there is exactly one `fadd` per element on
    // either side, no reassociation, no fused multiply-add anywhere in this kernel or its
    // lowering.
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

    // `cuLaunchKernel`'s `kernelParams` (what `CudaFunction::launch` calls `params`) is an
    // array of pointers, one per kernel argument, each pointing AT that argument's own
    // storage — never an array of the argument values themselves. The three device-pointer
    // arguments need a local `u64` holding `device_ptr()`'s value; the scalar `n` needs a
    // local `i32`. Every `params` slot below points into one of these locals, which stay in
    // scope (and therefore alive) for the rest of this function, well past the `launch` call.
    let mut a_dptr: u64 = a_buf.device_ptr();
    let mut b_dptr: u64 = b_buf.device_ptr();
    let mut c_dptr: u64 = c_buf.device_ptr();
    let mut n: i32 = N as i32;

    let mut params: [*mut c_void; 4] = [
        &mut a_dptr as *mut u64 as *mut c_void,
        &mut b_dptr as *mut u64 as *mut c_void,
        &mut c_dptr as *mut u64 as *mut c_void,
        &mut n as *mut i32 as *mut c_void,
    ];

    let block = (256u32, 1u32, 1u32);
    let grid = ((N as u32).div_ceil(block.0), 1u32, 1u32);

    function
        .launch(grid, block, 0, &mut params)
        .expect("launching vector_add");

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

    println!("PASS: vector_add on real GPU hardware, {N} elements bit-exact");
}
