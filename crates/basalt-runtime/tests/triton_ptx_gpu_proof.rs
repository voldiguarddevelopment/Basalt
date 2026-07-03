// The Triton counterpart to `ptx_gpu_proof.rs`: does a real `@triton.jit` kernel, compiled
// through the real `parse -> check_triton -> lower_triton -> basalt-ptx` pipeline (the same
// stages `basalt-cli`'s own `--triton --nvidia-ptx` runs), actually load and run on real CUDA
// hardware via the real Driver API? Two proofs, mirroring
// `crates/basalt-x86/tests/triton_link_and_run.rs`'s own two CPU proofs exactly (same kernel
// sources, byte-for-byte):
//
//   - `masked_triton_vector_add_runs_on_real_gpu_hardware`: the masked `vector_add`, launched
//     with a block size (1024) wider than the real array (1000), so the mask genuinely has to
//     guard real out-of-bounds device accesses for the result to be correct.
//   - `triton_matmul_runs_on_real_gpu_hardware`: the `tl.dot` matmul kernel, at the same
//     non-square M/N/K `triton_link_and_run.rs` already proves on CPU, checked against the same
//     host triple-loop reference.
//
// See `crates/basalt-sema/src/triton_lower.rs`'s module header for why `tl.dot` never lowers to
// BIR's `Op::Mma`: it is a real scalar triple loop, so nothing here exercises tensor cores —
// this proof is that the same scalar lowering that already runs correctly on the x86-64 oracle
// also runs correctly once `basalt-ptx` turns it into PTX and the real CUDA driver executes it.
//
// Both kernels carry a real runtime loop (the masked kernel's own `tl.arange`-driven tile fill,
// the matmul's `tl.dot` triple loop) — the first genuinely loop-bearing BIR this project's own
// PTX GPU proof has ever launched; `ptx_gpu_proof.rs`'s own `vector_add.cu` has none.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_ptx::Ptx;
use basalt_runtime::CudaDriver;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

/// Runs the real `parse -> check_triton -> lower_triton` pipeline over `src`, asserting no
/// diagnostics at any stage, then emits it through the real PTX backend and returns the text.
fn compile_triton_to_ptx(src: &str) -> String {
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

    let bir = basalt_passes::optimize(&bir);

    assert_eq!(Ptx.supports(&bir), Support::Supported);
    Ptx.emit(&bir, &EmitOpts::default())
        .expect("PTX emit succeeds")
        .as_text()
        .expect("the PTX backend emits a text payload, never bytes")
        .to_string()
}

/// Reads a Triton kernel source from `tests/kernels/`, relative to the workspace root — the
/// same file `basalt-cli`'s own `--triton` flag would be pointed at.
fn read_kernel(rel_path: &str) -> String {
    let path = workspace_root().join(rel_path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Opens the driver, or reports why it can't and lets the caller skip the rest of the test —
/// the exact pattern `ptx_gpu_proof.rs`/`cuda_driver.rs` use for every hardware-touching test in
/// this crate.
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
fn masked_triton_vector_add_runs_on_real_gpu_hardware() {
    // Compiling all the way to PTX text is real compiler work, not hardware-gated, and runs
    // unconditionally on every machine — only what follows (touching an actual CUDA driver) is
    // allowed to self-skip.
    let src = read_kernel("tests/kernels/tri_vadd.py");
    let ptx_text = compile_triton_to_ptx(&src);

    let Some(driver) = open_driver_or_skip("masked_triton_vector_add_runs_on_real_gpu_hardware")
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
        .expect("JIT-loading basalt-ptx's emitted PTX for the masked vector_add kernel");
    let function = module
        .get_function("vector_add")
        .expect("looking up the vector_add entry point declared in the emitted PTX");

    const N: usize = 1000;
    const BLOCK: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let a_byte_len = N * std::mem::size_of::<f32>();
    // `c` is this kernel's scratch pointer too (see `triton_lower.rs`'s own module header,
    // `Storage::Scratch`): every tile it materializes is carved out of real bytes past `c`'s
    // own `BLOCK`-float payload, so the device buffer must be far wider than `BLOCK` floats —
    // mirrors `examples/cpu_launch_triton_vadd.c`'s own `C_WORDS` convention exactly.
    const C_WORDS: usize = 32768;
    let c_byte_len = C_WORDS * std::mem::size_of::<f32>();

    let a_buf = ctx.alloc(a_byte_len).expect("allocating device buffer a");
    let b_buf = ctx.alloc(a_byte_len).expect("allocating device buffer b");
    let c_buf = ctx.alloc(c_byte_len).expect("allocating device buffer c");

    a_buf.copy_from_host(&a_bytes).expect("cuMemcpyHtoD a");
    b_buf.copy_from_host(&b_bytes).expect("cuMemcpyHtoD b");
    // Poison every c word so an untouched (masked-out) lane is caught if the mask fails to
    // guard it, exactly like the CPU proof's own C_WORDS buffer.
    let poison: Vec<u8> = vec![0xffu8; c_byte_len];
    c_buf
        .copy_from_host(&poison)
        .expect("cuMemcpyHtoD poison c");

    let mut a_dptr: u64 = a_buf.device_ptr();
    let mut b_dptr: u64 = b_buf.device_ptr();
    let mut c_dptr: u64 = c_buf.device_ptr();
    let mut n: i64 = N as i64;
    let mut block_size: i64 = BLOCK as i64;

    let mut params: [*mut c_void; 5] = [
        &mut a_dptr as *mut u64 as *mut c_void,
        &mut b_dptr as *mut u64 as *mut c_void,
        &mut c_dptr as *mut u64 as *mut c_void,
        &mut n as *mut i64 as *mut c_void,
        &mut block_size as *mut i64 as *mut c_void,
    ];

    // One program, exactly like the CPU oracle's own single-block scope (see
    // `triton_lower.rs`'s module header): `tl.program_id` always reads 0.
    function
        .launch((1, 1, 1), (1, 1, 1), 0, &mut params)
        .expect("launching vector_add");

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
        "PASS: masked triton vector_add on real GPU hardware, computed c[i]=a[i]+b[i] for \
         i<{N} and left the poisoned c[i] untouched for {N}<=i<{BLOCK}"
    );
}

#[test]
fn triton_matmul_runs_on_real_gpu_hardware() {
    let src = read_kernel("tests/kernels/tri_matmul.py");
    let ptx_text = compile_triton_to_ptx(&src);

    let Some(driver) = open_driver_or_skip("triton_matmul_runs_on_real_gpu_hardware") else {
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
        .expect("JIT-loading basalt-ptx's emitted PTX for the matmul kernel");
    let function = module
        .get_function("matmul_kernel")
        .expect("looking up the matmul_kernel entry point declared in the emitted PTX");

    const M: usize = 4;
    const N: usize = 3;
    const K: usize = 5;
    const OUT_WORDS: usize = 65536;

    let mut a = [0.0f32; M * K];
    for i in 0..M {
        for k in 0..K {
            a[i * K + k] = ((3 * i + k) % 7 + 1) as f32;
        }
    }
    let mut b = [0.0f32; K * N];
    for k in 0..K {
        for j in 0..N {
            b[k * N + j] = ((k + 5 * j) % 7 + 1) as f32;
        }
    }
    let mut c = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            c[i * N + j] = ((i + j) % 4) as f32;
        }
    }
    let mut expected = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = c[i * N + j];
            for k in 0..K {
                sum += a[i * K + k] * b[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let c_bytes: Vec<u8> = c.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let out_byte_len = OUT_WORDS * std::mem::size_of::<f32>();

    let a_buf = ctx
        .alloc(a_bytes.len())
        .expect("allocating device buffer a");
    let b_buf = ctx
        .alloc(b_bytes.len())
        .expect("allocating device buffer b");
    let c_buf = ctx
        .alloc(c_bytes.len())
        .expect("allocating device buffer c");
    let out_buf = ctx
        .alloc(out_byte_len)
        .expect("allocating device buffer out");

    a_buf.copy_from_host(&a_bytes).expect("cuMemcpyHtoD a");
    b_buf.copy_from_host(&b_bytes).expect("cuMemcpyHtoD b");
    c_buf.copy_from_host(&c_bytes).expect("cuMemcpyHtoD c");
    let poison: Vec<u8> = vec![0xffu8; out_byte_len];
    out_buf
        .copy_from_host(&poison)
        .expect("cuMemcpyHtoD poison out");

    let mut a_dptr: u64 = a_buf.device_ptr();
    let mut b_dptr: u64 = b_buf.device_ptr();
    let mut c_dptr: u64 = c_buf.device_ptr();
    let mut out_dptr: u64 = out_buf.device_ptr();
    let mut k_arg: i64 = K as i64;

    let mut params: [*mut c_void; 5] = [
        &mut a_dptr as *mut u64 as *mut c_void,
        &mut b_dptr as *mut u64 as *mut c_void,
        &mut c_dptr as *mut u64 as *mut c_void,
        &mut out_dptr as *mut u64 as *mut c_void,
        &mut k_arg as *mut i64 as *mut c_void,
    ];

    function
        .launch((1, 1, 1), (1, 1, 1), 0, &mut params)
        .expect("launching matmul_kernel");

    let mut out_bytes = vec![0u8; out_byte_len];
    out_buf
        .copy_to_host(&mut out_bytes)
        .expect("cuMemcpyDtoH out");
    let out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect();

    for i in 0..M {
        for j in 0..N {
            let got = out[i * N + j];
            let want = expected[i * N + j];
            assert_eq!(got, want, "mismatch at ({i},{j}): got {got}, want {want}");
        }
    }

    println!("PASS: triton matmul_kernel on real GPU hardware, D = A@B + C correct ({M}x{N}x{K})");
}
