// Exercises the CUDA driver loader end to end where a real driver is present, and asserts a
// clean, specific failure where it isn't — same "skip cleanly, don't fail the suite" pattern
// `basalt-x86`'s link/run tests use for an absent `cc` (see `cc_available` there).

use basalt_runtime::{CudaDriver, CudaError};

/// A minimal, hand-written PTX kernel that does nothing. Standing in for `basalt-ptx` output
/// without pulling in that crate just for this test: `cuModuleLoadData` doesn't care where
/// its PTX text came from.
const TRIVIAL_PTX: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry trivial()
{
    ret;
}
";

/// `CudaDriver::load()` either finds a real driver or fails with one of the three sanctioned
/// `CudaError` variants — either outcome is a pass. This is the one test in this file that
/// must succeed identically on every machine, with or without an NVIDIA driver installed.
#[test]
fn load_succeeds_or_fails_with_a_specific_error() {
    match CudaDriver::load() {
        Ok(_) => {
            eprintln!("cuda driver present: load() succeeded");
        }
        Err(err) => {
            let sensible = match &err {
                CudaError::DriverNotFound(msg) => !msg.is_empty(),
                CudaError::SymbolNotFound(sym) => !sym.is_empty(),
                CudaError::DriverCallFailed { call, .. } => !call.is_empty(),
            };
            assert!(sensible, "unexpectedly empty diagnostic in {err}");
            eprintln!("cuda driver unavailable, load() reported: {err}");
        }
    }
}

/// Opens the driver, or reports why it can't and lets the caller skip the rest of the test.
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
fn full_ptx_round_trip_loads_and_launches_a_trivial_kernel() {
    let Some(driver) =
        open_driver_or_skip("full_ptx_round_trip_loads_and_launches_a_trivial_kernel")
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
        .load_module(TRIVIAL_PTX)
        .expect("JIT-loading a minimal, syntactically valid PTX module");

    let function = module
        .get_function("trivial")
        .expect("looking up the entry point declared in TRIVIAL_PTX");

    function
        .launch((1, 1, 1), (1, 1, 1), 0, &mut [])
        .expect("launching a no-op kernel with no parameters");
}

#[test]
fn device_buffer_alloc_and_copy_round_trip() {
    let Some(driver) = open_driver_or_skip("device_buffer_alloc_and_copy_round_trip") else {
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

    let src: Vec<u8> = (0..256u32).map(|i| (i % 256) as u8).collect();
    let buf = ctx
        .alloc(src.len())
        .expect("allocating a small device buffer");
    assert_ne!(
        buf.device_ptr(),
        0,
        "cuMemAlloc must not hand back a null device pointer"
    );

    buf.copy_from_host(&src).expect("cuMemcpyHtoD");

    let mut dst = vec![0u8; src.len()];
    buf.copy_to_host(&mut dst).expect("cuMemcpyDtoH");

    assert_eq!(src, dst, "round-tripped bytes must match what was written");
}
