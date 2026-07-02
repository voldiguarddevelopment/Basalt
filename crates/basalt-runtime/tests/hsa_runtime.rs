// Exercises the HSA runtime loader end to end where a real ROCm install is present, and
// asserts a clean, specific failure where it isn't — same "skip cleanly, don't fail the suite"
// pattern `cuda_driver.rs` in this same directory uses for an absent CUDA driver.
//
// Unlike that file, there is no in-tree HSACO source to load yet: `basalt-amdgpu` (the backend
// that would emit one) is not implemented as of this test. So the executable/queue/dispatch
// path this crate builds (`HsaExecutable::load`, `HsaQueue::dispatch`) is exercised only up to
// what memory allocation and queue creation can prove without a real kernel image; the AQL
// packet layout itself has its own hardware-free structural test in `src/hsa/queue.rs`.

use basalt_runtime::{HsaDeviceType, HsaError, HsaRuntime};

/// `HsaRuntime::load()` either finds a real runtime or fails with one of the sanctioned
/// `HsaError` variants — either outcome is a pass. This is the one test in this file that must
/// succeed identically on every machine, with or without ROCm installed.
#[test]
fn load_succeeds_or_fails_with_a_specific_error() {
    match HsaRuntime::load() {
        Ok(_) => {
            eprintln!("HSA runtime present: load() succeeded");
        }
        Err(err) => {
            let sensible = match &err {
                HsaError::DriverNotFound(msg) => !msg.is_empty(),
                HsaError::SymbolNotFound(sym) => !sym.is_empty(),
                HsaError::RuntimeCallFailed { call, .. } => !call.is_empty(),
            };
            assert!(sensible, "unexpectedly empty diagnostic in {err}");
            eprintln!("HSA runtime unavailable, load() reported: {err}");
        }
    }
}

/// Opens the runtime, or reports why it can't and lets the caller skip the rest of the test.
fn open_runtime_or_skip(test_name: &str) -> Option<HsaRuntime> {
    match HsaRuntime::load() {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            eprintln!("skipping {test_name}: HSA runtime unavailable ({err})");
            None
        }
    }
}

#[test]
fn agent_enumeration_finds_a_sensible_gpu_subset() {
    let Some(runtime) = open_runtime_or_skip("agent_enumeration_finds_a_sensible_gpu_subset")
    else {
        return;
    };

    let agents = runtime
        .agents()
        .expect("hsa_iterate_agents on a successfully loaded runtime");
    let gpu_agents = runtime
        .gpu_agents()
        .expect("hsa_iterate_agents on a successfully loaded runtime");

    assert!(
        gpu_agents.len() <= agents.len(),
        "the GPU subset can never be larger than the full agent list"
    );
    for gpu in &gpu_agents {
        assert_eq!(gpu.device_type, HsaDeviceType::Gpu);
        assert!(
            !gpu.name.is_empty(),
            "hsa_agent_get_info(HSA_AGENT_INFO_NAME) should not report an empty name"
        );
    }
    eprintln!(
        "found {} agent(s), {} of them GPUs",
        agents.len(),
        gpu_agents.len()
    );
}

#[test]
fn kernarg_buffer_alloc_and_copy_round_trip() {
    let Some(runtime) = open_runtime_or_skip("kernarg_buffer_alloc_and_copy_round_trip") else {
        return;
    };

    let gpu_agents = match runtime.gpu_agents() {
        Ok(agents) => agents,
        Err(err) => {
            eprintln!("skipping: gpu_agents() failed ({err})");
            return;
        }
    };
    let Some(gpu) = gpu_agents.first() else {
        eprintln!("skipping: HSA runtime loaded but reports no GPU agents");
        return;
    };

    let region = match runtime.kernarg_region(gpu.agent) {
        Ok(region) => region,
        Err(err) => {
            eprintln!("skipping: no kernarg-capable region on this agent ({err})");
            return;
        }
    };

    let src: Vec<u8> = (0..256u32).map(|i| (i % 256) as u8).collect();
    let buf = runtime
        .alloc(region, src.len())
        .expect("allocating a small kernarg-region buffer");
    assert!(
        !buf.device_ptr().is_null(),
        "hsa_memory_allocate must not hand back a null pointer"
    );

    buf.copy_from_host(&src).expect("hsa_memory_copy to device");

    let mut dst = vec![0u8; src.len()];
    buf.copy_to_host(&mut dst).expect("hsa_memory_copy to host");

    assert_eq!(src, dst, "round-tripped bytes must match what was written");
}

#[test]
fn queue_create_and_destroy_on_a_gpu_agent() {
    let Some(runtime) = open_runtime_or_skip("queue_create_and_destroy_on_a_gpu_agent") else {
        return;
    };

    let gpu_agents = match runtime.gpu_agents() {
        Ok(agents) => agents,
        Err(err) => {
            eprintln!("skipping: gpu_agents() failed ({err})");
            return;
        }
    };
    let Some(gpu) = gpu_agents.first() else {
        eprintln!("skipping: HSA runtime loaded but reports no GPU agents");
        return;
    };

    let queue = runtime
        .create_queue(gpu.agent, 64)
        .expect("creating a small queue on a reported GPU agent");
    assert!(
        queue.last_fault().is_none(),
        "a freshly created queue must not already have a recorded fault"
    );
}
