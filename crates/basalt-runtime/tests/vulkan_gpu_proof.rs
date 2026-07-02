// The real GPU proof for the Vulkan runtime — the Vulkan counterpart to `ptx_gpu_proof.rs`'s
// CUDA one, but with a real, load-bearing complication that file doesn't have: `basalt-spirv`
// targets SPIR-V's `Kernel` execution model (see `crates/basalt-spirv/src/emit.rs`'s own header),
// while Vulkan's compute pipeline API requires `GLCompute`. This file exercises both halves of
// `basalt_runtime::vulkan`'s empirically-confirmed finding on that mismatch (see
// `src/vulkan/mod.rs`'s header for the full account):
//
//   1. `kernel_model_shader_module_loads_but_pipeline_creation_fails` compiles
//      `tests/kernels/vector_add.cu` through this project's real frontend/sema/passes/
//      `basalt-spirv` pipeline — unconditionally, not hardware-gated, same as
//      `ptx_gpu_proof.rs`'s own `compile_vector_add_to_ptx` — and proves, against a real Vulkan
//      loader/driver, that the resulting `Kernel`-model module loads via `vkCreateShaderModule`
//      but is refused by `vkCreateComputePipelines` with a specific, stable `VkResult`. This is
//      the honest, current boundary of what `basalt-spirv`'s output can do in a Vulkan compute
//      pipeline: not a bug in `basalt_runtime::vulkan`, and not something this file works around.
//
//   2. `vector_add_dispatches_through_real_vulkan_runtime_via_glcompute_stand_in` proves the
//      *runtime* itself — buffer allocation, descriptor sets, push constants, command buffer
//      recording, queue submit, fence wait, readback — genuinely works end to end against real
//      hardware, using `GLCOMPUTE_VECTOR_ADD_SPV`: a hand-written `GLCompute`-execution-model
//      SPIR-V module, semantically identical to `vector_add.cu`, compiled offline via
//      `glslangValidator` from the GLSL source quoted in that constant's own doc comment and
//      confirmed valid via `spirv-val`. It is emphatically **not** `basalt-spirv`'s output —
//      `basalt-spirv` cannot produce a `GLCompute` module today (see finding above) — this
//      constant exists solely to give the runtime itself something it can actually build a
//      pipeline from, so that everything downstream of pipeline creation can be proven for real
//      rather than left an assertion this crate cannot back up.
//
// Both tests follow `cuda_driver.rs`/`ptx_gpu_proof.rs`'s own "compile the pipeline
// unconditionally, gate everything hardware-touching on driver presence" pattern via
// `open_instance_or_skip`.

use std::path::{Path, PathBuf};

use basalt_backend::{Backend, EmitOpts, Support};
use basalt_frontend_c::PpOpts;
use basalt_runtime::{VulkanComputePipeline, VulkanDevice, VulkanError, VulkanInstance};
use basalt_spirv::Spirv;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

/// Runs the real lex/preprocess/parse/check/lower/optimize/emit pipeline over
/// `tests/kernels/vector_add.cu` through `basalt-spirv`, returning its emitted SPIR-V as `u32`
/// words (`VkShaderModuleCreateInfo`'s own native unit, and the unit `Assemble::assemble`
/// produces internally before `emit.rs` serializes it to little-endian bytes — reversed here via
/// plain `u32::from_le_bytes` rather than an alignment-sensitive pointer cast, since a `Vec<u8>`
/// carries no alignment guarantee a `*const u32` reinterpretation could rely on).
fn compile_vector_add_to_spirv_words() -> Vec<u32> {
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

    assert_eq!(Spirv.supports(&module), Support::Supported);
    let artifact = Spirv
        .emit(&module, &EmitOpts::default())
        .expect("SPIR-V emit succeeds for vector_add");
    let bytes = artifact
        .as_bytes()
        .expect("the SPIR-V backend emits a bytes payload, never text");

    assert_eq!(
        bytes.len() % 4,
        0,
        "a SPIR-V module is a whole number of 32-bit words"
    );
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// Opens the Vulkan loader, or reports why it can't and lets the caller skip the rest of the
/// test — the exact pattern `cuda_driver.rs`/`ptx_gpu_proof.rs` use for CUDA.
fn open_instance_or_skip(test_name: &str) -> Option<VulkanInstance> {
    match VulkanInstance::load() {
        Ok(instance) => Some(instance),
        Err(err) => {
            eprintln!("skipping {test_name}: Vulkan loader unavailable ({err})");
            None
        }
    }
}

/// Picks the first enumerated physical device and creates a logical device on its first
/// compute-capable queue family, or reports why it can't and lets the caller skip.
fn open_device_or_skip<'a>(
    instance: &'a VulkanInstance,
    test_name: &str,
) -> Option<VulkanDevice<'a>> {
    let devices = match instance.physical_devices() {
        Ok(devices) => devices,
        Err(err) => {
            eprintln!("skipping {test_name}: vkEnumeratePhysicalDevices failed ({err})");
            return None;
        }
    };
    let Some(physical) = devices.into_iter().next() else {
        eprintln!("skipping {test_name}: Vulkan loader present but reports zero physical devices");
        return None;
    };
    eprintln!(
        "{test_name}: using physical device {:?} ({:?})",
        physical.name, physical.device_type
    );

    let queue_family = match instance.find_compute_queue_family(&physical) {
        Ok(family) => family,
        Err(err) => {
            eprintln!("skipping {test_name}: {err}");
            return None;
        }
    };

    match VulkanDevice::create(instance, physical, queue_family) {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("skipping {test_name}: vkCreateDevice failed ({err})");
            None
        }
    }
}

#[test]
fn kernel_model_shader_module_loads_but_pipeline_creation_fails() {
    // The frontend-through-SPIR-V-emission half of this test is real compiler work, not
    // hardware-gated, and runs unconditionally on every machine — only what follows (touching a
    // real Vulkan loader) is allowed to self-skip.
    let words = compile_vector_add_to_spirv_words();

    let Some(instance) =
        open_instance_or_skip("kernel_model_shader_module_loads_but_pipeline_creation_fails")
    else {
        return;
    };
    let Some(device) = open_device_or_skip(
        &instance,
        "kernel_model_shader_module_loads_but_pipeline_creation_fails",
    ) else {
        return;
    };

    // `vector_add`'s BIR parameter list is `(a, b, c, n)`: three `Ty::Ptr(Global)` arguments and
    // one `i32` scalar — three storage-buffer bindings and a 4-byte push-constant range under
    // this crate's own resource-binding ABI (see `pipeline.rs`'s header).
    let result = VulkanComputePipeline::create(&device, &words, "vector_add", 3, 4);

    match result {
        Ok(_) => panic!(
            "vkCreateComputePipelines unexpectedly succeeded for a Kernel-execution-model \
             module; this contradicts this crate's own documented finding (see \
             src/vulkan/mod.rs) — either the driver on this machine behaves differently than the \
             one this was verified against, or something has silently changed"
        ),
        Err(VulkanError::CallFailed { call, code, .. }) => {
            assert_eq!(
                call, "vkCreateComputePipelines",
                "the Kernel-model module should fail at pipeline creation specifically, not \
                 earlier (vkCreateShaderModule is expected to succeed — see src/vulkan/mod.rs)"
            );
            eprintln!(
                "PASS: basalt-spirv's Kernel-model vector_add loaded via vkCreateShaderModule, \
                 then vkCreateComputePipelines failed as documented with VkResult {code}"
            );
        }
        Err(other) => panic!("unexpected failure mode: {other}"),
    }
}

/// A hand-written `GLCompute`-execution-model SPIR-V module, semantically identical to
/// `vector_add.cu`'s `c[i] = a[i] + b[i]` (guarded by `i < n`), compiled offline via
/// `glslangValidator -V` from:
///
/// ```glsl
/// #version 450
/// layout(local_size_x = 64) in;
/// layout(set = 0, binding = 0) readonly buffer ABuf { float a[]; };
/// layout(set = 0, binding = 1) readonly buffer BBuf { float b[]; };
/// layout(set = 0, binding = 2) writeonly buffer CBuf { float c[]; };
/// layout(push_constant) uniform PC { int n; } pc;
/// void main() {
///     uint i = gl_GlobalInvocationID.x;
///     if (i < uint(pc.n)) {
///         c[i] = a[i] + b[i];
///     }
/// }
/// ```
///
/// confirmed valid via `spirv-val` (SPIRV-Tools 1:1.4.350.1). This is **not** `basalt-spirv`
/// output — see this file's own header and `src/vulkan/mod.rs` for exactly why `basalt-spirv`
/// cannot produce a `GLCompute` module today — it exists solely so
/// `vector_add_dispatches_through_real_vulkan_runtime_via_glcompute_stand_in` below can prove
/// `basalt_runtime::vulkan`'s own dispatch plumbing against real hardware without overstating
/// what the SPIR-V backend itself can do.
#[rustfmt::skip]
const GLCOMPUTE_VECTOR_ADD_SPV: &[u32] = &[
    0x07230203, 0x00010000, 0x0008000b, 0x00000038, 0x00000000, 0x00020011,
    0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005,
    0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002,
    0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005,
    0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c,
    0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00030005, 0x00000012,
    0x00004350, 0x00040006, 0x00000012, 0x00000000, 0x0000006e, 0x00030005,
    0x00000014, 0x00006370, 0x00040005, 0x00000020, 0x66754243, 0x00000000,
    0x00040006, 0x00000020, 0x00000000, 0x00000063, 0x00030005, 0x00000022,
    0x00000000, 0x00040005, 0x00000025, 0x66754241, 0x00000000, 0x00040006,
    0x00000025, 0x00000000, 0x00000061, 0x00030005, 0x00000027, 0x00000000,
    0x00040005, 0x0000002d, 0x66754242, 0x00000000, 0x00040006, 0x0000002d,
    0x00000000, 0x00000062, 0x00030005, 0x0000002f, 0x00000000, 0x00040047,
    0x0000000b, 0x0000000b, 0x0000001c, 0x00030047, 0x00000012, 0x00000002,
    0x00050048, 0x00000012, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x0000001f, 0x00000006, 0x00000004, 0x00030047, 0x00000020, 0x00000003,
    0x00040048, 0x00000020, 0x00000000, 0x00000019, 0x00050048, 0x00000020,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000022, 0x00000019,
    0x00040047, 0x00000022, 0x00000021, 0x00000002, 0x00040047, 0x00000022,
    0x00000022, 0x00000000, 0x00040047, 0x00000024, 0x00000006, 0x00000004,
    0x00030047, 0x00000025, 0x00000003, 0x00040048, 0x00000025, 0x00000000,
    0x00000018, 0x00050048, 0x00000025, 0x00000000, 0x00000023, 0x00000000,
    0x00030047, 0x00000027, 0x00000018, 0x00040047, 0x00000027, 0x00000021,
    0x00000000, 0x00040047, 0x00000027, 0x00000022, 0x00000000, 0x00040047,
    0x0000002c, 0x00000006, 0x00000004, 0x00030047, 0x0000002d, 0x00000003,
    0x00040048, 0x0000002d, 0x00000000, 0x00000018, 0x00050048, 0x0000002d,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000002f, 0x00000018,
    0x00040047, 0x0000002f, 0x00000021, 0x00000001, 0x00040047, 0x0000002f,
    0x00000022, 0x00000000, 0x00040047, 0x00000037, 0x0000000b, 0x00000019,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015,
    0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007,
    0x00000006, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020,
    0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b,
    0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020,
    0x0000000d, 0x00000001, 0x00000006, 0x00040015, 0x00000011, 0x00000020,
    0x00000001, 0x0003001e, 0x00000012, 0x00000011, 0x00040020, 0x00000013,
    0x00000009, 0x00000012, 0x0004003b, 0x00000013, 0x00000014, 0x00000009,
    0x0004002b, 0x00000011, 0x00000015, 0x00000000, 0x00040020, 0x00000016,
    0x00000009, 0x00000011, 0x00020014, 0x0000001a, 0x00030016, 0x0000001e,
    0x00000020, 0x0003001d, 0x0000001f, 0x0000001e, 0x0003001e, 0x00000020,
    0x0000001f, 0x00040020, 0x00000021, 0x00000002, 0x00000020, 0x0004003b,
    0x00000021, 0x00000022, 0x00000002, 0x0003001d, 0x00000024, 0x0000001e,
    0x0003001e, 0x00000025, 0x00000024, 0x00040020, 0x00000026, 0x00000002,
    0x00000025, 0x0004003b, 0x00000026, 0x00000027, 0x00000002, 0x00040020,
    0x00000029, 0x00000002, 0x0000001e, 0x0003001d, 0x0000002c, 0x0000001e,
    0x0003001e, 0x0000002d, 0x0000002c, 0x00040020, 0x0000002e, 0x00000002,
    0x0000002d, 0x0004003b, 0x0000002e, 0x0000002f, 0x00000002, 0x0004002b,
    0x00000006, 0x00000035, 0x00000040, 0x0004002b, 0x00000006, 0x00000036,
    0x00000001, 0x0006002c, 0x00000009, 0x00000037, 0x00000035, 0x00000036,
    0x00000036, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007,
    0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d,
    0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f,
    0x0004003d, 0x00000006, 0x00000010, 0x00000008, 0x00050041, 0x00000016,
    0x00000017, 0x00000014, 0x00000015, 0x0004003d, 0x00000011, 0x00000018,
    0x00000017, 0x0004007c, 0x00000006, 0x00000019, 0x00000018, 0x000500b0,
    0x0000001a, 0x0000001b, 0x00000010, 0x00000019, 0x000300f7, 0x0000001d,
    0x00000000, 0x000400fa, 0x0000001b, 0x0000001c, 0x0000001d, 0x000200f8,
    0x0000001c, 0x0004003d, 0x00000006, 0x00000023, 0x00000008, 0x0004003d,
    0x00000006, 0x00000028, 0x00000008, 0x00060041, 0x00000029, 0x0000002a,
    0x00000027, 0x00000015, 0x00000028, 0x0004003d, 0x0000001e, 0x0000002b,
    0x0000002a, 0x0004003d, 0x00000006, 0x00000030, 0x00000008, 0x00060041,
    0x00000029, 0x00000031, 0x0000002f, 0x00000015, 0x00000030, 0x0004003d,
    0x0000001e, 0x00000032, 0x00000031, 0x00050081, 0x0000001e, 0x00000033,
    0x0000002b, 0x00000032, 0x00060041, 0x00000029, 0x00000034, 0x00000022,
    0x00000015, 0x00000023, 0x0003003e, 0x00000034, 0x00000033, 0x000200f9,
    0x0000001d, 0x000200f8, 0x0000001d, 0x000100fd, 0x00010038,
];

#[test]
fn vector_add_dispatches_through_real_vulkan_runtime_via_glcompute_stand_in() {
    let Some(instance) = open_instance_or_skip(
        "vector_add_dispatches_through_real_vulkan_runtime_via_glcompute_stand_in",
    ) else {
        return;
    };
    let Some(device) = open_device_or_skip(
        &instance,
        "vector_add_dispatches_through_real_vulkan_runtime_via_glcompute_stand_in",
    ) else {
        return;
    };

    let pipeline = VulkanComputePipeline::create(&device, GLCOMPUTE_VECTOR_ADD_SPV, "main", 3, 4)
        .expect("creating a compute pipeline from a real GLCompute-model module must succeed");

    const N: usize = 1024;
    // Same bit-exactness rationale as `ptx_gpu_proof.rs`: `i` and `i * 2` are both small
    // non-negative integers, exactly representable in f32, so the host- and device-computed
    // sums are bit-identical with no ULP tolerance needed — one `fadd` per element on either
    // side, no reassociation, no fused multiply-add anywhere in this shader.
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let byte_len = N * std::mem::size_of::<f32>();

    let a_buf = device
        .alloc_host_buffer(byte_len)
        .expect("allocating buffer a");
    let b_buf = device
        .alloc_host_buffer(byte_len)
        .expect("allocating buffer b");
    let c_buf = device
        .alloc_host_buffer(byte_len)
        .expect("allocating buffer c");

    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_ne_bytes()).collect();
    a_buf.copy_from_host(&a_bytes).expect("copying a to device");
    b_buf.copy_from_host(&b_bytes).expect("copying b to device");

    let n: i32 = N as i32;
    let group_count = (N as u32).div_ceil(64);

    pipeline
        .dispatch(
            &[&a_buf, &b_buf, &c_buf],
            &n.to_ne_bytes(),
            (group_count, 1, 1),
        )
        .expect("dispatching vector_add on the real Vulkan device");

    let mut c_bytes = vec![0u8; byte_len];
    c_buf
        .copy_to_host(&mut c_bytes)
        .expect("copying c from device");
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

    println!(
        "PASS: vector_add-equivalent dispatch on real Vulkan device {:?}, {N} elements bit-exact \
         (sample: c[0]={} c[1]={} c[{}]={} c[{}]={})",
        device.physical().name,
        c[0],
        c[1],
        N / 2,
        c[N / 2],
        N - 1,
        c[N - 1]
    );
}
