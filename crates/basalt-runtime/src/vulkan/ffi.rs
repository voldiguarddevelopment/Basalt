// Raw Vulkan 1.x core shapes: opaque handle types, the handful of structs and enum values this
// crate's loader touches, and function-pointer signatures matching the loader's stable,
// published C ABI. As with `../ffi.rs`'s CUDA transcription and `../hsa/ffi.rs`'s HSA
// transcription, every shape here was transcribed by hand rather than generated — but unlike
// those two, every field order/type/size below was cross-checked against a real
// `/usr/include/vulkan/vulkan_core.h` (Vulkan-Headers 1.4.350, installed on the machine this
// backend was verified against) and, for the two structs big enough to get wrong by hand
// (`VkPhysicalDeviceProperties`/`VkPhysicalDeviceMemoryProperties`), against `sizeof`/`_Alignof`/
// `offsetof` read back from a real compiled probe against that same header — see the
// `layout_matches_the_real_vulkan_header` test at the bottom of this file for the exact numbers.
//
// Scope: only the entry points and structs this crate's loader actually calls are transcribed.
// Anything this crate never constructs (`VkPhysicalDeviceFeatures`, `VkAllocationCallbacks`,
// `VkSpecializationInfo`, `VkDescriptorImageInfo`, `VkBufferView`) is passed as a null pointer at
// every call site and typed here as an untyped `*const c_void`/`*mut c_void`, exactly like this
// crate's `hsa_executable_create_alt` call already does for HSA's "give me your defaults" null
// options pointer.
//
// Handle representation: on every LP64 target (all Linux targets Basalt builds for), the
// `VK_DEFINE_HANDLE`/`VK_DEFINE_NON_DISPATCHABLE_HANDLE` macros both expand to
// `typedef struct object##_T *object;` — i.e. dispatchable handles (`VkInstance`, `VkDevice`,
// ...) and non-dispatchable handles (`VkBuffer`, `VkPipeline`, ...) are both just opaque
// pointers at the ABI level, the same "runtime-owned cookie Basalt never dereferences" shape
// `../ffi.rs`'s `CUcontext`/`CUmodule` newtypes already use for CUDA's handles.

use std::ffi::{c_char, c_void};

/// `VkResult`: a C enum, 4 bytes at the ABI level, negative on error. Only the values this
/// crate's calls can plausibly return are named; `describe_vk_result` below falls back to the
/// bare numeric code for anything else.
pub type VkResult = i32;
pub const VK_SUCCESS: VkResult = 0;

/// Renders a `VkResult` for `VulkanError::CallFailed.message`. Core Vulkan has no
/// `vkGetErrorString`-equivalent entry point (unlike `cuGetErrorString`/`hsa_status_string`), so
/// this crate carries its own fixed table of the codes its own calls are documented to return,
/// rather than guessing at one the driver doesn't actually provide.
pub fn describe_vk_result(code: VkResult) -> String {
    let name = match code {
        0 => "VK_SUCCESS",
        1 => "VK_NOT_READY",
        2 => "VK_TIMEOUT",
        -1 => "VK_ERROR_OUT_OF_HOST_MEMORY",
        -2 => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        -3 => "VK_ERROR_INITIALIZATION_FAILED",
        -4 => "VK_ERROR_DEVICE_LOST",
        -5 => "VK_ERROR_MEMORY_MAP_FAILED",
        -6 => "VK_ERROR_LAYER_NOT_PRESENT",
        -7 => "VK_ERROR_EXTENSION_NOT_PRESENT",
        -8 => "VK_ERROR_FEATURE_NOT_PRESENT",
        -9 => "VK_ERROR_INCOMPATIBLE_DRIVER",
        -10 => "VK_ERROR_TOO_MANY_OBJECTS",
        -11 => "VK_ERROR_FORMAT_NOT_SUPPORTED",
        -12 => "VK_ERROR_FRAGMENTED_POOL",
        -13 => "VK_ERROR_UNKNOWN",
        _ => return format!("VkResult({code})"),
    };
    name.to_string()
}

/// `VkStructureType`: every `sType` value this crate ever writes.
pub type VkStructureType = i32;
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: VkStructureType = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: VkStructureType = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: VkStructureType = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: VkStructureType = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: VkStructureType = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: VkStructureType = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: VkStructureType = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: VkStructureType = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: VkStructureType = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: VkStructureType = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: VkStructureType = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: VkStructureType = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: VkStructureType = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: VkStructureType = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: VkStructureType = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: VkStructureType = 42;

/// `VkPhysicalDeviceType`.
pub type VkPhysicalDeviceType = i32;
// `VK_PHYSICAL_DEVICE_TYPE_OTHER = 0` has no named constant here: it is exactly the fallback
// `VulkanDeviceType::Other` arm below already reduces every unrecognized value to, so there is
// nothing distinct for a `0`-valued constant to do.
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: VkPhysicalDeviceType = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: VkPhysicalDeviceType = 2;
pub const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: VkPhysicalDeviceType = 3;
pub const VK_PHYSICAL_DEVICE_TYPE_CPU: VkPhysicalDeviceType = 4;

/// `VkQueueFlagBits` (a bitmask, `VkQueueFlags`/`VkFlags` = `uint32_t`).
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;

/// `VkMemoryPropertyFlagBits`.
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;

/// `VkBufferUsageFlagBits`.
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;

/// `VkSharingMode`.
pub type VkSharingMode = i32;
pub const VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0;

/// `VkDescriptorType` (only the one kind of descriptor this backend's pointer-per-SSBO-binding
/// ABI uses — see `pipeline.rs`).
pub type VkDescriptorType = i32;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: VkDescriptorType = 7;

/// `VkShaderStageFlagBits`.
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;

/// `VkPipelineBindPoint`.
pub type VkPipelineBindPoint = i32;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: VkPipelineBindPoint = 1;

/// `VkCommandBufferLevel`.
pub type VkCommandBufferLevel = i32;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: VkCommandBufferLevel = 0;

/// `VkCommandBufferUsageFlagBits`.
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;

pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;
pub const VK_WHOLE_SIZE: u64 = !0u64;

/// `VkDeviceSize` is `uint64_t`; `VkBool32` is `uint32_t` (never `bool`-sized).
pub type VkDeviceSize = u64;
pub type VkBool32 = u32;

macro_rules! vk_handle {
    ($name:ident) => {
        /// Opaque handle: on every LP64 target this is a real pointer per the Vulkan spec's own
        /// `VK_DEFINE_HANDLE`/`VK_DEFINE_NON_DISPATCHABLE_HANDLE` macros (see this file's header
        /// note), but Basalt only ever passes it back into the loader, never dereferences it —
        /// the same stance `../ffi.rs`'s `CUcontext`/`CUmodule` newtypes take for CUDA's handles.
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct $name(pub *mut c_void);

        impl $name {
            pub const NULL: $name = $name(std::ptr::null_mut());
        }
    };
}

vk_handle!(VkInstance);
vk_handle!(VkPhysicalDevice);
vk_handle!(VkDevice);
vk_handle!(VkQueue);
vk_handle!(VkCommandBuffer);
vk_handle!(VkDeviceMemory);
vk_handle!(VkBuffer);
vk_handle!(VkShaderModule);
vk_handle!(VkPipelineLayout);
vk_handle!(VkPipeline);
vk_handle!(VkPipelineCache);
vk_handle!(VkDescriptorSetLayout);
vk_handle!(VkDescriptorPool);
vk_handle!(VkDescriptorSet);
vk_handle!(VkCommandPool);
vk_handle!(VkFence);

// ---- structs ----------------------------------------------------------------------------------

#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    /// `const VkPhysicalDeviceFeatures*` — always null (this backend enables no optional
    /// feature), so the pointee type is never observed and is left untyped.
    pub p_enabled_features: *const c_void,
}

/// `VkExtent3D`, embedded in `VkQueueFamilyProperties` below. Never read by this crate (only
/// `queue_flags` is consulted), but must be present for the struct's size/layout to match.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

/// `VkPhysicalDeviceLimits`: an opaque, correctly sized-and-aligned placeholder. This crate never
/// reads a single field of it — only `deviceType` and `deviceName`, both of which sit at lower
/// offsets in the enclosing `VkPhysicalDeviceProperties` (see below), are consulted — so getting
/// this struct's ~90 fields transcribed exactly right would be pure risk for zero benefit. What
/// *is* load-bearing is its size and alignment, since `vkGetPhysicalDeviceProperties` writes the
/// real, full-size struct through this memory: confirmed empirically as 504 bytes / 8-byte
/// alignment against the real header (see `layout_matches_the_real_vulkan_header` below), which
/// `[u64; 63]` reproduces exactly.
#[repr(C)]
pub struct VkPhysicalDeviceLimits {
    _opaque: [u64; 63],
}

/// `VkPhysicalDeviceSparseProperties`: same "opaque, size-and-align-only" treatment as
/// `VkPhysicalDeviceLimits` above, for the same reason — this crate never reads a sparse-residency
/// flag. Confirmed as 20 bytes / 4-byte alignment against the real header.
#[repr(C)]
pub struct VkPhysicalDeviceSparseProperties {
    _opaque: [u32; 5],
}

#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: VkPhysicalDeviceType,
    pub device_name: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    pub limits: VkPhysicalDeviceLimits,
    pub sparse_properties: VkPhysicalDeviceSparseProperties,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: u32,
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: VkSharingMode,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: VkDescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    /// `const VkSampler*` — always null (no samplers are bound by this backend).
    pub p_immutable_samplers: *const c_void,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkPushConstantRange {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    /// `const VkSpecializationInfo*` — always null (no specialization constants are used).
    pub p_specialization_info: *const c_void,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub ty: VkDescriptorType,
    pub descriptor_count: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: VkDescriptorType,
    /// `const VkDescriptorImageInfo*` — always null (this backend only ever writes storage-buffer
    /// descriptors).
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    /// `const VkBufferView*` — always null (no texel buffer views are used).
    pub p_texel_buffer_view: *const c_void,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: VkCommandBufferLevel,
    pub command_buffer_count: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    /// `const VkCommandBufferInheritanceInfo*` — always null (only primary command buffers with
    /// no secondary-buffer inheritance are used).
    pub p_inheritance_info: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const c_void,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const c_void,
}

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: u32,
}

// ---- function-pointer signatures ---------------------------------------------------------------

pub type FnVkCreateInstance = unsafe extern "C" fn(
    p_create_info: *const VkInstanceCreateInfo,
    p_allocator: *const c_void,
    p_instance: *mut VkInstance,
) -> VkResult;
pub type FnVkDestroyInstance =
    unsafe extern "C" fn(instance: VkInstance, p_allocator: *const c_void);
pub type FnVkEnumeratePhysicalDevices = unsafe extern "C" fn(
    instance: VkInstance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut VkPhysicalDevice,
) -> VkResult;
pub type FnVkGetPhysicalDeviceProperties = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    p_properties: *mut VkPhysicalDeviceProperties,
);
pub type FnVkGetPhysicalDeviceQueueFamilyProperties = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut VkQueueFamilyProperties,
);
pub type FnVkGetPhysicalDeviceMemoryProperties = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    p_memory_properties: *mut VkPhysicalDeviceMemoryProperties,
);
pub type FnVkCreateDevice = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    p_create_info: *const VkDeviceCreateInfo,
    p_allocator: *const c_void,
    p_device: *mut VkDevice,
) -> VkResult;
pub type FnVkDestroyDevice = unsafe extern "C" fn(device: VkDevice, p_allocator: *const c_void);
pub type FnVkGetDeviceQueue = unsafe extern "C" fn(
    device: VkDevice,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut VkQueue,
);
pub type FnVkDeviceWaitIdle = unsafe extern "C" fn(device: VkDevice) -> VkResult;

pub type FnVkCreateBuffer = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkBufferCreateInfo,
    p_allocator: *const c_void,
    p_buffer: *mut VkBuffer,
) -> VkResult;
pub type FnVkDestroyBuffer =
    unsafe extern "C" fn(device: VkDevice, buffer: VkBuffer, p_allocator: *const c_void);
pub type FnVkGetBufferMemoryRequirements = unsafe extern "C" fn(
    device: VkDevice,
    buffer: VkBuffer,
    p_memory_requirements: *mut VkMemoryRequirements,
);
pub type FnVkAllocateMemory = unsafe extern "C" fn(
    device: VkDevice,
    p_allocate_info: *const VkMemoryAllocateInfo,
    p_allocator: *const c_void,
    p_memory: *mut VkDeviceMemory,
) -> VkResult;
pub type FnVkFreeMemory =
    unsafe extern "C" fn(device: VkDevice, memory: VkDeviceMemory, p_allocator: *const c_void);
pub type FnVkBindBufferMemory = unsafe extern "C" fn(
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    memory_offset: VkDeviceSize,
) -> VkResult;
pub type FnVkMapMemory = unsafe extern "C" fn(
    device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    size: VkDeviceSize,
    flags: u32,
    pp_data: *mut *mut c_void,
) -> VkResult;
pub type FnVkUnmapMemory = unsafe extern "C" fn(device: VkDevice, memory: VkDeviceMemory);

pub type FnVkCreateDescriptorSetLayout = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkDescriptorSetLayoutCreateInfo,
    p_allocator: *const c_void,
    p_set_layout: *mut VkDescriptorSetLayout,
) -> VkResult;
pub type FnVkDestroyDescriptorSetLayout = unsafe extern "C" fn(
    device: VkDevice,
    descriptor_set_layout: VkDescriptorSetLayout,
    p_allocator: *const c_void,
);
pub type FnVkCreatePipelineLayout = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkPipelineLayoutCreateInfo,
    p_allocator: *const c_void,
    p_pipeline_layout: *mut VkPipelineLayout,
) -> VkResult;
pub type FnVkDestroyPipelineLayout = unsafe extern "C" fn(
    device: VkDevice,
    pipeline_layout: VkPipelineLayout,
    p_allocator: *const c_void,
);
pub type FnVkCreateShaderModule = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkShaderModuleCreateInfo,
    p_allocator: *const c_void,
    p_shader_module: *mut VkShaderModule,
) -> VkResult;
pub type FnVkDestroyShaderModule = unsafe extern "C" fn(
    device: VkDevice,
    shader_module: VkShaderModule,
    p_allocator: *const c_void,
);
pub type FnVkCreateComputePipelines = unsafe extern "C" fn(
    device: VkDevice,
    pipeline_cache: VkPipelineCache,
    create_info_count: u32,
    p_create_infos: *const VkComputePipelineCreateInfo,
    p_allocator: *const c_void,
    p_pipelines: *mut VkPipeline,
) -> VkResult;
pub type FnVkDestroyPipeline =
    unsafe extern "C" fn(device: VkDevice, pipeline: VkPipeline, p_allocator: *const c_void);

pub type FnVkCreateDescriptorPool = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkDescriptorPoolCreateInfo,
    p_allocator: *const c_void,
    p_descriptor_pool: *mut VkDescriptorPool,
) -> VkResult;
pub type FnVkDestroyDescriptorPool = unsafe extern "C" fn(
    device: VkDevice,
    descriptor_pool: VkDescriptorPool,
    p_allocator: *const c_void,
);
pub type FnVkAllocateDescriptorSets = unsafe extern "C" fn(
    device: VkDevice,
    p_allocate_info: *const VkDescriptorSetAllocateInfo,
    p_descriptor_sets: *mut VkDescriptorSet,
) -> VkResult;
pub type FnVkUpdateDescriptorSets = unsafe extern "C" fn(
    device: VkDevice,
    descriptor_write_count: u32,
    p_descriptor_writes: *const VkWriteDescriptorSet,
    descriptor_copy_count: u32,
    p_descriptor_copies: *const c_void,
);

pub type FnVkCreateCommandPool = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkCommandPoolCreateInfo,
    p_allocator: *const c_void,
    p_command_pool: *mut VkCommandPool,
) -> VkResult;
pub type FnVkDestroyCommandPool =
    unsafe extern "C" fn(device: VkDevice, command_pool: VkCommandPool, p_allocator: *const c_void);
pub type FnVkAllocateCommandBuffers = unsafe extern "C" fn(
    device: VkDevice,
    p_allocate_info: *const VkCommandBufferAllocateInfo,
    p_command_buffers: *mut VkCommandBuffer,
) -> VkResult;
pub type FnVkBeginCommandBuffer = unsafe extern "C" fn(
    command_buffer: VkCommandBuffer,
    p_begin_info: *const VkCommandBufferBeginInfo,
) -> VkResult;
pub type FnVkEndCommandBuffer = unsafe extern "C" fn(command_buffer: VkCommandBuffer) -> VkResult;

pub type FnVkCmdBindPipeline = unsafe extern "C" fn(
    command_buffer: VkCommandBuffer,
    pipeline_bind_point: VkPipelineBindPoint,
    pipeline: VkPipeline,
);
pub type FnVkCmdBindDescriptorSets = unsafe extern "C" fn(
    command_buffer: VkCommandBuffer,
    pipeline_bind_point: VkPipelineBindPoint,
    layout: VkPipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const VkDescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
);
pub type FnVkCmdPushConstants = unsafe extern "C" fn(
    command_buffer: VkCommandBuffer,
    layout: VkPipelineLayout,
    stage_flags: u32,
    offset: u32,
    size: u32,
    p_values: *const c_void,
);
pub type FnVkCmdDispatch = unsafe extern "C" fn(
    command_buffer: VkCommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
);

pub type FnVkCreateFence = unsafe extern "C" fn(
    device: VkDevice,
    p_create_info: *const VkFenceCreateInfo,
    p_allocator: *const c_void,
    p_fence: *mut VkFence,
) -> VkResult;
pub type FnVkDestroyFence =
    unsafe extern "C" fn(device: VkDevice, fence: VkFence, p_allocator: *const c_void);
pub type FnVkQueueSubmit = unsafe extern "C" fn(
    queue: VkQueue,
    submit_count: u32,
    p_submits: *const VkSubmitInfo,
    fence: VkFence,
) -> VkResult;
pub type FnVkWaitForFences = unsafe extern "C" fn(
    device: VkDevice,
    fence_count: u32,
    p_fences: *const VkFence,
    wait_all: VkBool32,
    timeout: u64,
) -> VkResult;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// Cross-checks this file's hand-transcribed struct sizes/alignments against the real
    /// numbers read back from `/usr/include/vulkan/vulkan_core.h` via a compiled C probe on the
    /// machine this backend was verified against (Vulkan-Headers 1.4.350):
    ///   sizeof(VkPhysicalDeviceLimits)              = 504, align 8
    ///   sizeof(VkPhysicalDeviceSparseProperties)    =  20, align 4
    ///   sizeof(VkPhysicalDeviceProperties)          = 824, align 8
    ///   offsetof(..., deviceType)                   =  16
    ///   offsetof(..., deviceName)                   =  20
    ///   offsetof(..., limits)                       = 296
    ///   offsetof(..., sparseProperties)              = 800
    ///   sizeof(VkPhysicalDeviceMemoryProperties)    = 520
    /// This test has no Vulkan library dependency of its own — it only checks that Rust's
    /// `repr(C)` layout of the structs above reproduces those numbers, so it runs on every
    /// machine regardless of whether a Vulkan loader is installed.
    #[test]
    fn layout_matches_the_real_vulkan_header() {
        assert_eq!(size_of::<VkPhysicalDeviceLimits>(), 504);
        assert_eq!(align_of::<VkPhysicalDeviceLimits>(), 8);
        assert_eq!(size_of::<VkPhysicalDeviceSparseProperties>(), 20);
        assert_eq!(align_of::<VkPhysicalDeviceSparseProperties>(), 4);
        assert_eq!(size_of::<VkPhysicalDeviceProperties>(), 824);
        assert_eq!(align_of::<VkPhysicalDeviceProperties>(), 8);
        assert_eq!(size_of::<VkPhysicalDeviceMemoryProperties>(), 520);

        let base = std::mem::MaybeUninit::<VkPhysicalDeviceProperties>::uninit();
        let base_ptr = base.as_ptr();
        // SAFETY: no field is read, only address arithmetic on an uninitialized-but-allocated
        // local, matching `hsa/queue.rs`'s identical `offset_of` technique for its AQL packet
        // layout test.
        let offset_of = |field_ptr: *const u8| -> usize {
            unsafe { field_ptr.offset_from(base_ptr.cast::<u8>()) as usize }
        };
        unsafe {
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).device_type).cast()),
                16
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).device_name).cast()),
                20
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).limits).cast()),
                296
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).sparse_properties).cast()),
                800
            );
        }
    }

    #[test]
    fn describe_vk_result_names_the_documented_codes_and_falls_back_for_the_rest() {
        assert_eq!(describe_vk_result(VK_SUCCESS), "VK_SUCCESS");
        assert_eq!(describe_vk_result(-13), "VK_ERROR_UNKNOWN");
        assert_eq!(describe_vk_result(-1000012000), "VkResult(-1000012000)");
    }
}
