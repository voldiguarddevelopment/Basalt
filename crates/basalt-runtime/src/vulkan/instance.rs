// The loaded Vulkan instance: a `dlopen` handle plus every core entry point this crate needs,
// resolved once via `dlsym` into a plain function-pointer table — the Vulkan counterpart to
// `../driver.rs`'s `CudaDriver`/`FnTable` and `../hsa/runtime.rs`'s `HsaRuntime`/`FnTable`.
//
// One deliberate simplification relative to "how a real Vulkan application is supposed to load":
// the spec's own documented bootstrap is a two-stage `vkGetInstanceProcAddr` chain (resolve
// global-level functions with a null instance, create an instance, then re-resolve instance- and
// device-level functions through that instance so a layer/multi-ICD-aware loader can dispatch
// correctly). This crate instead resolves every entry point — instance-level and device-level
// alike — directly via `dlsym` against `libvulkan.so.1` once, up front, exactly like
// `../driver.rs`/`../hsa/runtime.rs` already do for their own flat `FnTable`s. This is
// spec-legal (the loader exports every core trampoline as a real global symbol — confirmed via
// `nm -D libvulkan.so.1` on the machine this was developed on) and loses nothing this crate
// needs: no layers are enabled, no more than one physical device's functions are ever called
// through the same table, and correctness-first/simplicity-first is this project's own stated
// priority order.

use crate::dl::Library;
use crate::vulkan::error::VulkanError;
use crate::vulkan::ffi::{
    describe_vk_result, FnVkAllocateCommandBuffers, FnVkAllocateDescriptorSets, FnVkAllocateMemory,
    FnVkBeginCommandBuffer, FnVkBindBufferMemory, FnVkCmdBindDescriptorSets, FnVkCmdBindPipeline,
    FnVkCmdDispatch, FnVkCmdPushConstants, FnVkCreateBuffer, FnVkCreateCommandPool,
    FnVkCreateComputePipelines, FnVkCreateDescriptorPool, FnVkCreateDescriptorSetLayout,
    FnVkCreateDevice, FnVkCreateFence, FnVkCreateInstance, FnVkCreatePipelineLayout,
    FnVkCreateShaderModule, FnVkDestroyBuffer, FnVkDestroyCommandPool, FnVkDestroyDescriptorPool,
    FnVkDestroyDescriptorSetLayout, FnVkDestroyDevice, FnVkDestroyFence, FnVkDestroyInstance,
    FnVkDestroyPipeline, FnVkDestroyPipelineLayout, FnVkDestroyShaderModule, FnVkDeviceWaitIdle,
    FnVkEndCommandBuffer, FnVkEnumeratePhysicalDevices, FnVkFreeMemory,
    FnVkGetBufferMemoryRequirements, FnVkGetDeviceQueue, FnVkGetPhysicalDeviceMemoryProperties,
    FnVkGetPhysicalDeviceProperties, FnVkGetPhysicalDeviceQueueFamilyProperties, FnVkMapMemory,
    FnVkQueueSubmit, FnVkUnmapMemory, FnVkUpdateDescriptorSets, FnVkWaitForFences,
    VkApplicationInfo, VkExtent3D, VkInstance, VkInstanceCreateInfo, VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties, VkPhysicalDeviceProperties, VkPhysicalDeviceType,
    VkQueueFamilyProperties, VkResult, VK_QUEUE_COMPUTE_BIT, VK_STRUCTURE_TYPE_APPLICATION_INFO,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, VK_SUCCESS,
};

/// Every entry point this crate resolves, instance- and device-level alike (see the module-level
/// note on why both live in one flat table). Grouped in the same order the modules that use them
/// appear in this tree: instance/physical-device queries here, then device/buffer (`device.rs`),
/// then pipeline (`pipeline.rs`), then command-buffer dispatch (`dispatch.rs`).
pub(crate) struct FnTable {
    pub create_instance: FnVkCreateInstance,
    pub destroy_instance: FnVkDestroyInstance,
    pub enumerate_physical_devices: FnVkEnumeratePhysicalDevices,
    pub get_physical_device_properties: FnVkGetPhysicalDeviceProperties,
    pub get_physical_device_queue_family_properties: FnVkGetPhysicalDeviceQueueFamilyProperties,
    pub get_physical_device_memory_properties: FnVkGetPhysicalDeviceMemoryProperties,

    pub create_device: FnVkCreateDevice,
    pub destroy_device: FnVkDestroyDevice,
    pub get_device_queue: FnVkGetDeviceQueue,
    pub device_wait_idle: FnVkDeviceWaitIdle,

    pub create_buffer: FnVkCreateBuffer,
    pub destroy_buffer: FnVkDestroyBuffer,
    pub get_buffer_memory_requirements: FnVkGetBufferMemoryRequirements,
    pub allocate_memory: FnVkAllocateMemory,
    pub free_memory: FnVkFreeMemory,
    pub bind_buffer_memory: FnVkBindBufferMemory,
    pub map_memory: FnVkMapMemory,
    pub unmap_memory: FnVkUnmapMemory,

    pub create_descriptor_set_layout: FnVkCreateDescriptorSetLayout,
    pub destroy_descriptor_set_layout: FnVkDestroyDescriptorSetLayout,
    pub create_pipeline_layout: FnVkCreatePipelineLayout,
    pub destroy_pipeline_layout: FnVkDestroyPipelineLayout,
    pub create_shader_module: FnVkCreateShaderModule,
    pub destroy_shader_module: FnVkDestroyShaderModule,
    pub create_compute_pipelines: FnVkCreateComputePipelines,
    pub destroy_pipeline: FnVkDestroyPipeline,

    pub create_descriptor_pool: FnVkCreateDescriptorPool,
    pub destroy_descriptor_pool: FnVkDestroyDescriptorPool,
    pub allocate_descriptor_sets: FnVkAllocateDescriptorSets,
    pub update_descriptor_sets: FnVkUpdateDescriptorSets,
    pub create_command_pool: FnVkCreateCommandPool,
    pub destroy_command_pool: FnVkDestroyCommandPool,
    pub allocate_command_buffers: FnVkAllocateCommandBuffers,
    pub begin_command_buffer: FnVkBeginCommandBuffer,
    pub end_command_buffer: FnVkEndCommandBuffer,
    pub cmd_bind_pipeline: FnVkCmdBindPipeline,
    pub cmd_bind_descriptor_sets: FnVkCmdBindDescriptorSets,
    pub cmd_push_constants: FnVkCmdPushConstants,
    pub cmd_dispatch: FnVkCmdDispatch,
    pub create_fence: FnVkCreateFence,
    pub destroy_fence: FnVkDestroyFence,
    pub queue_submit: FnVkQueueSubmit,
    pub wait_for_fences: FnVkWaitForFences,
}

/// Resolves `name` and reinterprets the address as `F`. Same contract as `../driver.rs`'s
/// `resolve`/`../hsa/runtime.rs`'s `resolve`: `F` must be the exact `extern "C" fn` type matching
/// the documented Vulkan ABI for `name`.
fn resolve<F>(lib: &Library, name: &'static str) -> Result<F, VulkanError> {
    match lib.symbol(name) {
        // SAFETY: `ptr` is a non-null address `dlsym` resolved for `name`, one of the loader's
        // real exported symbols; per this function's documented contract, `F` is a
        // function-pointer type (pointer-sized) whose signature the caller has matched to
        // `name`'s real, published C prototype.
        Some(ptr) => Ok(unsafe { std::mem::transmute_copy(&ptr) }),
        None => Err(VulkanError::SymbolNotFound(name)),
    }
}

/// Turns a non-`VK_SUCCESS` `VkResult` into a `VulkanError::CallFailed`.
pub(crate) fn check(call: &'static str, code: VkResult) -> Result<(), VulkanError> {
    if code == VK_SUCCESS {
        return Ok(());
    }
    Err(VulkanError::CallFailed {
        call,
        code,
        message: describe_vk_result(code),
    })
}

/// A loaded Vulkan instance: an open `libvulkan.so*` handle, its resolved entry points, and a
/// live `VkInstance` handle.
pub struct VulkanInstance {
    _lib: Library,
    pub(crate) fns: FnTable,
    instance: VkInstance,
}

/// One enumerated physical device: its opaque handle plus the two `VkPhysicalDeviceProperties`
/// fields this crate actually consults (see `ffi.rs`'s note on why the rest of that struct is
/// left opaque).
#[derive(Debug, Clone)]
pub struct VulkanPhysicalDeviceInfo {
    pub(crate) handle: VkPhysicalDevice,
    pub name: String,
    pub device_type: VulkanDeviceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanDeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl From<VkPhysicalDeviceType> for VulkanDeviceType {
    fn from(raw: VkPhysicalDeviceType) -> Self {
        use crate::vulkan::ffi::{
            VK_PHYSICAL_DEVICE_TYPE_CPU, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
        };
        match raw {
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => VulkanDeviceType::IntegratedGpu,
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => VulkanDeviceType::DiscreteGpu,
            VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => VulkanDeviceType::VirtualGpu,
            VK_PHYSICAL_DEVICE_TYPE_CPU => VulkanDeviceType::Cpu,
            _ => VulkanDeviceType::Other,
        }
    }
}

impl VulkanInstance {
    /// `dlopen`s the Vulkan loader (`libvulkan.so.1`, the real installed SONAME, falling back to
    /// `libvulkan.so`, which typically only exists via a `-dev`/SDK symlink — same fallback
    /// convention `../driver.rs`/`../hsa/runtime.rs` already use), resolves every entry point
    /// this crate uses, then calls `vkCreateInstance` with no requested layers or extensions.
    pub fn load() -> Result<VulkanInstance, VulkanError> {
        let lib = Library::open_first(&["libvulkan.so.1", "libvulkan.so"])
            .map_err(VulkanError::DriverNotFound)?;

        let fns = FnTable {
            create_instance: resolve(&lib, "vkCreateInstance")?,
            destroy_instance: resolve(&lib, "vkDestroyInstance")?,
            enumerate_physical_devices: resolve(&lib, "vkEnumeratePhysicalDevices")?,
            get_physical_device_properties: resolve(&lib, "vkGetPhysicalDeviceProperties")?,
            get_physical_device_queue_family_properties: resolve(
                &lib,
                "vkGetPhysicalDeviceQueueFamilyProperties",
            )?,
            get_physical_device_memory_properties: resolve(
                &lib,
                "vkGetPhysicalDeviceMemoryProperties",
            )?,

            create_device: resolve(&lib, "vkCreateDevice")?,
            destroy_device: resolve(&lib, "vkDestroyDevice")?,
            get_device_queue: resolve(&lib, "vkGetDeviceQueue")?,
            device_wait_idle: resolve(&lib, "vkDeviceWaitIdle")?,

            create_buffer: resolve(&lib, "vkCreateBuffer")?,
            destroy_buffer: resolve(&lib, "vkDestroyBuffer")?,
            get_buffer_memory_requirements: resolve(&lib, "vkGetBufferMemoryRequirements")?,
            allocate_memory: resolve(&lib, "vkAllocateMemory")?,
            free_memory: resolve(&lib, "vkFreeMemory")?,
            bind_buffer_memory: resolve(&lib, "vkBindBufferMemory")?,
            map_memory: resolve(&lib, "vkMapMemory")?,
            unmap_memory: resolve(&lib, "vkUnmapMemory")?,

            create_descriptor_set_layout: resolve(&lib, "vkCreateDescriptorSetLayout")?,
            destroy_descriptor_set_layout: resolve(&lib, "vkDestroyDescriptorSetLayout")?,
            create_pipeline_layout: resolve(&lib, "vkCreatePipelineLayout")?,
            destroy_pipeline_layout: resolve(&lib, "vkDestroyPipelineLayout")?,
            create_shader_module: resolve(&lib, "vkCreateShaderModule")?,
            destroy_shader_module: resolve(&lib, "vkDestroyShaderModule")?,
            create_compute_pipelines: resolve(&lib, "vkCreateComputePipelines")?,
            destroy_pipeline: resolve(&lib, "vkDestroyPipeline")?,

            create_descriptor_pool: resolve(&lib, "vkCreateDescriptorPool")?,
            destroy_descriptor_pool: resolve(&lib, "vkDestroyDescriptorPool")?,
            allocate_descriptor_sets: resolve(&lib, "vkAllocateDescriptorSets")?,
            update_descriptor_sets: resolve(&lib, "vkUpdateDescriptorSets")?,
            create_command_pool: resolve(&lib, "vkCreateCommandPool")?,
            destroy_command_pool: resolve(&lib, "vkDestroyCommandPool")?,
            allocate_command_buffers: resolve(&lib, "vkAllocateCommandBuffers")?,
            begin_command_buffer: resolve(&lib, "vkBeginCommandBuffer")?,
            end_command_buffer: resolve(&lib, "vkEndCommandBuffer")?,
            cmd_bind_pipeline: resolve(&lib, "vkCmdBindPipeline")?,
            cmd_bind_descriptor_sets: resolve(&lib, "vkCmdBindDescriptorSets")?,
            cmd_push_constants: resolve(&lib, "vkCmdPushConstants")?,
            cmd_dispatch: resolve(&lib, "vkCmdDispatch")?,
            create_fence: resolve(&lib, "vkCreateFence")?,
            destroy_fence: resolve(&lib, "vkDestroyFence")?,
            queue_submit: resolve(&lib, "vkQueueSubmit")?,
            wait_for_fences: resolve(&lib, "vkWaitForFences")?,
        };

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: c"basalt".as_ptr(),
            application_version: 0,
            p_engine_name: std::ptr::null(),
            engine_version: 0,
            // Vulkan 1.1 is old enough to be present on any real driver/software-rasterizer
            // combination this crate targets, and new enough for everything this crate calls
            // (none of it is extension-gated).
            api_version: (1 << 22) | (1 << 12),
        };
        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };

        let mut instance = VkInstance::NULL;
        // SAFETY: matches `vkCreateInstance(const VkInstanceCreateInfo*, const
        // VkAllocationCallbacks*, VkInstance*)`; `create_info` (and the `app_info` it points at)
        // are valid, live local values for the duration of this call, and `instance` is a valid
        // writable out-pointer.
        let rc = unsafe { (fns.create_instance)(&create_info, std::ptr::null(), &mut instance) };
        check("vkCreateInstance", rc)?;

        Ok(VulkanInstance {
            _lib: lib,
            fns,
            instance,
        })
    }

    /// Every physical device Vulkan enumerates, each with its name and coarse device-kind
    /// classification (matching `HsaRuntime::agents`'s equivalent role for HSA).
    pub fn physical_devices(&self) -> Result<Vec<VulkanPhysicalDeviceInfo>, VulkanError> {
        let mut count: u32 = 0;
        // SAFETY: matches `vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*)`;
        // a null third argument is the documented way to query the count first.
        let rc = unsafe {
            (self.fns.enumerate_physical_devices)(self.instance, &mut count, std::ptr::null_mut())
        };
        check("vkEnumeratePhysicalDevices", rc)?;
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut handles = vec![VkPhysicalDevice::NULL; count as usize];
        // SAFETY: `handles` is a live, correctly-sized `Vec` (per the count just queried above)
        // for the duration of this call.
        let rc = unsafe {
            (self.fns.enumerate_physical_devices)(self.instance, &mut count, handles.as_mut_ptr())
        };
        check("vkEnumeratePhysicalDevices", rc)?;

        let mut out = Vec::with_capacity(handles.len());
        for handle in handles {
            let mut props = std::mem::MaybeUninit::<VkPhysicalDeviceProperties>::uninit();
            // SAFETY: matches `vkGetPhysicalDeviceProperties(VkPhysicalDevice,
            // VkPhysicalDeviceProperties*)`; `props` is large enough per `ffi.rs`'s
            // layout-verified struct (see `layout_matches_the_real_vulkan_header`), and the
            // driver fully initializes it before this call returns.
            unsafe { (self.fns.get_physical_device_properties)(handle, props.as_mut_ptr()) };
            // SAFETY: the call above fully initialized `props`.
            let props = unsafe { props.assume_init() };

            let name_bytes: Vec<u8> = props
                .device_name
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as u8)
                .collect();
            out.push(VulkanPhysicalDeviceInfo {
                handle,
                name: String::from_utf8_lossy(&name_bytes).into_owned(),
                device_type: VulkanDeviceType::from(props.device_type),
            });
        }
        Ok(out)
    }

    /// The index of the first queue family on `device` with `VK_QUEUE_COMPUTE_BIT` set —
    /// everything this crate dispatches needs only a compute-capable queue, never a
    /// graphics-only or transfer-only one.
    pub fn find_compute_queue_family(
        &self,
        device: &VulkanPhysicalDeviceInfo,
    ) -> Result<u32, VulkanError> {
        let mut count: u32 = 0;
        // SAFETY: matches `vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*,
        // VkQueueFamilyProperties*)`; a null third argument queries the count first.
        unsafe {
            (self.fns.get_physical_device_queue_family_properties)(
                device.handle,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        let mut families = Vec::with_capacity(count as usize);
        // A `VkQueueFamilyProperties` containing no meaningful data (only overwritten by the
        // call below) — needed purely to give the `Vec` `count` real elements to write into.
        families.resize_with(count as usize, || VkQueueFamilyProperties {
            queue_flags: 0,
            queue_count: 0,
            timestamp_valid_bits: 0,
            min_image_transfer_granularity: VkExtent3D {
                width: 0,
                height: 0,
                depth: 0,
            },
        });
        // SAFETY: `families` is a live, correctly-sized `Vec` (per the count just queried above)
        // for the duration of this call.
        unsafe {
            (self.fns.get_physical_device_queue_family_properties)(
                device.handle,
                &mut count,
                families.as_mut_ptr(),
            )
        };

        families
            .iter()
            .position(|f| f.queue_flags & VK_QUEUE_COMPUTE_BIT != 0)
            .map(|i| i as u32)
            .ok_or(VulkanError::NoComputeQueueFamily)
    }

    /// The full memory-type/heap table for `device`, needed to pick a memory type satisfying a
    /// buffer's own `memoryTypeBits` mask (see `device.rs`'s `alloc_host_buffer`).
    pub(crate) fn memory_properties(
        &self,
        device: &VulkanPhysicalDeviceInfo,
    ) -> VkPhysicalDeviceMemoryProperties {
        let mut props = std::mem::MaybeUninit::<VkPhysicalDeviceMemoryProperties>::uninit();
        // SAFETY: matches `vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,
        // VkPhysicalDeviceMemoryProperties*)`; `props` is exactly the struct's real size (see
        // `ffi.rs`'s layout test) and the driver fully initializes it before returning.
        unsafe {
            (self.fns.get_physical_device_memory_properties)(device.handle, props.as_mut_ptr())
        };
        // SAFETY: the call above fully initialized `props`.
        unsafe { props.assume_init() }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        // SAFETY: `self.instance` was produced by a successful `vkCreateInstance` and destroyed
        // at most once (`Drop` runs exactly once). Every `VulkanDevice` derived from this
        // instance borrows `&'a VulkanInstance`, so none can outlive this `drop` per the borrow
        // checker — by the time this runs, nothing else in the process still expects this
        // instance's devices to be valid.
        unsafe {
            (self.fns.destroy_instance)(self.instance, std::ptr::null());
        }
    }
}
