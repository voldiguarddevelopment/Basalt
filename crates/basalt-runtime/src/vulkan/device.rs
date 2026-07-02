// A created Vulkan logical device, its compute queue, and host-visible buffer allocation.
// `VulkanDevice<'a>` borrows `&'a VulkanInstance` the same way `CudaContext<'a>` borrows
// `&'a CudaDriver` (see `../context.rs`'s cross-resource drop-ordering note) — but everything
// built *on* a `VulkanDevice` (`VulkanBuffer` here, `VulkanComputePipeline` in `pipeline.rs`)
// deliberately deviates from that note's own pattern: instead of being a sibling borrowing the
// same outer lifetime (what `DeviceBuffer<'a>`/`HsaBuffer<'a>` both do relative to their
// driver/runtime), each one borrows `&'d VulkanDevice<'a>` directly, so the borrow checker
// itself forbids dropping a device while a buffer or pipeline built on it is still alive.
//
// That stronger tie is a real soundness requirement here, not just an available extra: CUDA's
// and HSA's own docs (see `../context.rs`'s note, `../hsa/executable.rs`'s identical one) are
// explicit that a stale handle used after its context/executable is destroyed comes back as a
// driver-reported error code, not memory corruption, because both runtimes validate opaque
// handles against their own live-object tables on every call. The Vulkan spec makes no such
// promise: using a `VkBuffer`/`VkPipeline` after `vkDestroyDevice` has run is documented invalid
// usage with no defined, safe failure mode (validation layers can catch it; a production driver
// is not required to). Since this crate has no validation layer wired in, the borrow checker is
// the only thing standing between a caller and that undefined behavior, so `VulkanBuffer`/
// `VulkanComputePipeline` are given the two-lifetime-parameter shape needed to make that real.

use crate::vulkan::buffer::VulkanBuffer;
use crate::vulkan::error::VulkanError;
use crate::vulkan::ffi::{
    VkBuffer, VkBufferCreateInfo, VkDevice, VkDeviceCreateInfo, VkDeviceMemory,
    VkDeviceQueueCreateInfo, VkMemoryAllocateInfo, VkQueue, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_SHARING_MODE_EXCLUSIVE, VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
};
use crate::vulkan::instance::{check, VulkanInstance, VulkanPhysicalDeviceInfo};

/// A logical device plus the one compute queue this crate ever opens on it. Every buffer
/// allocated through `alloc_host_buffer`, and every `VulkanComputePipeline`/dispatch built on top
/// of it (see `pipeline.rs`/`dispatch.rs`), borrows `&'a VulkanDevice` for the same reason
/// `CudaModule<'a>` borrows `&'a CudaDriver` — see `../context.rs`'s module-level note, which
/// applies identically here.
pub struct VulkanDevice<'a> {
    instance: &'a VulkanInstance,
    device: VkDevice,
    queue: VkQueue,
    queue_family_index: u32,
    physical: VulkanPhysicalDeviceInfo,
}

impl<'a> VulkanDevice<'a> {
    /// Creates a logical device on `physical` with a single queue from `queue_family_index`, and
    /// fetches that queue immediately (this crate never touches more than one queue per device).
    pub fn create(
        instance: &'a VulkanInstance,
        physical: VulkanPhysicalDeviceInfo,
        queue_family_index: u32,
    ) -> Result<VulkanDevice<'a>, VulkanError> {
        let priority: f32 = 1.0;
        let queue_create_info = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: &priority,
        };
        let create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
        };

        let mut device = VkDevice::NULL;
        // SAFETY: matches `vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const
        // VkAllocationCallbacks*, VkDevice*)`; `create_info` and the `queue_create_info`/
        // `priority` it points into are valid, live locals for the duration of this call, and
        // `device` is a valid writable out-pointer.
        let rc = unsafe {
            (instance.fns.create_device)(
                physical.handle,
                &create_info,
                std::ptr::null(),
                &mut device,
            )
        };
        check("vkCreateDevice", rc)?;

        let mut queue = VkQueue::NULL;
        // SAFETY: matches `vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue*)`; `device`
        // came from the successful call above, `queue_family_index` is the same one that device
        // was created with a queue from, and `queue` is a valid writable out-pointer.
        unsafe { (instance.fns.get_device_queue)(device, queue_family_index, 0, &mut queue) };

        Ok(VulkanDevice {
            instance,
            device,
            queue,
            queue_family_index,
            physical,
        })
    }

    pub fn physical(&self) -> &VulkanPhysicalDeviceInfo {
        &self.physical
    }

    pub(crate) fn handle(&self) -> VkDevice {
        self.device
    }

    pub(crate) fn queue(&self) -> VkQueue {
        self.queue
    }

    pub(crate) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub(crate) fn instance(&self) -> &'a VulkanInstance {
        self.instance
    }

    /// Allocates a `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` buffer of `bytes`, backed by
    /// host-visible, host-coherent memory, and maps it for the buffer's whole lifetime.
    ///
    /// Host-coherent memory (rather than device-local memory needing an explicit
    /// staging-buffer-and-copy-command round trip) is a deliberate simplification: it makes
    /// `VulkanBuffer::copy_from_host`/`copy_to_host` a plain `memcpy` with no command buffer of
    /// its own, at the cost of being unusable on a discrete GPU's own VRAM path — correct and
    /// simple first, matching this crate's other loaders (`DeviceBuffer`/`HsaBuffer` also do a
    /// direct copy with no intermediate staging step). `llvmpipe` and most integrated GPUs
    /// expose a host-visible+host-coherent memory type covering `VK_BUFFER_USAGE_STORAGE_BUFFER`
    /// usage directly, which is what every allocation through this crate's own tests exercises.
    pub fn alloc_host_buffer(&self, bytes: usize) -> Result<VulkanBuffer<'_, 'a>, VulkanError> {
        let create_info = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: bytes as u64,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };

        let mut buffer = VkBuffer::NULL;
        // SAFETY: matches `vkCreateBuffer(VkDevice, const VkBufferCreateInfo*, const
        // VkAllocationCallbacks*, VkBuffer*)`; `create_info` is a valid, live local for the
        // duration of this call.
        let rc = unsafe {
            (self.instance.fns.create_buffer)(
                self.device,
                &create_info,
                std::ptr::null(),
                &mut buffer,
            )
        };
        check("vkCreateBuffer", rc)?;

        let mut requirements = std::mem::MaybeUninit::uninit();
        // SAFETY: matches `vkGetBufferMemoryRequirements(VkDevice, VkBuffer,
        // VkMemoryRequirements*)`; `buffer` came from the successful call above.
        unsafe {
            (self.instance.fns.get_buffer_memory_requirements)(
                self.device,
                buffer,
                requirements.as_mut_ptr(),
            )
        };
        // SAFETY: the call above fully initialized `requirements`.
        let requirements = unsafe { requirements.assume_init() };

        let want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let mem_props = self.instance.memory_properties(&self.physical);
        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let bit_set = requirements.memory_type_bits & (1u32 << i) != 0;
                let flags_ok = mem_props.memory_types[i as usize].property_flags & want == want;
                bit_set && flags_ok
            })
            .ok_or(VulkanError::NoSuitableMemoryType)?;

        let allocate_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: requirements.size,
            memory_type_index,
        };
        let mut memory = VkDeviceMemory::NULL;
        // SAFETY: matches `vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const
        // VkAllocationCallbacks*, VkDeviceMemory*)`; `allocate_info` is a valid, live local for
        // the duration of this call.
        let rc = unsafe {
            (self.instance.fns.allocate_memory)(
                self.device,
                &allocate_info,
                std::ptr::null(),
                &mut memory,
            )
        };
        if let Err(err) = check("vkAllocateMemory", rc) {
            // SAFETY: `buffer` was created above and not yet bound to any memory; destroying it
            // here (the only place this function fails after creating it) leaks nothing.
            unsafe { (self.instance.fns.destroy_buffer)(self.device, buffer, std::ptr::null()) };
            return Err(err);
        }

        // SAFETY: matches `vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory,
        // VkDeviceSize)`; `buffer` and `memory` both came from the successful calls above, and
        // `memory`'s allocation size (`requirements.size`) is always >= `buffer`'s own
        // requirements, per `vkGetBufferMemoryRequirements`'s documented contract.
        let rc = unsafe { (self.instance.fns.bind_buffer_memory)(self.device, buffer, memory, 0) };
        check("vkBindBufferMemory", rc)?;

        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: matches `vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize,
        // VkMemoryMapFlags, void**)`; `memory` is host-visible per the property flags selected
        // above, `0`/`requirements.size` maps the whole allocation, and `mapped` is a valid
        // writable out-pointer.
        let rc = unsafe {
            (self.instance.fns.map_memory)(
                self.device,
                memory,
                0,
                requirements.size,
                0,
                &mut mapped,
            )
        };
        check("vkMapMemory", rc)?;

        Ok(VulkanBuffer::new(self, buffer, memory, mapped, bytes))
    }

    /// Blocks until every operation submitted to this device's queue has completed. Used by
    /// tests/callers that want a simple barrier without threading a fence through themselves;
    /// `dispatch.rs`'s own dispatch path uses a per-call fence instead (see its module header).
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        // SAFETY: matches `vkDeviceWaitIdle(VkDevice)`; `self.device` came from a successful
        // `vkCreateDevice` and is not destroyed before `&self` goes away (only this struct's own
        // `Drop` does that, which needs `&mut self`/ownership).
        let rc = unsafe { (self.instance.fns.device_wait_idle)(self.device) };
        check("vkDeviceWaitIdle", rc)
    }
}

impl<'a> Drop for VulkanDevice<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.device` was produced by a successful `vkCreateDevice` and destroyed at
        // most once (`Drop` runs exactly once). Every `VulkanBuffer`/`VulkanComputePipeline`
        // derived from this device borrows `&'d VulkanDevice<'a>` (see this file's module-level
        // note), so the borrow checker itself guarantees none of them are still alive at this
        // point — `drop(&mut self)` requires exclusive access, which cannot be granted while any
        // such borrow exists.
        unsafe {
            (self.instance.fns.destroy_device)(self.device, std::ptr::null());
        }
    }
}
