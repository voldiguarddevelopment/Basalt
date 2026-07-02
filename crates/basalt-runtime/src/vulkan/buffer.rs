// A host-visible, host-coherent Vulkan buffer allocation, persistently mapped for its whole
// lifetime. Mirrors `../context.rs`'s `DeviceBuffer`/`../hsa/runtime.rs`'s `HsaBuffer` in shape
// (bounds-checked host copies, a raw handle exposed for building a descriptor write), but ties
// its lifetime to the *device* that created it rather than the instance — see `device.rs`'s
// module header for why that's a real soundness requirement here, not just a style choice.

use std::ffi::c_void;

use crate::vulkan::device::VulkanDevice;
use crate::vulkan::error::VulkanError;
use crate::vulkan::ffi::{VkBuffer, VkDeviceMemory};

pub struct VulkanBuffer<'d, 'a> {
    device: &'d VulkanDevice<'a>,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    ptr: *mut c_void,
    len: usize,
}

impl<'d, 'a> VulkanBuffer<'d, 'a> {
    pub(crate) fn new(
        device: &'d VulkanDevice<'a>,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        ptr: *mut c_void,
        len: usize,
    ) -> Self {
        VulkanBuffer {
            device,
            buffer,
            memory,
            ptr,
            len,
        }
    }

    /// Copies `src` into this buffer's mapped memory. `src` must fit within the buffer's
    /// allocated length. Memory is host-coherent (see `device.rs`'s `alloc_host_buffer`), so no
    /// explicit `vkFlushMappedMemoryRanges` is required for the device to observe this write.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<(), VulkanError> {
        if src.len() > self.len {
            return Err(VulkanError::CallFailed {
                call: "vkMapMemory",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds buffer of {} bytes",
                    src.len(),
                    self.len
                ),
            });
        }
        // SAFETY: `self.ptr` was returned by a successful `vkMapMemory` in `alloc_host_buffer`
        // and stays mapped for this buffer's whole lifetime (unmapped only in `Drop`, which
        // needs exclusive access and so cannot run concurrently with this call); `src.len() <=
        // self.len`, the size the mapping covers, so this write stays in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.cast::<u8>(), src.len());
        }
        Ok(())
    }

    /// Copies from this buffer's mapped memory into `dst`. `dst` must fit within the buffer's
    /// allocated length. Host-coherent memory needs no explicit `vkInvalidateMappedMemoryRanges`
    /// before reading, but the caller is responsible for having already waited for the device
    /// work that produced this data to complete (a fence wait, e.g. `VulkanComputePipeline::
    /// dispatch`'s own `vkWaitForFences` call) — reading a mapped range with the device still
    /// writing to it is a race regardless of the coherency bit.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), VulkanError> {
        if dst.len() > self.len {
            return Err(VulkanError::CallFailed {
                call: "vkMapMemory",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds buffer of {} bytes",
                    dst.len(),
                    self.len
                ),
            });
        }
        // SAFETY: same contract as `copy_from_host`, reversed direction; `dst.len() <= self.len`
        // keeps the read within the mapped range.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.cast::<u8>(), dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    /// The raw `VkBuffer` handle, for building a `VkDescriptorBufferInfo` in a descriptor write.
    pub(crate) fn raw_handle(&self) -> VkBuffer {
        self.buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'d, 'a> Drop for VulkanBuffer<'d, 'a> {
    fn drop(&mut self) {
        let fns = &self.device.instance().fns;
        let device = self.device.handle();
        // SAFETY: matches `vkUnmapMemory(VkDevice, VkDeviceMemory)`; `self.memory` was mapped
        // exactly once in `alloc_host_buffer` and is unmapped here exactly once (`Drop` runs at
        // most once).
        unsafe { (fns.unmap_memory)(device, self.memory) };
        // SAFETY: matches `vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*)`;
        // `self.memory` was allocated in `alloc_host_buffer` and freed at most once.
        unsafe { (fns.free_memory)(device, self.memory, std::ptr::null()) };
        // SAFETY: matches `vkDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*)`;
        // `self.buffer` was created in `alloc_host_buffer` and destroyed at most once. The
        // return codes of `vkFreeMemory`/`vkDestroyBuffer` (both `void`-returning per the spec)
        // aren't checked because there is nothing to check — matches this crate's other `Drop`
        // impls, which discard the fallible ones for the same "a `Drop` cannot propagate a
        // `Result`" reason.
        unsafe { (fns.destroy_buffer)(device, self.buffer, std::ptr::null()) };
    }
}
