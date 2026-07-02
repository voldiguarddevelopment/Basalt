// Binds buffers and push constants to a `VulkanComputePipeline` and dispatches it, blocking
// until completion. There is no `cuLaunchKernel`-style single call in Vulkan: dispatching a
// pipeline means allocating a descriptor set, writing buffer bindings into it, recording a
// command buffer (bind pipeline, bind descriptor set, push constants, dispatch), submitting it to
// a queue, and waiting on a fence — the Vulkan counterpart to `../hsa/queue.rs`'s hand-built AQL
// packet, just built from library calls instead of a raw struct write.
//
// Every descriptor pool/command pool/fence used here is created fresh for this one call and
// destroyed before it returns, rather than being a persistent, reusable resource on
// `VulkanComputePipeline` itself — mirroring `HsaQueue::dispatch`'s own per-call completion
// signal (see `../hsa/queue.rs`'s header) rather than `CudaFunction::launch`'s reuse of an
// already-open context. Simpler to reason about at the cost of some allocator churn on a hot
// path — correct first, matching this project's stated priority order; a caller that wants to
// dispatch the same pipeline in a loop can still do so by calling this repeatedly.

use crate::vulkan::buffer::VulkanBuffer;
use crate::vulkan::error::VulkanError;
use crate::vulkan::ffi::{
    VkCommandBuffer, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo, VkCommandPool,
    VkCommandPoolCreateInfo, VkDescriptorBufferInfo, VkDescriptorPool, VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize, VkDescriptorSet, VkDescriptorSetAllocateInfo, VkFence, VkFenceCreateInfo,
    VkSubmitInfo, VkWriteDescriptorSet, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_PIPELINE_BIND_POINT_COMPUTE, VK_SHADER_STAGE_COMPUTE_BIT,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    VK_STRUCTURE_TYPE_SUBMIT_INFO, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, VK_WHOLE_SIZE,
};
use crate::vulkan::instance::check;
use crate::vulkan::pipeline::VulkanComputePipeline;

impl<'d, 'a> VulkanComputePipeline<'d, 'a> {
    /// Binds `buffers` (one `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` descriptor per buffer, at
    /// `binding = ` its index — see `pipeline.rs`'s module header for this ABI), pushes
    /// `push_constants` verbatim at offset 0, dispatches `group_counts` work groups, and blocks
    /// until the dispatch completes. `buffers.len()` must equal
    /// `self.num_storage_buffers()`; `push_constants.len()` must equal whatever
    /// `push_constant_bytes` this pipeline's descriptor set layout was built with (0 is valid —
    /// no push constants are then bound).
    pub fn dispatch(
        &self,
        buffers: &[&VulkanBuffer<'_, '_>],
        push_constants: &[u8],
        group_counts: (u32, u32, u32),
    ) -> Result<(), VulkanError> {
        let device = self.device();
        let fns = &device.instance().fns;
        let vk_device = device.handle();

        let pool_size = VkDescriptorPoolSize {
            ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: self.num_storage_buffers(),
        };
        let pool_ci = VkDescriptorPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool = VkDescriptorPool::NULL;
        // SAFETY: matches `vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*,
        // const VkAllocationCallbacks*, VkDescriptorPool*)`; `pool_ci` is a valid, live local for
        // the duration of this call.
        let rc = unsafe {
            (fns.create_descriptor_pool)(
                vk_device,
                &pool_ci,
                std::ptr::null(),
                &mut descriptor_pool,
            )
        };
        check("vkCreateDescriptorPool", rc)?;

        let result =
            self.dispatch_with_pool(descriptor_pool, buffers, push_constants, group_counts);

        // SAFETY: matches `vkDestroyDescriptorPool(VkDevice, VkDescriptorPool, const
        // VkAllocationCallbacks*)`; destroying the pool also frees every descriptor set
        // allocated from it (the documented behavior of a non-`FREE_DESCRIPTOR_SET`-flagged
        // pool), so no separate `vkFreeDescriptorSets` call is needed.
        unsafe { (fns.destroy_descriptor_pool)(vk_device, descriptor_pool, std::ptr::null()) };

        result
    }

    fn dispatch_with_pool(
        &self,
        descriptor_pool: VkDescriptorPool,
        buffers: &[&VulkanBuffer<'_, '_>],
        push_constants: &[u8],
        group_counts: (u32, u32, u32),
    ) -> Result<(), VulkanError> {
        let device = self.device();
        let fns = &device.instance().fns;
        let vk_device = device.handle();
        let descriptor_set_layout = self.descriptor_set_layout();

        let set_alloc_info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
        };
        let mut descriptor_set = VkDescriptorSet::NULL;
        // SAFETY: matches `vkAllocateDescriptorSets(VkDevice, const
        // VkDescriptorSetAllocateInfo*, VkDescriptorSet*)`; `set_alloc_info` is a valid, live
        // local, and `descriptor_pool` was just created by the caller with exactly enough
        // capacity (`max_sets = 1`, one storage-buffer descriptor per binding) for this one
        // allocation.
        let rc = unsafe {
            (fns.allocate_descriptor_sets)(vk_device, &set_alloc_info, &mut descriptor_set)
        };
        check("vkAllocateDescriptorSets", rc)?;

        let buffer_infos: Vec<VkDescriptorBufferInfo> = buffers
            .iter()
            .map(|b| VkDescriptorBufferInfo {
                buffer: b.raw_handle(),
                offset: 0,
                range: VK_WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<VkWriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| VkWriteDescriptorSet {
                s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: descriptor_set,
                dst_binding: i as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: info,
                p_texel_buffer_view: std::ptr::null(),
            })
            .collect();
        // SAFETY: matches `vkUpdateDescriptorSets(VkDevice, uint32_t, const
        // VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet*)`; `writes` and the
        // `buffer_infos` each entry points into are both valid, live locals for the duration of
        // this call; zero copies are requested.
        unsafe {
            (fns.update_descriptor_sets)(
                vk_device,
                writes.len() as u32,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            )
        };

        let pool_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: device.queue_family_index(),
        };
        let mut command_pool = VkCommandPool::NULL;
        // SAFETY: matches `vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const
        // VkAllocationCallbacks*, VkCommandPool*)`; `pool_ci` is a valid, live local, and
        // `device.queue_family_index()` is the same family this device's one queue was created
        // from.
        let rc = unsafe {
            (fns.create_command_pool)(vk_device, &pool_ci, std::ptr::null(), &mut command_pool)
        };
        check("vkCreateCommandPool", rc)?;

        let result =
            self.record_and_submit(command_pool, descriptor_set, push_constants, group_counts);

        // SAFETY: matches `vkDestroyCommandPool(VkDevice, VkCommandPool, const
        // VkAllocationCallbacks*)`; destroying the pool also frees every command buffer
        // allocated from it, the documented behavior.
        unsafe { (fns.destroy_command_pool)(vk_device, command_pool, std::ptr::null()) };

        result
    }

    fn record_and_submit(
        &self,
        command_pool: VkCommandPool,
        descriptor_set: VkDescriptorSet,
        push_constants: &[u8],
        group_counts: (u32, u32, u32),
    ) -> Result<(), VulkanError> {
        let device = self.device();
        let fns = &device.instance().fns;
        let vk_device = device.handle();

        let cmd_alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut command_buffer = VkCommandBuffer::NULL;
        // SAFETY: matches `vkAllocateCommandBuffers(VkDevice, const
        // VkCommandBufferAllocateInfo*, VkCommandBuffer*)`; `cmd_alloc_info` is a valid, live
        // local, and `command_pool` was just created by the caller with room for this one
        // allocation.
        let rc = unsafe {
            (fns.allocate_command_buffers)(vk_device, &cmd_alloc_info, &mut command_buffer)
        };
        check("vkAllocateCommandBuffers", rc)?;

        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: std::ptr::null(),
        };
        // SAFETY: matches `vkBeginCommandBuffer(VkCommandBuffer, const
        // VkCommandBufferBeginInfo*)`; `command_buffer` came from the successful call above,
        // `begin_info` is valid and live for the duration of this call.
        let rc = unsafe { (fns.begin_command_buffer)(command_buffer, &begin_info) };
        check("vkBeginCommandBuffer", rc)?;

        // SAFETY: matches `vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint,
        // VkPipeline)`; `command_buffer` is mid-recording (between begin/end) and
        // `self.pipeline()` is a live compute pipeline owned by `self`.
        unsafe {
            (fns.cmd_bind_pipeline)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline(),
            )
        };
        // SAFETY: matches `vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint,
        // VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const
        // uint32_t*)`; `descriptor_set` was allocated and populated by the caller against this
        // same pipeline's descriptor set layout.
        unsafe {
            (fns.cmd_bind_descriptor_sets)(
                command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline_layout(),
                0,
                1,
                &descriptor_set,
                0,
                std::ptr::null(),
            )
        };
        if !push_constants.is_empty() {
            // SAFETY: matches `vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout,
            // VkShaderStageFlags, uint32_t, uint32_t, const void*)`; `push_constants` is a live
            // slice for the duration of this call, and its length matches the push-constant
            // range this pipeline's layout was built with (the caller's responsibility, exactly
            // like `cuLaunchKernel`'s own unchecked parameter array).
            unsafe {
                (fns.cmd_push_constants)(
                    command_buffer,
                    self.pipeline_layout(),
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push_constants.len() as u32,
                    push_constants.as_ptr().cast(),
                )
            };
        }
        // SAFETY: matches `vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t)`.
        unsafe {
            (fns.cmd_dispatch)(
                command_buffer,
                group_counts.0,
                group_counts.1,
                group_counts.2,
            )
        };

        // SAFETY: matches `vkEndCommandBuffer(VkCommandBuffer)`; ends the same recording begun
        // above.
        let rc = unsafe { (fns.end_command_buffer)(command_buffer) };
        check("vkEndCommandBuffer", rc)?;

        let fence_ci = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
        };
        let mut fence = VkFence::NULL;
        // SAFETY: matches `vkCreateFence(VkDevice, const VkFenceCreateInfo*, const
        // VkAllocationCallbacks*, VkFence*)`; created unsignaled (`flags = 0`), the state
        // `vkQueueSubmit`/`vkWaitForFences` below expect.
        let rc = unsafe { (fns.create_fence)(vk_device, &fence_ci, std::ptr::null(), &mut fence) };
        check("vkCreateFence", rc)?;

        let submit_info = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        // SAFETY: matches `vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence)`;
        // `submit_info` and `command_buffer` are both live for the duration of this call, and
        // `fence` was just created unsignaled above.
        let rc = unsafe { (fns.queue_submit)(device.queue(), 1, &submit_info, fence) };
        if let Err(err) = check("vkQueueSubmit", rc) {
            // SAFETY: `fence` was created above and not yet waited on or destroyed.
            unsafe { (fns.destroy_fence)(vk_device, fence, std::ptr::null()) };
            return Err(err);
        }

        // SAFETY: matches `vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32,
        // uint64_t)`; `VK_TRUE` (1) for `wait_all` and `u64::MAX` for an unbounded timeout are
        // the documented "block until this one fence signals, however long that takes" values.
        let rc = unsafe { (fns.wait_for_fences)(vk_device, 1, &fence, 1, u64::MAX) };
        let wait_result = check("vkWaitForFences", rc);

        // SAFETY: matches `vkDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)`;
        // `fence` was created above and destroyed at most once here regardless of the wait's
        // outcome.
        unsafe { (fns.destroy_fence)(vk_device, fence, std::ptr::null()) };

        wait_result
    }
}
