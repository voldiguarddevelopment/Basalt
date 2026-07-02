// Shader module + descriptor-set-layout + pipeline-layout + compute-pipeline construction. This
// is where this crate's central empirical finding about `basalt-spirv`'s `Kernel`-execution-model
// output actually manifests as two real API calls with two real, different outcomes — see the
// comments on `vkCreateShaderModule`/`vkCreateComputePipelines` below, and
// `../../basalt-spirv/src/emit.rs`'s own header for why that backend targets `Kernel` rather than
// `GLCompute` in the first place.
//
// Resource-binding ABI: this crate invents the simplest possible one, since neither BIR nor
// `basalt-spirv` carries any descriptor-binding information (see `emit.rs`'s header on exactly
// this gap) — one `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` binding per buffer argument, bound in
// argument order at `set = 0`, and at most one push-constant range (offset 0) for every scalar
// argument packed by the caller. This ABI is defined and used only by this loader's own tests
// (see `tests/vulkan_gpu_proof.rs`), matching a hand-written GLCompute shader's own `layout(set =
// 0, binding = N)`/`layout(push_constant)` declarations — it is not, and does not attempt to be,
// a general answer to the resource-binding-ABI gap `emit.rs` describes.

use std::ffi::CString;

use crate::vulkan::device::VulkanDevice;
use crate::vulkan::error::VulkanError;
use crate::vulkan::ffi::{
    VkComputePipelineCreateInfo, VkDescriptorSetLayout, VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutCreateInfo, VkPipeline, VkPipelineCache, VkPipelineLayout,
    VkPipelineLayoutCreateInfo, VkPipelineShaderStageCreateInfo, VkPushConstantRange,
    VkShaderModule, VkShaderModuleCreateInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_SHADER_STAGE_COMPUTE_BIT, VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
};
use crate::vulkan::instance::check;

/// A compute pipeline plus the descriptor-set-layout/pipeline-layout it was built with. Borrows
/// `&'d VulkanDevice<'a>` directly — see `device.rs`'s module header for why that tie (rather
/// than sibling-under-the-instance) is load-bearing here, not just a style choice.
pub struct VulkanComputePipeline<'d, 'a> {
    device: &'d VulkanDevice<'a>,
    descriptor_set_layout: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    shader_module: VkShaderModule,
    pipeline: VkPipeline,
    num_storage_buffers: u32,
}

impl<'d, 'a> VulkanComputePipeline<'d, 'a> {
    /// Builds a compute pipeline from `spirv_words` (a whole SPIR-V module, as `u32` words — the
    /// same unit `basalt_spirv::Spirv::emit` returns as bytes and `VkShaderModuleCreateInfo`
    /// itself is specified in terms of), looking up `entry_point` as the pipeline's single
    /// shader stage. `num_storage_buffers` and `push_constant_bytes` describe the resource-
    /// binding ABI this call builds (see this file's module header) — the caller is responsible
    /// for having compiled/written a shader matching it.
    pub fn create(
        device: &'d VulkanDevice<'a>,
        spirv_words: &[u32],
        entry_point: &str,
        num_storage_buffers: u32,
        push_constant_bytes: u32,
    ) -> Result<VulkanComputePipeline<'d, 'a>, VulkanError> {
        let entry_cstr = CString::new(entry_point).map_err(|_| VulkanError::CallFailed {
            call: "vkCreateComputePipelines",
            code: -1,
            message: "entry point name contains an interior NUL byte".to_string(),
        })?;

        let fns = &device.instance().fns;
        let vk_device = device.handle();

        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..num_storage_buffers)
            .map(|binding| VkDescriptorSetLayoutBinding {
                binding,
                descriptor_type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let dsl_create_info = VkDescriptorSetLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut descriptor_set_layout = VkDescriptorSetLayout::NULL;
        // SAFETY: matches `vkCreateDescriptorSetLayout(VkDevice, const
        // VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*,
        // VkDescriptorSetLayout*)`; `dsl_create_info` and the `bindings` it points into are
        // valid, live locals for the duration of this call.
        let rc = unsafe {
            (fns.create_descriptor_set_layout)(
                vk_device,
                &dsl_create_info,
                std::ptr::null(),
                &mut descriptor_set_layout,
            )
        };
        check("vkCreateDescriptorSetLayout", rc)?;

        let push_constant_range = VkPushConstantRange {
            stage_flags: VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_constant_bytes,
        };
        let pipeline_layout_ci = VkPipelineLayoutCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: if push_constant_bytes > 0 { 1 } else { 0 },
            p_push_constant_ranges: &push_constant_range,
        };
        let mut pipeline_layout = VkPipelineLayout::NULL;
        // SAFETY: matches `vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*,
        // const VkAllocationCallbacks*, VkPipelineLayout*)`; `pipeline_layout_ci` and everything
        // it points into are valid, live locals for the duration of this call.
        let rc = unsafe {
            (fns.create_pipeline_layout)(
                vk_device,
                &pipeline_layout_ci,
                std::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if let Err(err) = check("vkCreatePipelineLayout", rc) {
            // SAFETY: `descriptor_set_layout` was created above and not yet referenced by
            // anything else (this is the only failure path after it that returns early).
            unsafe {
                (fns.destroy_descriptor_set_layout)(
                    vk_device,
                    descriptor_set_layout,
                    std::ptr::null(),
                )
            };
            return Err(err);
        }

        let code_size_bytes = std::mem::size_of_val(spirv_words);
        let shader_ci = VkShaderModuleCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: code_size_bytes,
            p_code: spirv_words.as_ptr(),
        };
        let mut shader_module = VkShaderModule::NULL;
        // SAFETY: matches `vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const
        // VkAllocationCallbacks*, VkShaderModule*)`; `shader_ci` and `spirv_words` are valid,
        // live for the duration of this call.
        //
        // This call succeeding even when `spirv_words` is a `Kernel`-execution-model module (see
        // `../../basalt-spirv/src/emit.rs`'s header) is the first half of this crate's real,
        // empirically-confirmed finding on this project's llvmpipe test machine: the loader/
        // driver combination parses the SPIR-V binary's structure at module-creation time
        // without validating the entry point's execution model against any particular pipeline
        // stage — that check only happens where it is actually meaningful, at
        // `vkCreateComputePipelines` below.
        let rc = unsafe {
            (fns.create_shader_module)(vk_device, &shader_ci, std::ptr::null(), &mut shader_module)
        };
        if let Err(err) = check("vkCreateShaderModule", rc) {
            // SAFETY: both handles were created above and not yet referenced by anything else.
            unsafe { (fns.destroy_pipeline_layout)(vk_device, pipeline_layout, std::ptr::null()) };
            unsafe {
                (fns.destroy_descriptor_set_layout)(
                    vk_device,
                    descriptor_set_layout,
                    std::ptr::null(),
                )
            };
            return Err(err);
        }

        let stage = VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry_cstr.as_ptr(),
            p_specialization_info: std::ptr::null(),
        };
        let compute_ci = VkComputePipelineCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: VkPipeline::NULL,
            base_pipeline_index: -1,
        };
        let mut pipeline = VkPipeline::NULL;
        // SAFETY: matches `vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t, const
        // VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*)`;
        // `compute_ci`, its embedded `stage`, and `entry_cstr` (NUL-terminated, kept alive across
        // this whole call) are all valid for its duration. `VkPipelineCache::NULL` requests no
        // caching, the documented meaning of a null cache handle.
        //
        // This is the second half of the finding, and the real, load-bearing boundary
        // `basalt-spirv`'s `Kernel`-model output cannot currently cross: on real llvmpipe, this
        // exact call returns `VK_ERROR_UNKNOWN` (-13) for a `Kernel`-execution-model module
        // (confirmed directly against this backend's own emitted bytes for `vector_add.cu`) and
        // `VK_SUCCESS` for a semantically-identical, hand-written `GLCompute`-model module (see
        // `tests/vulkan_gpu_proof.rs`, which exercises both). Vulkan's compute pipeline creation
        // is specified to require the `GLCompute` execution model unconditionally — there is no
        // capability or extension that relaxes this — so this failure is an expected, permanent
        // property of `basalt-spirv`'s current output, not a defect in this loader.
        let rc = unsafe {
            (fns.create_compute_pipelines)(
                vk_device,
                VkPipelineCache::NULL,
                1,
                &compute_ci,
                std::ptr::null(),
                &mut pipeline,
            )
        };
        if let Err(err) = check("vkCreateComputePipelines", rc) {
            // SAFETY: all three handles were created above and not yet referenced by anything
            // else (pipeline creation itself is what failed).
            unsafe { (fns.destroy_shader_module)(vk_device, shader_module, std::ptr::null()) };
            unsafe { (fns.destroy_pipeline_layout)(vk_device, pipeline_layout, std::ptr::null()) };
            unsafe {
                (fns.destroy_descriptor_set_layout)(
                    vk_device,
                    descriptor_set_layout,
                    std::ptr::null(),
                )
            };
            return Err(err);
        }

        Ok(VulkanComputePipeline {
            device,
            descriptor_set_layout,
            pipeline_layout,
            shader_module,
            pipeline,
            num_storage_buffers,
        })
    }

    pub(crate) fn device(&self) -> &'d VulkanDevice<'a> {
        self.device
    }

    pub(crate) fn descriptor_set_layout(&self) -> VkDescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub(crate) fn pipeline_layout(&self) -> VkPipelineLayout {
        self.pipeline_layout
    }

    pub(crate) fn pipeline(&self) -> VkPipeline {
        self.pipeline
    }

    pub(crate) fn num_storage_buffers(&self) -> u32 {
        self.num_storage_buffers
    }
}

impl<'d, 'a> Drop for VulkanComputePipeline<'d, 'a> {
    fn drop(&mut self) {
        let fns = &self.device.instance().fns;
        let vk_device = self.device.handle();
        // SAFETY: every handle here was produced by a successful call in `create` and is
        // destroyed at most once (`Drop` runs exactly once). Destruction order (pipeline before
        // the shader module/pipeline layout it was built from, those before the descriptor set
        // layout) is the conservative direction, though the Vulkan spec permits destroying a
        // shader module immediately after pipeline creation regardless of the pipeline's own
        // lifetime.
        unsafe { (fns.destroy_pipeline)(vk_device, self.pipeline, std::ptr::null()) };
        unsafe { (fns.destroy_shader_module)(vk_device, self.shader_module, std::ptr::null()) };
        unsafe { (fns.destroy_pipeline_layout)(vk_device, self.pipeline_layout, std::ptr::null()) };
        unsafe {
            (fns.destroy_descriptor_set_layout)(
                vk_device,
                self.descriptor_set_layout,
                std::ptr::null(),
            )
        };
    }
}
