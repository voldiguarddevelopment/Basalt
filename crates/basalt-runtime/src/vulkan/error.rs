use std::fmt;

/// Everything that can go wrong loading or driving the Vulkan loader/driver, from a missing
/// `libvulkan.so.1` through a failed API call. Mirrors `../error.rs`'s `CudaError` and
/// `../hsa/error.rs`'s `HsaError` shape and the same reasoning for staying independent of
/// `basalt-diag`'s E-codes: this crate has no consuming diagnostic stage yet.
///
/// One real difference from the other two loaders: core Vulkan has no `vkGetErrorString`-style
/// entry point at all (unlike `cuGetErrorString`/`hsa_status_string`), so `CallFailed.message`
/// is produced locally from a fixed table of the `VkResult` values this crate's own calls can
/// plausibly return (see `describe_vk_result` in `ffi.rs`), falling back to the bare numeric
/// code for anything outside that table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanError {
    /// Neither `libvulkan.so.1` nor `libvulkan.so` could be `dlopen`ed; the string is the
    /// dynamic linker's own diagnostic for the last name tried.
    DriverNotFound(String),
    /// The loader library opened, but a required entry point isn't exported under the symbol
    /// name this crate knows to try.
    SymbolNotFound(&'static str),
    /// A resolved entry point ran and returned a non-`VK_SUCCESS` `VkResult`.
    CallFailed {
        call: &'static str,
        code: i32,
        message: String,
    },
    /// `vkGetPhysicalDeviceQueueFamilyProperties` reported no queue family with
    /// `VK_QUEUE_COMPUTE_BIT` set on the physical device this crate was asked to use.
    NoComputeQueueFamily,
    /// `vkGetPhysicalDeviceMemoryProperties` reported no memory type satisfying the requested
    /// property flags intersected with a buffer's own `memoryTypeBits` mask.
    NoSuitableMemoryType,
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanError::DriverNotFound(msg) => {
                write!(f, "Vulkan loader library not found: {msg}")
            }
            VulkanError::SymbolNotFound(sym) => {
                write!(f, "Vulkan entry point not found: {sym}")
            }
            VulkanError::CallFailed {
                call,
                code,
                message,
            } => {
                write!(f, "{call} failed with VkResult {code}: {message}")
            }
            VulkanError::NoComputeQueueFamily => {
                write!(f, "no queue family with VK_QUEUE_COMPUTE_BIT was found")
            }
            VulkanError::NoSuitableMemoryType => {
                write!(f, "no memory type satisfies the requested property flags")
            }
        }
    }
}

impl std::error::Error for VulkanError {}
