// Raw HSA Core Runtime API shapes: opaque handle types, status/attribute enums, and
// function-pointer signatures matching the HSA Runtime's stable, published C ABI (as
// documented by the HSA Foundation's Core Runtime specification and mirrored by every
// ROCm `hsa.h`). As with `../ffi.rs`'s CUDA transcription, nothing here is generated from a
// real header — there is none in this build — so every shape is transcribed by hand and must
// stay in lockstep with the spec.
//
// The one piece of this file with no CUDA analogue is the AQL (Architected Queuing Language)
// kernel-dispatch packet: HSA has no `cuLaunchKernel`-style entry point, so dispatching a
// kernel means writing this exact 64-byte struct into a queue's ring buffer by hand. Field
// order and width below match the HSA spec's packet layout precisely; `queue.rs`'s
// structural self-consistency test checks the byte offsets this produces.

use std::ffi::{c_char, c_void};

/// `hsa_status_t`: a C enum, 4 bytes at the ABI level. `0x0` is `HSA_STATUS_SUCCESS`.
pub type HsaStatus = i32;
pub const HSA_STATUS_SUCCESS: HsaStatus = 0x0;

/// Opaque runtime-owned handles. Every one of these is a 64-bit integer cookie the runtime
/// hands back and validates on later calls — never a pointer Basalt dereferences itself, so a
/// plain `#[repr(C)]` wrapper is a faithful, safe-to-copy representation (mirrors `ffi.rs`'s
/// `CUcontext`/`CUmodule` newtypes, which make the same choice for CUDA's pointer-shaped
/// handles).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaAgent {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaRegion {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaSignal {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaExecutableHandle {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaExecutableSymbol {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaCodeObjectReader {
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HsaLoadedCodeObject {
    pub handle: u64,
}

/// `hsa_queue_t`: unlike the handles above, this one is a real, publicly-documented struct —
/// user code is expected to read `base_address` and `doorbell_signal` directly to construct
/// and submit AQL packets by hand, which is exactly what `queue.rs` does.
#[repr(C)]
#[derive(Debug)]
pub struct HsaQueueRaw {
    pub queue_type: HsaQueueType32,
    pub features: u32,
    pub base_address: *mut c_void,
    pub doorbell_signal: HsaSignal,
    pub size: u32,
    pub reserved1: u32,
    pub id: u64,
}

pub type HsaQueueType32 = u32;
pub const HSA_QUEUE_TYPE_MULTI: HsaQueueType32 = 0;

pub type HsaAgentInfoAttr = i32;
pub const HSA_AGENT_INFO_NAME: HsaAgentInfoAttr = 0;
pub const HSA_AGENT_INFO_DEVICE: HsaAgentInfoAttr = 17;

/// `hsa_device_type_t`.
pub type HsaDeviceTypeRaw = i32;
pub const HSA_DEVICE_TYPE_CPU: HsaDeviceTypeRaw = 0;
pub const HSA_DEVICE_TYPE_GPU: HsaDeviceTypeRaw = 1;
pub const HSA_DEVICE_TYPE_DSP: HsaDeviceTypeRaw = 2;

pub type HsaRegionInfoAttr = i32;
pub const HSA_REGION_INFO_SEGMENT: HsaRegionInfoAttr = 0;
pub const HSA_REGION_INFO_GLOBAL_FLAGS: HsaRegionInfoAttr = 1;

/// `hsa_region_segment_t`.
pub type HsaRegionSegmentRaw = i32;
pub const HSA_REGION_SEGMENT_GLOBAL: HsaRegionSegmentRaw = 0;

/// `hsa_region_global_flag_t`, a bitmask, not an exclusive enum: a global region can be
/// kernarg-capable and fine- or coarse-grained at once.
pub const HSA_REGION_GLOBAL_FLAG_KERNARG: u32 = 1;
pub const HSA_REGION_GLOBAL_FLAG_FINE_GRAINED: u32 = 2;
pub const HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED: u32 = 4;

pub type HsaExecutableSymbolInfoAttr = i32;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: HsaExecutableSymbolInfoAttr = 11;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: HsaExecutableSymbolInfoAttr = 13;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: HsaExecutableSymbolInfoAttr = 14;
pub const HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT: HsaExecutableSymbolInfoAttr = 22;

pub type HsaProfileRaw = i32;
pub const HSA_PROFILE_BASE: HsaProfileRaw = 0;

pub type HsaDefaultFloatRoundingModeRaw = i32;
pub const HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT: HsaDefaultFloatRoundingModeRaw = 0;

pub type HsaSignalValue = i64;

pub type HsaSignalConditionRaw = i32;
pub const HSA_SIGNAL_CONDITION_LT: HsaSignalConditionRaw = 2;

pub type HsaWaitStateRaw = i32;
pub const HSA_WAIT_STATE_BLOCKED: HsaWaitStateRaw = 0;

// --- AQL kernel-dispatch packet -------------------------------------------------------------
//
// Field-for-field transcription of `hsa_kernel_dispatch_packet_t` from the HSA Core Runtime
// specification: 64 bytes total, no implicit padding (every field's own alignment already
// satisfies the ones after it, so `#[repr(C)]` alone reproduces the documented layout).

pub const HSA_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;

/// Bit offsets within a packet's 16-bit `header`, per `hsa_packet_header_t` /
/// `hsa_packet_header_width_t`. Bit 8 (`HSA_PACKET_HEADER_BARRIER`) is omitted here: this
/// crate never sets it, since it never pipelines more than one outstanding dispatch per queue
/// and so never needs a barrier dependency on the packet ahead of it.
pub const HSA_PACKET_HEADER_TYPE_SHIFT: u16 = 0;
pub const HSA_PACKET_HEADER_SCACQUIRE_FENCE_SHIFT: u16 = 9;
pub const HSA_PACKET_HEADER_SCRELEASE_FENCE_SHIFT: u16 = 11;

/// `hsa_fence_scope_t`.
pub const HSA_FENCE_SCOPE_SYSTEM: u16 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HsaKernelDispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub reserved0: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: u64,
    pub kernarg_address: *mut c_void,
    pub reserved2: u64,
    pub completion_signal: HsaSignal,
}

/// Builds a kernel-dispatch packet header with both the acquire and release fences scoped to
/// the whole system (the conservative, always-correct choice — narrower scopes are an
/// optimization this crate does not attempt) and the barrier bit clear, since this crate never
/// pipelines more than one outstanding dispatch per queue.
pub fn build_kernel_dispatch_header() -> u16 {
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE_SHIFT)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SHIFT)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SHIFT)
}

// --- Function-pointer signatures ------------------------------------------------------------

pub type FnHsaInit = unsafe extern "C" fn() -> HsaStatus;
pub type FnHsaShutDown = unsafe extern "C" fn() -> HsaStatus;
pub type FnHsaStatusString =
    unsafe extern "C" fn(status: HsaStatus, status_string: *mut *const c_char) -> HsaStatus;

pub type HsaIterateAgentsCallback = extern "C" fn(agent: HsaAgent, data: *mut c_void) -> HsaStatus;
pub type FnHsaIterateAgents =
    unsafe extern "C" fn(callback: HsaIterateAgentsCallback, data: *mut c_void) -> HsaStatus;

pub type FnHsaAgentGetInfo = unsafe extern "C" fn(
    agent: HsaAgent,
    attribute: HsaAgentInfoAttr,
    value: *mut c_void,
) -> HsaStatus;

pub type HsaIterateRegionsCallback =
    extern "C" fn(region: HsaRegion, data: *mut c_void) -> HsaStatus;
pub type FnHsaAgentIterateRegions = unsafe extern "C" fn(
    agent: HsaAgent,
    callback: HsaIterateRegionsCallback,
    data: *mut c_void,
) -> HsaStatus;
pub type FnHsaRegionGetInfo = unsafe extern "C" fn(
    region: HsaRegion,
    attribute: HsaRegionInfoAttr,
    value: *mut c_void,
) -> HsaStatus;

pub type FnHsaMemoryAllocate =
    unsafe extern "C" fn(region: HsaRegion, size: usize, ptr: *mut *mut c_void) -> HsaStatus;
pub type FnHsaMemoryFree = unsafe extern "C" fn(ptr: *mut c_void) -> HsaStatus;
pub type FnHsaMemoryCopy =
    unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, size: usize) -> HsaStatus;

pub type FnHsaCodeObjectReaderCreateFromMemory = unsafe extern "C" fn(
    code_object: *const c_void,
    size: usize,
    reader: *mut HsaCodeObjectReader,
) -> HsaStatus;
pub type FnHsaCodeObjectReaderDestroy =
    unsafe extern "C" fn(reader: HsaCodeObjectReader) -> HsaStatus;

pub type FnHsaExecutableCreateAlt = unsafe extern "C" fn(
    profile: HsaProfileRaw,
    default_float_rounding_mode: HsaDefaultFloatRoundingModeRaw,
    options: *const c_char,
    executable: *mut HsaExecutableHandle,
) -> HsaStatus;
pub type FnHsaExecutableDestroy =
    unsafe extern "C" fn(executable: HsaExecutableHandle) -> HsaStatus;
pub type FnHsaExecutableLoadAgentCodeObject = unsafe extern "C" fn(
    executable: HsaExecutableHandle,
    agent: HsaAgent,
    reader: HsaCodeObjectReader,
    options: *const c_char,
    loaded_code_object: *mut HsaLoadedCodeObject,
) -> HsaStatus;
pub type FnHsaExecutableFreeze =
    unsafe extern "C" fn(executable: HsaExecutableHandle, options: *const c_char) -> HsaStatus;

pub type FnHsaExecutableGetSymbolByName = unsafe extern "C" fn(
    executable: HsaExecutableHandle,
    symbol_name: *const c_char,
    agent: *const HsaAgent,
    symbol: *mut HsaExecutableSymbol,
) -> HsaStatus;
pub type FnHsaExecutableSymbolGetInfo = unsafe extern "C" fn(
    symbol: HsaExecutableSymbol,
    attribute: HsaExecutableSymbolInfoAttr,
    value: *mut c_void,
) -> HsaStatus;

/// The queue-level asynchronous error callback `hsa_queue_create` accepts — this crate's
/// ABEND-wiring hook. Note this is the HSA Core spec's own signature: `(status, source queue,
/// data)`, with no packet index parameter. Correlating a fault against a specific packet index
/// or dispatch snapshot is part of the fuller diagnostic system this task does not build; see
/// `mod.rs`'s scope note.
pub type HsaQueueErrorCallback =
    extern "C" fn(status: HsaStatus, source: *mut HsaQueueRaw, data: *mut c_void);

pub type FnHsaQueueCreate = unsafe extern "C" fn(
    agent: HsaAgent,
    size: u32,
    queue_type: HsaQueueType32,
    callback: Option<HsaQueueErrorCallback>,
    data: *mut c_void,
    private_segment_size: u32,
    group_segment_size: u32,
    queue: *mut *mut HsaQueueRaw,
) -> HsaStatus;
pub type FnHsaQueueDestroy = unsafe extern "C" fn(queue: *mut HsaQueueRaw) -> HsaStatus;

pub type FnHsaSignalCreate = unsafe extern "C" fn(
    initial_value: HsaSignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    signal: *mut HsaSignal,
) -> HsaStatus;
pub type FnHsaSignalDestroy = unsafe extern "C" fn(signal: HsaSignal) -> HsaStatus;

/// Real HSA queue-index and signal-store/wait primitives are plain value-returning functions,
/// not `hsa_status_t`-returning calls — there is nothing to fail short of a corrupt queue
/// pointer, which is a caller bug, not a runtime error this crate's `check()` models.
pub type FnHsaQueueAddWriteIndexRelaxed =
    unsafe extern "C" fn(queue: *const HsaQueueRaw, value: u64) -> u64;
pub type FnHsaQueueLoadReadIndexRelaxed = unsafe extern "C" fn(queue: *const HsaQueueRaw) -> u64;

pub type FnHsaSignalStoreRelaxed = unsafe extern "C" fn(signal: HsaSignal, value: HsaSignalValue);
pub type FnHsaSignalWaitScacquire = unsafe extern "C" fn(
    signal: HsaSignal,
    condition: HsaSignalConditionRaw,
    compare_value: HsaSignalValue,
    timeout_hint: u64,
    wait_expectancy_hint: HsaWaitStateRaw,
) -> HsaSignalValue;
