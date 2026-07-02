// The loaded HSA runtime: a `dlopen` handle plus every Core Runtime entry point this crate
// needs, resolved once via `dlsym` into a plain function-pointer table — the HSA counterpart to
// `../driver.rs`'s `CudaDriver`/`FnTable`. `HsaAgent`/`HsaRegion` values handed out here are
// opaque runtime cookies, not borrows, so unlike CUDA's `CudaContext<'a>` they carry no
// lifetime of their own; `HsaExecutable`/`HsaQueue`/`HsaBuffer` are what borrow `&'a HsaRuntime`
// for their whole lifetime.
//
// Memory: the base HSA spec's region API (`hsa_agent_iterate_regions` /
// `hsa_region_get_info` / `hsa_memory_allocate` / `hsa_memory_free` / `hsa_memory_copy`) is used
// here rather than the `hsa_amd_memory_pool_*` vendor extension. Real ROCm programs reach for
// the AMD pool API mainly to pick apart fine- vs. coarse-grained VRAM pools for performance; the
// base region API is guaranteed present on any HSA-conformant runtime, requires no vendor
// extension table lookup, and is sufficient to find a kernarg-capable region and a
// general-purpose global region, which is everything dispatch needs. Revisiting this for actual
// VRAM-locality control is future work, once real hardware is available to validate against.

use std::ffi::{c_char, c_void};
use std::sync::{Arc, Mutex};

use crate::dl::Library;
use crate::hsa::error::HsaError;
use crate::hsa::ffi::{
    FnHsaAgentGetInfo, FnHsaAgentIterateRegions, FnHsaCodeObjectReaderCreateFromMemory,
    FnHsaCodeObjectReaderDestroy, FnHsaExecutableCreateAlt, FnHsaExecutableDestroy,
    FnHsaExecutableFreeze, FnHsaExecutableGetSymbolByName, FnHsaExecutableLoadAgentCodeObject,
    FnHsaExecutableSymbolGetInfo, FnHsaInit, FnHsaIterateAgents, FnHsaMemoryAllocate,
    FnHsaMemoryCopy, FnHsaMemoryFree, FnHsaQueueAddWriteIndexRelaxed, FnHsaQueueCreate,
    FnHsaQueueDestroy, FnHsaQueueLoadReadIndexRelaxed, FnHsaRegionGetInfo, FnHsaShutDown,
    FnHsaSignalCreate, FnHsaSignalDestroy, FnHsaSignalStoreRelaxed, FnHsaSignalWaitScacquire,
    FnHsaStatusString, HsaAgent, HsaDeviceTypeRaw, HsaQueueErrorCallback, HsaRegion,
    HsaRegionInfoAttr, HsaStatus, HSA_AGENT_INFO_DEVICE, HSA_AGENT_INFO_NAME, HSA_DEVICE_TYPE_CPU,
    HSA_DEVICE_TYPE_DSP, HSA_DEVICE_TYPE_GPU, HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED,
    HSA_REGION_GLOBAL_FLAG_FINE_GRAINED, HSA_REGION_GLOBAL_FLAG_KERNARG,
    HSA_REGION_INFO_GLOBAL_FLAGS, HSA_REGION_INFO_SEGMENT, HSA_REGION_SEGMENT_GLOBAL,
    HSA_STATUS_SUCCESS,
};
use crate::hsa::queue::{queue_error_callback, HsaQueue};

pub(crate) struct FnTable {
    pub hsa_shut_down: FnHsaShutDown,
    pub hsa_status_string: Option<FnHsaStatusString>,
    pub hsa_iterate_agents: FnHsaIterateAgents,
    pub hsa_agent_get_info: FnHsaAgentGetInfo,
    pub hsa_agent_iterate_regions: FnHsaAgentIterateRegions,
    pub hsa_region_get_info: FnHsaRegionGetInfo,
    pub hsa_memory_allocate: FnHsaMemoryAllocate,
    pub hsa_memory_free: FnHsaMemoryFree,
    pub hsa_memory_copy: FnHsaMemoryCopy,
    pub hsa_code_object_reader_create_from_memory: FnHsaCodeObjectReaderCreateFromMemory,
    pub hsa_code_object_reader_destroy: FnHsaCodeObjectReaderDestroy,
    pub hsa_executable_create_alt: FnHsaExecutableCreateAlt,
    pub hsa_executable_destroy: FnHsaExecutableDestroy,
    pub hsa_executable_load_agent_code_object: FnHsaExecutableLoadAgentCodeObject,
    pub hsa_executable_freeze: FnHsaExecutableFreeze,
    pub hsa_executable_get_symbol_by_name: FnHsaExecutableGetSymbolByName,
    pub hsa_executable_symbol_get_info: FnHsaExecutableSymbolGetInfo,
    pub hsa_queue_create: FnHsaQueueCreate,
    pub hsa_queue_destroy: FnHsaQueueDestroy,
    pub hsa_signal_create: FnHsaSignalCreate,
    pub hsa_signal_destroy: FnHsaSignalDestroy,
    pub hsa_queue_add_write_index_relaxed: FnHsaQueueAddWriteIndexRelaxed,
    pub hsa_queue_load_read_index_relaxed: FnHsaQueueLoadReadIndexRelaxed,
    pub hsa_signal_store_relaxed: FnHsaSignalStoreRelaxed,
    pub hsa_signal_wait_scacquire: FnHsaSignalWaitScacquire,
}

/// Resolves `symbol` under each candidate in `names` (first hit wins) and reinterprets the
/// address as `F`. Same contract as `../driver.rs`'s `resolve`: `F` must be the exact
/// `extern "C" fn` type matching the documented HSA ABI for every name tried.
fn resolve<F>(lib: &Library, names: &[&'static str]) -> Result<F, HsaError> {
    for name in names {
        if let Some(ptr) = lib.symbol(name) {
            // SAFETY: `ptr` is a non-null address `dlsym` resolved for `name`, one of HSA's
            // real exported symbols; per this function's documented contract, `F` is a
            // function-pointer type (pointer-sized) whose signature the caller has matched to
            // `name`'s real C prototype.
            let f: F = unsafe { std::mem::transmute_copy(&ptr) };
            return Ok(f);
        }
    }
    Err(HsaError::SymbolNotFound(names[0]))
}

/// Turns a non-success `hsa_status_t` into an `HsaError::RuntimeCallFailed`, using
/// `hsa_status_string` for a human-readable message when the runtime exports it (it is part of
/// the Core API and always should be, but this crate treats it as best-effort like
/// `cuGetErrorString`'s alternate-name fallback in the CUDA loader).
pub(crate) fn check(fns: &FnTable, call: &'static str, code: HsaStatus) -> Result<(), HsaError> {
    if code == HSA_STATUS_SUCCESS {
        return Ok(());
    }
    let message = match fns.hsa_status_string {
        Some(status_string) => {
            let mut msg_ptr: *const c_char = std::ptr::null();
            // SAFETY: `status_string` was resolved against `hsa_status_string`, documented as
            // `hsa_status_t hsa_status_string(hsa_status_t status, const char **status_string)`:
            // it writes a pointer to a static, runtime-owned string into `*status_string` and
            // returns `HSA_STATUS_SUCCESS` on a recognized code. `msg_ptr` is a valid, aligned,
            // writable location for the call.
            let rc = unsafe { status_string(code, &mut msg_ptr) };
            if rc == HSA_STATUS_SUCCESS && !msg_ptr.is_null() {
                // SAFETY: `msg_ptr` was just confirmed non-null; per `hsa_status_string`'s
                // contract it points at a NUL-terminated, runtime-owned static string.
                unsafe { std::ffi::CStr::from_ptr(msg_ptr) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("unknown HSA status {code}")
            }
        }
        None => format!("HSA status {code} (hsa_status_string unavailable)"),
    };
    Err(HsaError::RuntimeCallFailed {
        call,
        code,
        message,
    })
}

/// Which kind of compute device an `HsaAgentInfo` describes, per `hsa_device_type_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsaDeviceType {
    Cpu,
    Gpu,
    Dsp,
    Other(HsaDeviceTypeRaw),
}

impl From<HsaDeviceTypeRaw> for HsaDeviceType {
    fn from(raw: HsaDeviceTypeRaw) -> Self {
        match raw {
            HSA_DEVICE_TYPE_CPU => HsaDeviceType::Cpu,
            HSA_DEVICE_TYPE_GPU => HsaDeviceType::Gpu,
            HSA_DEVICE_TYPE_DSP => HsaDeviceType::Dsp,
            other => HsaDeviceType::Other(other),
        }
    }
}

/// A compute agent enumerated via `hsa_iterate_agents`, with the two properties this crate
/// actually consults: its human-readable name and its device kind.
#[derive(Debug, Clone)]
pub struct HsaAgentInfo {
    pub agent: HsaAgent,
    pub name: String,
    pub device_type: HsaDeviceType,
}

extern "C" fn collect_agent_cb(agent: HsaAgent, data: *mut c_void) -> HsaStatus {
    // SAFETY: `data` is always a `&mut Vec<HsaAgent>` passed by `HsaRuntime::raw_agents`, which
    // keeps that vector alive on its own stack for the whole `hsa_iterate_agents` call — the
    // only place this callback is ever invoked from.
    let agents = unsafe { &mut *data.cast::<Vec<HsaAgent>>() };
    agents.push(agent);
    HSA_STATUS_SUCCESS
}

struct RegionEntry {
    region: HsaRegion,
    segment: HsaRegionInfoAttr,
    flags: u32,
}

struct RegionCollectCtx {
    get_info: FnHsaRegionGetInfo,
    regions: Vec<RegionEntry>,
}

extern "C" fn collect_region_cb(region: HsaRegion, data: *mut c_void) -> HsaStatus {
    // SAFETY: `data` is always a `&mut RegionCollectCtx` passed by `HsaRuntime::regions`, kept
    // alive on that function's stack for the whole `hsa_agent_iterate_regions` call.
    let ctx = unsafe { &mut *data.cast::<RegionCollectCtx>() };
    let mut segment: HsaRegionInfoAttr = 0;
    // SAFETY: matches `hsa_region_get_info(hsa_region_t, hsa_region_info_t, void*)`, called
    // through a function pointer resolved from the same live library `region` was produced by.
    let rc = unsafe {
        (ctx.get_info)(
            region,
            HSA_REGION_INFO_SEGMENT,
            (&mut segment as *mut HsaRegionInfoAttr).cast(),
        )
    };
    if rc != HSA_STATUS_SUCCESS {
        return rc;
    }
    let mut flags: u32 = 0;
    // SAFETY: same contract as the call above, different attribute.
    let rc = unsafe {
        (ctx.get_info)(
            region,
            HSA_REGION_INFO_GLOBAL_FLAGS,
            (&mut flags as *mut u32).cast(),
        )
    };
    if rc != HSA_STATUS_SUCCESS {
        return rc;
    }
    ctx.regions.push(RegionEntry {
        region,
        segment,
        flags,
    });
    HSA_STATUS_SUCCESS
}

/// A loaded HSA runtime: an open `libhsa-runtime64.so` handle plus its resolved entry points.
/// `hsa_init` has already succeeded by the time `load()` returns.
pub struct HsaRuntime {
    _lib: Library,
    fns: FnTable,
}

impl HsaRuntime {
    /// `dlopen`s the HSA runtime, resolves every entry point this crate uses, then calls
    /// `hsa_init()` — required before any other HSA call in the process, exactly like the CUDA
    /// loader's `cuInit`.
    pub fn load() -> Result<HsaRuntime, HsaError> {
        let lib = Library::open_first(&["libhsa-runtime64.so.1", "libhsa-runtime64.so"])
            .map_err(HsaError::DriverNotFound)?;

        let hsa_init: FnHsaInit = resolve(&lib, &["hsa_init"])?;
        let fns = FnTable {
            hsa_shut_down: resolve(&lib, &["hsa_shut_down"])?,
            hsa_status_string: resolve(&lib, &["hsa_status_string"]).ok(),
            hsa_iterate_agents: resolve(&lib, &["hsa_iterate_agents"])?,
            hsa_agent_get_info: resolve(&lib, &["hsa_agent_get_info"])?,
            hsa_agent_iterate_regions: resolve(&lib, &["hsa_agent_iterate_regions"])?,
            hsa_region_get_info: resolve(&lib, &["hsa_region_get_info"])?,
            hsa_memory_allocate: resolve(&lib, &["hsa_memory_allocate"])?,
            hsa_memory_free: resolve(&lib, &["hsa_memory_free"])?,
            hsa_memory_copy: resolve(&lib, &["hsa_memory_copy"])?,
            hsa_code_object_reader_create_from_memory: resolve(
                &lib,
                &["hsa_code_object_reader_create_from_memory"],
            )?,
            hsa_code_object_reader_destroy: resolve(&lib, &["hsa_code_object_reader_destroy"])?,
            hsa_executable_create_alt: resolve(&lib, &["hsa_executable_create_alt"])?,
            hsa_executable_destroy: resolve(&lib, &["hsa_executable_destroy"])?,
            hsa_executable_load_agent_code_object: resolve(
                &lib,
                &["hsa_executable_load_agent_code_object"],
            )?,
            hsa_executable_freeze: resolve(&lib, &["hsa_executable_freeze"])?,
            hsa_executable_get_symbol_by_name: resolve(
                &lib,
                &["hsa_executable_get_symbol_by_name"],
            )?,
            hsa_executable_symbol_get_info: resolve(&lib, &["hsa_executable_symbol_get_info"])?,
            hsa_queue_create: resolve(&lib, &["hsa_queue_create"])?,
            hsa_queue_destroy: resolve(&lib, &["hsa_queue_destroy"])?,
            hsa_signal_create: resolve(&lib, &["hsa_signal_create"])?,
            hsa_signal_destroy: resolve(&lib, &["hsa_signal_destroy"])?,
            hsa_queue_add_write_index_relaxed: resolve(
                &lib,
                &["hsa_queue_add_write_index_relaxed"],
            )?,
            hsa_queue_load_read_index_relaxed: resolve(
                &lib,
                &["hsa_queue_load_read_index_relaxed"],
            )?,
            hsa_signal_store_relaxed: resolve(&lib, &["hsa_signal_store_relaxed"])?,
            hsa_signal_wait_scacquire: resolve(&lib, &["hsa_signal_wait_scacquire"])?,
        };

        // SAFETY: `hsa_init` was resolved against `hsa_init(void)`; it takes no arguments.
        let rc = unsafe { hsa_init() };
        check(&fns, "hsa_init", rc)?;

        Ok(HsaRuntime { _lib: lib, fns })
    }

    pub(crate) fn fns(&self) -> &FnTable {
        &self.fns
    }

    fn raw_agents(&self) -> Result<Vec<HsaAgent>, HsaError> {
        let mut agents: Vec<HsaAgent> = Vec::new();
        // SAFETY: matches `hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*),
        // void *data)`; `collect_agent_cb` is a real `extern "C" fn` matching that signature,
        // and `data` points at `agents`, live on this stack frame for the whole call.
        let rc = unsafe {
            (self.fns.hsa_iterate_agents)(
                collect_agent_cb,
                (&mut agents as *mut Vec<HsaAgent>).cast(),
            )
        };
        check(&self.fns, "hsa_iterate_agents", rc)?;
        Ok(agents)
    }

    fn agent_name(&self, agent: HsaAgent) -> Result<String, HsaError> {
        // `HSA_AGENT_INFO_NAME` writes into a fixed 64-byte `char[64]` buffer per the spec.
        let mut buf = [0u8; 64];
        // SAFETY: matches `hsa_agent_get_info(hsa_agent_t, hsa_agent_info_t, void*)`; `buf` is
        // exactly the 64-byte fixed-size buffer this attribute writes into.
        let rc = unsafe {
            (self.fns.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_NAME,
                buf.as_mut_ptr().cast::<c_void>(),
            )
        };
        check(&self.fns, "hsa_agent_get_info", rc)?;
        let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..nul]).into_owned())
    }

    fn agent_device_type(&self, agent: HsaAgent) -> Result<HsaDeviceType, HsaError> {
        let mut device: HsaDeviceTypeRaw = 0;
        // SAFETY: same contract as `agent_name`'s call, `HSA_AGENT_INFO_DEVICE` writes a
        // 4-byte `hsa_device_type_t` value.
        let rc = unsafe {
            (self.fns.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                (&mut device as *mut HsaDeviceTypeRaw).cast(),
            )
        };
        check(&self.fns, "hsa_agent_get_info", rc)?;
        Ok(HsaDeviceType::from(device))
    }

    /// Every compute agent the runtime knows about (CPU and GPU alike).
    pub fn agents(&self) -> Result<Vec<HsaAgentInfo>, HsaError> {
        let mut out = Vec::new();
        for agent in self.raw_agents()? {
            out.push(HsaAgentInfo {
                agent,
                name: self.agent_name(agent)?,
                device_type: self.agent_device_type(agent)?,
            });
        }
        Ok(out)
    }

    /// Just the GPU agents, per `HSA_AGENT_INFO_DEVICE == HSA_DEVICE_TYPE_GPU` — the common
    /// case for a compiler that wants somewhere to run a HSACO image.
    pub fn gpu_agents(&self) -> Result<Vec<HsaAgentInfo>, HsaError> {
        Ok(self
            .agents()?
            .into_iter()
            .filter(|a| a.device_type == HsaDeviceType::Gpu)
            .collect())
    }

    fn regions(&self, agent: HsaAgent) -> Result<Vec<RegionEntry>, HsaError> {
        let mut ctx = RegionCollectCtx {
            get_info: self.fns.hsa_region_get_info,
            regions: Vec::new(),
        };
        // SAFETY: matches `hsa_agent_iterate_regions(hsa_agent_t, hsa_status_t
        // (*callback)(hsa_region_t, void*), void *data)`; `collect_region_cb` matches that
        // signature and `data` points at `ctx`, live for the whole call.
        let rc = unsafe {
            (self.fns.hsa_agent_iterate_regions)(
                agent,
                collect_region_cb,
                (&mut ctx as *mut RegionCollectCtx).cast(),
            )
        };
        check(&self.fns, "hsa_agent_iterate_regions", rc)?;
        Ok(ctx.regions)
    }

    /// The region to allocate a kernel's argument buffer in: a global region flagged
    /// kernarg-capable, required by `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE`'s
    /// own documented usage.
    pub fn kernarg_region(&self, agent: HsaAgent) -> Result<HsaRegion, HsaError> {
        self.regions(agent)?
            .into_iter()
            .find(|r| {
                r.segment == HSA_REGION_SEGMENT_GLOBAL
                    && (r.flags & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0
            })
            .map(|r| r.region)
            .ok_or(HsaError::RuntimeCallFailed {
                call: "hsa_agent_iterate_regions",
                code: -1,
                message: "no kernarg-capable global region found on this agent".to_string(),
            })
    }

    /// A general-purpose global region for device buffers, preferring coarse-grained (typical
    /// for dedicated VRAM), then fine-grained, then any global region at all — the same
    /// preference order real HSA programs apply when a specific grain isn't required.
    pub fn device_region(&self, agent: HsaAgent) -> Result<HsaRegion, HsaError> {
        let regions = self.regions(agent)?;
        let is_global = |r: &&RegionEntry| r.segment == HSA_REGION_SEGMENT_GLOBAL;
        regions
            .iter()
            .filter(is_global)
            .find(|r| (r.flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0)
            .or_else(|| {
                regions
                    .iter()
                    .filter(is_global)
                    .find(|r| (r.flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0)
            })
            .or_else(|| regions.iter().find(is_global))
            .map(|r| r.region)
            .ok_or(HsaError::RuntimeCallFailed {
                call: "hsa_agent_iterate_regions",
                code: -1,
                message: "no global memory region found on this agent".to_string(),
            })
    }

    /// Allocates `bytes` in `region` via `hsa_memory_allocate`.
    pub fn alloc(&self, region: HsaRegion, bytes: usize) -> Result<HsaBuffer<'_>, HsaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: matches `hsa_memory_allocate(hsa_region_t, size_t, void **ptr)`; `ptr` is a
        // valid, writable location for the duration of the call.
        let rc = unsafe { (self.fns.hsa_memory_allocate)(region, bytes, &mut ptr) };
        check(&self.fns, "hsa_memory_allocate", rc)?;
        Ok(HsaBuffer {
            runtime: self,
            ptr,
            len: bytes,
        })
    }

    /// Creates a command queue on `agent` with `size` packet slots (must be a power of two not
    /// exceeding the agent's `HSA_AGENT_INFO_QUEUE_MAX_SIZE`, per the spec — this crate does
    /// not second-guess the caller's choice, matching `cuLaunchKernel`'s own no-validation
    /// stance on caller-supplied dimensions).
    ///
    /// This is the ABEND-wiring hook: an `Arc<Mutex<Option<HsaError>>>` is threaded through as
    /// the callback's user data, so a real runtime-reported queue fault becomes a structured
    /// `HsaError` retrievable via `HsaQueue::last_fault` after the fact. See the module-level
    /// scope note in `mod.rs`.
    pub fn create_queue(&self, agent: HsaAgent, size: u32) -> Result<HsaQueue<'_>, HsaError> {
        let fault: Arc<Mutex<Option<HsaError>>> = Arc::new(Mutex::new(None));
        // One strong reference is handed to the runtime as opaque `data`; `HsaQueue` keeps the
        // other. The runtime's copy is reclaimed in `HsaQueue::drop` via `Arc::from_raw`.
        let callback_data_raw = Arc::into_raw(fault.clone()) as *mut c_void;

        let mut queue_ptr = std::ptr::null_mut();
        let callback: HsaQueueErrorCallback = queue_error_callback;
        // SAFETY: matches `hsa_queue_create(hsa_agent_t, uint32_t, hsa_queue_type32_t,
        // void(*)(hsa_status_t, hsa_queue_t*, void*), void*, uint32_t, uint32_t,
        // hsa_queue_t**)`. `callback` is a real `extern "C" fn` of the documented signature;
        // `callback_data_raw` is a live `Arc` pointer that outlives this queue (reclaimed only
        // in `Drop`); `u32::MAX` for both segment-size arguments asks the runtime for its own
        // default, per the spec's documented meaning of that sentinel; `queue_ptr` is a valid,
        // writable out-pointer.
        let rc = unsafe {
            (self.fns.hsa_queue_create)(
                agent,
                size,
                crate::hsa::ffi::HSA_QUEUE_TYPE_MULTI,
                Some(callback),
                callback_data_raw,
                u32::MAX,
                u32::MAX,
                &mut queue_ptr,
            )
        };
        if let Err(err) = check(&self.fns, "hsa_queue_create", rc) {
            // SAFETY: `callback_data_raw` was produced by `Arc::into_raw` immediately above and
            // the runtime never took ownership of it (the call failed before installing the
            // callback), so reclaiming it here is the only owner drop that will ever happen.
            unsafe {
                drop(Arc::from_raw(
                    callback_data_raw as *const Mutex<Option<HsaError>>,
                ));
            }
            return Err(err);
        }

        Ok(HsaQueue::new(
            self,
            queue_ptr,
            agent,
            fault,
            callback_data_raw,
        ))
    }
}

impl Drop for HsaRuntime {
    fn drop(&mut self) {
        // SAFETY: `hsa_shut_down` takes no arguments; every `HsaQueue`/`HsaExecutable`/
        // `HsaBuffer` borrows `&'a HsaRuntime`, so none can outlive this `drop` per the borrow
        // checker — by the time this runs, nothing else in the process still expects HSA calls
        // to succeed. The return code is intentionally discarded, matching this crate's CUDA
        // loader's `Drop` impls (a `Drop` cannot propagate a `Result`).
        unsafe {
            let _ = (self.fns.hsa_shut_down)();
        }
    }
}

/// A host- or device-visible allocation from an HSA region. Mirrors `../context.rs`'s
/// `DeviceBuffer` closely: bounds-checked host copies, a raw pointer exposed for building a
/// kernarg buffer or AQL packet field.
pub struct HsaBuffer<'a> {
    runtime: &'a HsaRuntime,
    ptr: *mut c_void,
    len: usize,
}

impl<'a> HsaBuffer<'a> {
    /// Copies `src` into this buffer via `hsa_memory_copy`. `src` must fit within the buffer's
    /// allocated length.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<(), HsaError> {
        if src.len() > self.len {
            return Err(HsaError::RuntimeCallFailed {
                call: "hsa_memory_copy",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds HSA buffer of {} bytes",
                    src.len(),
                    self.len
                ),
            });
        }
        // SAFETY: matches `hsa_memory_copy(void *dst, const void *src, size_t size)`;
        // `src.as_ptr()` is valid for `src.len()` bytes (guaranteed by the slice), and
        // `src.len() <= self.len`, the size `hsa_memory_allocate` was asked for.
        let rc = unsafe {
            (self.runtime.fns.hsa_memory_copy)(self.ptr, src.as_ptr().cast::<c_void>(), src.len())
        };
        check(&self.runtime.fns, "hsa_memory_copy", rc)
    }

    /// Copies from this buffer into `dst` via `hsa_memory_copy`. `dst` must fit within the
    /// buffer's allocated length.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), HsaError> {
        if dst.len() > self.len {
            return Err(HsaError::RuntimeCallFailed {
                call: "hsa_memory_copy",
                code: -1,
                message: format!(
                    "host slice of {} bytes exceeds HSA buffer of {} bytes",
                    dst.len(),
                    self.len
                ),
            });
        }
        // SAFETY: matches `hsa_memory_copy(void *dst, const void *src, size_t size)`;
        // `dst.as_mut_ptr()` is valid for `dst.len()` writable bytes, and `dst.len() <=
        // self.len`, so the read from `self.ptr` stays within the allocated range.
        let rc = unsafe {
            (self.runtime.fns.hsa_memory_copy)(
                dst.as_mut_ptr().cast::<c_void>(),
                self.ptr as *const c_void,
                dst.len(),
            )
        };
        check(&self.runtime.fns, "hsa_memory_copy", rc)
    }

    /// The raw pointer, for building a kernarg buffer or AQL packet field directly (HSA
    /// addresses are real host-mapped pointers, unlike CUDA's opaque `CUdeviceptr` integer).
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> Drop for HsaBuffer<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was produced by a successful `hsa_memory_allocate` above and freed
        // at most once (`Drop` runs exactly once). The return code is discarded for the same
        // reason as every other `Drop` impl in this crate.
        unsafe {
            let _ = (self.runtime.fns.hsa_memory_free)(self.ptr);
        }
    }
}
