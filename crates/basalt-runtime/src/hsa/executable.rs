// A loaded HSA executable (the result of reading a HSACO image via a code object reader and
// loading it for a specific agent) and the kernels looked up within it. See `../context.rs`'s
// cross-resource drop-ordering note, which applies equally here.

use std::ffi::CString;

use crate::hsa::error::HsaError;
use crate::hsa::ffi::{
    HsaAgent, HsaCodeObjectReader, HsaExecutableHandle, HsaExecutableSymbol,
    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
    HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, HSA_PROFILE_BASE,
};
use crate::hsa::runtime::{check, HsaRuntime};

/// A kernel found within a loaded executable, carrying exactly what `HsaQueue::dispatch` needs
/// to build an AQL packet. Carries no `Drop`/borrow of its own — like `../module.rs`'s
/// `CudaFunction`, the runtime validates the embedded `kernel_object` handle against its own
/// live-executable table on every dispatch, not a raw pointer Basalt dereferences, so a stale
/// handle used after its executable is destroyed comes back as a dispatch-time runtime error,
/// not memory corruption.
#[derive(Debug, Clone, Copy)]
pub struct HsaKernel {
    pub(crate) kernel_object: u64,
    pub(crate) kernarg_segment_size: u32,
    pub(crate) group_segment_size: u32,
    pub(crate) private_segment_size: u32,
}

/// A loaded HSA executable, bound to the agent it was loaded for.
pub struct HsaExecutable<'a> {
    runtime: &'a HsaRuntime,
    executable: HsaExecutableHandle,
    agent: HsaAgent,
}

impl<'a> HsaExecutable<'a> {
    /// Wraps `hsaco_bytes` in a code object reader, creates an executable, loads the image for
    /// `agent`, and freezes it — the four-call sequence the HSA spec documents for going from a
    /// raw HSACO buffer to a dispatch-ready executable.
    pub fn load(
        runtime: &'a HsaRuntime,
        agent: HsaAgent,
        hsaco_bytes: &[u8],
    ) -> Result<HsaExecutable<'a>, HsaError> {
        let fns = runtime.fns();

        let mut reader = HsaCodeObjectReader { handle: 0 };
        // SAFETY: matches `hsa_code_object_reader_create_from_memory(const void*, size_t,
        // hsa_code_object_reader_t*)`; `hsaco_bytes` is a live, in-bounds slice for its own
        // length, kept alive across the call by the borrow in this function's signature.
        let rc = unsafe {
            (fns.hsa_code_object_reader_create_from_memory)(
                hsaco_bytes.as_ptr().cast(),
                hsaco_bytes.len(),
                &mut reader,
            )
        };
        check(fns, "hsa_code_object_reader_create_from_memory", rc)?;

        let mut executable = HsaExecutableHandle { handle: 0 };
        // SAFETY: matches `hsa_executable_create_alt(hsa_profile_t,
        // hsa_default_float_rounding_mode_t, const char*, hsa_executable_t*)`; a null options
        // pointer is the documented way to request the runtime's default options.
        let rc = unsafe {
            (fns.hsa_executable_create_alt)(
                HSA_PROFILE_BASE,
                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                std::ptr::null(),
                &mut executable,
            )
        };
        check(fns, "hsa_executable_create_alt", rc)?;

        let mut loaded = crate::hsa::ffi::HsaLoadedCodeObject { handle: 0 };
        // SAFETY: matches `hsa_executable_load_agent_code_object(hsa_executable_t, hsa_agent_t,
        // hsa_code_object_reader_t, const char*, hsa_loaded_code_object_t*)`; `executable` and
        // `reader` both came from the successful calls immediately above.
        let rc = unsafe {
            (fns.hsa_executable_load_agent_code_object)(
                executable,
                agent,
                reader,
                std::ptr::null(),
                &mut loaded,
            )
        };
        check(fns, "hsa_executable_load_agent_code_object", rc)?;

        // SAFETY: matches `hsa_executable_freeze(hsa_executable_t, const char*)`; must run
        // after every `hsa_executable_load_agent_code_object` call and before any
        // `hsa_executable_get_symbol_by_name` lookup, per the spec's documented executable
        // lifecycle.
        let rc = unsafe { (fns.hsa_executable_freeze)(executable, std::ptr::null()) };
        check(fns, "hsa_executable_freeze", rc)?;

        // SAFETY: matches `hsa_code_object_reader_destroy(hsa_code_object_reader_t)`; the
        // reader has already been fully consumed by the load call above and the spec does not
        // require it to outlive the executable.
        let rc = unsafe { (fns.hsa_code_object_reader_destroy)(reader) };
        check(fns, "hsa_code_object_reader_destroy", rc)?;

        Ok(HsaExecutable {
            runtime,
            executable,
            agent,
        })
    }

    /// Looks up a kernel by its exported symbol name via `hsa_executable_get_symbol_by_name`,
    /// then reads back everything `HsaQueue::dispatch` needs about it.
    pub fn get_kernel(&self, name: &str) -> Result<HsaKernel, HsaError> {
        let fns = self.runtime.fns();
        let cname = CString::new(name).map_err(|_| HsaError::RuntimeCallFailed {
            call: "hsa_executable_get_symbol_by_name",
            code: -1,
            message: "kernel name contains an interior NUL byte".to_string(),
        })?;

        let mut symbol = HsaExecutableSymbol { handle: 0 };
        // SAFETY: matches `hsa_executable_get_symbol_by_name(hsa_executable_t, const char*,
        // const hsa_agent_t*, hsa_executable_symbol_t*)`; `self.executable` came from a
        // successful `load()` and has not been destroyed (only this struct's own `Drop` does
        // that, which cannot run before this call returns since it needs `&self`); `cname` is
        // NUL-terminated and kept alive across the call; `&self.agent` is a valid pointer to
        // the agent this executable was loaded for.
        let rc = unsafe {
            (fns.hsa_executable_get_symbol_by_name)(
                self.executable,
                cname.as_ptr(),
                &self.agent,
                &mut symbol,
            )
        };
        check(fns, "hsa_executable_get_symbol_by_name", rc)?;

        let kernel_object =
            self.symbol_info_u64(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT)?;
        let kernarg_segment_size = self.symbol_info_u32(
            symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        )?;
        let group_segment_size =
            self.symbol_info_u32(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE)?;
        let private_segment_size = self.symbol_info_u32(
            symbol,
            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        )?;

        Ok(HsaKernel {
            kernel_object,
            kernarg_segment_size,
            group_segment_size,
            private_segment_size,
        })
    }

    fn symbol_info_u64(
        &self,
        symbol: HsaExecutableSymbol,
        attribute: crate::hsa::ffi::HsaExecutableSymbolInfoAttr,
    ) -> Result<u64, HsaError> {
        let mut value: u64 = 0;
        // SAFETY: matches `hsa_executable_symbol_get_info(hsa_executable_symbol_t,
        // hsa_executable_symbol_info_t, void*)`; `attribute` is documented to write an 8-byte
        // value for `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`, matching `value`'s width.
        let rc = unsafe {
            (self.runtime.fns().hsa_executable_symbol_get_info)(
                symbol,
                attribute,
                (&mut value as *mut u64).cast(),
            )
        };
        check(self.runtime.fns(), "hsa_executable_symbol_get_info", rc)?;
        Ok(value)
    }

    fn symbol_info_u32(
        &self,
        symbol: HsaExecutableSymbol,
        attribute: crate::hsa::ffi::HsaExecutableSymbolInfoAttr,
    ) -> Result<u32, HsaError> {
        let mut value: u32 = 0;
        // SAFETY: same contract as `symbol_info_u64`, but every other kernel attribute this
        // crate queries (kernarg/group/private segment size) is documented as a 4-byte value.
        let rc = unsafe {
            (self.runtime.fns().hsa_executable_symbol_get_info)(
                symbol,
                attribute,
                (&mut value as *mut u32).cast(),
            )
        };
        check(self.runtime.fns(), "hsa_executable_symbol_get_info", rc)?;
        Ok(value)
    }
}

impl<'a> Drop for HsaExecutable<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.executable` was produced by a successful `hsa_executable_create_alt`
        // and destroyed at most once. See `../context.rs`'s module-level note on
        // cross-resource drop ordering: any `HsaKernel` derived from this executable that
        // outlives it will see its next dispatch fail with a handle-validation error, not a
        // crash. The return code is discarded, matching every other `Drop` impl in this crate.
        unsafe {
            let _ = (self.runtime.fns().hsa_executable_destroy)(self.executable);
        }
    }
}
