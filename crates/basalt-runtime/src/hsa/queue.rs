// A created HSA command queue and the manual AQL dispatch mechanism built on top of it. There
// is no `cuLaunchKernel` equivalent in HSA: dispatching a kernel means writing a
// `HsaKernelDispatchPacket` into the queue's own ring buffer at an index the queue hands out,
// then ringing its doorbell signal. See `mod.rs` for the module-wide overview and the
// ABEND-wiring scope note this file's error callback implements.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use crate::hsa::error::HsaError;
use crate::hsa::executable::HsaKernel;
use crate::hsa::ffi::{
    build_kernel_dispatch_header, HsaAgent, HsaKernelDispatchPacket, HsaQueueRaw, HsaSignal,
    HsaStatus, HSA_SIGNAL_CONDITION_LT, HSA_WAIT_STATE_BLOCKED,
};
use crate::hsa::runtime::{check, HsaBuffer, HsaRuntime};

/// The runtime's own queue-level asynchronous error callback (see `../runtime.rs`'s
/// `create_queue`, which installs this). Turns HSA's own status code into this crate's
/// `HsaError`, stashed in the `Arc<Mutex<Option<HsaError>>>` a caller can inspect afterward via
/// `HsaQueue::last_fault`.
///
/// Scope: this is the whole of this crate's fault-handling story today — a real callback wired
/// to a real, structured `HsaError`. It does not attempt to correlate the fault against a
/// tracked allocation or capture a dispatch snapshot; that fuller diagnostic system is separate,
/// later work (see `mod.rs`).
pub(crate) extern "C" fn queue_error_callback(
    status: HsaStatus,
    _source: *mut HsaQueueRaw,
    data: *mut c_void,
) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` was produced by `Arc::into_raw` on an `Arc<Mutex<Option<HsaError>>>` in
    // `HsaRuntime::create_queue` and handed to `hsa_queue_create` as this queue's own callback
    // data; it stays a valid `Arc`-owned allocation until `HsaQueue::drop` reclaims the extra
    // reference, which cannot race a callback invocation (both happen on calls tied to this
    // same queue, and the runtime does not invoke this callback after `hsa_queue_destroy`
    // returns). `Arc::from_raw` here does not take ownership away from the runtime's copy: the
    // reference count is restored with `mem::forget` below.
    let fault = unsafe { Arc::from_raw(data.cast::<Mutex<Option<HsaError>>>()) };
    let err = HsaError::RuntimeCallFailed {
        call: "hsa_queue_error_callback",
        code: status,
        message: format!("HSA runtime reported an asynchronous queue error (status {status})"),
    };
    if let Ok(mut slot) = fault.lock() {
        *slot = Some(err);
    }
    std::mem::forget(fault);
}

/// A created HSA command queue. Borrows `&'a HsaRuntime` for the same reason `CudaContext<'a>`
/// borrows `&'a CudaDriver` — see `../context.rs`'s module-level note on cross-resource drop
/// ordering, which applies identically here.
pub struct HsaQueue<'a> {
    runtime: &'a HsaRuntime,
    queue: *mut HsaQueueRaw,
    agent: HsaAgent,
    fault: Arc<Mutex<Option<HsaError>>>,
    callback_data_raw: *mut c_void,
}

impl<'a> HsaQueue<'a> {
    pub(crate) fn new(
        runtime: &'a HsaRuntime,
        queue: *mut HsaQueueRaw,
        agent: HsaAgent,
        fault: Arc<Mutex<Option<HsaError>>>,
        callback_data_raw: *mut c_void,
    ) -> Self {
        HsaQueue {
            runtime,
            queue,
            agent,
            fault,
            callback_data_raw,
        }
    }

    /// Takes and returns any fault the queue's error callback has recorded since the last time
    /// this was called. `None` means no asynchronous error has been reported — not a guarantee
    /// none occurred moments ago and hasn't been delivered yet, same caveat as any async
    /// callback-driven error channel.
    pub fn last_fault(&self) -> Option<HsaError> {
        self.fault.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Dispatches `kernel` with the given grid/workgroup dimensions and kernarg bytes, and
    /// blocks until it completes. Mirrors `CudaFunction::launch`'s fold-the-sync-in design (see
    /// `../module.rs`) — this crate's only consumers need the result before they can check
    /// anything, so there is no fire-and-forget path.
    ///
    /// `kernarg_bytes` is copied verbatim into a kernarg-region buffer sized to the kernel's own
    /// `kernarg_segment_size` (or the caller's byte count if larger); laying out those bytes to
    /// match the kernel's actual parameter list is the caller's responsibility, exactly like
    /// `cuLaunchKernel`'s parameter array.
    pub fn dispatch(
        &self,
        kernel: &HsaKernel,
        grid: (u32, u32, u32),
        workgroup: (u16, u16, u16),
        kernarg_bytes: &[u8],
    ) -> Result<(), HsaError> {
        let fns = self.runtime.fns();

        let kernarg_region = self.runtime.kernarg_region(self.agent)?;
        let kernarg_len = kernarg_bytes
            .len()
            .max(kernel.kernarg_segment_size as usize);
        let kernarg_buf = self.runtime.alloc(kernarg_region, kernarg_len.max(1))?;
        if !kernarg_bytes.is_empty() {
            kernarg_buf.copy_from_host(kernarg_bytes)?;
        }

        let mut completion_signal = HsaSignal { handle: 0 };
        // SAFETY: matches `hsa_signal_create(hsa_signal_value_t initial_value, uint32_t
        // num_consumers, const hsa_agent_t *consumers, hsa_signal_t *signal)`; zero consumers
        // with a null consumer list asks for a signal visible to every agent, which is what a
        // host-observed completion signal needs. `completion_signal` is a valid out-pointer.
        let rc = unsafe { (fns.hsa_signal_create)(1, 0, std::ptr::null(), &mut completion_signal) };
        check(fns, "hsa_signal_create", rc)?;

        let result =
            self.dispatch_with_signal(kernel, grid, workgroup, &kernarg_buf, completion_signal);

        // SAFETY: matches `hsa_signal_destroy(hsa_signal_t)`; `completion_signal` was created
        // above and is not touched again after this point regardless of `result`.
        let destroy_rc = unsafe { (fns.hsa_signal_destroy)(completion_signal) };

        result?;
        check(fns, "hsa_signal_destroy", destroy_rc)?;

        if let Some(fault) = self.last_fault() {
            return Err(fault);
        }
        Ok(())
    }

    fn dispatch_with_signal(
        &self,
        kernel: &HsaKernel,
        grid: (u32, u32, u32),
        workgroup: (u16, u16, u16),
        kernarg_buf: &HsaBuffer<'_>,
        completion_signal: HsaSignal,
    ) -> Result<(), HsaError> {
        let fns = self.runtime.fns();
        let dims: u16 = if grid.2 > 1 || workgroup.2 > 1 {
            3
        } else if grid.1 > 1 || workgroup.1 > 1 {
            2
        } else {
            1
        };

        // SAFETY: matches `uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t*,
        // uint64_t)`; `self.queue` came from a successful `hsa_queue_create` and is not
        // destroyed before `&self` goes away (only `HsaQueue::drop` destroys it).
        let index = unsafe { (fns.hsa_queue_add_write_index_relaxed)(self.queue, 1) };

        // SAFETY: `self.queue` is valid per the note above; `size` is a plain field read.
        let queue_size = unsafe { (*self.queue).size } as u64;

        // A real dispatcher must not overwrite a ring-buffer slot the hardware hasn't
        // consumed yet. This crate never pipelines more than one outstanding dispatch per
        // queue, so this only spins on a freshly-created queue that somehow already has a
        // packet in flight, which cannot happen through this crate's own API — kept anyway
        // because a caller-visible `HsaQueue` could in principle be shared and misused, and
        // spinning is the documented-correct response, not a bug to paper over.
        loop {
            // SAFETY: same pointer-validity contract as the write-index call above.
            let read_index = unsafe { (fns.hsa_queue_load_read_index_relaxed)(self.queue) };
            if index - read_index < queue_size {
                break;
            }
            std::hint::spin_loop();
        }

        let slot = (index % queue_size) as usize;
        // SAFETY: `base_address` was populated by `hsa_queue_create` and points at a ring
        // buffer of `queue_size` packets, each `size_of::<HsaKernelDispatchPacket>()` (64)
        // bytes wide, matching the AQL spec; `slot < queue_size` per the wait loop above, so
        // this stays in bounds.
        let packet_ptr = unsafe {
            (*self.queue)
                .base_address
                .cast::<HsaKernelDispatchPacket>()
                .add(slot)
        };

        // SAFETY: `packet_ptr` is exclusively ours to write until the doorbell ring below
        // hands the slot to the hardware; every field except `header` is written here, with
        // `header` stored last (below) using release ordering, so the hardware never observes
        // a partially-initialized packet — the AQL spec's documented write protocol.
        unsafe {
            (*packet_ptr).setup = dims;
            (*packet_ptr).workgroup_size_x = workgroup.0;
            (*packet_ptr).workgroup_size_y = workgroup.1;
            (*packet_ptr).workgroup_size_z = workgroup.2;
            (*packet_ptr).reserved0 = 0;
            (*packet_ptr).grid_size_x = grid.0;
            (*packet_ptr).grid_size_y = grid.1;
            (*packet_ptr).grid_size_z = grid.2;
            (*packet_ptr).private_segment_size = kernel.private_segment_size;
            (*packet_ptr).group_segment_size = kernel.group_segment_size;
            (*packet_ptr).kernel_object = kernel.kernel_object;
            (*packet_ptr).kernarg_address = kernarg_buf.device_ptr();
            (*packet_ptr).reserved2 = 0;
            (*packet_ptr).completion_signal = completion_signal;
        }

        // SAFETY: `HsaKernelDispatchPacket`'s first field is `header: u16`, so reinterpreting
        // its address as `*const AtomicU16` and storing through it is a same-size,
        // same-alignment reinterpretation of the exact bytes the plain writes above would have
        // touched — the standard way to perform the release-ordered header store the AQL
        // protocol requires (real dispatch code, e.g. ROCr's own samples, does the same).
        unsafe {
            let header_atomic = packet_ptr.cast::<AtomicU16>();
            (*header_atomic).store(build_kernel_dispatch_header(), Ordering::Release);
        }

        // SAFETY: matches `void hsa_signal_store_relaxed(hsa_signal_t, hsa_signal_value_t)`;
        // rings the queue's doorbell with the index of the packet just written, the documented
        // way to notify the hardware a new packet is ready.
        unsafe {
            (fns.hsa_signal_store_relaxed)((*self.queue).doorbell_signal, index as i64);
        }

        // SAFETY: matches `hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t,
        // hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t)`; blocks
        // until the completion signal (initialized to 1 in `dispatch`) is decremented below 1
        // by the hardware when the dispatch finishes, i.e. reaches 0. `u64::MAX` requests an
        // unbounded wait, per the spec's documented meaning of that sentinel.
        let _ = unsafe {
            (fns.hsa_signal_wait_scacquire)(
                completion_signal,
                HSA_SIGNAL_CONDITION_LT,
                1,
                u64::MAX,
                HSA_WAIT_STATE_BLOCKED,
            )
        };

        Ok(())
    }
}

impl<'a> Drop for HsaQueue<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.queue` was produced by a successful `hsa_queue_create` and destroyed at
        // most once (`Drop` runs exactly once); the return code is discarded, matching every
        // other `Drop` impl in this crate.
        unsafe {
            let _ = (self.runtime.fns().hsa_queue_destroy)(self.queue);
        }
        // SAFETY: `self.callback_data_raw` is the pointer handed to `hsa_queue_create` as the
        // error callback's `data`, originally produced by `Arc::into_raw`; the queue has just
        // been destroyed above so the runtime can no longer invoke the callback with it, making
        // this the single point where that extra reference is reclaimed.
        unsafe {
            drop(Arc::from_raw(
                self.callback_data_raw.cast::<Mutex<Option<HsaError>>>(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// Structural self-consistency check for the AQL packet layout: no test here talks to real
    /// hardware, but the byte size and every field's offset must match what the HSA spec
    /// documents, since `dispatch_with_signal` writes this struct directly into a queue's ring
    /// buffer with no serialization step to catch a layout mistake.
    #[test]
    fn kernel_dispatch_packet_matches_the_documented_aql_layout() {
        assert_eq!(
            size_of::<HsaKernelDispatchPacket>(),
            64,
            "AQL kernel-dispatch packets are always exactly 64 bytes"
        );
        assert_eq!(align_of::<HsaKernelDispatchPacket>(), 8);

        let base = std::mem::MaybeUninit::<HsaKernelDispatchPacket>::uninit();
        let base_ptr = base.as_ptr();
        // SAFETY: no field is read, only address arithmetic on an uninitialized-but-allocated
        // local; `offset_from` on raw pointers derived from the same allocation is well-defined
        // regardless of the pointee's init state.
        let offset_of = |field_ptr: *const u8| -> usize {
            unsafe { field_ptr.offset_from(base_ptr.cast::<u8>()) as usize }
        };

        // SAFETY: `addr_of!` never reads through the pointer, only computes a field address,
        // which is sound even though `base` is uninitialized.
        unsafe {
            assert_eq!(offset_of(std::ptr::addr_of!((*base_ptr).header).cast()), 0);
            assert_eq!(offset_of(std::ptr::addr_of!((*base_ptr).setup).cast()), 2);
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).workgroup_size_x).cast()),
                4
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).workgroup_size_y).cast()),
                6
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).workgroup_size_z).cast()),
                8
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).reserved0).cast()),
                10
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).grid_size_x).cast()),
                12
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).grid_size_y).cast()),
                16
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).grid_size_z).cast()),
                20
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).private_segment_size).cast()),
                24
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).group_segment_size).cast()),
                28
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).kernel_object).cast()),
                32
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).kernarg_address).cast()),
                40
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).reserved2).cast()),
                48
            );
            assert_eq!(
                offset_of(std::ptr::addr_of!((*base_ptr).completion_signal).cast()),
                56
            );
        }
    }

    #[test]
    fn header_encodes_kernel_dispatch_type_and_system_fences() {
        let header = build_kernel_dispatch_header();
        let packet_type = header & 0xff;
        assert_eq!(
            packet_type, 2,
            "low 8 bits of the header are HSA_PACKET_TYPE_KERNEL_DISPATCH"
        );
        let acquire_scope = (header >> 9) & 0x3;
        let release_scope = (header >> 11) & 0x3;
        assert_eq!(
            acquire_scope, 2,
            "acquire fence scoped to HSA_FENCE_SCOPE_SYSTEM"
        );
        assert_eq!(
            release_scope, 2,
            "release fence scoped to HSA_FENCE_SCOPE_SYSTEM"
        );
        let barrier_bit = (header >> 8) & 0x1;
        assert_eq!(barrier_bit, 0, "no barrier dependency on the prior packet");
    }
}
