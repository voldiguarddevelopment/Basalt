// Hand-rolled `dlopen`/`dlsym`/`dlclose` bindings. A Rust binary on Linux already links
// against libc (and libc links against libdl, or exposes the dl* symbols directly on glibc
// >= 2.34), so these `extern "C"` declarations cost nothing extra at link time — no `libc`
// crate dependency needed for three functions.

use std::ffi::{c_char, c_int, c_void, CString};

const RTLD_NOW: c_int = 2;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
}

/// An open shared-library handle. `Drop` calls `dlclose`; nothing in this crate reaches back
/// into the library's memory after the handle is dropped (all resolved function pointers are
/// copied out as plain `usize`/fn-pointer values before the handle can go away).
pub struct Library {
    handle: *mut c_void,
}

impl Library {
    /// Tries each name in `names` in order, returning the first that `dlopen`s successfully.
    pub fn open_first(names: &[&str]) -> Result<Library, String> {
        let mut last_err = String::new();
        for name in names {
            let cname = CString::new(*name).expect("library name has no interior NUL");
            // SAFETY: `cname` is a valid NUL-terminated C string that outlives the call;
            // `dlopen` either returns a valid handle or null, both of which are checked below.
            let handle = unsafe { dlopen(cname.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                return Ok(Library { handle });
            }
            last_err = last_dlerror().unwrap_or_else(|| format!("dlopen({name}) failed"));
        }
        Err(last_err)
    }

    /// Resolves `symbol` to a raw pointer, or `None` if the library exposes no such symbol.
    /// None of the CUDA driver entry points this crate resolves are legitimately
    /// null-valued, so a null return is treated uniformly as "not found".
    pub fn symbol(&self, symbol: &str) -> Option<*mut c_void> {
        let cname = CString::new(symbol).expect("symbol name has no interior NUL");
        // SAFETY: `self.handle` is a live handle owned by this `Library` (checked non-null at
        // construction, never closed before this call since it requires `&self`); `cname` is
        // a valid NUL-terminated C string, live for the duration of the call.
        let ptr = unsafe { dlsym(self.handle, cname.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was returned by a successful `dlopen` in `open_first` and
            // has not been closed before (single owner, dropped once); no code retains raw
            // pointers into the library after this point (see the struct-level note above).
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

fn last_dlerror() -> Option<String> {
    // SAFETY: `dlerror` takes no arguments and returns either null or a pointer to a
    // NUL-terminated static string owned by the dynamic linker (valid until the next dl*
    // call on this thread); it is copied into an owned `String` immediately.
    let msg = unsafe { dlerror() };
    if msg.is_null() {
        return None;
    }
    // SAFETY: `msg` was just checked non-null and points at a NUL-terminated C string per
    // `dlerror`'s contract.
    let cstr = unsafe { std::ffi::CStr::from_ptr(msg) };
    Some(cstr.to_string_lossy().into_owned())
}
