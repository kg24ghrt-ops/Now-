use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

// Opaque handle types.
pub struct HwSession;
pub struct HwProfile;
pub struct HwCanvas;

// We'll use type aliases for the C API.
pub type HwSessionHandle = *mut HwSession;
pub type HwProfileHandle = *mut HwProfile;
pub type HwCanvasHandle = *mut HwCanvas;

/// Error codes returned by the API.
#[repr(C)]
pub enum HwError {
    HwSuccess = 0,
    HwErrorInvalidArgument = 1,
    HwErrorOutOfMemory = 2,
    HwErrorInternal = 3,
    HwErrorNotFound = 4,
}

/// Create a new session.
/// Returns a handle, or null on error.
#[no_mangle]
pub extern "C" fn hw_session_create() -> HwSessionHandle {
    // Placeholder: create a session from hw-core.
    // For now, return a dummy pointer.
    Box::into_raw(Box::new(HwSession))
}

/// Destroy a session.
#[no_mangle]
pub unsafe extern "C" fn hw_session_destroy(session: HwSessionHandle) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}

/// Load a profile for a specific script (e.g., "myanmar", "latin").
/// Returns a profile handle, or null on error.
#[no_mangle]
pub extern "C" fn hw_load_profile(script_name: *const c_char) -> HwProfileHandle {
    // Parse script name.
    if script_name.is_null() {
        return ptr::null_mut();
    }
    let cstr = unsafe { CStr::from_ptr(script_name) };
    let name = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    // For now, just return dummy.
    Box::into_raw(Box::new(HwProfile))
}

/// Feed text into the session.
#[no_mangle]
pub extern "C" fn hw_session_feed_text(
    session: HwSessionHandle,
    text: *const c_char,
) -> HwError {
    if session.is_null() || text.is_null() {
        return HwError::HwErrorInvalidArgument;
    }
    let cstr = unsafe { CStr::from_ptr(text) };
    let s = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return HwError::HwErrorInvalidArgument,
    };
    // Pass to hw-core.
    // For now, just print.
    println!("Received text: {}", s);
    HwError::HwSuccess
}

/// Render the current session to a canvas (canvas handle is opaque).
#[no_mangle]
pub extern "C" fn hw_render(
    session: HwSessionHandle,
    canvas: HwCanvasHandle,
) -> HwError {
    if session.is_null() || canvas.is_null() {
        return HwError::HwErrorInvalidArgument;
    }
    // Render using hw-core & hw-render-wgpu.
    HwError::HwSuccess
}

/// Export the rendered result to a bitmap (returns a pointer to raw RGBA data? We'll define later).