//! C ABI boundary for the handwriting engine.
//! Provides opaque handles and error codes for foreign callers.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::Mutex;

// Bring in the core session.
use hw_core::Session;

// Opaque handle types.
pub struct HwSession;
pub struct HwProfile;
pub struct HwCanvas;

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
    HwErrorFontLoadFailed = 5,
    HwErrorShapingFailed = 6,
    HwErrorRenderFailed = 7,
}

// We need to store the actual Session in a global or in the opaque handle.
// We'll store it as a raw pointer in the HwSession handle.
// For simplicity, we'll use a boxed Session.

/// Create a new session with a given font file (raw bytes) and seed.
/// `font_data` must be a valid TrueType/OpenType font.
/// Returns a handle, or null on error.
#[no_mangle]
pub extern "C" fn hw_session_create(
    font_data: *const u8,
    font_len: usize,
    seed: u64,
) -> HwSessionHandle {
    if font_data.is_null() || font_len == 0 {
        return ptr::null_mut();
    }

    // Convert the raw pointer to a slice.
    let data = unsafe { std::slice::from_raw_parts(font_data, font_len) };

    // Create the session.
    let session = match Session::new(data, seed) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    // Box it and return as a raw pointer.
    let boxed = Box::new(session);
    Box::into_raw(boxed) as HwSessionHandle
}

/// Destroy a session.
#[no_mangle]
pub unsafe extern "C" fn hw_session_destroy(session: HwSessionHandle) {
    if !session.is_null() {
        // Reconstruct the Box and drop it.
        drop(Box::from_raw(session as *mut Session));
    }
}

/// Feed text into the session.
/// This replaces any previous text.
#[no_mangle]
pub extern "C" fn hw_session_feed_text(session: HwSessionHandle, text: *const c_char) -> HwError {
    if session.is_null() || text.is_null() {
        return HwError::HwErrorInvalidArgument;
    }

    let cstr = unsafe { CStr::from_ptr(text) };
    let s = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return HwError::HwErrorInvalidArgument,
    };

    // Get the session.
    let session_ref = unsafe { &mut *(session as *mut Session) };
    session_ref.feed_text(s);

    // Process the text to generate the mesh.
    if let Err(e) = session_ref.process() {
        eprintln!("Processing error: {}", e);
        return HwError::HwErrorShapingFailed;
    }

    HwError::HwSuccess
}

/// Set the ink color (RGBA, 0..1).
#[no_mangle]
pub extern "C" fn hw_session_set_ink_color(
    session: HwSessionHandle,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> HwError {
    if session.is_null() {
        return HwError::HwErrorInvalidArgument;
    }

    let session_ref = unsafe { &mut *(session as *mut Session) };
    session_ref.set_ink_color(r, g, b, a);
    HwError::HwSuccess
}

/// Set the wetness (0..1).
#[no_mangle]
pub extern "C" fn hw_session_set_wetness(session: HwSessionHandle, wetness: f32) -> HwError {
    if session.is_null() {
        return HwError::HwErrorInvalidArgument;
    }

    let session_ref = unsafe { &mut *(session as *mut Session) };
    session_ref.set_wetness(wetness);
    HwError::HwSuccess
}

/// Set the base stroke width in pixels.
#[no_mangle]
pub extern "C" fn hw_session_set_base_width(session: HwSessionHandle, width: f32) -> HwError {
    if session.is_null() {
        return HwError::HwErrorInvalidArgument;
    }

    let session_ref = unsafe { &mut *(session as *mut Session) };
    session_ref.set_base_width(width);
    HwError::HwSuccess
}

/// Render the current session to a bitmap.
/// `width` and `height` are the desired output dimensions in pixels.
/// The output is RGBA (8 bits per channel), row-major.
/// The caller must free the returned buffer with `hw_free_bitmap`.
#[no_mangle]
pub extern "C" fn hw_session_render_to_bitmap(
    session: HwSessionHandle,
    width: u32,
    height: u32,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> HwError {
    if session.is_null() || out_data.is_null() || out_len.is_null() {
        return HwError::HwErrorInvalidArgument;
    }

    if width == 0 || height == 0 {
        return HwError::HwErrorInvalidArgument;
    }

    let session_ref = unsafe { &mut *(session as *mut Session) };

    // Get the mesh.
    let mesh = match session_ref.mesh() {
        Some(m) => m,
        None => return HwError::HwErrorNotFound,
    };

    // Get the paper texture.
    let paper = session_ref.paper_texture();

    // Create the renderer.
    let renderer = match pollster::block_on(hw_render_wgpu::Renderer::new(width, height)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Renderer creation error: {}", e);
            return HwError::HwErrorRenderFailed;
        }
    };

    // Load the paper texture.
    if let Err(e) = renderer.load_paper_texture(paper) {
        eprintln!("Paper load error: {}", e);
        return HwError::HwErrorRenderFailed;
    }

    // Render the mesh.
    let ink_color = session_ref.ink_color();
    let wetness = session_ref.wetness();
    if let Err(e) = renderer.render_mesh(mesh, ink_color, wetness) {
        eprintln!("Render error: {}", e);
        return HwError::HwErrorRenderFailed;
    }

    // Read back the pixels.
    let pixels = match renderer.read_pixels() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Readback error: {}", e);
            return HwError::HwErrorRenderFailed;
        }
    };

    // Allocate a buffer for the caller.
    let len = pixels.len();
    let buffer = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(len, 1).unwrap()) };
    if buffer.is_null() {
        return HwError::HwErrorOutOfMemory;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), buffer, len);
        *out_data = buffer;
        *out_len = len;
    }

    HwError::HwSuccess
}

/// Free a bitmap buffer previously allocated by `hw_session_render_to_bitmap`.
#[no_mangle]
pub unsafe extern "C" fn hw_free_bitmap(data: *mut u8, len: usize) {
    if !data.is_null() {
        std::alloc::dealloc(data, std::alloc::Layout::from_size_align(len, 1).unwrap());
    }
}

// ===== Convenience functions =====

/// Load a font from a file path and create a session.
/// This is a convenience wrapper for hw_session_create.
#[no_mangle]
pub extern "C" fn hw_session_create_from_file(
    font_path: *const c_char,
    seed: u64,
) -> HwSessionHandle {
    if font_path.is_null() {
        return ptr::null_mut();
    }

    let cstr = unsafe { CStr::from_ptr(font_path) };
    let path = match cstr.to_str() {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    // Read the font file.
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return ptr::null_mut(),
    };

    // Create the session.
    let session = match Session::new(&data, seed) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let boxed = Box::new(session);
    Box::into_raw(boxed) as HwSessionHandle
}
