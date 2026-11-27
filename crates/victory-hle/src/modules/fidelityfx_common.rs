//! The platform-independent half of the FidelityFX interception (#375): the
//! SDK's descriptions and return codes, the backend link, and the shared
//! logging macro. `fidelityfx` serves the title's imports from these; the
//! Windows build forwards to a PC FidelityFX, the macOS build (M2) serves
//! the upscaler with MetalFX.

// Return codes.
pub(crate) const OK: u32 = 0;
pub(crate) const UNKNOWN_TYPE: u32 = 2;
pub(crate) const NO_PROVIDER: u32 = 4;
pub(crate) const PARAMETER: u32 = 6;

// Description types.
pub(crate) const CREATE_UPSCALE: u64 = 0x1_0000;
pub(crate) const DISPATCH_UPSCALE: u64 = 0x1_0001;
pub(crate) const CONSOLE_BACKEND: u64 = 0x100_0002;
pub(crate) const HOST_BACKEND: u64 = 2;
pub(crate) const GET_VERSIONS: u64 = 4;
pub(crate) const GET_PROVIDER_VERSION: u64 = 6;
pub(crate) const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Header {
    pub(crate) kind: u64,
    pub(crate) next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CreateUpscale {
    pub(crate) header: Header,
    pub(crate) flags: u32,
    pub(crate) max_render: [u32; 2],
    pub(crate) max_output: [u32; 2],
    pub(crate) message: usize,
}

/// The backend description: a device.
#[repr(C)]
pub(crate) struct Backend {
    pub(crate) header: Header,
    pub(crate) device: *mut c_void,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Resource {
    pub(crate) resource: *mut c_void,
    pub(crate) description: [u32; 8],
    pub(crate) state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DispatchUpscale {
    pub(crate) header: Header,
    pub(crate) list: *mut c_void,
    pub(crate) resources: [Resource; 7],
    pub(crate) rest: [u32; 18],
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);


/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
pub(crate) struct GetVersions {
    pub(crate) header: Header,
    pub(crate) create_kind: u64,
    pub(crate) device: *mut c_void,
    pub(crate) count: *mut u64,
    pub(crate) ids: *mut u64,
    pub(crate) names: *mut *const c_char,
}

#[repr(C)]
pub(crate) struct ProviderVersion {
    pub(crate) header: Header,
    pub(crate) id: u64,
    pub(crate) name: *const c_char,
}

pub(crate) type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
pub(crate) type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
pub(crate) type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;


macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

use std::ffi::{c_char, c_void};

pub(crate) use once;

pub(crate) fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    let after = upscale.header.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    upscale.header.next = &raw mut backend.header;
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
pub(crate) fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

