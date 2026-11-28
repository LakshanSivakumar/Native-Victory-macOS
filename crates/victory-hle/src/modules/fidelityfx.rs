//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`). Without it, or if it offers no
//! upscaler for the host GPU, this module stays out of the registry and the
//! title's own console DLLs run as before.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Only the upscaler: other
//! effects (frame generation) are refused.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeSet;
use std::ffi::{c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::Mutex;

use super::fidelityfx_common::{host_state, link};
use super::fidelityfx_common::*;

use std::ffi::CStr;
#[cfg(windows)]
use std::sync::OnceLock;
#[cfg(windows)]
use victory_gpu::object::Object;
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
#[cfg(windows)]
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
#[cfg(windows)]
const DLL: &str = "amd_fidelityfx_upscaler_dx12.dll";

#[cfg(windows)]
struct Host {
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
#[cfg(windows)]
unsafe impl Send for Host {}
// SAFETY: as above.
#[cfg(windows)]
unsafe impl Sync for Host {}

#[cfg(windows)]
static HOST: OnceLock<Host> = OnceLock::new();


/// The title's upscaler context under `VICTORY_METALFX=1`: what the create
/// description carries, before the scaler itself exists (M2 builds it from
/// the first dispatch, where the formats show up).
struct UpscalerConfig {
    render: [u32; 2],
    output: [u32; 2],
    flags: u32,
}

impl UpscalerConfig {
    /// What MetalFX requires before a scaler can be built at all. The
    /// device's own scale bounds are checked at creation (M2).
    fn validate(&self) -> Result<(), String> {
        let [rw, rh] = self.render;
        let [ow, oh] = self.output;
        if rw == 0 || rh == 0 || ow == 0 || oh == 0 {
            return Err("a size is zero".into());
        }
        if ow < rw {
            return Err("output smaller than render: MetalFX upscales".into());
        }
        let scale = ow as f64 / rw.max(1) as f64;
        if scale > 4.0 {
            return Err(format!("scale {scale:.2}x is beyond MetalFX's range"));
        }
        Ok(())
    }
}

/// The MetalFX contexts we handed the title, by address. The handles are
/// `Box<UpscalerConfig>` until M2 turns them into scalers.
static METALFX_CONTEXTS: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());

/// Whether the title's upscaler is served with MetalFX (`VICTORY_METALFX=1`).
fn metalfx_enabled() -> bool {
    std::env::var_os("VICTORY_METALFX").is_some_and(|v| v != "0")
}

fn metalfx_holds(context: *mut *mut c_void) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: the caller's context points at our handle.
    let handle = unsafe { *context } as usize;
    METALFX_CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&handle)
}

/// Creates the MetalFX context from the title's upscaler description.
///
/// # Safety
/// `desc` names an upscaler description; `context` is the title's out-slot.
unsafe fn metalfx_create(
    context: *mut *mut c_void,
    desc: *mut Header,
) -> u32 {
    // SAFETY: an upscaler description, by its type.
    let upscale = unsafe { desc.cast::<CreateUpscale>().read() };
    let config = UpscalerConfig {
        render: upscale.max_render,
        output: upscale.max_output,
        flags: upscale.flags,
    };
    if let Err(why) = config.validate() {
        tracing::error!("FidelityFX MetalFX: refused the upscaler: {why}");
        return PARAMETER;
    }
    // A real provider's context is a large structure and the title reads fields of
    // it past the description it gave us (a bare 24-byte config read past its end
    // faulted the run right after the first dispatch, image+0x55e56f8). Hand out a
    // zero-padded block of the same scale instead: reads past the config land on
    // zeros, the way the PC build's unused fields do.
    const CONTEXT_BYTES: usize = 0x1000;
    // SAFETY: a valid layout, and the block is freed in `metalfx_destroy` with the
    // same one.
    let handle = unsafe {
        std::alloc::alloc_zeroed(
            std::alloc::Layout::from_size_align(CONTEXT_BYTES, 16).unwrap(),
        )
    } as *mut UpscalerConfig;
    // SAFETY: the block is at least as large as the config.
    unsafe { handle.write(config) };
    // SAFETY: the title's out-slot.
    unsafe { context.write(handle.cast()) };
    METALFX_CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(handle as usize);
    let (render, output, flags) = unsafe {
        // SAFETY: the handle was just boxed from a valid config.
        let c = handle.as_ref().unwrap();
        (c.render, c.output, c.flags)
    };
    tracing::info!(
        render = ?render,
        output = ?output,
        flags = format_args!("{flags:#x}"),
        "FidelityFX: MetalFX upscaler context created (the scaler itself lands at the first dispatch, M2)"
    );
    OK
}

/// Destroys a MetalFX context; `None` when the handle is not ours.
///
/// # Safety
/// `context` points at a handle this module handed out or the title's own.
unsafe fn metalfx_destroy(context: *mut *mut c_void) -> Option<u32> {
    if context.is_null() {
        return None;
    }
    // SAFETY: as above.
    let handle = unsafe { *context } as usize;
    let removed = METALFX_CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&handle);
    if removed {
        // SAFETY: the handle came from `metalfx_create`, which allocated it with this
        // layout.
        unsafe {
            std::alloc::dealloc(
                handle as *mut u8,
                std::alloc::Layout::from_size_align(0x1000, 16).unwrap(),
            )
        };
        Some(OK)
    } else {
        None
    }
}

/// The MetalFX dispatch on macOS: the title's upscaling is recorded into its
/// own list through the native host (`native::metalfx_record`), which resolves
/// the dispatch's textures to host copies; the temporal scaler itself is built
/// from them at the first encode (`victory_metal::metalfx::encode_temporal`).
/// `None` when the dispatch is not ours.
///
/// # Safety
/// `context` and `desc` are the title's pointers; `desc` names a dispatch.
#[cfg(target_os = "macos")]
unsafe fn metalfx_dispatch(
    context: *mut *mut c_void,
    desc: *mut Header,
) -> Option<u32> {
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE || !metalfx_holds(context) {
        return None;
    }
    // SAFETY: a DispatchUpscale description, by its type.
    let d = unsafe { desc.cast::<DispatchUpscale>().read() };
    // The FidelityFX order is colour, depth, motion vectors, exposure,
    // reactive, transparency, output (`fidelityfx_common`'s comment); the
    // temporal scaler takes the first five and the output last.
    let textures: [victory_gpu::object::This; 7] = d
        .resources
        .map(|r| r.resource.cast::<victory_gpu::object::Object>().cast_const());
    // `rest` opens with the jitter offsets (`fidelityfx_common`'s comment:
    // "then jitter, motion-vector scale, sizes, ...").
    let jitter = [f32::from_bits(d.rest[0]), f32::from_bits(d.rest[1])];
    // The scaler's history starts at the context's first dispatch.
    static DISPATCHED: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
    let reset = DISPATCHED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(context as usize);
    // SAFETY: the title's list and texture objects, read per their ABI.
    if unsafe { victory_gpu::native::metalfx_record(d.list.cast(), context as usize, &textures, reset, jitter) } {
        once!(
            info,
            "FidelityFX MetalFX: the upscaling was recorded into the title's list"
        );
        Some(OK)
    } else {
        once!(
            warn,
            "FidelityFX MetalFX: a texture or the list did not resolve; the upscaling did not run"
        );
        Some(PARAMETER)
    }
}

/// The M1 dispatch path (Windows only): a MetalFX context's upscaling is
/// refused there, where the PC build serves instead. `None` when the dispatch
/// is not ours.
///
/// # Safety
/// `context` and `desc` are the title's pointers; `desc` names a dispatch.
#[cfg(windows)]
unsafe fn metalfx_dispatch_refusal(
    context: *mut *mut c_void,
    desc: *mut Header,
) -> Option<u32> {
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind == DISPATCH_UPSCALE && metalfx_holds(context) {
        once!(
            warn,
            "FidelityFX MetalFX: the scaler build lands at the first dispatch (M2); the upscaling did not run"
        );
        return Some(PARAMETER);
    }
    None
}

/// The provider the MetalFX path reports in version queries. Our own id:
/// create ignores the requested provider, and the real DLL's values should
/// be dumped on Windows (the PC `versions()` helper logs them) if the title
/// ever cross-checks.
const METALFX_PROVIDER_ID: u64 = 0x6d65_7461; // "meta"
const METALFX_PROVIDER_NAME: &CStr = c"MetalFX (victory)";

/// Answers the version negotiation the PC build answers on Windows. Without
/// this the title concludes there is no provider and never creates a
/// context. `None` when the query is not one of the version kinds.
///
/// # Safety
/// `desc` names a GetVersions or ProviderVersion description whose buffers
/// the title owns.
unsafe fn metalfx_query(desc: *mut Header) -> Option<u32> {
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    match kind {
        GET_VERSIONS => {
            // SAFETY: a GetVersions description, by its type.
            let gv = unsafe { desc.cast::<GetVersions>().read() };
            if gv.create_kind != CREATE_UPSCALE || gv.count.is_null() {
                return Some(PARAMETER);
            }
            // SAFETY: the title owns the count slot.
            unsafe { *gv.count = 1 };
            if !gv.ids.is_null() {
                // SAFETY: the title owns the ids array.
                unsafe { *gv.ids = METALFX_PROVIDER_ID };
            }
            if !gv.names.is_null() {
                // SAFETY: the title owns the names array.
                unsafe { *gv.names = METALFX_PROVIDER_NAME.as_ptr().cast() };
            }
            once!(
                info,
                id = format_args!("{METALFX_PROVIDER_ID:#x}"),
                "FidelityFX MetalFX: version query answered with one provider"
            );
            Some(OK)
        }
        GET_PROVIDER_VERSION => {
            // SAFETY: a ProviderVersion description, by its type.
            unsafe {
                (*desc.cast::<ProviderVersion>()).id = METALFX_PROVIDER_ID;
                (*desc.cast::<ProviderVersion>()).name = METALFX_PROVIDER_NAME.as_ptr().cast();
            }
            once!(info, "FidelityFX MetalFX: provider version answered");
            Some(OK)
        }
        _ => None,
    }
}

/// MetalFX path's configure: settings and debug callbacks are accepted and
/// ignored (the PC build forwards them to the DLL, which only logs).
fn metalfx_configure() -> u32 {
    once!(info, "FidelityFX MetalFX: configure accepted and ignored");
    OK
}

/// Serves the title's FidelityFX calls with the PC upscaler in `dir`
/// (`VICTORY_FIDELITYFX_DIR`) if it offers one for the host GPU; else the
/// title's own DLLs run. Call once, before the import registry is first
/// used: it holds this module only when serving.
#[cfg(windows)]
pub fn use_host_dir(dir: Option<&Path>) {
    let Some(dir) = dir else {
        tracing::info!(
            "no FidelityFX folder (VICTORY_FIDELITYFX_DIR): a title's own FidelityFX DLLs run"
        );
        return;
    };
    match load(dir) {
        Ok((host, versions)) => {
            tracing::info!(folder = %dir.display(), versions, "FidelityFX: the host's upscaler serves the title's");
            let _ = HOST.set(host);
        }
        Err(why) => tracing::warn!(
            folder = %dir.display(),
            "FidelityFX: {why}; the title's own DLLs run"
        ),
    }
}

/// Off-Windows hosts have no PC FidelityFX build to load; MetalFX
/// (`VICTORY_METALFX=1`) serves instead when it lands.
#[cfg(not(windows))]
pub fn use_host_dir(_dir: Option<&Path>) {
    tracing::info!("FidelityFX: no PC build on this host; MetalFX serves when VICTORY_METALFX=1");
}

/// Whether the host's FidelityFX serves the title's: with the PC build on
/// Windows, or with MetalFX where `VICTORY_METALFX=1` (macOS).
pub(crate) fn serving() -> bool {
    #[cfg(windows)]
    if HOST.get().is_some() {
        return true;
    }
    metalfx_enabled()
}

#[cfg(windows)]
fn load(dir: &Path) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(dir.join(DLL)).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!("no {DLL} there"));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{DLL} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{DLL} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query: std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?),
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            device,
        }
    };
    let versions = versions(&host)?;
    Ok((host, versions))
}

/// The upscaler versions the host DLL offers for the host GPU, by name; an
/// error if none.
#[cfg(windows)]
fn versions(host: &Host) -> Result<String, String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind: CREATE_UPSCALE,
        device: host.device.as_raw(),
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { (host.query)(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no upscaler for the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    query.names = names.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    unsafe { (host.query)(null_mut(), &mut query.header) };
    Ok(names
        .iter()
        .filter(|n| !n.is_null())
        // SAFETY: NUL-terminated names the DLL keeps.
        .map(|&n| unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", "))
}

/// Puts the host backend after the title's upscaler description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if metalfx_enabled() && kind == CREATE_UPSCALE {
        // SAFETY: the kind was just read from `desc`.
        return unsafe { metalfx_create(context, desc) };
    }
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    #[cfg(windows)]
    {
        // SAFETY: the title's pointers; `desc` names an upscaler description.
        if let Some(code) = unsafe { pc_create(context, desc, memory) } {
            return code;
        }
    }
    NO_PROVIDER
}

/// Forwards the title's upscaler to the host's PC FidelityFX build
/// (Windows only; the macOS provider is MetalFX).
///
/// # Safety
/// The title's pointers; `desc` names an upscaler description.
#[cfg(windows)]
unsafe fn pc_create(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> Option<u32> {
    let host = HOST.get()?;
    // SAFETY: an upscaler description, by its type.
    let mut upscale = unsafe { desc.cast::<CreateUpscale>().read() };
    let mut backend = Backend {
        header: Header {
            kind: HOST_BACKEND,
            next: null_mut(),
        },
        device: host.device.as_raw(),
    };
    link(&mut upscale, &mut backend);
    // SAFETY: a PC chain; the title's later descriptions live for the call.
    let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
    if code != OK {
        tracing::error!(code, render = ?upscale.max_render, output = ?upscale.max_output, flags = format_args!("{:#x}", upscale.flags), "FidelityFX: the host refused the title's upscaler");
        return Some(code);
    }
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    unsafe { (host.query)(context, &mut version.header) };
    // SAFETY: a NUL-terminated name the DLL keeps, if any.
    let name = (!version.name.is_null())
        .then(|| unsafe { CStr::from_ptr(version.name) }.to_string_lossy());
    tracing::info!(
        provider = name.as_deref().unwrap_or("?"),
        id = format_args!("{:#x}", version.id),
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        "FidelityFX: upscaler context on the host"
    );
    Some(OK)
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if metalfx_enabled() {
        // SAFETY: the title passes its context.
        if let Some(code) = unsafe { metalfx_destroy(context) } {
            return code;
        }
    }
    #[cfg(windows)]
    {
        // SAFETY: the title passes its context.
        if let Some(code) = unsafe { pc_destroy(context, memory) } {
            return code;
        }
    }
    NO_PROVIDER
}

/// Forwards to the host's PC build (Windows only).
///
/// # Safety
/// The title's pointers.
#[cfg(windows)]
unsafe fn pc_destroy(context: *mut *mut c_void, memory: *const c_void) -> Option<u32> {
    // SAFETY: the title passes its context.
    HOST.get().map(|h| unsafe { (h.destroy)(context, memory) })
}

/// Settings (key-value pairs, debug messages) name nothing console-specific.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if metalfx_enabled() {
        return metalfx_configure();
    }
    #[cfg(windows)]
    {
        // SAFETY: the title passes its context and descriptions.
        if let Some(code) = unsafe { pc_configure(context, desc) } {
            return code;
        }
    }
    NO_PROVIDER
}

/// Forwards to the host's PC build (Windows only).
///
/// # Safety
/// The title's pointers.
#[cfg(windows)]
unsafe fn pc_configure(context: *mut *mut c_void, desc: *mut Header) -> Option<u32> {
    // SAFETY: the title passes its context and descriptions.
    HOST.get().map(|h| unsafe { (h.configure)(context, desc) })
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if metalfx_enabled() {
        // SAFETY: the title's pointers.
        if let Some(code) = unsafe { metalfx_query(desc) } {
            return code;
        }
    }
    // These carry the console device.
    if kind == GET_VERSIONS || kind == MEMORY_USAGE_WITH_DEVICE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: a query naming a device was refused"
        );
        return PARAMETER;
    }
    #[cfg(windows)]
    {
        // SAFETY: the title passes its pointers.
        if let Some(code) = unsafe { pc_query(context, desc) } {
            return code;
        }
    }
    NO_PROVIDER
}

/// Forwards to the host's PC build (Windows only).
///
/// # Safety
/// The title's pointers.
#[cfg(windows)]
unsafe fn pc_query(context: *mut *mut c_void, desc: *mut Header) -> Option<u32> {
    let host = HOST.get()?;
    // SAFETY: as above.
    Some(unsafe { (host.query)(context, desc) })
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if metalfx_enabled() {
        #[cfg(target_os = "macos")]
        // SAFETY: the title's pointers; the kind was just read.
        if let Some(code) = unsafe { metalfx_dispatch(context, desc) } {
            return code;
        }
        #[cfg(windows)]
        // SAFETY: the title's pointers; the kind was just read.
        if let Some(code) = unsafe { metalfx_dispatch_refusal(context, desc) } {
            return code;
        }
    }
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    #[cfg(windows)]
    {
        // SAFETY: the title's pointers; `desc` names an upscaling description.
        if let Some(code) = unsafe { pc_dispatch(context, desc) } {
            return code;
        }
    }
    NO_PROVIDER
}

/// Forwards the upscaling to the host's PC FidelityFX build (Windows only).
///
/// # Safety
/// The title's pointers; `desc` names an upscaling description.
#[cfg(windows)]
unsafe fn pc_dispatch(context: *mut *mut c_void, desc: *mut Header) -> Option<u32> {
    let host = HOST.get()?;
    // SAFETY: an upscaling, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    let list = dispatch.list.cast::<Object>().cast_const();
    // Alive for the call (the list holds them too).
    let mut textures = Vec::new();
    for r in dispatch
        .resources
        .iter_mut()
        .filter(|r| !r.resource.is_null())
    {
        let wanted = host_state(r.state);
        let texture = wanted.and_then(|w| {
            victory_gpu::external::host_texture(list, r.resource.cast::<Object>(), w)
                .filter(|&(_, state)| state == w)
        });
        let Some((texture, _)) = texture else {
            once!(
                warn,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return Some(PARAMETER);
        };
        r.resource = texture.as_raw();
        textures.push(texture);
    }
    once!(info, formats = ?textures.iter().map(|t| {
        // SAFETY: a live texture.
        unsafe { t.GetDesc() }.Format.0
    }).collect::<Vec<_>>(), "FidelityFX: first upscaling, host formats of its textures");
    Some(
        victory_gpu::external::record(list, |host_list| {
            dispatch.list = host_list.as_raw();
            // SAFETY: the PC dispatch on host objects; the context is the host's.
            unsafe { (host.dispatch)(context, &raw mut dispatch.header) }
        })
        .unwrap_or(PARAMETER),
    )
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        // The loader answers the SDK's version queries; the upscaler DLL is
        // what the title's loader LoadLibraryWs next — on macOS MetalFX serves
        // both with the same five-function API (kernel.rs refused the xbox
        // upscaler as "outside the host policy" before this).
        dlls: &["amd_fidelityfx_loader_xbox.dll", "amd_fidelityfx_upscaler_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::fidelityfx_common::{host_state, link};

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[cfg(windows)]
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).err().unwrap().contains(DLL));
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }
}

#[cfg(test)]
mod metalfx_tests {
    use super::*;

    #[test]
    fn context_roundtrip_and_dispatch_refusal() {
        let mut ctx: *mut c_void = null_mut();
        let mut upscale = CreateUpscale {
            header: Header {
                kind: CREATE_UPSCALE,
                next: null_mut(),
            },
            flags: 1,
            max_render: [1280, 720],
            max_output: [2560, 1440],
            message: 0,
        };
        // SAFETY: a fabricated description and our own out-slot.
        assert_eq!(unsafe { metalfx_create(&mut ctx, &mut upscale.header) }, OK);
        let handle = ctx as usize;
        assert!(METALFX_CONTEXTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&handle));

        let mut dispatch = DispatchUpscale {
            header: Header {
                kind: DISPATCH_UPSCALE,
                next: null_mut(),
            },
            list: null_mut(),
            resources: [Resource {
                resource: null_mut(),
                description: [0; 8],
                state: 0,
            }; 7],
            rest: [0; 18],
        };
        let mut dc: *mut c_void = ctx;
        // SAFETY: our own context and description. A null list resolves to
        // nothing on either path, so the dispatch is refused either way.
        #[cfg(target_os = "macos")]
        assert_eq!(unsafe { metalfx_dispatch(&mut dc, &mut dispatch.header) }.unwrap(), PARAMETER);
        #[cfg(windows)]
        assert_eq!(unsafe { metalfx_dispatch_refusal(&mut dc, &mut dispatch.header) }.unwrap(), PARAMETER);

        // SAFETY: our own context.
        assert_eq!(unsafe { metalfx_destroy(&mut ctx) }, Some(OK));
        assert!(!METALFX_CONTEXTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&handle));
    }

    #[test]
    fn zero_sizes_are_refused() {
        let cfg = UpscalerConfig {
            render: [0, 0],
            output: [0, 0],
            flags: 0,
        };
        assert!(cfg.validate().is_err());
        let cfg = UpscalerConfig {
            render: [1280, 720],
            output: [640, 360],
            flags: 0,
        };
        assert!(cfg.validate().is_err());
    }
}

    #[test]
    fn metalfx_query_negotiates_one_provider() {
        let mut count = 0u64;
        let mut id = 0u64;
        let mut names: [*const c_char; 1] = [null_mut()];
        let mut gv = GetVersions {
            header: Header {
                kind: GET_VERSIONS,
                next: null_mut(),
            },
            create_kind: CREATE_UPSCALE,
            device: null_mut(),
            count: &raw mut count,
            ids: &raw mut id,
            names: names.as_mut_ptr(),
        };
        // SAFETY: our own description and buffers.
        assert_eq!(unsafe { metalfx_query(&raw mut gv.header) }.unwrap(), OK);
        assert_eq!(count, 1);
        assert_eq!(id, METALFX_PROVIDER_ID);
        assert!(!names[0].is_null());

        // A provider-version query gets the same id back.
        let mut pv = ProviderVersion {
            header: Header {
                kind: GET_PROVIDER_VERSION,
                next: null_mut(),
            },
            id: 0,
            name: null(),
        };
        // SAFETY: our own description.
        assert_eq!(unsafe { metalfx_query(&raw mut pv.header) }.unwrap(), OK);
        assert_eq!(pv.id, METALFX_PROVIDER_ID);
        assert!(!pv.name.is_null());
    }
