//! Mapping a PE image into a macOS process: no Windows loader, so we do
//! what it does. The image is laid out at its preferred base if free (else
//! anywhere), relocated, and given its final page protections by
//! [`NativeImage::finalize`] once imports are bound.
//!
//! Nothing here runs title code and nothing is x86-64 specific, so an arm64
//! host (the recompiler's) maps images the same way.

use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use victory_pe::layout::{self, dir};
use victory_pe::{ExportInfo, ImageInfo, ImportSymbol};
use victory_platform::mem::{self, Protect};

use super::seh;

struct MappedMeta {
    base: usize,
    has_tls: bool,
    exports: Vec<ExportInfo>,
}

static MAPPED: Mutex<Vec<MappedMeta>> = Mutex::new(Vec::new());

fn mapped() -> std::sync::MutexGuard<'static, Vec<MappedMeta>> {
    MAPPED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `base` is a currently mapped [`NativeImage`], and if so whether it
/// has a static TLS directory.
#[cfg(target_arch = "x86_64")]
pub(super) fn has_tls(base: usize) -> Option<bool> {
    mapped().iter().find(|m| m.base == base).map(|m| m.has_tls)
}

/// Export lookup result on a mapped [`NativeImage`].
pub(super) enum MappedExport {
    Address(usize),
    Forward(String),
    Missing,
}

/// Looks up `symbol` in the mapped [`NativeImage`] at `base`, or `None` if no
/// image is mapped at `base`.
pub(super) fn export_at(base: usize, symbol: &ImportSymbol) -> Option<MappedExport> {
    let guard = mapped();
    let meta = guard.iter().find(|m| m.base == base)?;
    let found = match symbol {
        ImportSymbol::Name { name, .. } => meta
            .exports
            .iter()
            .find(|e| e.name.as_deref() == Some(name.as_str())),
        ImportSymbol::Ordinal(o) => meta.exports.iter().find(|e| e.ordinal == *o),
    };
    Some(match found.map(|f| (f.rva, f.forward.as_deref())) {
        Some((Some(rva), _)) => MappedExport::Address(base + rva as usize),
        Some((None, Some(forward))) => MappedExport::Forward(forward.to_owned()),
        _ => MappedExport::Missing,
    })
}

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_MEM_READ: u32 = 0x4000_0000;
const SCN_MEM_WRITE: u32 = 0x8000_0000;
const REL_ABSOLUTE: u16 = 0;
const REL_DIR64: u16 = 10;
const REL_HIGHLOW: u16 = 3;

/// The image's static TLS, as its TLS directory describes it (addresses
/// relocated, so they point into the mapped image).
#[derive(Debug, Clone, Copy)]
pub struct TlsTemplate {
    /// Start of the initial data every thread's block begins with.
    pub start: usize,
    pub end: usize,
    pub zero_fill: usize,
    /// Where the image keeps its TLS index (`_tls_index`).
    pub index_at: usize,
}

impl TlsTemplate {
    /// Bytes of one thread's block.
    pub fn block_len(&self) -> usize {
        self.end
            .saturating_sub(self.start)
            .saturating_add(self.zero_fill)
    }
}

/// A title image mapped into this process.
#[derive(Debug)]
pub struct NativeImage {
    base: usize,
    len: usize,
    info: ImageInfo,
}

// SAFETY: the base is just an address; the image has no interior state.
unsafe impl Send for NativeImage {}
// SAFETY: as above.
unsafe impl Sync for NativeImage {}

impl NativeImage {
    /// Maps `bytes`, a PE32+ file. `base` forces the load address (to test
    /// relocation, or to place a DLL); by default the preferred base, then
    /// any free range if that is taken.
    pub fn map(bytes: &[u8], base: Option<usize>) -> Result<NativeImage> {
        let info = ImageInfo::parse(bytes)?;
        if info.size_of_image == 0 {
            bail!("image has zero SizeOfImage");
        }
        let len = (info.size_of_image as usize).next_multiple_of(mem::page_size());
        let preferred = info.image_base as usize;
        let at = base.unwrap_or(preferred);
        // SAFETY: a fresh allocation; it fails rather than replacing anything.
        let region = unsafe { mem::alloc(Some(at), len, Protect::READ_WRITE) }
            .or_else(|first| {
                if base.is_some() || !info.has_relocations {
                    return Err(first);
                }
                // SAFETY: as above, anywhere this time.
                unsafe { mem::alloc(None, len, Protect::READ_WRITE) }
            })
            .with_context(|| format!("mapping {len:#x} bytes of image at {at:#x}"))?;
        let image = NativeImage {
            base: region as usize,
            len,
            info,
        };
        if let Err(err) = image.copy_sections(bytes).and_then(|()| image.relocate(bytes)) {
            // SAFETY: fresh mapping not yet exposed to unwind tables or title code.
            let _ = unsafe { mem::release(image.base as *mut u8, image.len) };
            return Err(err);
        }
        let pdata = layout::data_directory_offset(bytes, dir::EXCEPTION).map(|at| {
            (
                layout::read_u32(bytes, at).unwrap_or(0),
                layout::read_u32(bytes, at + 4).unwrap_or(0),
            )
        });
        seh::unwind::add_image(image.base, image.len, pdata);
        {
            let mut guard = mapped();
            guard.retain(|m| m.base != image.base);
            guard.push(MappedMeta {
                base: image.base,
                has_tls: image.info.tls.is_some(),
                exports: image.info.exports.clone(),
            });
        }
        tracing::info!(
            base = format_args!("{:#x}", image.base),
            preferred = format_args!("{preferred:#x}"),
            "native image mapped"
        );
        Ok(image)
    }

    pub fn base(&self) -> usize {
        self.base
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Parsed from the original image (imports, entry point, TLS callbacks).
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    pub fn entry(&self) -> Option<usize> {
        (self.info.entry_rva != 0).then(|| self.base + self.info.entry_rva as usize)
    }

    /// Address of a named export.
    pub fn export(&self, name: &str) -> Option<usize> {
        let rva = self
            .info
            .exports
            .iter()
            .find(|e| e.name.as_deref() == Some(name))?
            .rva?;
        Some(self.base + rva as usize)
    }

    /// Address of an export, by name or ordinal.
    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match symbol {
            ImportSymbol::Name { name, .. } => self.export(name),
            ImportSymbol::Ordinal(o) => {
                let rva = self.info.exports.iter().find(|e| e.ordinal == *o)?.rva?;
                Some(self.base + rva as usize)
            }
        }
    }

    fn copy_sections(&self, bytes: &[u8]) -> Result<()> {
        let opt = layout::optional_header_offset(bytes).context("no optional header")?;
        // SizeOfHeaders is at offset 60 of the optional header.
        let headers = layout::read_u32(bytes, opt + 60).context("no SizeOfHeaders")? as usize;
        let copy = |rva: usize, from: usize, len: usize| -> Result<()> {
            let src = bytes
                .get(from..from.checked_add(len).context("section overflows")?)
                .with_context(|| format!("section data {from:#x}+{len:#x} past the file"))?;
            if rva.checked_add(len).is_none_or(|end| end > self.len) {
                bail!("section at {rva:#x}+{len:#x} past the image");
            }
            // SAFETY: inside our own writable mapping, checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), (self.base + rva) as *mut u8, len)
            };
            Ok(())
        };
        copy(0, 0, headers.min(bytes.len()))?;
        for s in &self.info.sections {
            let len = if s.virtual_size == 0 {
                s.file_size
            } else {
                s.file_size.min(s.virtual_size)
            };
            copy(s.rva as usize, s.file_offset as usize, len as usize)?;
        }
        Ok(())
    }

    /// Applies base relocations for the distance between the load address
    /// and the preferred base.
    fn relocate(&self, bytes: &[u8]) -> Result<()> {
        let delta = self.base.wrapping_sub(self.info.image_base as usize);
        if delta == 0 {
            return Ok(());
        }
        let at = layout::data_directory_offset(bytes, dir::BASERELOC).context("no directory")?;
        let rva = layout::read_u32(bytes, at).unwrap_or(0) as usize;
        let size = layout::read_u32(bytes, at + 4).unwrap_or(0) as usize;
        if rva == 0 || size == 0 {
            bail!("image is not at its preferred base and has no relocations");
        }
        let end = rva
            .checked_add(size)
            .filter(|&end| end <= self.len)
            .with_context(|| format!("relocation directory {rva:#x}+{size:#x} past the image"))?;
        let mut block = rva;
        while block.checked_add(8).is_some_and(|h| h <= end) {
            let page = self.checked_read::<u32>(block)? as usize;
            let block_size = self.checked_read::<u32>(block + 4)? as usize;
            if block_size < 8 {
                break;
            }
            let next = block
                .checked_add(block_size)
                .filter(|&next| next <= end)
                .with_context(|| {
                    format!("relocation block {block:#x}+{block_size:#x} past directory")
                })?;
            for i in 0..(block_size - 8) / 2 {
                let entry = self.checked_read::<u16>(block + 8 + i * 2)?;
                let target = page
                    .checked_add(usize::from(entry & 0xFFF))
                    .context("relocation target overflows")?;
                match entry >> 12 {
                    REL_ABSOLUTE => {}
                    REL_DIR64 => {
                        let v = self.checked_read::<u64>(target)?.wrapping_add(delta as u64);
                        self.checked_write(target, v)?;
                    }
                    REL_HIGHLOW => {
                        let v = self.checked_read::<u32>(target)?.wrapping_add(delta as u32);
                        self.checked_write(target, v)?;
                    }
                    other => bail!("unsupported relocation type {other}"),
                }
            }
            block = next;
        }
        Ok(())
    }

    fn checked_read<T: Copy>(&self, rva: usize) -> Result<T> {
        if rva.checked_add(size_of::<T>()).is_none_or(|end| end > self.len) {
            bail!("rva {rva:#x} outside the image");
        }
        // SAFETY: inside the mapping (checked above); PE data is unaligned.
        Ok(unsafe { ((self.base + rva) as *const T).read_unaligned() })
    }

    fn checked_write<T>(&self, rva: usize, value: T) -> Result<()> {
        if rva.checked_add(size_of::<T>()).is_none_or(|end| end > self.len) {
            bail!("rva {rva:#x} outside the image");
        }
        // SAFETY: inside our writable mapping (checked above).
        unsafe { ((self.base + rva) as *mut T).write_unaligned(value) };
        Ok(())
    }

    /// The image's static TLS, if it has any.
    pub fn tls(&self) -> Option<TlsTemplate> {
        let dir = self.info.tls.as_ref()?.directory_rva as usize;
        let start = self.checked_read::<u64>(dir).ok()? as usize;
        let end = self.checked_read::<u64>(dir + 8).ok()? as usize;
        let index_at = self.checked_read::<u64>(dir + 16).ok()? as usize;
        let zero_fill = self.checked_read::<u32>(dir + 32).ok()? as usize;
        (end >= start).then_some(TlsTemplate {
            start,
            end,
            index_at,
            zero_fill,
        })
    }

    /// Fills IAT slots: `(slot rva, address)` pairs.
    pub fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        for &(rva, address) in slots {
            self.checked_write(rva as usize, address as u64)
                .with_context(|| format!("IAT slot {rva:#x} outside the image"))?;
        }
        Ok(())
    }

    /// Sets each page's protection from the sections on it (the union where
    /// two share a page, as the host's page may be larger than the PE's).
    /// Call it once imports are bound; the headers become read-only.
    pub fn finalize(&self) -> Result<()> {
        let page = mem::page_size();
        let mut pages = vec![0u32; self.len / page];
        pages[0] |= SCN_MEM_READ; // headers
        for s in &self.info.sections {
            let start = s.rva as usize / page;
            let end = (s.rva as usize + s.virtual_size.max(s.file_size) as usize).div_ceil(page);
            let flags = s.characteristics & (SCN_MEM_EXECUTE | SCN_MEM_READ | SCN_MEM_WRITE);
            for p in pages.iter_mut().take(end.min(self.len / page)).skip(start) {
                *p |= flags;
            }
        }
        let protect_of = |f: u32| match (f & SCN_MEM_WRITE != 0, f & SCN_MEM_EXECUTE != 0) {
            (true, true) => Protect::READ_WRITE_EXECUTE,
            (true, false) => Protect::READ_WRITE,
            (false, true) => Protect::READ_EXECUTE,
            (false, false) => Protect::READ,
        };
        let mut i = 0;
        while i < pages.len() {
            let mut j = i + 1;
            while j < pages.len() && protect_of(pages[j]) == protect_of(pages[i]) {
                j += 1;
            }
            // SAFETY: a run of pages of the mapping this image owns.
            unsafe {
                mem::protect(
                    (self.base + i * page) as *mut u8,
                    (j - i) * page,
                    protect_of(pages[i]),
                )
                .with_context(|| format!("protecting image pages {i}..{j}"))?;
                mem::flush_icache((self.base + i * page) as *mut u8, (j - i) * page);
            }
            i = j;
        }
        Ok(())
    }

    /// Unmaps the image.
    ///
    /// # Safety
    /// No title code may run or hold pointers into it afterwards.
    pub unsafe fn unmap(self) {
        mapped().retain(|m| m.base != self.base);
        #[cfg(target_arch = "x86_64")]
        super::run::forget_image(self.base);
        seh::unwind::remove_image(self.base);
        // SAFETY: forwarded from the caller.
        unsafe { mem::release(self.base as *mut u8, self.len) }.ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_pe(size_of_image: u32, reloc_rva: u32, reloc_size: u32, reloc_body: &[u8]) -> Vec<u8> {
        let mut pe = vec![0u8; 0x400];
        let put16 = |d: &mut [u8], at: usize, v: u16| d[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let put32 = |d: &mut [u8], at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put64 = |d: &mut [u8], at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
        pe[0..2].copy_from_slice(b"MZ");
        put32(&mut pe, 0x3C, 0x40);
        pe[0x40..0x44].copy_from_slice(b"PE\0\0");
        put16(&mut pe, 0x44, victory_pe::MACHINE_AMD64);
        put16(&mut pe, 0x46, 1);
        put16(&mut pe, 0x54, 240);
        put16(&mut pe, 0x56, 0x2022);
        let opt = 0x58;
        put16(&mut pe, opt, 0x20B);
        put64(&mut pe, opt + 24, 0x1_8000_0000);
        put32(&mut pe, opt + 32, 0x1000);
        put32(&mut pe, opt + 36, 0x200);
        put32(&mut pe, opt + 56, size_of_image);
        put32(&mut pe, opt + 60, 0x200);
        put32(&mut pe, opt + 108, 16);
        put32(&mut pe, opt + 112 + dir::BASERELOC * 8, reloc_rva);
        put32(&mut pe, opt + 112 + dir::BASERELOC * 8 + 4, reloc_size);
        let sec = opt + 240;
        pe[sec..sec + 6].copy_from_slice(b".reloc");
        put32(&mut pe, sec + 8, 0x200);
        put32(&mut pe, sec + 12, 0x1000);
        put32(&mut pe, sec + 16, 0x200);
        put32(&mut pe, sec + 20, 0x200);
        put32(&mut pe, sec + 36, SCN_MEM_READ);
        pe[0x200..0x200 + reloc_body.len()].copy_from_slice(reloc_body);
        pe
    }

    #[test]
    fn malformed_relocations_return_error_and_release_mapping() {
        let len = 0x2000usize.next_multiple_of(mem::page_size());
        // Pick a free address so we can check whether a failed map leaked it.
        // Far above where the system places allocations made anywhere: a range
        // freed there stays free while the other tests allocate, so step 3
        // fails only if one of the failed maps below kept it (a range found
        // with `alloc(None)` is the next one handed out, and other threads
        // took it about 1 run in 3).
        let forced = (0..64)
            .map(|i| 0x5000_0000_0000 + i * 0x1_0000_0000)
            .find(|&at| {
                // SAFETY: a fresh allocation at a fixed address; it fails
                // rather than replacing anything.
                match unsafe { mem::alloc(Some(at), len, Protect::READ_WRITE) } {
                    // SAFETY: just allocated.
                    Ok(p) => unsafe { mem::release(p, len) }.is_ok(),
                    Err(_) => false,
                }
            })
            .expect("a free range far above the allocations");

        // 1. Relocation block whose size runs past its directory.
        let mut bad_block = Vec::new();
        bad_block.extend_from_slice(&0x1000u32.to_le_bytes());
        bad_block.extend_from_slice(&0x400u32.to_le_bytes());
        let pe = synthetic_pe(0x2000, 0x1000, 16, &bad_block);
        assert!(NativeImage::map(&pe, Some(forced)).is_err());

        // 2. Relocation target pointing past the end of the image.
        let mut bad_target = Vec::new();
        bad_target.extend_from_slice(&0x20000u32.to_le_bytes());
        bad_target.extend_from_slice(&10u32.to_le_bytes());
        bad_target.extend_from_slice(&((REL_DIR64 << 12) | 0x10).to_le_bytes());
        let pe = synthetic_pe(0x2000, 0x1000, 10, &bad_target);
        assert!(NativeImage::map(&pe, Some(forced)).is_err());

        // 3. The forced range must still be free (not leaked by either failure).
        let mut valid_reloc = Vec::new();
        valid_reloc.extend_from_slice(&0x1000u32.to_le_bytes());
        valid_reloc.extend_from_slice(&8u32.to_le_bytes());
        let pe = synthetic_pe(0x2000, 0x1000, 8, &valid_reloc);
        let ok = NativeImage::map(&pe, Some(forced)).expect("failed maps must release their range");
        assert_eq!(ok.base(), forced);
        // SAFETY: no title code ran.
        unsafe { ok.unmap() };

        // 4. Zero SizeOfImage is rejected before allocation/finalize.
        let zero = synthetic_pe(0, 0, 0, &[]);
        assert!(NativeImage::map(&zero, None).is_err());
    }

    #[test]
    fn tls_rejects_inverted_bounds_without_panicking() {
        let mut body = vec![0u8; 40];
        // start > end in IMAGE_TLS_DIRECTORY64.
        body[0..8].copy_from_slice(&0x1_8000_2000u64.to_le_bytes());
        body[8..16].copy_from_slice(&0x1_8000_1000u64.to_le_bytes());
        body[16..24].copy_from_slice(&0x1_8000_1040u64.to_le_bytes());
        let mut pe = synthetic_pe(0x2000, 0, 0, &body);
        let opt = 0x58;
        // Data directory 9 (TLS): RVA 0x1000, size 40.
        pe[opt + 112 + 9 * 8..opt + 112 + 9 * 8 + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        pe[opt + 112 + 9 * 8 + 4..opt + 112 + 9 * 8 + 8].copy_from_slice(&40u32.to_le_bytes());
        // Add a valid empty reloc block so map can place the image anywhere.
        let mut reloc = [0u8; 8];
        reloc[0..4].copy_from_slice(&0x1000u32.to_le_bytes());
        reloc[4..8].copy_from_slice(&8u32.to_le_bytes());
        pe[0x200 + 64..0x200 + 72].copy_from_slice(&reloc);
        pe[opt + 112 + dir::BASERELOC * 8..opt + 112 + dir::BASERELOC * 8 + 4]
            .copy_from_slice(&0x1040u32.to_le_bytes());
        pe[opt + 112 + dir::BASERELOC * 8 + 4..opt + 112 + dir::BASERELOC * 8 + 8]
            .copy_from_slice(&8u32.to_le_bytes());

        let image = NativeImage::map(&pe, None).unwrap();
        assert!(image.tls().is_none());
        // SAFETY: no title code ran.
        unsafe { image.unmap() };
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn disable_thread_calls_and_thread_detach_order_follow_windows() {
        use crate::native::{
            disable_thread_calls, register_dll_thread_notices, register_thread_callbacks,
            spawn_title_thread,
        };

        // Build a relocatable PE32+ image with code at RVA 0x1000 (`DllMain` at 0x1000, TLS
        // callback at 0x1040, optional static TLS directory at 0x1080).
        fn callback_pe(id: u8, with_tls: bool, log_addr: usize, count_addr: usize) -> Vec<u8> {
            let mut pe = vec![0u8; 0x600];
            let put16 = |d: &mut [u8], at: usize, v: u16| d[at..at + 2].copy_from_slice(&v.to_le_bytes());
            let put32 = |d: &mut [u8], at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
            let put64 = |d: &mut [u8], at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
            pe[0..2].copy_from_slice(b"MZ");
            put32(&mut pe, 0x3C, 0x40);
            pe[0x40..0x44].copy_from_slice(b"PE\0\0");
            put16(&mut pe, 0x44, victory_pe::MACHINE_AMD64);
            put16(&mut pe, 0x46, 1);
            put16(&mut pe, 0x54, 240);
            put16(&mut pe, 0x56, 0x2022);
            let opt = 0x58;
            put16(&mut pe, opt, 0x20B);
            put32(&mut pe, opt + 16, 0x1000); // AddressOfEntryPoint = 0x1000
            put64(&mut pe, opt + 24, 0x1_8000_0000);
            put32(&mut pe, opt + 32, 0x1000);
            put32(&mut pe, opt + 36, 0x200);
            put32(&mut pe, opt + 56, 0x2000);
            put32(&mut pe, opt + 60, 0x200);
            put32(&mut pe, opt + 108, 16);
            put32(&mut pe, opt + 112 + dir::BASERELOC * 8, 0x1100);
            put32(&mut pe, opt + 112 + dir::BASERELOC * 8 + 4, 8);
            if with_tls {
                put32(&mut pe, opt + 112 + 9 * 8, 0x1080);
                put32(&mut pe, opt + 112 + 9 * 8 + 4, 40);
            }
            let sec = opt + 240;
            pe[sec..sec + 5].copy_from_slice(b".text");
            put32(&mut pe, sec + 8, 0x400);
            put32(&mut pe, sec + 12, 0x1000);
            put32(&mut pe, sec + 16, 0x400);
            put32(&mut pe, sec + 20, 0x200);
            put32(&mut pe, sec + 36, SCN_MEM_READ | SCN_MEM_WRITE | SCN_MEM_EXECUTE);

            // Stub (`rcx = base, edx = reason`):
            //   mov r8, count_addr; mov r9, log_addr
            //   mov eax, 1; lock xadd [r8], eax
            //   shl edx, 16; or edx, (id << 8 | kind)
            //   mov [r9 + rax*4], edx; mov eax, 1; ret
            let emit_stub = |d: &mut [u8], at: usize, kind: u8| {
                let mut code = Vec::new();
                code.extend_from_slice(&[0x49, 0xB8]);
                code.extend_from_slice(&(count_addr as u64).to_le_bytes());
                code.extend_from_slice(&[0x49, 0xB9]);
                code.extend_from_slice(&(log_addr as u64).to_le_bytes());
                code.extend_from_slice(&[0xB8, 1, 0, 0, 0]); // mov eax, 1
                code.extend_from_slice(&[0xF0, 0x41, 0x0F, 0xC1, 0x00]); // lock xadd [r8], eax
                code.extend_from_slice(&[0xC1, 0xE2, 16]); // shl edx, 16
                code.extend_from_slice(&[0x81, 0xCA, kind, id, 0, 0]); // or edx, imm32
                code.extend_from_slice(&[0x41, 0x89, 0x14, 0x81]); // mov [r9 + rax*4], edx
                code.extend_from_slice(&[0xB8, 1, 0, 0, 0, 0xC3]); // mov eax, 1; ret
                d[at..at + code.len()].copy_from_slice(&code);
            };
            emit_stub(&mut pe, 0x200, 1); // DllMain (kind = 1) at RVA 0x1000
            emit_stub(&mut pe, 0x240, 2); // TLS callback (kind = 2) at RVA 0x1040

            if with_tls {
                // IMAGE_TLS_DIRECTORY64 at RVA 0x1080 (file 0x280), callback array at RVA 0x10B0 (file 0x2B0).
                let pref = 0x1_8000_0000u64;
                put64(&mut pe, 0x280, pref + 0x10C0); // StartAddressOfRawData
                put64(&mut pe, 0x288, pref + 0x10C4); // EndAddressOfRawData
                put64(&mut pe, 0x290, pref + 0x10C8); // AddressOfIndex
                put64(&mut pe, 0x298, pref + 0x10B0); // AddressOfCallBacks
                put64(&mut pe, 0x2B0, pref + 0x1040); // callback[0] = RVA 0x1040
                put64(&mut pe, 0x2B8, 0); // terminator
                // Relocate the 5 absolute VAs at RVA 0x1080..0x1098 and 0x10B0.
                put32(&mut pe, opt + 112 + dir::BASERELOC * 8 + 4, 20);
                put32(&mut pe, 0x300, 0x1000);
                put32(&mut pe, 0x304, 20);
                for (k, off) in [0x80u16, 0x88, 0x90, 0x98, 0xB0, 0].into_iter().enumerate() {
                    let entry = if off == 0 { 0 } else { (REL_DIR64 << 12) | off };
                    put16(&mut pe, 0x308 + k * 2, entry);
                }
            } else {
                put32(&mut pe, 0x300, 0x1000);
                put32(&mut pe, 0x304, 8);
            }
            pe
        }

        let log = Box::new([0u32; 64]);
        let count = Box::new(0u32);
        let log_addr = log.as_ptr() as usize;
        let count_addr = (&*count) as *const u32 as usize;

        assert_eq!(disable_thread_calls(0x1234), None);

        // DLL A (no static TLS): calls `disable_thread_calls` BEFORE `register_dll_thread_notices`,
        // just as a real DLL does inside `DllMain(DLL_PROCESS_ATTACH)`.
        let dll_a = NativeImage::map(&callback_pe(0xA, false, log_addr, count_addr), None).unwrap();
        dll_a.finalize().unwrap();
        assert_eq!(disable_thread_calls(dll_a.base()), Some(true));
        register_dll_thread_notices(&dll_a);

        // DLL B (has static TLS + TLS callback + DllMain): `disable_thread_calls` fails with `Some(false)`.
        let dll_b = NativeImage::map(&callback_pe(0xB, true, log_addr, count_addr), None).unwrap();
        dll_b.finalize().unwrap();
        assert_eq!(disable_thread_calls(dll_b.base()), Some(false));
        register_dll_thread_notices(&dll_b);

        // DLL C (no static TLS, keeps DllMain enabled) + separate TLS-callback-only image D after B.
        let dll_c = NativeImage::map(&callback_pe(0xC, false, log_addr, count_addr), None).unwrap();
        dll_c.finalize().unwrap();
        register_dll_thread_notices(&dll_c);

        let exe_d = NativeImage::map(&callback_pe(0xD, true, log_addr, count_addr), None).unwrap();
        exe_d.finalize().unwrap();
        register_thread_callbacks(&exe_d);

        extern "win64" fn thread_noop(_: usize) -> u32 {
            0
        }
        // SAFETY: `thread_noop` is a valid win64 entry point.
        let worker = unsafe {
            spawn_title_thread("notice-order-probe", 1 << 20, thread_noop as *const () as usize, 0, |_| {})
                .unwrap()
        };
        assert_eq!(worker.join(), 0);

        let total = unsafe { (count_addr as *const u32).read_volatile() } as usize;
        let events: Vec<(u32, u8, u8)> = log[..total]
            .iter()
            .map(|&w| (w >> 16, (w >> 8) as u8, w as u8))
            .filter(|&(_, id, _)| matches!(id, 0xA | 0xB | 0xC | 0xD))
            .collect();

        // DLL A was silenced before registration, so zero 0xA events fired;
        // attach (reason 2): B.tls(2), B.main(1), C.main(1), D.tls(2);
        // detach (reason 3): D.tls(2), C.main(1), B.main(1), B.tls(2) (reverse load order, DllMain before TLS).
        assert_eq!(
            events,
            vec![
                (2, 0xB, 2),
                (2, 0xB, 1),
                (2, 0xC, 1),
                (2, 0xD, 2),
                (3, 0xD, 2),
                (3, 0xC, 1),
                (3, 0xB, 1),
                (3, 0xB, 2),
            ]
        );

        // SAFETY: worker thread has exited; unmapping cleans `MAPPED`, `THREAD_NOTICES`, and `DISABLED`.
        unsafe {
            exe_d.unmap();
            dll_c.unmap();
            dll_b.unmap();
            dll_a.unmap();
        }
    }
}
