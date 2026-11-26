//! The pool-offset contract: title memory at addresses a shader computes, on the native host.
//!
//! A translated shader reads a pointer it computed (`global_load`, `s_load` from a loaded address,
//! stores, atomics, BVH nodes) by looking its 40-bit title address up in a page table, the root SRV
//! `t0` in `victory_shader::emit::MEMORY_SPACE` (translator notes, "Title memory"): per 64 KiB title
//! page a resource-heap index of a raw buffer view and the low dword of the title address that view
//! starts at, so that the byte offset in the view is `address low dword - start`. DXIL has no way to
//! dereference a 64-bit address, and the converter's raw buffers are views `{GPU address, size}`
//! read from the descriptor heap, so this is also what Metal can express; the translator stays as
//! it is, and the host fills the table differently:
//!
//! * D3D12: views of host *copies* of title memory, refreshed per submission and copied back.
//! * Here: views of the pools themselves. A mapping `(title address, length, pool, offset)` of the
//!   pool table gives each of its 64 KiB pages `(pair of views of the pool buffer, start)`. There
//!   is one SRV/UAV pair of descriptors (two identical entries) per `(pool, 2 GiB chunk)`, a
//!   [`CHUNK`] plus a page long, because a view's size is 32 bits. The pairs live at fixed
//!   resource-heap indices just past the title's own entries of the heap the shader runs with
//!   (a heap is [`RESERVED`] entries longer than the title made it, and has a table of its own,
//!   whose view indices say where). Pair 0 is the null view (size 0, bounds-checked by the
//!   converter): what a page no pool shows reads as, with its writes dropped.
//! * `Options::zero_copy_memory`: the shader marks no pages and binds no bitmap, so the table is
//!   all the host binds besides the heap.
//!
//! A heap's table ([`Table`]) is built whole once and then kept current in place: when the pool
//! table changed, the host rewrites the entries of the pages whose mappings changed, in the buffer
//! the GPU reads (as a console's GPU sees the title map and unmap pages at once). At 64 KiB a page
//! and 8 bytes an entry, 16 GiB of mappings is 2 MiB of words. A 64 KiB title page shows a pool only
//! if all of it shows one run of that pool (one mapping, or mappings that follow each other in title
//! memory and in the pool); any other page (a pool boundary, a hole) is a null page. [`build`] is the
//! whole-table form of the same layout, for mappings taken as they are.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::rc::Rc;
use victory_backend::backend::ComputeBackend;

use super::memory::PageTarget;

/// Bytes of a title page of the table (the console's allocation granularity).
pub const PAGE: u64 = 1 << 16;
/// How much of a pool one view pair covers before the next one starts: `size` is 32 bits.
pub const CHUNK: u64 = 1 << 31;
/// Pairs of views the host writes into every resource heap (pair 0 is the null view).
pub const PAIRS: usize = 256;
/// Entries the host adds to every resource heap, for the view pairs.
pub const RESERVED: usize = 2 * PAIRS;

/// Whether a resource heap of `count` entries has its pairs below the translator's clamp of heap
/// indices (and its own null view).
pub fn fits(count: usize) -> bool {
    count + RESERVED < victory_shader::emit::NULL_BUFFER_VIEW as usize
}

/// Level 1 of the table: 4096 dwords (address bits 28-39), a level-2 block number.
const LEVEL_1: usize = 4096;
/// A level-2 block: 4096 entries of two dwords (address bits 16-27).
const BLOCK: usize = 4096 * 2;

/// What a view pair shows: `[chunk * CHUNK, ..)` of pool `pool` (whose mapping is at `base`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PairKey {
    pub pool: usize,
    pub base: usize,
    /// The pool's length.
    pub len: u64,
    pub chunk: u64,
}

/// A mapping of the pool table, as the table builder reads it.
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub at: u64,
    pub len: u64,
    pub pool: usize,
    pub base: usize,
    pub pool_len: u64,
    pub offset: u64,
}

/// The view pairs made so far: pair `n` (n >= 1) is `keys[n - 1]`. Only ever grows, so a heap
/// that has the first k pairs written only needs the ones after them.
#[derive(Default, Debug)]
pub struct Pairs {
    pub keys: Vec<PairKey>,
    /// The last key asked for and its pair: the pages of a table come mostly in runs of one pool.
    last: Option<(PairKey, u32)>,
}

impl Pairs {
    fn pair(&mut self, key: PairKey) -> Option<u32> {
        if let Some((k, pair)) = self.last
            && k == key
        {
            return Some(pair);
        }
        let n = match self.keys.iter().rposition(|k| *k == key) {
            Some(n) => n,
            None if self.keys.len() + 1 < PAIRS => {
                self.keys.push(key);
                self.keys.len() - 1
            }
            None => return None,
        };
        let pair = n as u32 + 1;
        self.last = Some((key, pair));
        Some(pair)
    }

    /// The bytes of the pool a pair shows: from `chunk * CHUNK`, a chunk and a page long, as far
    /// as the pool goes.
    pub fn span(key: &PairKey) -> (u64, u32) {
        let from = key.chunk * CHUNK;
        (from, (key.len - from).min(CHUNK + PAGE) as u32)
    }
}

/// Heap index of the SRV of pair `n` in a heap whose pairs start at `first` (its title entries).
pub fn pair_index(first: u32, n: u32) -> u32 {
    first + 2 * n
}

/// The page table, in dwords: level 1, then the null block 0, then the blocks the mappings reach.
pub fn build(maps: &[Mapping], pairs: &mut Pairs, first: u32) -> Vec<u32> {
    let null = pair_index(first, 0);
    let mut words = vec![0u32; LEVEL_1];
    for _ in 0..BLOCK / 2 {
        words.extend([null, 0]);
    }
    for m in maps {
        let from = m.at.next_multiple_of(PAGE);
        let end = (m.at + m.len) & !(PAGE - 1);
        for page in (from..end).step_by(PAGE as usize) {
            // Addresses of 40 bits only; the translator sends the rest to block 0.
            if page >> 40 != 0 {
                break;
            }
            let pool_at = m.offset + (page - m.at);
            if pool_at + PAGE > m.pool_len {
                continue;
            }
            let key = PairKey {
                pool: m.pool,
                base: m.base,
                len: m.pool_len,
                chunk: pool_at / CHUNK,
            };
            let Some(pair) = pairs.pair(key) else { continue };
            let l1 = (page >> 28) as usize;
            if words[l1] == 0 {
                words[l1] = (words.len() - LEVEL_1) as u32 / BLOCK as u32;
                words.extend((0..BLOCK / 2).flat_map(|_| [null, 0]));
            }
            let at = LEVEL_1 + words[l1] as usize * BLOCK + 2 * ((page >> 16) as usize & 0xFFF);
            let in_view = pool_at - key.chunk * CHUNK;
            words[at] = pair_index(first, pair);
            words[at + 1] = (page as u32).wrapping_sub(in_view as u32);
        }
    }
    words
}

/// Blocks a [`Table`] has room for at first (two MiB of entries, besides level 1); it doubles when
/// the title's mappings reach more 256 MiB regions than that.
const FIRST_BLOCKS: usize = 64;
/// Blocks any table can use: one per level-1 entry, and the null block.
const MAX_BLOCKS: usize = LEVEL_1 + 1;

/// The page table of one resource heap (its view indices are that heap's), kept current in place:
/// the host writes only the pages whose mappings changed ([`super::memory::PoolTable::pages_since`])
/// into the buffer the GPU reads, as the console's own page tables change under GPU work when the
/// title maps memory. Rebuilding a table for every change of the pool table (each mapping the
/// title's streaming makes) cost a snapshot of every mapping and a new buffer, under the pool and
/// title locks, many times per submission; and a buffer superseded more than a few times behind the
/// GPU was freed while queued work still read it. A buffer that must grow is copied into one twice
/// its size, and the old one is kept for the work that may still read it.
pub struct Table<B: ComputeBackend> {
    pages: Rc<Pages<B>>,
    /// Superseded buffers, for the work queued before they were.
    // shortcut: kept as long as the table; it grows a few times at most (it doubles, to MAX_BLOCKS).
    old: Vec<Rc<Pages<B>>>,
    /// Level 1 as the buffer has it: the block of each 256 MiB of title addresses (0: none).
    level1: Vec<u32>,
    /// Blocks in use, the null block 0 included, and blocks the buffer has room for.
    blocks: usize,
    capacity: usize,
    /// The heap's title entries: its pair 0 (the null view) is at this index.
    first: u32,
    /// The pool-table generation the table is current for.
    pub generation: u64,
    /// Pages left null because no view pair was left for their pool and chunk.
    pub unpaired: u64,
}

impl<B: ComputeBackend> Table<B> {
    /// The table of a heap of `first` title entries, showing `pages` (current for `generation`).
    pub fn new(
        backend: &B,
        first: u32,
        generation: u64,
        pages: &[(u64, PageTarget)],
        pairs: &mut Pairs,
    ) -> Result<Table<B>, String> {
        let mut table = Table {
            pages: Rc::new(Self::alloc(backend, first, FIRST_BLOCKS)?),
            old: Vec::new(),
            level1: vec![0; LEVEL_1],
            blocks: 1,
            capacity: FIRST_BLOCKS,
            first,
            generation,
            unpaired: 0,
        };
        for &(page, target) in pages {
            table.set(backend, page, Some(target), pairs)?;
        }
        Ok(table)
    }

    /// The buffer the GPU reads now.
    pub fn buffer(&self) -> &Rc<Pages<B>> {
        &self.pages
    }

    /// Blocks in use (the null block included) and superseded buffers kept.
    pub fn size(&self) -> (usize, usize) {
        (self.blocks, self.old.len())
    }

    /// A buffer with room for `blocks` blocks: level 1 all block 0, every entry the null view.
    fn alloc(backend: &B, first: u32, blocks: usize) -> Result<Pages<B>, String> {
        let words = LEVEL_1 + blocks * BLOCK;
        let pages = Pages::new(backend, words * 4)?;
        let null = pair_index(first, 0);
        let mut block = vec![0u32; BLOCK];
        for e in block.chunks_exact_mut(2) {
            e[0] = null;
        }
        for b in 0..blocks {
            pages.write_at(LEVEL_1 + b * BLOCK, &block);
        }
        Ok(pages)
    }

    fn at(block: usize, page: u64) -> usize {
        LEVEL_1 + block * BLOCK + 2 * ((page >> 16) as usize & 0xFFF)
    }

    /// Writes what the 64 KiB title page `page` shows (`None`: nothing, the null view). Pages above
    /// 40 bits are never looked up (the translator sends them to block 0).
    pub fn set(
        &mut self,
        backend: &B,
        page: u64,
        target: Option<PageTarget>,
        pairs: &mut Pairs,
    ) -> Result<(), String> {
        if page >> 40 != 0 {
            return Ok(());
        }
        let null = [pair_index(self.first, 0), 0];
        let entry = match target {
            None => null,
            Some(t) => {
                let key = PairKey {
                    pool: t.pool,
                    base: t.base,
                    len: t.pool_len,
                    chunk: t.pool_at / CHUNK,
                };
                match pairs.pair(key) {
                    Some(pair) => {
                        let in_view = t.pool_at - key.chunk * CHUNK;
                        [pair_index(self.first, pair), (page as u32).wrapping_sub(in_view as u32)]
                    }
                    None => {
                        self.unpaired += 1;
                        null
                    }
                }
            }
        };
        let l1 = (page >> 28) as usize;
        if self.level1[l1] != 0 {
            self.pages.write_entry(Self::at(self.level1[l1] as usize, page), entry);
            return Ok(());
        }
        if entry == null {
            // Its block is the null block already.
            return Ok(());
        }
        if self.blocks == self.capacity {
            self.grow(backend)?;
        }
        let block = self.blocks;
        self.blocks += 1;
        // The new block's entries are null already: the entry, then level 1 points at the block (so
        // a lookup never reads a block before its entry is in).
        self.pages.write_entry(Self::at(block, page), entry);
        self.pages.write_at(l1, &[block as u32]);
        self.level1[l1] = block as u32;
        Ok(())
    }

    fn grow(&mut self, backend: &B) -> Result<(), String> {
        let capacity = (self.capacity * 2).min(MAX_BLOCKS);
        if capacity <= self.capacity {
            return Err("a page table has a block for every 256 MiB already".into());
        }
        let bigger = Self::alloc(backend, self.first, capacity)?;
        bigger.copy_from(&self.pages, LEVEL_1 + self.blocks * BLOCK);
        let old = std::mem::replace(&mut self.pages, Rc::new(bigger));
        self.old.push(old);
        self.capacity = capacity;
        Ok(())
    }

    /// The words the GPU reads, as far as they are in use (for tests and checks).
    pub fn words(&self) -> Vec<u32> {
        self.pages.read(LEVEL_1 + self.blocks * BLOCK)
    }
}

/// A root SRV or UAV address as the translator's root views want it (`emit`, "Root SRVs and root
/// UAVs"): the heap index of the view pair's SRV (the UAV is the next entry) and the byte offset of
/// the address in the view. An address no mapping shows is the null view at offset 0.
pub fn root_view(maps: &[Mapping], pairs: &mut Pairs, first: u32, address: u64) -> (u32, u32) {
    let null = (pair_index(first, 0), 0);
    let Some(m) = maps.iter().find(|m| address >= m.at && address - m.at < m.len) else {
        return null;
    };
    let pool_at = m.offset + (address - m.at);
    let key = PairKey {
        pool: m.pool,
        base: m.base,
        len: m.pool_len,
        chunk: pool_at / CHUNK,
    };
    match pairs.pair(key) {
        Some(pair) => (pair_index(first, pair), (pool_at - key.chunk * CHUNK) as u32),
        None => null,
    }
}

/// What the translated shader does with `words` for the address `va` (the lookup of
/// `emit::lower::memory_entry`): the heap index of the view and the offset in it, or `None` for the
/// null view.
pub fn lookup(words: &[u32], va: u64) -> Option<(u32, u32)> {
    let block = if va >> 40 == 0 {
        words[(va >> 28) as usize] as usize
    } else {
        0
    };
    let at = LEVEL_1 + block * BLOCK + 2 * ((va >> 16) as usize & 0xFFF);
    let (view, start) = (words[at], words[at + 1]);
    (view != words[LEVEL_1]).then(|| (view, (va as u32).wrapping_sub(start)))
}

/// Zeroed, page-aligned host memory wrapped as a GPU buffer: the table, the null view.
pub struct Pages<B: ComputeBackend> {
    pub buf: Rc<B::Buffer>,
    ptr: *mut u8,
    layout: Layout,
}

impl<B: ComputeBackend> Pages<B> {
    /// `bytes` rounded up to 16 KiB (a page on Apple silicon; a multiple of every page size here).
    pub fn new(backend: &B, bytes: usize) -> Result<Pages<B>, String> {
        let layout = Layout::from_size_align(bytes.max(1).next_multiple_of(1 << 14), 1 << 14)
            .map_err(|e| e.to_string())?;
        // SAFETY: a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(format!("no memory for {} bytes", layout.size()));
        }
        match backend.wrap_title_memory(ptr, layout.size()) {
            Ok(buf) => Ok(Pages {
                buf: Rc::new(buf),
                ptr,
                layout,
            }),
            Err(e) => {
                // SAFETY: allocated above with this layout, not wrapped.
                unsafe { dealloc(ptr, layout) };
                Err(e)
            }
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Writes `words` from the start.
    pub fn write(&self, words: &[u32]) {
        assert!(words.len() * 4 <= self.layout.size());
        // SAFETY: inside the allocation; the GPU does not read a table before it is bound.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), self.ptr.cast(), words.len()) };
    }

    /// Writes `words` from word `index` on, each word in one store (the GPU may be reading them).
    pub fn write_at(&self, index: usize, words: &[u32]) {
        assert!((index + words.len()) * 4 <= self.layout.size());
        let at = self.ptr.cast::<u32>();
        for (i, &w) in words.iter().enumerate() {
            // SAFETY: inside the allocation, aligned (the allocation is page aligned).
            unsafe { at.add(index + i).write_volatile(w) };
        }
    }

    /// Writes the two-word entry at the even word `index` in one store, so a lookup the GPU makes
    /// meanwhile sees the old entry or the new one, not half of each.
    pub fn write_entry(&self, index: usize, entry: [u32; 2]) {
        assert!(index.is_multiple_of(2) && (index + 2) * 4 <= self.layout.size());
        let value = u64::from(entry[0]) | u64::from(entry[1]) << 32;
        // SAFETY: inside the allocation, 8-byte aligned (an even word of a page-aligned allocation).
        unsafe { self.ptr.cast::<u64>().add(index / 2).write_volatile(value.to_le()) };
    }

    /// Copies the first `words` words of `other` here.
    pub fn copy_from(&self, other: &Pages<B>, words: usize) {
        assert!(words * 4 <= self.layout.size() && words * 4 <= other.layout.size());
        // SAFETY: inside both allocations, which are distinct.
        unsafe { std::ptr::copy_nonoverlapping(other.ptr.cast::<u32>(), self.ptr.cast::<u32>(), words) };
    }

    /// The first `words` words.
    pub fn read(&self, words: usize) -> Vec<u32> {
        assert!(words * 4 <= self.layout.size());
        let at = self.ptr.cast::<u32>();
        // SAFETY: inside the allocation, aligned.
        (0..words).map(|i| unsafe { at.add(i).read_volatile() }).collect()
    }
}

impl<B: ComputeBackend> Drop for Pages<B> {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_view_is_a_pair_and_an_offset_in_it() {
        let maps = [
            map(0x10_0000_0000, 3 * CHUNK, 7, 3 * CHUNK, 0),
            map(0x20_0000_0000, PAGE, 8, PAGE, 0),
        ];
        let mut pairs = Pairs::default();
        let first = 1000;
        // The second chunk of pool 7, 0x123 bytes in.
        let a = 0x10_0000_0000 + CHUNK + 0x123;
        assert_eq!(root_view(&maps, &mut pairs, first, a), (pair_index(first, 1), 0x123));
        // The same pool and chunk is the same pair; another pool is the next.
        assert_eq!(root_view(&maps, &mut pairs, first, a + 8), (pair_index(first, 1), 0x12b));
        assert_eq!(root_view(&maps, &mut pairs, first, 0x20_0000_0010).0, pair_index(first, 2));
        // Nothing mapped: the null view.
        assert_eq!(root_view(&maps, &mut pairs, first, 0x30_0000_0000), (pair_index(first, 0), 0));
    }

    /// Where a heap of a thousand title entries has its pairs.
    const FIRST: u32 = 1000;

    fn map(at: u64, len: u64, pool: usize, pool_len: u64, offset: u64) -> Mapping {
        Mapping {
            at,
            len,
            pool,
            base: 0x1_0000_0000 * pool,
            pool_len,
            offset,
        }
    }

    /// What the lookup says an address of the title is: the pair's view and where in the pool.
    fn resolve(words: &[u32], pairs: &Pairs, va: u64) -> Option<(usize, u64)> {
        let (view, offset) = lookup(words, va)?;
        let key = pairs.keys[((view - FIRST) / 2 - 1) as usize];
        Some((key.pool, key.chunk * CHUNK + u64::from(offset)))
    }

    #[test]
    fn a_page_resolves_to_its_pool_offset() {
        let mut pairs = Pairs::default();
        // A pool of 4 MiB shown at two addresses: one above 4 GiB, one below, at an offset.
        let maps = [
            map(0x12_3450_0000, 0x40_0000, 1, 0x40_0000, 0),
            map(0x7000_0000, 0x10_0000, 1, 0x40_0000, 0x2_0000),
            map(0x20_0000_0000, 0x2_0000, 2, 0x2_0000, 0),
        ];
        let words = build(&maps, &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x12_3450_0000), Some((1, 0)));
        assert_eq!(resolve(&words, &pairs, 0x12_3450_1234), Some((1, 0x1234)));
        assert_eq!(resolve(&words, &pairs, 0x12_347F_FFFC), Some((1, 0x2F_FFFC)));
        assert_eq!(resolve(&words, &pairs, 0x7000_0010), Some((1, 0x2_0010)));
        assert_eq!(resolve(&words, &pairs, 0x7000_0000 + 0xF_FFFC), Some((1, 0x2_0000 + 0xF_FFFC)));
        assert_eq!(resolve(&words, &pairs, 0x20_0001_0000), Some((2, 0x1_0000)));
        // Nothing past the mappings, nor in another 256 MiB region, nor above 40 bits.
        for va in [0x12_3490_0000, 0x7010_0000, 0x6FFF_0000, 0x55_0000_0040, 1 << 41, 0] {
            assert_eq!(resolve(&words, &pairs, va), None, "{va:#x}");
        }
        // Two views of the pool: one pair.
        assert_eq!(pairs.keys.len(), 2);
    }

    #[test]
    fn a_pool_bigger_than_a_chunk_has_a_pair_per_chunk_and_offsets_stay_in_the_view() {
        let mut pairs = Pairs::default();
        let len = 3 * CHUNK + 0x3_0000;
        let at = 0x40_0000_0000u64;
        let words = build(&[map(at, len, 5, len, 0)], &mut pairs, FIRST);
        for off in [0, CHUNK - 4, CHUNK, CHUNK + 4, 2 * CHUNK + 0x1234, 3 * CHUNK + 0x2_FFFC] {
            let r = resolve(&words, &pairs, at + off).unwrap();
            assert_eq!(r, (5, off), "{off:#x}");
            // The offset is inside the view of that pair (a chunk and a page long, to the pool's end).
            let (view, o) = lookup(&words, at + off).unwrap();
            let key = pairs.keys[((view - FIRST) / 2 - 1) as usize];
            let (_, size) = Pairs::span(&key);
            assert!(u64::from(o) + 4 <= u64::from(size), "{off:#x}");
        }
        assert_eq!(pairs.keys.len(), 4);
    }

    #[test]
    fn mappings_off_the_64k_grid_leave_null_pages() {
        let mut pairs = Pairs::default();
        // From 16 KiB past a 64 KiB boundary, to a 16 KiB-aligned pool offset: the first partial
        // page is null, the four whole ones after it are not, and the last partial one is null.
        let words = build(&[map(0x1000_4000, 0x4_C000, 1, 0x10_0000, 0x4000)], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x1000_4000), None);
        for va in [0x1001_0000u64, 0x1004_FFFC] {
            assert_eq!(resolve(&words, &pairs, va), Some((1, 0x4000 + va - 0x1000_4000)));
        }
        assert_eq!(resolve(&words, &pairs, 0x1005_0000), None);
        assert_eq!(resolve(&words, &pairs, 0x1010_0000), None);
        // A mapping whose last page runs past the pool is null there.
        let mut pairs = Pairs::default();
        let words = build(&[map(0x2000_0000, 0x2_0000, 1, 0x1_8000, 0)], &mut pairs, FIRST);
        assert_eq!(resolve(&words, &pairs, 0x2000_0000), Some((1, 0)));
        assert_eq!(resolve(&words, &pairs, 0x2001_0000), None);
    }

    #[test]
    fn views_are_stable_and_bounded() {
        let mut pairs = Pairs::default();
        let a = map(0x1000_0000, 0x1_0000, 1, 0x1_0000, 0);
        build(&[a], &mut pairs, FIRST);
        let first = pairs.keys.clone();
        // The same pool at another address, and one more pool: earlier pairs keep their numbers.
        build(&[a, map(0x2000_0000, 0x1_0000, 2, 0x1_0000, 0)], &mut pairs, FIRST);
        assert_eq!(pairs.keys[..1], first[..]);
        assert_eq!(pairs.keys.len(), 2);
        // Past the heap's reserved entries pages are null, not another pair's.
        let many: Vec<Mapping> = (0..PAIRS as u64 * 2)
            .map(|i| map(0x10_0000_0000 + i * PAGE, PAGE, 100 + i as usize, PAGE, 0))
            .collect();
        let words = build(&many, &mut pairs, FIRST);
        assert_eq!(pairs.keys.len(), PAIRS - 1);
        assert_eq!(resolve(&words, &pairs, 0x10_0000_0000 + 2 * PAGE).unwrap().0, 102);
        assert_eq!(resolve(&words, &pairs, 0x10_0000_0000 + (PAIRS as u64 * 2 - 1) * PAGE), None);
        // The pairs and the translator's own null views fit below the clamp.
        assert!(fits(1_000_000) && !fits(victory_shader::emit::NULL_BUFFER_VIEW as usize));
    }
}
