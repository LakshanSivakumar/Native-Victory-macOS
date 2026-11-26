//! Memory for lists' upload blocks (list method 0x4a0) on macOS: a few large pools below 2^40,
//! each told to the pool table once, from which blocks are carved. A pool per block changed the
//! table at every allocation and every free, so the page table translated shaders read was made
//! again every frame, and each block cost a mapping, a Metal buffer and a residency entry.

use super::memory::{PoolInfo, PoolObserver, PoolTable};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Bytes of an arena pool, unless a block needs more.
pub(crate) const POOL: usize = 64 << 20;
/// What a Metal buffer over a pool needs (the kernel's page, also under Rosetta, whose pages say
/// 4 KiB), as the title's pools have.
const PAGE: usize = 0x4000;
/// Blocks start and end on this.
pub(crate) const ALIGN: usize = 16;
/// Pool ids of arena pools: bit 63 set. `xmem_physical` numbers the title's pools from 1 up and
/// never reaches it, so the two never meet in the pool table or the host's buffers by pool.
pub(crate) const IDS: usize = 1 << (usize::BITS - 1);

/// One pool: its address and size, and its free ranges by address (never touching: freeing
/// merges a range with its neighbours).
struct Region {
    base: usize,
    len: usize,
    free: BTreeMap<usize, usize>,
}

impl Region {
    /// The first free range that holds `len` bytes, its start taken.
    fn take(&mut self, len: usize) -> Option<usize> {
        let (&at, &have) = self.free.iter().find(|&(_, &have)| have >= len)?;
        self.free.remove(&at);
        if have > len {
            self.free.insert(at + len, have - len);
        }
        Some(at)
    }

    fn give(&mut self, mut at: usize, mut len: usize) {
        if let Some(next) = self.free.remove(&(at + len)) {
            len += next;
        }
        if let Some((&before, &have)) = self.free.range(..at).next_back()
            && before + have == at
        {
            self.free.remove(&before);
            at = before;
            len += have;
        }
        self.free.insert(at, len);
    }
}

/// Upload memory: pools below 2^40 in `table`, and the blocks carved from them.
pub(crate) struct Arena {
    table: Arc<PoolTable>,
    regions: Vec<Region>,
    next_id: usize,
}

impl Arena {
    pub(crate) fn new(table: Arc<PoolTable>) -> Arena {
        Arena {
            table,
            regions: Vec::new(),
            next_id: 0,
        }
    }

    /// `len` bytes (rounded up to 16), zeroed, 16-byte aligned, below 2^40: their address and
    /// rounded size, to give back to [`Arena::free`]. First fit over the pools, oldest first; a
    /// new pool (of [`POOL`] bytes, or the block's size if larger) when none has room.
    pub(crate) fn alloc(&mut self, len: usize) -> Option<(usize, usize)> {
        let len = len.max(1).checked_next_multiple_of(ALIGN)?;
        let at = match self.regions.iter_mut().find_map(|r| r.take(len)) {
            Some(at) => at,
            None => {
                let region = self.region(len.max(POOL))?;
                self.regions.push(region);
                self.regions.last_mut()?.take(len)?
            }
        };
        // Reused bytes hold an earlier block's: a block is zeroed, as a fresh Windows one is.
        // SAFETY: `[at, at + len)` lies in one of our pools and is no one's now.
        unsafe { (at as *mut u8).write_bytes(0, len) };
        Some((at, len))
    }

    /// Gives back a block from [`Arena::alloc`]: only once the GPU is past every use of it
    /// (`Title::retire`), since the next block may be carved from the same bytes.
    pub(crate) fn free(&mut self, at: usize, len: usize) {
        if let Some(r) = self
            .regions
            .iter_mut()
            .find(|r| (r.base..r.base + r.len).contains(&at))
        {
            r.give(at, len);
        }
    }

    /// A pool of `len` bytes (rounded up to 16 KiB) in [4 GiB, 2^40), where shaders find title
    /// memory (`XMemVirtualAlloc` reserves there too), mapped at its own address in the table.
    // shortcut: pools are never given back (a burst of uploads keeps its peak, 64 MiB at a time);
    // release an empty pool after a while if the footprint matters.
    fn region(&mut self, len: usize) -> Option<Region> {
        use victory_platform::mem;
        let len = len.checked_next_multiple_of(PAGE)?;
        // SAFETY: fresh address space, ours for good.
        let at = unsafe { mem::reserve_in(1 << 32, (1 << 40) - 1, len, PAGE) }.ok()?;
        // SAFETY: the reservation just made; zero filled.
        if unsafe { mem::commit(at, len, mem::Protect::READ_WRITE) }.is_err() {
            // SAFETY: as above.
            let _ = unsafe { mem::release(at, len) };
            return None;
        }
        let base = at as usize;
        let pool = PoolInfo {
            id: IDS | self.next_id,
            base,
            len,
            memory_entry: None,
        };
        self.next_id += 1;
        self.table.pool_created(&pool);
        self.table.mapped(base, len, &pool, 0);
        Some(Region {
            base,
            len,
            free: BTreeMap::from([(base, len)]),
        })
    }

    #[cfg(test)]
    pub(crate) fn pools(&self) -> usize {
        self.regions.len()
    }

    /// Whether byte `at` lies in a free range.
    #[cfg(test)]
    pub(crate) fn is_free(&self, at: usize) -> bool {
        self.regions.iter().any(|r| {
            r.free
                .range(..=at)
                .next_back()
                .is_some_and(|(&a, &n)| at < a + n)
        })
    }
}

/// The arena of the native host's pool table, which list upload blocks come from. One lock; it
/// is taken under the host's lock (blocks retired there are freed there) and never takes it.
pub(crate) fn arena() -> &'static Mutex<Arena> {
    static ARENA: OnceLock<Mutex<Arena>> = OnceLock::new();
    ARENA.get_or_init(|| Mutex::new(Arena::new(super::glue::pools().clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena() -> (Arc<PoolTable>, Arena) {
        let table = Arc::new(PoolTable::default());
        (table.clone(), Arena::new(table))
    }

    #[test]
    fn blocks_are_title_memory_below_2_40_in_pools_of_their_own() {
        let (table, mut a) = arena();
        let (at, len) = a.alloc(40).unwrap();
        assert_eq!(len, 48);
        assert!((1 << 32..1 << 40).contains(&at), "{at:#x}");
        assert_eq!(at % PAGE, 0, "the first block starts the pool");
        // SAFETY: the block's bytes.
        unsafe { (at as *mut u8).add(8).write(0x5a) };
        let (pool, p) = table.resolve(at as u64 + 8, 16).expect("a pool");
        assert_ne!(pool.id & IDS, 0, "apart from the title's pools");
        assert_eq!((pool.base, pool.len), (at, POOL));
        assert_eq!(p, (at + 8) as *const u8);
        // SAFETY: what the GPU's buffer shows is the CPU's byte.
        assert_eq!(unsafe { *p }, 0x5a);
    }

    #[test]
    fn blocks_are_aligned_reused_and_merged() {
        let (_, mut a) = arena();
        let blocks: Vec<_> = [1, 17, 0x100, 3].map(|n| a.alloc(n).unwrap()).into();
        for &(at, len) in &blocks {
            assert_eq!((at % ALIGN, len % ALIGN), (0, 0));
        }
        // Carved one after the other.
        for w in blocks.windows(2) {
            assert_eq!(w[0].0 + w[0].1, w[1].0);
        }
        let base = blocks[0].0;
        // A freed range comes back first fit.
        a.free(blocks[1].0, blocks[1].1);
        assert!(a.is_free(blocks[1].0) && !a.is_free(blocks[2].0));
        assert_eq!(a.alloc(17).unwrap(), blocks[1]);
        // Neighbours freed in any order merge into the whole pool again.
        for &i in &[0, 2, 3, 1] {
            a.free(blocks[i].0, blocks[i].1);
        }
        assert_eq!(a.regions[0].free.len(), 1);
        assert_eq!(a.alloc(POOL).unwrap(), (base, POOL));
        assert_eq!(a.pools(), 1);
    }

    #[test]
    fn a_reused_block_is_zeroed() {
        let (_, mut a) = arena();
        let (at, len) = a.alloc(64).unwrap();
        // SAFETY: the block's bytes.
        unsafe { (at as *mut u8).write_bytes(0xff, len) };
        a.free(at, len);
        assert_eq!(a.alloc(64).unwrap(), (at, len));
        // SAFETY: as above.
        let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, len) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_block_larger_than_a_pool_gets_a_pool_of_its_size() {
        let (table, mut a) = arena();
        a.alloc(16).unwrap();
        let (at, len) = a.alloc(POOL + 1).unwrap();
        assert_eq!(a.pools(), 2);
        assert_eq!(len, POOL + ALIGN);
        let (pool, _) = table.resolve(at as u64, len as u64).expect("one pool");
        assert_eq!((pool.base, pool.len), (at, POOL + PAGE));
        // The first pool still serves small blocks.
        let (small, _) = a.alloc(16).unwrap();
        assert!(!(at..at + len).contains(&small));
    }

    #[test]
    fn the_pool_table_stays_still_after_warming_up() {
        let (table, mut a) = arena();
        let frame = |a: &mut Arena| {
            let blocks: Vec<_> = (1..50).map(|n| a.alloc(n * 100).unwrap()).collect();
            for (at, len) in blocks.into_iter().rev() {
                a.free(at, len);
            }
        };
        frame(&mut a);
        let (pools, generation) = (a.pools(), table.generation());
        for _ in 0..100 {
            frame(&mut a);
        }
        assert_eq!((a.pools(), table.generation()), (pools, generation));
        assert_eq!(table.mappings().1.len(), 1);
    }

    /// The Metal host wraps an arena pool in a buffer and finds a block's address in it.
    #[test]
    fn the_native_host_resolves_blocks() {
        let Ok(metal) = victory_metal::metal::Metal::new() else {
            eprintln!("skipped: no Metal device");
            return;
        };
        let (table, mut a) = arena();
        let host = crate::native::Host::with_memory(metal, table);
        let (at, _) = a.alloc(0x100).unwrap();
        let (b, _) = a.alloc(0x100).unwrap();
        let (_, p) = host.resolve(b as u64 + 0x40, 0x40).expect("resolved");
        assert_eq!(p, (b + 0x40) as *const u8);
        assert!(
            host.resolve(at as u64, 0x200).is_ok(),
            "both blocks, one buffer"
        );
    }
}
