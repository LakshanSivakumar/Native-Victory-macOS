//! Title memory for the native host: which physical-page pool a title address shows.
//!
//! On the native host a GPU buffer wraps a pool's own mapping (one `MTLBuffer` per pool, zero copy),
//! so title writes and GPU writes are the same bytes. There is no shadow copy, no write guard, no
//! copy-back: `shadow.rs`, `watch.rs` and the page bitmaps of `victory_shader::emit` are Windows
//! only and never reached from here. What is left is a table from title pages to
//! `(pool, offset)`, which the pool observer of `victory_hle::modules::xmem_physical` (branch
//! `native/host-mem`, merged by `native/xmem-rest`) keeps current. The types below have the
//! observer's shape, so the host implements `PoolObserver` for [`PoolTable`] with a one-line
//! forward per method; this crate does not depend on `victory-hle`.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

/// Bytes of a title page of the GPU's page table ([`super::addr::PAGE`]).
const TABLE_PAGE: usize = 1 << 16;

/// Mapping changes kept for page tables that are behind ([`PoolTable::pages_since`]); a table
/// further behind than the oldest one kept is built again whole.
const JOURNAL: usize = if cfg!(test) { 64 } else { 1 << 18 };

/// What a 64 KiB title page shows, for the page table ([`super::addr::Table`]): the pool, and where
/// in it the page starts. A page shows a pool only when all of it shows one run of that pool (one
/// mapping, or mappings that follow each other in title memory and in the pool).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageTarget {
    pub pool: usize,
    /// The pool's own mapping ([`PoolInfo::base`]).
    pub base: usize,
    pub pool_len: u64,
    /// The page's offset in the pool.
    pub pool_at: u64,
}

/// `xmem_physical::PoolInfo`.
#[derive(Clone, Debug)]
pub struct PoolInfo {
    pub id: usize,
    /// The pool's own read-write mapping, 16 KiB aligned, never moved until `pool_released`.
    pub base: usize,
    pub len: usize,
    pub memory_entry: Option<u32>,
}

/// `xmem_physical::Observer`.
pub trait PoolObserver: Send + Sync {
    fn pool_created(&self, pool: &PoolInfo);
    fn pool_released(&self, pool: &PoolInfo);
    fn mapped(&self, at: usize, len: usize, pool: &PoolInfo, offset: usize);
    fn unmapped(&self, at: usize, len: usize);
}

#[derive(Clone, Copy)]
struct Map {
    len: usize,
    pool: usize,
    offset: usize,
}

#[derive(Default)]
struct Inner {
    pools: BTreeMap<usize, PoolInfo>,
    /// Title address of a mapping to what it shows; mappings never overlap.
    maps: BTreeMap<usize, Map>,
    /// Pools released since the last [`PoolTable::take_released`]: their GPU buffers must go.
    released: Vec<usize>,
    /// Counts every change to pools and mappings ([`PoolTable::generation`]).
    generation: u64,
    /// The title ranges whose mappings changed, with the generation of each change, oldest first.
    journal: VecDeque<(u64, usize, usize)>,
    /// The generation of the newest change dropped from `journal` (0: none dropped).
    journal_lost: u64,
}

impl Inner {
    /// Keeps `[at, at + len)` as changed at the current generation.
    fn note(&mut self, at: usize, len: usize) {
        if len == 0 {
            return;
        }
        if self.journal.len() >= JOURNAL
            && let Some((g, _, _)) = self.journal.pop_front()
        {
            self.journal_lost = g;
        }
        self.journal.push_back((self.generation, at, len));
    }

    /// What the 64 KiB title page at `page` shows now.
    fn page_target(&self, page: usize) -> Option<PageTarget> {
        let (pool, ptr) = self.resolve(page, TABLE_PAGE)?;
        Some(PageTarget {
            pool: pool.id,
            base: pool.base,
            pool_len: pool.len as u64,
            pool_at: (ptr as usize - pool.base) as u64,
        })
    }

    /// Removes `[at, at + len)` from the mappings, keeping what is left of any it cuts.
    fn cut(&mut self, at: usize, len: usize) {
        self.generation += 1;
        self.note(at, len);
        let end = at + len;
        // Only the immediate predecessor can overlap the start of the cut.
        let mut preceding = self
            .maps
            .range(..at)
            .next_back()
            .filter(|&(&a, m)| a + m.len > at)
            .map(|(&a, _)| a);
        while let Some(a) = preceding
            .take()
            .or_else(|| self.maps.range(at..end).next().map(|(&a, _)| a))
        {
            let m = self.maps.remove(&a).expect("overlapping mapping");
            if a < at {
                self.maps.insert(a, Map { len: at - a, ..m });
            }
            if a + m.len > end {
                let cut = end - a;
                self.maps.insert(
                    end,
                    Map {
                        len: m.len - cut,
                        offset: m.offset + cut,
                        ..m
                    },
                );
                break;
            }
        }
    }

    fn resolve(&self, addr: usize, len: usize) -> Option<(&PoolInfo, *const u8)> {
        let end = addr.checked_add(len)?;
        let (&start, first) = self.maps.range(..=addr).next_back()?;
        let mut covered = start.checked_add(first.len)?;
        if addr >= covered {
            return None;
        }
        if end > covered {
            // Scatter maps can report one mapping per page, even for consecutive pool pages.
            for (&at, next) in self.maps.range(covered..end) {
                if at != covered
                    || next.pool != first.pool
                    || next.offset != first.offset.checked_add(at - start)?
                {
                    return None;
                }
                covered = at.checked_add(next.len)?;
            }
            if covered < end {
                return None;
            }
        }
        let pool = self.pools.get(&first.pool)?;
        let offset = first.offset.checked_add(addr - start)?;
        if offset.checked_add(len)? > pool.len {
            return None;
        }
        Some((pool, pool.base.checked_add(offset)? as *const u8))
    }
}

/// The title page to pool table. Thread safe: the observer is called from title threads.
#[derive(Default)]
pub struct PoolTable {
    inner: Mutex<Inner>,
    #[cfg(all(test, target_os = "macos"))]
    mapping_snapshots: std::sync::atomic::AtomicUsize,
}

impl PoolTable {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pool `[addr, addr + len)` lies in, and the pool-mapping address of `addr`. The range may
    /// run over several mappings when each follows the one before it both in title memory and in
    /// the same pool (the pages of one allocation are reported one mapping at a time, and titles
    /// build buffer views across them). A range over a gap, over two pools, or over mappings that
    /// are not next to each other in the pool is refused: one pointer cannot show it.
    pub fn resolve(&self, addr: u64, len: u64) -> Option<(PoolInfo, *const u8)> {
        let (addr, len) = (
            usize::try_from(addr).ok()?,
            usize::try_from(len.max(1)).ok()?,
        );
        let memory = self.lock();
        let (pool, address) = memory.resolve(addr, len)?;
        Some((pool.clone(), address))
    }

    fn try_contains(&self, addr: u64) -> Option<bool> {
        let addr = usize::try_from(addr).ok()?;
        let memory = self.inner.try_lock().ok()?;
        Some(memory.resolve(addr, 1).is_some())
    }

    /// A number that changes whenever a pool or a mapping does: what a table built from
    /// [`PoolTable::mappings`] is current for.
    pub fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// Every mapping, by title address: `(at, len, pool, offset in the pool)`.
    pub fn mappings(&self) -> (u64, Vec<(usize, usize, PoolInfo, usize)>) {
        let g = self.lock();
        #[cfg(all(test, target_os = "macos"))]
        self.mapping_snapshots
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let maps = g
            .maps
            .iter()
            .filter_map(|(&at, m)| Some((at, m.len, g.pools.get(&m.pool)?.clone(), m.offset)))
            .collect();
        (g.generation, maps)
    }

    /// Every 64 KiB title page a mapping shows all of, and what it shows (a page table built whole),
    /// with the generation that is current for.
    pub fn pages(&self) -> (u64, Vec<(u64, PageTarget)>) {
        let g = self.lock();
        let mut out = Vec::new();
        // Mappings come in address order and never overlap: a page two of them share is seen twice
        // in a row, and looked at once.
        let mut last = None;
        for (&at, m) in &g.maps {
            let end = at + m.len;
            let mut page = at & !(TABLE_PAGE - 1);
            while page < end {
                if last.is_none_or(|l| page > l) {
                    last = Some(page);
                    if let Some(t) = g.page_target(page) {
                        out.push((page as u64, t));
                    }
                }
                page += TABLE_PAGE;
            }
        }
        (g.generation, out)
    }

    /// The 64 KiB title pages whose mappings changed after generation `since`, and what each shows
    /// now (`None`: nothing, or not all of it one run of one pool), with the generation that is
    /// current for. `None` when changes that old are no longer kept: build the table whole
    /// ([`PoolTable::pages`]).
    pub fn pages_since(&self, since: u64) -> Option<(u64, Vec<(u64, Option<PageTarget>)>)> {
        let g = self.lock();
        if since < g.journal_lost {
            return None;
        }
        let from = g.journal.partition_point(|&(generation, _, _)| generation <= since);
        let mut pages: Vec<usize> = Vec::new();
        for &(_, at, len) in g.journal.range(from..) {
            let end = at + len;
            let mut page = at & !(TABLE_PAGE - 1);
            while page < end {
                pages.push(page);
                page += TABLE_PAGE;
            }
        }
        pages.sort_unstable();
        pages.dedup();
        let out = pages.into_iter().map(|p| (p as u64, g.page_target(p))).collect();
        Some((g.generation, out))
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn mapping_snapshots(&self) -> usize {
        self.mapping_snapshots
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Pools released since the last call; the host drops their buffers.
    pub fn take_released(&self) -> Vec<usize> {
        std::mem::take(&mut self.lock().released)
    }
}

type BufferViews = BTreeMap<u64, BTreeMap<usize, (u64, u64)>>;

/// Pool and descriptor metadata that a crash report can read independently of the native host.
pub(crate) struct CrashAddresses {
    memory: Arc<PoolTable>,
    views: Mutex<BufferViews>,
}

impl CrashAddresses {
    pub(crate) fn new(memory: Arc<PoolTable>) -> Self {
        Self {
            memory,
            views: Mutex::default(),
        }
    }

    fn lock_views(&self) -> MutexGuard<'_, BufferViews> {
        self.views.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn set_buffer_view(&self, heap: u64, index: usize, address: u64, len: u64) {
        self.lock_views()
            .entry(heap)
            .or_default()
            .insert(index, (address, len));
    }

    pub(crate) fn remove_buffer_view(&self, heap: u64, index: usize) {
        if let Some(slots) = self.lock_views().get_mut(&heap) {
            slots.remove(&index);
        }
    }

    pub(crate) fn remove_heap(&self, heap: u64) {
        self.lock_views().remove(&heap);
    }

    /// Report the parts of the metadata available without waiting on a lock.
    pub(crate) fn describe(&self, at: u64) -> String {
        let pool = match self.memory.try_contains(at) {
            Some(true) => "in a pool",
            Some(false) => "in no pool",
            None => "pool mapping table unavailable",
        };
        let Ok(views) = self.views.try_lock() else {
            return format!("{pool}, buffer view table unavailable");
        };
        let over: Vec<String> = views
            .iter()
            .flat_map(|(&heap, slots)| {
                slots
                    .iter()
                    .filter(move |&(_, &(start, len))| at >= start && at - start < len)
                    .map(move |(&index, &(start, len))| {
                        format!("heap {heap:#x}[{index}] {start:#x}+{len:#x}")
                    })
            })
            .take(8)
            .collect();
        format!(
            "{pool}, {} buffer views over it {}",
            over.len(),
            over.join(", ")
        )
    }
}

impl PoolObserver for PoolTable {
    fn pool_created(&self, pool: &PoolInfo) {
        let mut g = self.lock();
        g.generation += 1;
        g.pools.insert(pool.id, pool.clone());
    }

    fn pool_released(&self, pool: &PoolInfo) {
        let mut g = self.lock();
        g.generation += 1;
        g.pools.remove(&pool.id);
        let gone: Vec<(usize, usize)> = g
            .maps
            .iter()
            .filter(|(_, m)| m.pool == pool.id)
            .map(|(&at, m)| (at, m.len))
            .collect();
        for (at, len) in gone {
            g.maps.remove(&at);
            g.note(at, len);
        }
        g.released.push(pool.id);
    }

    fn mapped(&self, at: usize, len: usize, pool: &PoolInfo, offset: usize) {
        let mut g = self.lock();
        g.pools.entry(pool.id).or_insert_with(|| pool.clone());
        g.cut(at, len);
        g.maps.insert(
            at,
            Map {
                len,
                pool: pool.id,
                offset,
            },
        );
    }

    fn unmapped(&self, at: usize, len: usize) {
        self.lock().cut(at, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(id: usize) -> PoolInfo {
        PoolInfo {
            id,
            base: 0x1000_0000 * id,
            len: 0x100_0000,
            memory_entry: None,
        }
    }

    #[test]
    fn maps_cut_and_resolve() {
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        t.mapped(0x7000_0000, 0x40_0000, &p, 0x1_0000);
        let (info, ptr) = t.resolve(0x7000_0100, 16).unwrap();
        assert_eq!((info.id, ptr as usize), (1, p.base + 0x1_0100));
        // Unmapping the middle leaves both ends, still showing the right offsets.
        t.unmapped(0x7010_0000, 0x10_0000);
        assert!(t.resolve(0x7010_0000, 1).is_none());
        let (_, ptr) = t.resolve(0x7020_0000, 4).unwrap();
        assert_eq!(ptr as usize, p.base + 0x1_0000 + 0x20_0000);
        // A range across the hole is refused; a remap over it replaces what it showed.
        assert!(t.resolve(0x700F_FFF0, 0x20).is_none());
        let q = pool(2);
        t.mapped(0x7000_0000, 0x1000, &q, 0);
        assert_eq!(t.resolve(0x7000_0010, 1).unwrap().0.id, 2);
        assert_eq!(t.resolve(0x7000_1000, 1).unwrap().0.id, 1);
        t.pool_released(&p);
        assert!(t.resolve(0x7000_1000, 1).is_none());
        assert_eq!(t.take_released(), vec![1]);
    }

    #[test]
    fn a_range_can_cross_consecutive_mappings_of_one_pool() {
        const PAGE: usize = 64 << 10;
        let t = PoolTable::default();
        let p = pool(1);
        let base = 0x7000_0000;
        t.mapped(base, PAGE, &p, 2 * PAGE);
        t.mapped(base + PAGE, PAGE, &p, 3 * PAGE);
        t.mapped(base + 2 * PAGE, PAGE, &p, 4 * PAGE);

        let start = (base + PAGE - 8) as u64;
        let len = (PAGE + 16) as u64;
        let (info, ptr) = t.resolve(start, len).unwrap();
        assert_eq!(info.id, p.id);
        assert_eq!(ptr as usize, p.base + 3 * PAGE - 8);

        // Repeating the same physical page (as tiled mappings can) is not contiguous.
        t.mapped(base + PAGE, PAGE, &p, 2 * PAGE);
        assert!(t.resolve(start, len).is_none());

        let q = pool(2);
        t.mapped(base + PAGE, PAGE, &q, 3 * PAGE);
        assert!(t.resolve(start, len).is_none());

        t.unmapped(base + PAGE, PAGE);
        assert!(t.resolve(start, len).is_none());
    }

    #[test]
    fn a_crash_report_can_read_either_table_while_the_other_is_locked() {
        let memory = Arc::new(PoolTable::default());
        let address = 0x7000_0000;
        memory.mapped(address as usize, 0x1_0000, &pool(1), 0);
        let info = CrashAddresses::new(memory.clone());
        let heap = 0x8000_0000;
        info.set_buffer_view(heap, 3, address, 64);

        {
            let _pool_lock = memory.lock();
            let report = info.describe(address);
            assert!(
                report.starts_with("pool mapping table unavailable,"),
                "{report}"
            );
            assert!(report.contains("heap 0x80000000[3]"), "{report}");
        }
        {
            let _view_lock = info.lock_views();
            assert_eq!(
                info.describe(address),
                "in a pool, buffer view table unavailable"
            );
        }

        info.remove_buffer_view(heap, 3);
        assert!(info.describe(address).contains("0 buffer views"));
        info.set_buffer_view(heap, 4, address, 64);
        info.remove_heap(heap);
        assert!(info.describe(address).contains("0 buffer views"));
    }

    #[test]
    fn a_range_over_adjacent_mappings_resolves_when_the_pool_continues() {
        let t = PoolTable::default();
        let (p, q) = (pool(1), pool(2));
        t.pool_created(&p);
        t.pool_created(&q);
        // Three 64 KiB pages of one allocation, reported one at a time, in pool order.
        for i in 0..3 {
            t.mapped(0x7000_0000 + i * 0x1_0000, 0x1_0000, &p, 0x4_0000 + i * 0x1_0000);
        }
        // Over one boundary, and over all three.
        let (info, ptr) = t.resolve(0x7000_fff0, 0x20).unwrap();
        assert_eq!((info.id, ptr as usize), (1, p.base + 0x4_fff0));
        let (_, ptr) = t.resolve(0x7000_8000, 0x2_0000).unwrap();
        assert_eq!(ptr as usize, p.base + 0x4_8000);
        // Up to the last byte, not past it.
        assert!(t.resolve(0x7000_0000, 0x3_0000).is_some());
        assert!(t.resolve(0x7000_0000, 0x3_0001).is_none());
        // The next title page shows the same pool, but not the bytes that follow in it.
        t.mapped(0x7003_0000, 0x1_0000, &p, 0x9_0000);
        assert!(t.resolve(0x7002_fff0, 0x20).is_none());
        assert!(t.resolve(0x7003_0000, 0x10).is_some());
        // The next title page shows another pool at the offset that would follow.
        t.mapped(0x7003_0000, 0x1_0000, &q, 0x7_0000);
        assert!(t.resolve(0x7002_fff0, 0x20).is_none());
        // A length that wraps is refused, not followed.
        assert!(t.resolve(0x7000_0000, u64::MAX).is_none());
    }

    #[test]
    fn an_empty_mapping_ends_the_walk() {
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        t.mapped(0x7000_0000, 0x1_0000, &p, 0x4_0000);
        // Empty, at the next address, with the pool and offset that would continue the first.
        t.mapped(0x7001_0000, 0, &p, 0x5_0000);
        assert!(t.resolve(0x7000_fff0, 0x20).is_none());
        // What lies wholly in the first mapping still resolves.
        assert_eq!(t.resolve(0x7000_fff0, 0x10).unwrap().1 as usize, p.base + 0x4_fff0);
        // An address in the empty mapping is in no mapping.
        t.unmapped(0x7000_0000, 0x1_0000);
        assert!(t.resolve(0x7001_0000, 1).is_none());
    }

    #[test]
    fn pages_since_names_the_pages_that_changed_and_what_they_show() {
        const P: usize = TABLE_PAGE;
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        for i in 0..4 {
            t.mapped(0x7000_0000 + i * P, P, &p, i * P);
        }
        let at = |pool_at: usize| PageTarget {
            pool: 1,
            base: p.base,
            pool_len: p.len as u64,
            pool_at: pool_at as u64,
        };
        let (g, pages) = t.pages();
        assert_eq!(pages.len(), 4);
        assert_eq!(pages[2], (0x7002_0000, at(2 * P)));
        assert_eq!(t.pages_since(g).unwrap(), (g, vec![]));

        // An unmap, a remap to another pool offset, a quarter page (not all of a page) and a pool
        // creation (no page).
        t.unmapped(0x7001_0000, P);
        t.mapped(0x7003_0000, P, &p, 8 * P);
        t.mapped(0x7010_4000, 0x4000, &p, 0x4000);
        t.pool_created(&pool(2));
        let (g2, changed) = t.pages_since(g).unwrap();
        assert!(g2 > g);
        assert_eq!(
            changed,
            vec![(0x7001_0000, None), (0x7003_0000, Some(at(8 * P))), (0x7010_0000, None)]
        );
        assert_eq!(t.pages_since(g2).unwrap(), (g2, vec![]));

        // The three quarters left, in pool order: now all of the page shows the pool.
        for i in [0, 2, 3] {
            t.mapped(0x7010_0000 + i * 0x4000, 0x4000, &p, i * 0x4000);
        }
        let (g3, changed) = t.pages_since(g2).unwrap();
        assert_eq!(changed, vec![(0x7010_0000, Some(at(0)))]);

        // A released pool takes its pages with it.
        t.pool_released(&p);
        let (_, changed) = t.pages_since(g3).unwrap();
        assert_eq!(
            changed,
            vec![(0x7000_0000, None), (0x7002_0000, None), (0x7003_0000, None), (0x7010_0000, None)]
        );
        assert_eq!(t.pages().1, vec![]);
    }

    #[test]
    fn a_table_further_behind_than_the_journal_is_built_again() {
        let t = PoolTable::default();
        let p = pool(1);
        t.pool_created(&p);
        let start = t.generation();
        for i in 0..JOURNAL + 1 {
            t.mapped(0x7000_0000 + (i % 8) * TABLE_PAGE, TABLE_PAGE, &p, 0);
        }
        assert!(t.pages_since(start).is_none());
        // A table current a few changes ago is still told what changed.
        let recent = t.generation() - 3;
        assert_eq!(t.pages_since(recent).unwrap().1.len(), 3);
    }

    #[test]
    fn cut_spans_multiple_mappings_and_keeps_their_edges() {
        let t = PoolTable::default();
        let p = pool(1);
        for (base, offset) in [(0x1000, 0), (0x3000, 0x4000), (0x6000, 0x8000)] {
            t.mapped(base, 0x2000, &p, offset);
        }
        let before = t.generation();
        t.unmapped(0x2000, 0x5000);
        assert_eq!(t.generation(), before + 1);
        assert_eq!(t.resolve(0x1fff, 1).unwrap().1 as usize, p.base + 0xfff);
        for addr in [0x2000, 0x3000, 0x6000, 0x6fff] {
            assert!(t.resolve(addr, 1).is_none());
        }
        assert_eq!(t.resolve(0x7000, 1).unwrap().1 as usize, p.base + 0x9000);
    }
}
