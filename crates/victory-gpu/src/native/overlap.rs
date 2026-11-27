//! `VICTORY_TRACE_TEXTURE_OVERLAP=1`: how host texture copies overlap in title memory (#352).
//! Once a second: of the bytes the copies hold more than once, those under copies with an
//! identical layout at the same start (one texture could serve them) and those under different
//! layouts; and how often an upload took a range from another copy. Nothing here changes which
//! texture a draw samples.

use std::collections::{BTreeMap, HashMap};
use victory_formats::texture::Layout;

/// The overlap trace's state, kept only while the trace is on.
#[derive(Default)]
pub struct Overlap {
    /// The copy that last uploaded each range: start, to (end, key). Ranges do not overlap.
    owners: BTreeMap<usize, (usize, usize)>,
    /// Uploads that took bytes from another copy, and those bytes: from a live copy with an
    /// identical layout at the same start, from a live copy with another, from a copy no longer
    /// there (released or given up: memory reused).
    changes: [u64; 3],
    bytes: [u64; 3],
    pub(super) logged: Option<std::time::Instant>,
}

/// What an earlier owner of a range was to the copy that took it.
#[derive(Clone, Copy)]
pub(super) enum Was {
    Identical = 0,
    Different = 1,
    Gone = 2,
}

pub(super) fn on() -> bool {
    env_set!("VICTORY_TRACE_TEXTURE_OVERLAP")
}

impl Overlap {
    /// Copy `key` uploaded `[start, end)`: it owns that range now. Returns the other copies it
    /// took bytes from, with how many.
    pub(super) fn take(&mut self, start: usize, end: usize, key: usize) -> HashMap<usize, u64> {
        let mut from = HashMap::new();
        let hit: Vec<usize> = self
            .owners
            .range(..end)
            .rev()
            .take_while(|&(_, &(e, _))| e > start)
            .map(|(&s, _)| s)
            .collect();
        for s in hit {
            let (e, k) = self.owners.remove(&s).expect("listed");
            if k != key {
                *from.entry(k).or_default() += (e.min(end) - s.max(start)) as u64;
            }
            if s < start {
                self.owners.insert(s, (start, k));
            }
            if e > end {
                self.owners.insert(end, (e, k));
            }
        }
        self.owners.insert(start, (end, key));
        from
    }

    pub(super) fn count(&mut self, was: Was, bytes: u64) {
        self.changes[was as usize] += 1;
        self.bytes[was as usize] += bytes;
    }

    /// The counts since the last call: (uploads, bytes) for each [`Was`].
    pub(super) fn take_counts(&mut self) -> [(u64, u64); 3] {
        let (c, b) = (std::mem::take(&mut self.changes), std::mem::take(&mut self.bytes));
        std::array::from_fn(|i| (c[i], b[i]))
    }
}

/// Of the bytes `copies` (start, size, layout) hold more than once (each byte counted once less
/// than the copies over it): those held again by copies with an identical layout at the same start,
/// and those held again by copies that differ (another layout, or another start over the same bytes).
pub(super) fn excess(copies: &[(usize, usize, &Layout)]) -> (u64, u64) {
    let mut kinds: Vec<(usize, usize, &Layout)> = Vec::new();
    let mut identical = 0u64;
    for &(start, size, layout) in copies {
        if kinds.iter().any(|&(s, _, l)| s == start && l == layout) {
            identical += size as u64;
        } else {
            kinds.push((start, size, layout));
        }
    }
    let total: u64 = kinds.iter().map(|k| k.1 as u64).sum();
    let distinct = super::textures::distinct_bytes(kinds.iter().map(|k| (k.0, k.1)).collect());
    (identical, total - distinct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Swizzle};

    fn layout(width: u32, swizzle: Swizzle) -> Layout {
        let desc = Desc { dimension: Dimension::Tex2d, width, height: 16, depth: 1, mips: 1, element_bytes: 4, block: 1, samples: 1, swizzle };
        Layout::new(&desc).unwrap()
    }

    #[test]
    fn an_upload_takes_the_bytes_it_covers_from_their_last_owner() {
        let mut o = Overlap::default();
        assert!(o.take(100, 200, 1).is_empty(), "nobody owned it");
        assert!(o.take(100, 200, 1).is_empty(), "its own range again");
        assert_eq!(o.take(150, 250, 2), HashMap::from([(1, 50)]));
        // Key 3 spans the rest of 1 and part of 2.
        assert_eq!(o.take(120, 160, 3), HashMap::from([(1, 30), (2, 10)]));
        assert_eq!(o.take(0, 1000, 4), HashMap::from([(1, 20), (3, 40), (2, 90)]));
        assert_eq!(o.owners, BTreeMap::from([(0, (1000, 4))]));
    }

    #[test]
    fn the_excess_splits_into_identical_and_different_layouts() {
        let (a, b) = (layout(16, Swizzle::LINEAR), layout(16, Swizzle::SW_64KB_S));
        let size = a.size as usize;
        assert_eq!(excess(&[(0, size, &a)]), (0, 0));
        // Three identical copies at one start: two held again.
        assert_eq!(excess(&[(0, size, &a), (0, size, &a), (0, size, &a)]), (2 * size as u64, 0));
        // Another layout over the same bytes, and the same layout starting inside them.
        assert_eq!(excess(&[(0, size, &a), (0, size, &b)]), (0, size.min(b.size as usize) as u64));
        assert_eq!(excess(&[(0, size, &a), (size / 2, size, &a)]), (0, (size / 2) as u64));
        // Both kinds at once.
        assert_eq!(excess(&[(0, size, &a), (0, size, &a), (0, size, &b)]), (size as u64, size.min(b.size as usize) as u64));
    }
}
