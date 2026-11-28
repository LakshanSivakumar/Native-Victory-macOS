//! Metal mesh output storage, including the alignment between signature elements.

use crate::dxbc::u32_at;
use crate::{Container, FourCC};

pub const METAL_MESH_OUTPUT_BYTES: u32 = 32_768;

/// Whether the 32 KiB mesh output guard is on: `VICTORY_MESH_OVER_LIMIT=1` turns it off, for
/// every check (#528: on an M5 Max with macOS 26.5, Metal made and ran pipelines of 33,792 to
/// 38,400 declared bytes).
pub fn mesh_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MESH_OVER_LIMIT").as_deref() != Ok("1"))
}

/// Whether a mesh program declaring `bytes` of outputs is refused for size, with the guard `on`.
pub fn mesh_over_limit(bytes: u32, on: bool) -> bool {
    on && bytes > METAL_MESH_OUTPUT_BYTES
}

/// Same layout rule as the Wine host's `metal_mesh_bytes`: distance scalars, position,
/// then signature order, each field aligned to its own size. Counts are declared maxima,
/// not the number of vertices a particular dispatch happens to emit. Fail closed on bad DXIL.
pub fn metal_mesh_bytes(dxil: &[u8]) -> Option<u32> {
    let c = Container::parse(dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?.data;
    let info = psv.get(4..4usize.checked_add(u32_at(psv, 0).ok()? as usize)?)?;
    if *info.get(24)? != 13 {
        return None;
    }
    let elements = |kind| -> Option<Vec<(u32, u32)>> {
        let Some(part) = c.part(kind) else {
            return Some(Vec::new());
        };
        let count = u32_at(part.data, 0).ok()? as usize;
        let base = u32_at(part.data, 4).ok()? as usize;
        let records = part
            .data
            .get(base..base.checked_add(count.checked_mul(32)?)?)?;
        records
            .as_chunks::<32>().0.iter()
            .map(|e| {
                let mask = e[24];
                if mask & !15 != 0 {
                    return None;
                }
                Some((u32_at(e, 12).ok()?, mask.count_ones()))
            })
            .collect()
    };
    let layout = |sizes: &mut dyn Iterator<Item = u32>| {
        sizes.filter(|&s| s > 0).try_fold(0u32, |end, size| {
            end.checked_add(size - 1)?
                .checked_div(size)?
                .checked_mul(size)?
                .checked_add(size)
        })
    };
    let vertex = elements(FourCC::OSG1)?;
    let distances: u32 = vertex
        .iter()
        .filter(|e| matches!(e.0, 2 | 3))
        .map(|e| e.1)
        .sum();
    let vb = layout(
        &mut std::iter::repeat_n(4, distances as usize)
            .chain([16])
            .chain(vertex.iter().filter(|e| e.0 != 1).map(|e| 4 * e.1)),
    )?;
    let pb = layout(&mut elements(FourCC::PSG1)?.iter().map(|e| 4 * e.1))?;
    let half = |at| {
        Some(u32::from(u16::from_le_bytes(
            info.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    half(12)?
        .checked_mul(vb)?
        .checked_add(half(14)?.checked_mul(pb)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxbc;
    fn shader(
        vertices: u16,
        primitives: u16,
        vertex: &[(u32, u8)],
        primitive: &[(u32, u8)],
    ) -> Vec<u8> {
        let sig = |els: &[(u32, u8)]| {
            let mut b = vec![0u8; 8 + els.len() * 32];
            b[..4].copy_from_slice(&(els.len() as u32).to_le_bytes());
            b[4..8].copy_from_slice(&8u32.to_le_bytes());
            for (i, &(sv, mask)) in els.iter().enumerate() {
                let at = 8 + i * 32;
                b[at + 12..at + 16].copy_from_slice(&sv.to_le_bytes());
                b[at + 24] = mask;
            }
            b
        };
        let mut psv = vec![0u8; 52];
        psv[..4].copy_from_slice(&48u32.to_le_bytes());
        psv[16..18].copy_from_slice(&vertices.to_le_bytes());
        psv[18..20].copy_from_slice(&primitives.to_le_bytes());
        psv[28] = 13;
        dxbc::build(&[
            (FourCC::PSV0, &psv),
            (FourCC::OSG1, &sig(vertex)),
            (FourCC::PSG1, &sig(primitive)),
        ])
    }
    #[test]
    fn mesh_size_alignment_counts_and_limit() {
        // Position 16, scalar 4, float3 aligned at 24, float4 aligned at 48 => 64.
        assert_eq!(
            metal_mesh_bytes(&shader(
                128,
                64,
                &[(1, 15), (0, 1), (0, 7), (0, 15)],
                &[(0, 1)]
            )),
            Some(128 * 64 + 64 * 4)
        );
        let mut outputs = vec![(1, 15)];
        outputs.extend([(0, 15); 15]);
        assert_eq!(
            metal_mesh_bytes(&shader(128, 0, &outputs, &[])),
            Some(METAL_MESH_OUTPUT_BYTES)
        );
        outputs.push((0, 1));
        assert!(
            metal_mesh_bytes(&shader(128, 0, &outputs, &[])).unwrap() > METAL_MESH_OUTPUT_BYTES
        );
        // Clip/cull scalars precede position, and remain in signature order, as in the reference.
        assert_eq!(
            metal_mesh_bytes(&shader(1, 0, &[(1, 15), (2, 3)], &[])),
            Some(40)
        );
    }
    #[test]
    fn mesh_size_rejects_malformed_or_non_mesh() {
        let b = shader(3, 1, &[(1, 15)], &[]);
        for end in 0..b.len() {
            assert_eq!(metal_mesh_bytes(&b[..end]), None);
        }
        assert_eq!(metal_mesh_bytes(&shader(3, 1, &[(1, 255)], &[])), None);
        let mut b = b;
        // First part starts after the table, PSV runtime shader kind at part + 8 + 28.
        let at = u32_at(&b, 32).unwrap() as usize;
        b[at + 36] = 1;
        assert_eq!(metal_mesh_bytes(&b), None);
    }

    /// The guard refuses only over 32 KiB and only while it is on; every check asks this (#528).
    #[test]
    fn the_mesh_output_guard_refuses_over_32_kib_only_while_on() {
        assert!(mesh_over_limit(33_792, true));
        assert!(!mesh_over_limit(32_768, true));
        assert!(!mesh_over_limit(38_400, false));
    }
}
