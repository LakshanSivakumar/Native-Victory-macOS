//! Scans of the DXIL that `victory-shader` emits, shared by the host (`Binding`) and the Metal
//! backend (which converter switches a module needs, and so its cache key).

/// Whether a DXBC-wrapped DXIL module emitted by `victory-shader` calls `dx.op.bufferLoad.*`,
/// `dx.op.bufferStore.*`, or `dx.op.getDimensions` on a `TypedBuffer` (`ResourceProperties` kind
/// `10`). Defaults conservatively to `true` on unrecognized or abbreviated bitcode.
pub fn dxil_uses_typed_buffers(dxil: &[u8]) -> bool {
    dxil_bitcode(dxil)
        .and_then(scan_dxil_typed_buffers)
        .unwrap_or(true)
}

fn dxil_bitcode(dxil: &[u8]) -> Option<&[u8]> {
    if dxil.len() < 32 || &dxil[..4] != b"DXBC" {
        return None;
    }
    let parts = u32::from_le_bytes(dxil[28..32].try_into().ok()?) as usize;
    if 32usize.checked_add(parts.checked_mul(4)?)? > dxil.len() {
        return None;
    }
    for i in 0..parts {
        let off = u32::from_le_bytes(dxil[32 + i * 4..36 + i * 4].try_into().ok()?) as usize;
        let hdr = dxil.get(off..off.checked_add(8)?)?;
        if &hdr[..4] != b"DXIL" {
            continue;
        }
        let part_len = u32::from_le_bytes(hdr[4..8].try_into().ok()?) as usize;
        let part = dxil.get(off + 8..(off + 8).checked_add(part_len)?)?;
        if part.len() < 24 || &part[8..12] != b"DXIL" {
            return None;
        }
        let bc_off = 8usize.checked_add(u32::from_le_bytes(part[16..20].try_into().ok()?) as usize)?;
        let bc_len = u32::from_le_bytes(part[20..24].try_into().ok()?) as usize;
        let bc = part.get(bc_off..bc_off.checked_add(bc_len)?)?;
        return (bc.len() >= 4 && &bc[..4] == b"BC\xC0\xDE" && bc.len().is_multiple_of(4))
            .then(|| &bc[4..]);
    }
    None
}

fn scan_dxil_typed_buffers(bc: &[u8]) -> Option<bool> {
    struct BitReader<'a> {
        words: &'a [u8],
        bit: usize,
    }
    impl BitReader<'_> {
        fn bits(&mut self, width: u32, end: usize) -> Option<u64> {
            let mut left = width as usize;
            if self.bit.checked_add(left)? > end {
                return None;
            }
            let (mut out, mut shift) = (0u64, 0usize);
            while left > 0 {
                let word_idx = self.bit >> 5;
                let bit_in_word = self.bit & 31;
                let bytes: [u8; 4] = self.words.get(word_idx * 4..word_idx * 4 + 4)?.try_into().ok()?;
                let word = u32::from_le_bytes(bytes);
                let take = left.min(32 - bit_in_word);
                let mask = if take == 32 { u64::from(u32::MAX) } else { (1u64 << take) - 1 };
                out |= ((u64::from(word) >> bit_in_word) & mask) << shift;
                self.bit += take;
                shift += take;
                left -= take;
            }
            Some(out)
        }
        fn vbr(&mut self, width: u32, end: usize) -> Option<u64> {
            let step = width - 1;
            let hi = 1u64 << step;
            let (mut out, mut shift) = (0u64, 0u32);
            loop {
                let chunk = self.bits(width, end)?;
                if shift >= 64 {
                    return None;
                }
                out |= (chunk & (hi - 1)) << shift;
                if chunk & hi == 0 {
                    return Some(out);
                }
                shift += step;
            }
        }
        fn align32(&mut self) {
            self.bit = (self.bit + 31) & !31;
        }
    }

    let total_bits = bc.len().checked_mul(8)?;
    let mut r = BitReader { words: bc, bit: 0 };
    if r.bits(2, total_bits)? != 1 || r.vbr(8, total_bits)? != 8 {
        return None;
    }
    let mod_w = r.vbr(4, total_bits)? as u32;
    r.align32();
    let mod_words = r.bits(32, total_bits)? as usize;
    let mod_end = r.bit.checked_add(mod_words.checked_mul(32)?)?;
    if mod_end > total_bits {
        return None;
    }

    let mut props_ty: Option<u64> = None;
    let mut value_base = 0u64;
    let mut has_typed_props = false;
    let mut has_get_dims = false;
    let mut scratch = Vec::new();
    while r.bit < mod_end {
        match r.bits(mod_w, mod_end)? {
            0 => break,
            3 => {
                let code = r.vbr(6, mod_end)?;
                let n = r.vbr(6, mod_end)? as usize;
                for _ in 0..n {
                    r.vbr(6, mod_end)?;
                }
                if code == 7 || code == 8 {
                    value_base += 1;
                }
            }
            1 => {
                let sub_id = r.vbr(8, mod_end)?;
                let sub_w = r.vbr(4, mod_end)? as u32;
                r.align32();
                let sub_words = r.bits(32, mod_end)? as usize;
                let sub_end = r.bit.checked_add(sub_words.checked_mul(32)?)?;
                if sub_end > mod_end {
                    return None;
                }
                match sub_id {
                    // TYPE_BLOCK: locate `dx.types.ResourceProperties`.
                    17 => {
                        let (mut ty_idx, mut is_props_name) = (0u64, false);
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    if code == 19 {
                                        scratch.clear();
                                        for _ in 0..n {
                                            scratch.push(r.vbr(6, sub_end)? as u8);
                                        }
                                        is_props_name = scratch == b"dx.types.ResourceProperties";
                                    } else {
                                        for _ in 0..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                        if code != 1 {
                                            if code == 20 && is_props_name {
                                                props_ty = Some(ty_idx);
                                            }
                                            is_props_name = false;
                                            ty_idx += 1;
                                        }
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    // CONSTANTS_BLOCK: check for `ResourceProperties` with resource kind 10 (`TypedBuffer`).
                    11 => {
                        let mut cur_ty = None;
                        let mut int_low: Vec<Option<u8>> = Vec::new();
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    if code == 1 && n >= 1 {
                                        cur_ty = Some(r.vbr(6, sub_end)?);
                                        for _ in 1..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                    } else if code == 4 && n >= 1 {
                                        let raw = r.vbr(6, sub_end)?;
                                        for _ in 1..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                        let val = if raw & 1 == 0 {
                                            (raw >> 1) as i64
                                        } else {
                                            -((raw >> 1) as i64)
                                        };
                                        int_low.push(Some((val as u64 & 0xFF) as u8));
                                    } else if code == 7
                                        && cur_ty == props_ty
                                        && props_ty.is_some()
                                        && n >= 1
                                    {
                                        let id0 = r.vbr(6, sub_end)?;
                                        for _ in 1..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                        if let Some(idx) = id0.checked_sub(value_base)
                                            && int_low.get(idx as usize).copied().flatten()
                                                == Some(10)
                                        {
                                            has_typed_props = true;
                                        }
                                        int_low.push(None);
                                    } else {
                                        for _ in 0..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                        int_low.push(None);
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    // VALUE_SYMTAB_BLOCK: inspect kept `dx.op.*` declarations.
                    14 => {
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    if code == 1 && n >= 1 {
                                        let _val_id = r.vbr(6, sub_end)?;
                                        scratch.clear();
                                        for _ in 1..n {
                                            scratch.push(r.vbr(6, sub_end)? as u8);
                                        }
                                        if scratch.starts_with(b"dx.op.bufferLoad.")
                                            || scratch.starts_with(b"dx.op.bufferStore.")
                                        {
                                            return Some(true);
                                        }
                                        if scratch == b"dx.op.getDimensions" {
                                            has_get_dims = true;
                                        }
                                    } else {
                                        for _ in 0..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    _ => r.bit = sub_end,
                }
            }
            _ => return None,
        }
    }
    Some(has_get_dims && has_typed_props)
}
