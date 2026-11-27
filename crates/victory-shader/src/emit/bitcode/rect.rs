//! Replay a vertex program for the three supplied corners, preserving its control flow.

use super::*;

impl Module {
    /// Output channels whose live value graph includes the fourth-corner sum and difference.
    #[cfg(test)]
    pub(crate) fn completed_outputs(&self) -> Vec<(u32, u8)> {
        let plan = self.plan();
        let mut outputs = Vec::new();
        for block in plan.layout.iter().map(|&b| &self.blocks[b]) {
            for &i in &block.body {
                let Op::Call(f, args) = &self.insts[i as usize].op else {
                    continue;
                };
                if !self.funcs[*f as usize]
                    .name
                    .starts_with("dx.op.storeOutput.")
                {
                    continue;
                }
                let mut seen = std::collections::HashSet::new();
                let mut work = vec![args[4]];
                let mut completed = false;
                while let Some(v) = work.pop() {
                    let V::Inst(i) = v else { continue };
                    if !seen.insert(i) {
                        continue;
                    }
                    let inst = &self.insts[i as usize];
                    if let Op::Bin(Bin::Sub, V::Inst(sum), _) = inst.op
                        && self.types[inst.ty.0 as usize] == TypeDef::Float
                        && matches!(self.insts[sum as usize].op, Op::Bin(Bin::Add, _, _))
                    {
                        completed = true;
                    }
                    for_each_operand(&inst.op, |v| work.push(v));
                }
                if completed {
                    outputs.push((
                        self.int_value(args[1]).unwrap() as u32,
                        self.int_value(args[3]).unwrap() as u8,
                    ));
                }
            }
        }
        outputs
    }

    /// Replace the entry point with three successive copies of its body. System input
    /// `vertex_id` takes `ids[k]` in copy k; instance IDs and resource reads are unchanged.
    /// Output stores become SSA values, joined at every exit, so conditional exports and
    /// loops complete before the next evaluation. The caller writes the final outputs.
    fn rectangle_corners(
        &mut self,
        vertex_id: u32,
        ids: [V; 3],
        body: Vec<Block>,
        insts: Vec<Inst>,
        outputs: &[(u32, u8, Ty)],
    ) -> (usize, Vec<Vec<V>>) {
        let mut preds = vec![Vec::new(); body.len()];
        for (b, block) in body.iter().enumerate() {
            for &to in successors(block.term).iter() {
                if !preds[to].contains(&b) {
                    preds[to].push(b);
                }
            }
        }
        let mut from = 0;
        let mut corners = Vec::new();
        for id in ids {
            let base = self.blocks.len();
            self.blocks.extend(body.iter().cloned());
            let end = self.new_block();
            self.terminate(from, Term::Br(base));
            let first = self.insts.len() as u32;
            let value = |v| match v {
                V::Inst(i) => V::Inst(first + i),
                _ => v,
            };
            for inst in &insts {
                let mut inst = inst.clone();
                inst.op = match inst.op {
                    Op::Bin(op, a, b) => Op::Bin(op, value(a), value(b)),
                    Op::Cast(op, a) => Op::Cast(op, value(a)),
                    Op::Cmp(op, a, b) => Op::Cmp(op, value(a), value(b)),
                    Op::Select(a, b, c) => Op::Select(value(a), value(b), value(c)),
                    Op::Extract(a, n) => Op::Extract(value(a), n),
                    Op::Call(f, args) => {
                        if self.funcs[f as usize].name == "dx.op.loadInput.i32"
                            && self.int_value(args[1]) == Some(i64::from(vertex_id))
                        {
                            Op::Bin(Bin::Add, id, self.i32(0))
                        } else {
                            Op::Call(f, args.into_iter().map(value).collect())
                        }
                    }
                    Op::Phi(inc) => {
                        Op::Phi(inc.into_iter().map(|(v, b)| (value(v), base + b)).collect())
                    }
                    Op::Gep(t, p, idx) => {
                        Op::Gep(t, value(p), idx.into_iter().map(value).collect())
                    }
                    Op::Load(p) => Op::Load(value(p)),
                    Op::Store(p, v) => Op::Store(value(p), value(v)),
                    Op::AtomicRmw(op, p, v) => Op::AtomicRmw(op, value(p), value(v)),
                    Op::CmpXchg(p, a, b) => Op::CmpXchg(value(p), value(a), value(b)),
                };
                self.insts.push(inst);
            }
            // The value on entry to a block, or the last export in that block, for each
            // output component. Phis also carry a prior export through loops without one.
            let incoming: Vec<Vec<_>> = (0..body.len())
                .map(|b| {
                    outputs
                        .iter()
                        .map(|&(_, _, ty)| {
                            if b == 0 {
                                self.undef(ty)
                            } else {
                                self.phi(base + b, ty)
                            }
                        })
                        .collect()
                })
                .collect();
            let mut outgoing = incoming.clone();
            let mut exits = Vec::new();
            for (b, original) in body.iter().enumerate() {
                let mut kept = Vec::new();
                for &i in &original.body {
                    if let Op::Call(f, args) = &insts[i as usize].op
                        && self.funcs[*f as usize]
                            .name
                            .starts_with("dx.op.storeOutput.")
                    {
                        let key = (
                            self.int_value(args[1]).unwrap() as u32,
                            self.int_value(args[3]).unwrap() as u8,
                        );
                        let o = outputs
                            .iter()
                            .position(|&(row, col, _)| (row, col) == key)
                            .unwrap();
                        outgoing[b][o] = value(args[4]);
                    } else {
                        kept.push(first + i);
                    }
                }
                let block = &mut self.blocks[base + b];
                // New output phis were appended after the original program's phis.
                for phi in block.phis.iter_mut().take(original.phis.len()) {
                    *phi += first;
                }
                block.body = kept;
                block.term = Some(match original.term.unwrap_or(Term::Ret) {
                    Term::Br(to) => Term::Br(base + to),
                    Term::CondBr(c, a, z) => Term::CondBr(value(c), base + a, base + z),
                    Term::Ret => {
                        exits.push(b);
                        Term::Br(end)
                    }
                });
            }
            for b in 1..body.len() {
                for (o, &phi) in incoming[b].iter().enumerate() {
                    self.set_phi(
                        phi,
                        preds[b]
                            .iter()
                            .map(|&p| (outgoing[p][o], base + p))
                            .collect(),
                    );
                }
            }
            let result = outputs
                .iter()
                .enumerate()
                .map(|(o, &(_, _, ty))| {
                    let phi = self.phi(end, ty);
                    self.set_phi(
                        phi,
                        exits.iter().map(|&b| (outgoing[b][o], base + b)).collect(),
                    );
                    phi
                })
                .collect();
            corners.push(result);
            from = end;
        }
        (from, corners)
    }

    /// Six converted SV_VertexIDs per rectangle. The host doubles first vertex in both
    /// Metal's vertexStart and its draw parameters; the converter removes that base,
    /// so replayed console IDs stay relative to the draw, just as in ordinary shaders.
    /// The original program is saved before making the new prologue.
    pub(crate) fn rectangle_values(
        &mut self,
        vertex_id: u32,
        load: V,
        outputs: &[(u32, u8, Ty)],
    ) -> (usize, V, Vec<Vec<V>>) {
        let body = std::mem::take(&mut self.blocks);
        let insts = std::mem::take(&mut self.insts);
        self.new_block();
        let i32t = self.int(32);
        let args = vec![
            self.i32(4),
            self.i32(vertex_id),
            self.i32(0),
            self.i8(0),
            self.undef(i32t),
        ];
        let vid = self.call(0, load, args);
        let six = self.i32(6);
        let three = self.i32(3);
        let rect = self.bin(0, Bin::UDiv, vid, six);
        let base = self.bin(0, Bin::Mul, rect, three);
        let corner = self.bin(0, Bin::URem, vid, six);
        let one = self.i32(1);
        let two = self.i32(2);
        let ids = [
            base,
            self.bin(0, Bin::Add, base, one),
            self.bin(0, Bin::Add, base, two),
        ];
        let (end, values) = self.rectangle_corners(vertex_id, ids, body, insts, outputs);
        (end, corner, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{container::SigElement, shader::Shader};

    /// Execute only the scalar operations used by this fixture, following the generated CFG.
    /// Reads before definitions and phi edges from the wrong predecessor fail the test.
    fn outputs(m: &Module, vertex: u32, instance: u32) -> Vec<u32> {
        let mut values = HashMap::new();
        let mut result = vec![0; 5];
        let live = m.live();
        let get = |v: V, values: &HashMap<u32, u32>| match v {
            V::Const(i) => match m.consts[i as usize].1 {
                Const::Int(x) => x as u32,
                Const::Float(x) => x,
                Const::Undef => 0,
                _ => panic!("not a scalar"),
            },
            V::Inst(i) => *values
                .get(&i)
                .unwrap_or_else(|| panic!("read before definition: {v:?}")),
            _ => panic!("not a value"),
        };
        let (mut block, mut pred_block) = (0, usize::MAX);
        for _ in 0..1000 {
            let b = &m.blocks[block];
            let phis: Vec<_> = b
                .phis
                .iter()
                .filter(|&&i| live[i as usize])
                .map(|&i| {
                    let Op::Phi(inc) = &m.insts[i as usize].op else {
                        panic!("not a phi")
                    };
                    let v = inc
                        .iter()
                        .find(|v| v.1 == pred_block)
                        .expect("phi predecessor")
                        .0;
                    (i, get(v, &values))
                })
                .collect();
            values.extend(phis);
            for &i in b.body.iter().filter(|&&i| live[i as usize]) {
                let inst = &m.insts[i as usize];
                let v = match &inst.op {
                    Op::Bin(op, a, b) => {
                        let (a, b) = (get(*a, &values), get(*b, &values));
                        if m.types[inst.ty.0 as usize] == TypeDef::Float {
                            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                            match op {
                                Bin::Add => a + b,
                                Bin::Sub => a - b,
                                Bin::Mul => a * b,
                                _ => panic!("float op"),
                            }
                            .to_bits()
                        } else {
                            match op {
                                Bin::Add => a.wrapping_add(b),
                                Bin::Sub => a.wrapping_sub(b),
                                Bin::Mul => a.wrapping_mul(b),
                                Bin::UDiv => a / b,
                                Bin::URem => a % b,
                                Bin::And => a & b,
                                _ => panic!("integer op"),
                            }
                        }
                    }
                    Op::Cmp(op, a, b) => {
                        let (a, b) = (get(*a, &values), get(*b, &values));
                        u32::from(match *op {
                            pred::EQ => a == b,
                            pred::IULT => a < b,
                            pred::OGT => f32::from_bits(a) > f32::from_bits(b),
                            _ => panic!("comparison"),
                        })
                    }
                    Op::Select(c, t, f) => {
                        get(if get(*c, &values) != 0 { *t } else { *f }, &values)
                    }
                    Op::Cast(Cast::UiToFp, v) => (get(*v, &values) as f32).to_bits(),
                    Op::Call(f, args) => match m.funcs[*f as usize].name.as_str() {
                        "dx.op.loadInput.i32" => {
                            if get(args[1], &values) == 0 {
                                vertex
                            } else {
                                instance
                            }
                        }
                        "dx.op.storeOutput.f32" => {
                            let row = get(args[1], &values);
                            let col = get(args[3], &values);
                            result[if row == 0 { col as usize } else { 4 }] = get(args[4], &values);
                            0
                        }
                        name => panic!("unexpected call {name}"),
                    },
                    op => panic!("unexpected operation {op:?}"),
                };
                values.insert(i, v);
            }
            let next = match b.term.unwrap() {
                Term::Br(to) => to,
                Term::CondBr(c, t, f) => {
                    if get(c, &values) != 0 {
                        t
                    } else {
                        f
                    }
                }
                Term::Ret => return result,
            };
            pred_block = block;
            block = next;
        }
        panic!("loop did not finish");
    }

    #[test]
    fn replay_keeps_loop_phis_conditional_exports_and_instance_ids() {
        let mut s = Shader::new(false);
        let element = |name: &str, kind, comp, row, cols| SigElement {
            name: name.into(),
            kind,
            comp,
            row: Some(row),
            cols,
            index: 0,
            col: 0,
            interp: 0,
            usage: (1 << cols) - 1,
            half: false,
        };
        s.inputs = vec![
            element("SV_VertexID", 1, 5, 0, 1),
            element("SV_InstanceID", 2, 5, 1, 1),
        ];
        s.outputs = vec![
            element("SV_Position", 3, 9, 0, 4),
            element("TEXCOORD", 0, 9, 1, 1),
        ];
        let inputs: Vec<_> = (0..2)
            .map(|id| {
                let args = [s.m.i32(id), s.m.i32(0), s.m.i8(0), s.m.undef(s.t.i32)];
                s.op(0, "loadInput.i32", Attr::ReadNone, s.t.i32, 4, &args)
            })
            .collect();
        let (zero, one, two, three) = (s.m.i32(0), s.m.i32(1), s.m.i32(2), s.m.i32(3));
        let (fz, fo) = (s.m.f32(0), s.m.f32(1f32.to_bits()));
        let c = s.m.bin(0, Bin::URem, inputs[0], three);
        let r = s.m.bin(0, Bin::UDiv, inputs[0], three);
        let r = s.m.cast(0, Cast::UiToFp, r, s.t.f32);
        let inst = s.m.cast(0, Cast::UiToFp, inputs[1], s.t.f32);
        let cf = s.m.cast(0, Cast::UiToFp, c, s.t.f32);
        let x = s.m.cmp(0, pred::EQ, c, two);
        let y = s.m.cmp(0, pred::EQ, c, one);
        let x = s.m.select(0, x, fo, fz);
        let y = s.m.select(0, y, fo, fz);
        let x = s.m.bin(0, Bin::Add, r, x);
        let y = s.m.bin(0, Bin::Add, inst, y);
        // A shear makes the supplied parallelogram non-orthogonal. The pivot stays p0.
        let half = s.m.f32(0.5f32.to_bits());
        let shear = s.m.bin(0, Bin::Mul, x, half);
        let y = s.m.bin(0, Bin::Add, y, shear);
        let (head, body, end, left, right) = (
            s.m.new_block(),
            s.m.new_block(),
            s.m.new_block(),
            s.m.new_block(),
            s.m.new_block(),
        );
        s.m.terminate(0, Term::Br(head));
        let n = s.m.phi(head, s.t.i32);
        let sum = s.m.phi(head, s.t.f32);
        let more = s.m.cmp(head, pred::IULT, n, c);
        s.m.terminate(head, Term::CondBr(more, body, end));
        let next = s.m.bin(body, Bin::Add, n, one);
        let added = s.m.bin(body, Bin::Add, sum, cf);
        s.m.terminate(body, Term::Br(head));
        s.m.set_phi(n, vec![(zero, 0), (next, body)]);
        s.m.set_phi(sum, vec![(fz, 0), (added, body)]);
        // Position is exported before the branch, the varying on either of two return paths.
        for (col, v) in [x, y, fz, fo].into_iter().enumerate() {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), v];
            s.op(end, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        let is_zero = s.m.cmp(end, pred::EQ, c, zero);
        s.m.terminate(end, Term::CondBr(is_zero, left, right));
        let varying = s.m.bin(right, Bin::Add, sum, inst);
        for (b, v) in [(left, inst), (right, varying)] {
            let args = [s.m.i32(1), s.m.i32(0), s.m.i8(0), v];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
            s.m.terminate(b, Term::Ret);
        }
        crate::emit::rect::expand_vertex(&mut s).unwrap();
        for instance in [0, 7] {
            for vid in 0..12 {
                let r = (vid / 6) as f32;
                let i = instance as f32 + r * 0.5;
                let (x, y, u) = [
                    (r, i, instance as f32),
                    (r, i + 1., instance as f32 + 1.),
                    (r + 1., i + 0.5, instance as f32 + 4.),
                    (r + 1., i + 0.5, instance as f32 + 4.),
                    (r, i + 1., instance as f32 + 1.),
                    (r + 1., i + 1.5, instance as f32 + 5.),
                ][vid as usize % 6];
                assert_eq!(
                    outputs(&s.m, vid, instance),
                    [x, y, 0., 1., u].map(f32::to_bits),
                    "vertex {vid}, instance {instance}"
                );
            }
        }
        s.finish(); // Also exercise dead-code pruning, phi remapping and serialization.
    }
}
