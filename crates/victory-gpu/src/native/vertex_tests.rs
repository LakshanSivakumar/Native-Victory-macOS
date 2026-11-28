//! Capture a native host submission at the backend boundary without a GPU or shader converter.

use super::host::NativeHost;
use super::list::NativeList;
use super::memory::{PoolInfo, PoolObserver};
use std::cell::RefCell;
use std::rc::Rc;
use victory_backend::backend::{ComputeBackend, RootLayout, Submission};
use victory_backend::resources::*;

// Group counts, resolved indirect-record address, and resolved count-buffer address/index.
type CapturedDispatch = ([usize; 3], Option<usize>, Option<(usize, u32)>);

#[test]
fn bounded_compute_passes_keep_dispatch_order_and_render_boundaries() {
    let mut host = NativeHost::new(Capture::default());
    host.compute_dispatches_per_pass = 2;
    let mut list = NativeList::default();
    list.set_compute_pipeline(&Rc::new(super::list::ComputePso {
        pso: (), threads: [1; 3], binding: None,
    }));
    for x in 1..=5 { list.dispatch(x, 1, 1).unwrap(); }
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    list.dispatch(6, 1, 1).unwrap();
    list.dispatch(7, 1, 1).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    assert_eq!(*host.backend.pass_kinds.borrow(), ["compute", "compute", "compute", "render", "compute"]);
    let groups: Vec<Vec<_>> = host.backend.compute.borrow().iter()
        .map(|pass| pass.iter().map(|d| d.0[0]).collect()).collect();
    assert_eq!(groups, [vec![1, 2], vec![3, 4], vec![5], vec![6, 7]]);
}

#[test]
fn bounded_compute_passes_count_expanded_indirect_records() {
    let mut host = NativeHost::new(Capture::default());
    host.compute_dispatches_per_pass = 2;
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.set_compute_pipeline(&Rc::new(super::list::ComputePso {
        pso: (), threads: [1; 3], binding: None,
    }));
    list.indirect_records(alias + 0x100, 16, 5, Some(alias + 0x200), |l, r, c| {
        let (count, index) = c.unwrap();
        l.dispatch_indirect_counted(r, count, index)
    }).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    let captured = host.backend.compute.borrow();
    assert_eq!(captured.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 1]);
    let base = mem.as_ptr() as usize;
    let records: Vec<_> = captured.iter().flatten()
        .map(|d| (d.1.unwrap() - base, d.2.unwrap().0 - base, d.2.unwrap().1)).collect();
    assert_eq!(records, [(0x100, 0x200, 0), (0x110, 0x200, 1), (0x120, 0x200, 2),
                         (0x130, 0x200, 3), (0x140, 0x200, 4)]);
}

#[test]
fn zero_compute_limit_keeps_one_pass() {
    let mut host = NativeHost::new(Capture::default());
    host.compute_dispatches_per_pass = 0;
    let mut list = NativeList::default();
    list.set_compute_pipeline(&Rc::new(super::list::ComputePso {
        pso: (), threads: [1; 3], binding: None,
    }));
    for _ in 0..5 { list.dispatch(1, 1, 1).unwrap(); }
    host.execute(&(), &[&list], &[], &[]).unwrap();
    assert_eq!(host.backend.compute.borrow().iter().map(Vec::len).collect::<Vec<_>>(), [5]);
}

#[test]
fn a_draw_with_only_vertex_slot_two_submits_its_buffer() {
    let host = NativeHost::new(Capture::default());
    let pool = PoolInfo {
        id: 1,
        base: 0x40000,
        len: 0x4000,
        memory_entry: None,
    };
    host.memory().mapped(0x80000, pool.len, &pool, 0);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.set_vertex_buffer(2, 0x80100);
    list.draw_instanced(3, 1, 0, 0).unwrap();
    host.execute(&(), &[&list], &[], &[]).unwrap();

    assert_eq!(host.unmapped(), 0, "empty vertex slots dropped the draw");
    assert_eq!(*host.backend.draws.borrow(), [vec![(2, 0x40000, 0x40100)]]);
}

#[derive(Default)]
pub(super) struct Capture {
    compute: RefCell<Vec<Vec<CapturedDispatch>>>,
    pass_kinds: RefCell<Vec<&'static str>>,
    pub(super) views: RefCell<Vec<ViewDesc>>,
    // For each draw, the (slot, pool base, resolved address) bindings received.
    draws: RefCell<Vec<Vec<(usize, usize, usize)>>>,
    host_buffers: RefCell<Vec<Box<[u8]>>>,
    // (heap index, pool base, resolved address, size) for each buffer descriptor write.
    pub(super) buffer_writes: RefCell<Vec<(usize, usize, usize, u32)>>,
    // When set, `submit_passes` fails with it: (whether the command buffer committed, the pass
    // indices dropped from it).
    pub(super) refuse: std::cell::RefCell<Option<(bool, Vec<usize>)>>,
}

impl GraphicsBackend for Capture {
    type RenderPipeline = ();

    fn submit_passes(&self, _: &(), work: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        if let Some((committed, refused)) = self.refuse.borrow().as_ref() {
            return Err(SubmitFailure {
                committed: *committed,
                refused: refused.clone(),
                error: "refused for the test".to_string(),
            });
        }
        for pass in work.passes {
            self.pass_kinds.borrow_mut().push(match pass {
                Pass::Compute(dispatches) => {
                    self.compute.borrow_mut().push(dispatches.iter().map(|d| (
                        d.groups, d.indirect.map(|(_, at)| at as usize),
                        d.count.map(|(_, at, index)| (at as usize, index)),
                    )).collect());
                    "compute"
                }
                Pass::Render(_) => "render",
                Pass::Blit(_) => "blit",
                Pass::ClearColorRects { .. } => "clear",
            });
            let Pass::Render(render) = pass else {
                continue;
            };
            for draw in &render.draws {
                self.draws.borrow_mut().push(
                    draw.vertex_buffers
                        .iter()
                        .map(|&(slot, base, address)| (slot, *base, address as usize))
                        .collect(),
                );
            }
        }
        Ok(())
    }

    fn create_render_pipeline(&self, _: &RenderPipelineDesc<'_>) -> Result<(), String> {
        unreachable!()
    }
}

impl ComputeBackend for Capture {
    type Queue = ();
    type Fence = ();
    type Buffer = usize;
    type Pipeline = ();
    type Heap = RefCell<Vec<Option<TextureDesc>>>;

    fn wrap_title_memory(&self, base: *mut u8, _: usize) -> Result<usize, String> {
        Ok(base as usize)
    }

    fn create_queue(&self) -> Result<(), String> {
        unreachable!()
    }
    fn create_fence(&self, _: u64) -> Result<(), String> {
        unreachable!()
    }
    fn host_buffer(&self, len: usize) -> Result<(usize, *mut u8), String> {
        let mut bytes = vec![0; len].into_boxed_slice();
        let ptr = bytes.as_mut_ptr();
        self.host_buffers.borrow_mut().push(bytes);
        Ok((ptr as usize, ptr))
    }
    fn gpu_address(&self, _: &usize, _: *const u8) -> u64 {
        unreachable!()
    }
    fn create_compute_pipeline(&self, _: &[u8]) -> Result<(), String> {
        unreachable!()
    }
    fn create_compute_pipeline_with(&self, _: &[u8], _: &RootLayout) -> Result<(), String> {
        unreachable!()
    }
    fn submit(&self, _: &(), _: &Submission<'_, Self>) -> Result<(), String> {
        unreachable!()
    }
    fn signal(&self, _: &(), _: u64) {
        unreachable!()
    }
    fn completed(&self, _: &()) -> u64 {
        unreachable!()
    }
    fn wait(&self, _: &(), _: u64, _: u64) -> bool {
        unreachable!()
    }
}

impl ResourceBackend for Capture {
    type Texture = TextureDesc;
    type Sampler = ();

    fn create_heap(&self, _: HeapKind, count: usize) -> Result<Self::Heap, String> {
        Ok(RefCell::new(vec![None; count]))
    }
    fn heap_address(&self, _: &Self::Heap, index: usize) -> u64 {
        (index * 32) as u64
    }
    fn write_buffer(&self, _: &Self::Heap, index: usize, base: &usize, address: *const u8, size: u32) {
        self.buffer_writes.borrow_mut().push((index, *base, address as usize, size));
    }
    fn write_texture(&self, heap: &Self::Heap, slot: usize, tex: &TextureDesc, _: f32) {
        heap.borrow_mut()[slot] = Some(*tex);
    }
    fn write_sampler(&self, _: &Self::Heap, _: usize, _: &(), _: f32) {
        unreachable!()
    }
    fn clear_entry(&self, _: &Self::Heap, _: usize) {
        unreachable!()
    }
    fn create_texture(&self, desc: &TextureDesc) -> Result<TextureDesc, String> {
        Ok(*desc)
    }
    fn create_texture_view(&self, tex: &TextureDesc, view: &ViewDesc) -> Result<TextureDesc, String> {
        self.views.borrow_mut().push(*view);
        Ok(*tex)
    }
    fn create_texture_over_memory(
        &self,
        _: &usize,
        _: *const u8,
        _: &TextureDesc,
        _: usize,
    ) -> Result<TextureDesc, String> {
        unreachable!()
    }
    fn linear_alignment(&self, _: Format) -> usize {
        unreachable!()
    }
    fn upload_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &[u8], _: usize, _: usize) {
        unreachable!()
    }
    fn read_texture(&self, _: &TextureDesc, _: usize, _: usize, _: &mut [u8], _: usize, _: usize) {
        unreachable!()
    }
    fn create_sampler(&self, _: &SamplerDesc) -> Result<(), String> {
        unreachable!()
    }
}

/// A pool of real memory shown to the "title" at an alias, the way `title_tests` makes one.
fn aliased_pool(host: &NativeHost<Capture>, mem: &[u8]) -> u64 {
    let base = mem.as_ptr() as usize;
    let alias = (base + 0x8000_0000) & !0xfff;
    let pool = PoolInfo { id: 1, base, len: mem.len(), memory_entry: None };
    host.memory().mapped(alias, pool.len, &pool, 0);
    alias as u64
}

#[test]
fn a_submission_that_never_committed_pays_its_value_writes_on_the_cpu() {
    let backend = Capture::default();
    backend.refuse.replace(Some((false, Vec::new())));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    list.fill(alias + 0x200, 4, 0x5A);
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused the submission");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the lost submission's first fill was not paid");
    assert_eq!(&mem[0x200..0x204], &[0x5A; 4], "the lost submission's second fill was not paid");
}

#[test]
fn a_committed_failure_leaves_the_fills_to_the_backend() {
    let backend = Capture::default();
    // The command buffer commits, but pass 0 (the first fill) and pass 1 (the draw) are dropped.
    // The backend re-encodes the refused pass's fills itself (#80), in the GPU's order: the host
    // pays none of them here, where a CPU write would let a waiter go on before this command
    // buffer's earlier GPU passes have run.
    backend.refuse.replace(Some((true, vec![0, 1])));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.set_graphics_pipeline(&Rc::new(()), Topology::Triangles);
    list.set_render_targets(&[Rc::new(TextureDesc::d2(Format::Rgba8Unorm, 1, 1))], None);
    list.fill(alias + 0x100, 8, 0xAB); // pass 0: refused
    list.set_vertex_buffer(2, alias + 0x300);
    list.draw_instanced(3, 1, 0, 0).unwrap(); // pass 1: refused
    list.fill(alias + 0x200, 4, 0x5A); // pass 2: kept, runs on the GPU
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused passes 0 and 1");
    assert_eq!(&mem[0x100..0x108], &[0; 8], "a committed failure's fills are the backend's (re-encoded)");
    assert_eq!(&mem[0x200..0x204], &[0; 4], "a kept pass's fill must stay with the GPU");
}

/// A submission that never committed lost every pass, and no GPU work of its is in flight: the
/// host pays all of its fills on the CPU, so a waiter on one goes on (#80).
#[test]
fn an_uncommitted_failure_pays_every_fill_on_the_cpu() {
    let backend = Capture::default();
    backend.refuse.replace(Some((false, vec![])));
    let host = NativeHost::new(backend);
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    list.fill(alias + 0x200, 4, 0x5A);
    let err = host.execute(&(), &[&list], &[], &[]);
    assert!(err.is_err(), "the backend refused the submission");
    assert_eq!(&mem[0x100..0x108], &[0xAB; 8], "the first fill was paid");
    assert_eq!(&mem[0x200..0x204], &[0x5A; 4], "the second fill was paid");
}

#[test]
fn a_submitted_piece_leaves_its_value_writes_to_the_gpu() {
    let host = NativeHost::new(Capture::default());
    let mem = vec![0u8; 0x4000].into_boxed_slice();
    let alias = aliased_pool(&host, &mem);
    let mut list = NativeList::default();
    list.fill(alias + 0x100, 8, 0xAB);
    host.execute(&(), &[&list], &[], &[]).unwrap();
    assert_eq!(&mem[0x100..0x108], &[0; 8], "the host wrote a fill the GPU was given");
}
