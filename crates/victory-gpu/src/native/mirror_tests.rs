//! CPU-only host regressions: mirror stamps must follow precisely the resources and mappings
//! they resolved, without repeating backend writes for an unchanged entry.
use super::super::memory::{PoolInfo, PoolObserver};
use super::*;
use victory_backend::backend::{ComputeBackend, RootLayout, Submission};

#[derive(Default)]
pub(in crate::native) struct Recording {
    textures: std::cell::Cell<usize>,
    writes: std::cell::Cell<u64>,
    wraps: RefCell<Vec<(usize, usize)>>,
    spans: RefCell<Vec<Vec<(usize, usize, usize)>>>,
}
pub(in crate::native) struct RecordedBuffer {
    _memory: Option<Box<[u32]>>,
}
type Heap = RefCell<HashMap<usize, (usize, u32)>>;

impl ComputeBackend for Recording {
    type Queue = ();
    type Fence = std::cell::Cell<u64>;
    type Buffer = RecordedBuffer;
    type Pipeline = ();
    type Heap = Heap;
    fn create_queue(&self) -> Result<(), String> {
        Ok(())
    }
    fn create_fence(&self, v: u64) -> Result<Self::Fence, String> {
        Ok(v.into())
    }
    fn wrap_title_memory(&self, p: *mut u8, n: usize) -> Result<RecordedBuffer, String> {
        self.wraps.borrow_mut().push((p as usize, n));
        Ok(RecordedBuffer { _memory: None })
    }
    fn wrap_title_span(&self, n: usize, pieces: &[(usize, *mut u8, usize)]) -> Result<(RecordedBuffer, *mut u8), String> {
        self.spans.borrow_mut().push(pieces.iter().map(|&(at, p, len)| (at, p as usize, len)).collect());
        self.host_buffer(n)
    }
    fn host_buffer(&self, len: usize) -> Result<(RecordedBuffer, *mut u8), String> {
        let mut memory = vec![0u32; len.div_ceil(4)].into_boxed_slice();
        let ptr = memory.as_mut_ptr().cast();
        Ok((
            RecordedBuffer {
                _memory: Some(memory),
            },
            ptr,
        ))
    }
    fn gpu_address(&self, _: &RecordedBuffer, p: *const u8) -> u64 {
        p as u64
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
    fn signal(&self, f: &Self::Fence, v: u64) {
        f.set(v);
    }
    fn completed(&self, f: &Self::Fence) -> u64 {
        f.get()
    }
    fn wait(&self, f: &Self::Fence, v: u64, _: u64) -> bool {
        f.get() >= v
    }
}

impl ResourceBackend for Recording {
    type Texture = usize;
    type Sampler = ();
    fn create_heap(&self, _: HeapKind, _: usize) -> Result<Heap, String> {
        Ok(Heap::default())
    }
    fn heap_address(&self, h: &Heap, i: usize) -> u64 {
        h as *const Heap as u64 + i as u64 * 32
    }
    fn write_buffer(&self, h: &Heap, i: usize, _: &RecordedBuffer, p: *const u8, n: u32) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (p as usize, n));
    }
    fn write_texture(&self, h: &Heap, i: usize, t: &usize, _: f32) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (*t, u32::MAX));
    }
    fn write_sampler(&self, _: &Heap, _: usize, _: &(), _: f32) {
        unreachable!()
    }
    fn clear_entry(&self, h: &Heap, i: usize) {
        self.writes.set(self.writes.get() + 1);
        h.borrow_mut().insert(i, (0, 0));
    }
    fn create_texture(&self, _: &TextureDesc) -> Result<usize, String> {
        self.textures.set(self.textures.get() + 1);
        Ok(self.textures.get())
    }
    fn create_texture_view(&self, t: &usize, _: &ViewDesc) -> Result<usize, String> {
        Ok(*t)
    }
    fn create_texture_over_memory(
        &self,
        _: &RecordedBuffer,
        _: *const u8,
        _: &TextureDesc,
        _: usize,
    ) -> Result<usize, String> {
        unreachable!()
    }
    fn linear_alignment(&self, _: Format) -> usize {
        unreachable!()
    }
    fn upload_texture(&self, _: &usize, _: usize, _: usize, _: &[u8], _: usize, _: usize) {}
    fn read_texture(&self, _: &usize, _: usize, _: usize, _: &mut [u8], _: usize, _: usize) {
        unreachable!()
    }
    fn create_sampler(&self, _: &SamplerDesc) -> Result<(), String> {
        unreachable!()
    }
}

impl GraphicsBackend for Recording {
    type RenderPipeline = ();
    fn create_render_pipeline(&self, _: &RenderPipelineDesc<'_>) -> Result<(), String> {
        unreachable!()
    }
    fn submit_passes(&self, _: &(), _: &PassSubmission<'_, Self>) -> Result<(), SubmitFailure> {
        Ok(())
    }
}

impl PresentBackend for Recording {
    type Surface = ();
    fn create_surface(&self, _: *mut std::ffi::c_void) -> Result<(), String> { Ok(()) }
    fn present(&self, _: &(), _: &(), _: &[&usize]) -> Result<(), String> { Ok(()) }
}

#[test]
fn unrelated_resources_do_not_rewrite_a_mirrored_table_entry() {
    let h = NativeHost::new(Recording::default());
    let (texture, other) = (Rc::new(11), Rc::new(22));
    let (key, unrelated) = (Rc::as_ptr(&texture) as usize, Rc::as_ptr(&other) as usize);
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);

    h.register_texture(unrelated, &other);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "an unrelated registration only revalidates");
    assert_eq!(heap.mirrored.borrow()[&0].epoch, h.resources_epoch.get(), "the next scan is cheap again");
    h.release_resource(unrelated);
    h.remirror_tables(&roots, None);
    h.register_buffer(unrelated, 0x1000, 256);
    h.remirror_tables(&roots, None);
    h.forget_resource(unrelated);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "unrelated buffers and removals are also irrelevant");
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX));
    crate::descriptor::forget(key);
}

#[test]
fn a_texture_key_replacement_and_release_follow_the_new_texture() {
    let h = NativeHost::new(Recording::default());
    let texture = Rc::new(11);
    let key = Rc::as_ptr(&texture) as usize;
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    let record = record_at(handle);
    let first_version = h.resource_version(key);

    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2, "replacement without release invalidates too");
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    h.release_resource(key);
    assert_eq!(heap.heap.borrow()[&0], (0, 0), "released textures clear their entries immediately");
    assert_eq!(h.resource_version(key), 0);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3, "an absent resource must not restore its old texture");
    h.register_texture(key, &Rc::new(33));
    assert!(h.resource_version(key) > first_version);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 4);
    assert_eq!(heap.heap.borrow()[&0], (33, u32::MAX));
    assert_eq!(record_at(handle), record, "resource identity, not descriptor bytes, changed");
    crate::descriptor::forget(key);
}

#[test]
fn a_device_mirror_skips_identical_records_but_writes_a_changed_record() {
    let h = NativeHost::new(Recording::default());
    let (a, b) = (Rc::new(11), Rc::new(22));
    let (key_a, key_b) = (Rc::as_ptr(&a) as usize, Rc::as_ptr(&b) as usize);
    h.register_texture(key_a, &a);
    h.register_texture(key_b, &b);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let write = |key| crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    write(key_a);
    assert!(h.mirror(handle).unwrap());
    let record = record_at(handle);
    write(key_a);
    assert_eq!(record_at(handle), record, "writing the same view preserves the record");
    assert!(h.mirror(handle).unwrap());
    assert_eq!(h.backend.writes.get(), 1);
    write(key_b);
    assert_ne!(record_at(handle), record);
    assert!(h.mirror(handle).unwrap());
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    crate::descriptor::forget(key_a);
    crate::descriptor::forget(key_b);
}

#[test]
fn placed_texture_stamps_follow_the_placement_key_and_keep_the_pool_generation() {
    let h = NativeHost::new(Recording::default());
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut res = [0u32; 14];
    res[0] = 3;
    let mut view = [0u32; 16];
    (view[0], view[1], view[5], view[10]) = (28, 4, 1, 0x7000_0000);
    let key = super::super::textures::placed_key(u64::from(view[10]), &res);
    h.register_texture(key, &Rc::new(11));
    crate::descriptor::create_placed_srv(std::ptr::null_mut(), &res, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    h.register_texture(1, &Rc::new(99));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    h.memory.unmapped(0x7000_0000, 65536);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3, "placed images still depend on pool mappings");
}

#[test]
fn a_buffer_resource_version_follows_its_address_and_forgetting_it() {
    let h = NativeHost::new(Recording::default());
    let mut bytes = vec![0u32; 16384];
    let pool = PoolInfo { id: 1, base: bytes.as_mut_ptr() as usize, len: 65536, memory_entry: None };
    h.memory.mapped(0x7000_0000, pool.len, &pool, 0);
    let key = bytes.as_ptr() as usize;
    h.register_buffer(key, 0x7000_0000, 256);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    h.register_texture(1, &Rc::new(99));
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_buffer(key, 0x7000_0100, 512);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2);
    assert_eq!(heap.heap.borrow()[&0], (pool.base + 256, 512));
    let version = h.resource_version(key);
    h.forget_resource(key);
    assert_eq!(h.resource_version(key), 0);
    h.register_buffer(key, 0x7000_0200, 128);
    assert!(h.resource_version(key) > version);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 3);
    assert_eq!(heap.heap.borrow()[&0], (pool.base + 512, 128));
    crate::descriptor::forget(key);
}

#[test]
fn a_uav_with_a_counter_keeps_the_global_resource_dependency() {
    let h = NativeHost::new(Recording::default());
    let (texture, counter) = (Rc::new(11), Box::new(0u32));
    let (key, counter_key) = (Rc::as_ptr(&texture) as usize, &*counter as *const u32 as usize);
    h.register_texture(key, &texture);
    h.register_buffer(counter_key, 0x1000, 4);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut view = [0u32; 10];
    (view[0], view[1]) = (28, 4);
    crate::descriptor::create_uav(std::ptr::null_mut(), key as _, counter_key as _, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1);
    h.register_buffer(counter_key, 0x2000, 4);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 2, "both resources can change a counter UAV");
    crate::descriptor::forget(key);
    crate::descriptor::forget(counter_key);
}

#[test]
fn unchanged_uav_follows_remaps_and_recovers_after_an_unmapped_attempt() {
    let h = NativeHost::new(Recording::default());
    let mut old = vec![0u32; 16384];
    let mut new = vec![0u32; 16384];
    let pool = |id, base| PoolInfo {
        id,
        base,
        len: 65536,
        memory_entry: None,
    };
    let (a, b) = (
        pool(1, old.as_mut_ptr() as usize),
        pool(2, new.as_mut_ptr() as usize),
    );
    let address = 0x7000_0000usize;
    h.memory.pool_created(&a);
    h.memory.pool_created(&b);
    h.memory.mapped(address, 65536, &a, 0);
    let mut slots = [0u32; 8];
    let handle = slots.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    let mut res = [0u32; 14];
    res[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8], view[10]) = (1, 1, 1, address as u32);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &res, &view, handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    let write = |value: u32| {
        h.remirror_tables(&roots, None);
        let (ptr, size) = heap.heap.borrow()[&0];
        if size >= 4 {
            // SAFETY: the recording backend's non-null view resolves into the live vectors.
            unsafe {
                (ptr as *mut u32).write(value);
            }
        }
    };
    write(11);
    assert_eq!((old[0], new[0]), (11, 0));
    let record = record_at(handle);
    h.memory.mapped(address, 65536, &b, 0);
    write(22);
    assert_eq!(record_at(handle), record, "title descriptor did not change");
    assert_eq!(
        (old[0], new[0]),
        (11, 22),
        "recycled memory must not receive the next UAV write"
    );
    h.memory.unmapped(address, 65536);
    write(33);
    assert_eq!((old[0], new[0]), (11, 22));
    assert!(
        !heap.mirrored.borrow().contains_key(&0),
        "failed mirrors are not successful cache entries"
    );
    h.memory.mapped(address, 65536, &a, 0);
    write(44);
    assert_eq!((old[0], new[0]), (44, 22));
}

#[test]
fn submitted_table_buffer_owns_memory_after_more_than_eight_remaps() {
    let h = NativeHost::new(Recording::default());
    let heap = h.host_heap().unwrap();
    let first = h.memory_table(&heap).unwrap();
    first.write(&[0x1234abcd]);
    let ptr = first.ptr();
    // A backend submission retains the buffer, not the Pages wrapper.
    let submitted = first.buf.clone();
    let wrapper = Rc::downgrade(&first);
    drop(first);
    for _ in 0..12 {
        h.memory.unmapped(0x7000_0000, 65536);
        h.memory_table(&heap).unwrap();
    }
    assert!(
        wrapper.upgrade().is_none(),
        "no arbitrary retired-table queue"
    );
    assert!(submitted._memory.is_some());
    // SAFETY: the retained buffer owns the allocation at ptr.
    assert_eq!(unsafe { ptr.cast::<u32>().read() }, 0x1234abcd);
}

#[test]
fn forgetting_a_texture_keeps_retained_entries_until_the_resource_is_replaced() {
    let h = NativeHost::new(Recording::default());
    let texture = Rc::new(11);
    let key = Rc::as_ptr(&texture) as usize;
    h.register_texture(key, &texture);
    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, handle, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, std::ptr::null(), handle);
    let roots = [Some(Root::Table(heap.clone(), 0))];
    h.remirror_tables(&roots, None);
    let record = record_at(handle);
    h.remirror_tables(&roots, None);
    assert_eq!(h.backend.writes.get(), 1, "an unchanged entry does not rewrite the backend");
    h.forget_resource(key);
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX), "upstream forget retains the heap view");
    h.remirror_tables(&roots, None);
    assert_eq!(heap.heap.borrow()[&0], (11, u32::MAX));
    h.register_texture(key, &Rc::new(22));
    h.remirror_tables(&roots, None);
    assert_eq!(record_at(handle), record);
    assert_eq!(heap.heap.borrow()[&0], (22, u32::MAX));
    crate::descriptor::forget(key);
}

#[test]
fn shared_depth_keys_use_one_twin_and_colour_reads_follow_its_planes() {
    let h = NativeHost::new(Recording::default());
    let desc = TextureDesc::d2(Format::D32FloatS8, 8, 8);
    let depth = h.texture(&desc).unwrap();
    let keys = [0x123400, 0x123401];
    for key in keys {
        h.register_texture(key, &depth);
        h.register_depth(key, &depth, &desc);
    }
    let srv = crate::descriptor::View::Shader(keys[1], Some([41, 4, 0, 0, 0, 1, 0, 0, 0, 0]));
    assert!(h.twin_plane(keys[1], &depth, &srv).is_none(), "a colour read alone makes no twin");
    let uav = crate::descriptor::View::Unordered(keys[0], 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
    let version = h.resource_version(keys[1]);
    let (write, format) = h.twin_plane(keys[0], &depth, &uav).unwrap();
    assert_eq!(format, Format::R32Float, "console 16-bit depth with stencil uses host float depth");
    assert!(h.resource_version(keys[1]) > version, "existing aliases are remirrored after twin creation");
    let (read, _) = h.twin_plane(keys[1], &depth, &srv).unwrap();
    assert!(Rc::ptr_eq(&read, &write));
    let stencil = crate::descriptor::View::Unordered(keys[1], 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0]));
    let (stencil, format) = h.twin_plane(keys[1], &depth, &stencil).unwrap();
    assert_eq!(format, Format::R8Uint);
    assert!(!Rc::ptr_eq(&stencil, &write));
    assert_eq!(h.backend.textures.get(), 3, "one depth texture and its two planes");
    let twins = h.twins.borrow();
    assert!(Rc::ptr_eq(twins[&keys[0]].as_ref().unwrap(), twins[&keys[1]].as_ref().unwrap()));
    let twin = twins[&keys[0]].as_ref().unwrap();
    assert_eq!(twin.need(twin::TWIN), Some(false));
    twin.newer.set(twin::TWIN);
    assert_eq!(twin.need(twin::DEPTH), Some(true));
    assert_eq!(twin.copy(true).len(), 4, "both depth and stencil copy through staging");
}

/// Follow every copy through remapping, a gap (the whole view becomes null), and recovery.
#[test]
fn copied_wraps_follow_remaps_and_gaps_with_one_comparison_walk() {
    let h = NativeHost::new(Recording::default());
    const PAGE: usize = 16384;
    const VA: usize = 0x7100_0000;
    let pool = PoolInfo { id: 1, base: 0x1000_0000, len: 4 * PAGE, memory_entry: None };
    h.memory.mapped(VA, PAGE, &pool, 0);
    h.memory.mapped(VA + PAGE, PAGE, &pool, 2 * PAGE);
    let records = [0u32; 24];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 3).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[6], view[8], view[10]) = (1, 64, 1, (VA + PAGE - 128) as u32);
    for i in 0..3 {
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, start + 32 * i);
        assert!(h.mirror(start + 32 * i).unwrap());
    }
    let walks = || SIGNATURES.with(|n| n.replace(0));
    h.memory.pool_created(&PoolInfo { id: 2, base: 0x2000_0000, len: PAGE, memory_entry: None });
    walks();
    h.follow_title_pages();
    assert_eq!(walks(), 1);
    let old = h.aliases.borrow()[&(VA as u64, (VA + 2 * PAGE) as u64)].2.clone();
    let writes = h.backend.writes.get();
    h.memory.mapped(VA + PAGE, PAGE, &pool, 3 * PAGE);
    h.follow_title_pages();
    assert_eq!(walks(), 4, "one comparison walk and three fresh wrap validations; no extra walk to track each result");
    assert_eq!(h.backend.writes.get() - writes, 3);
    let aliases = h.aliases.borrow();
    let (signature, _, new) = &aliases[&(VA as u64, (VA + 2 * PAGE) as u64)];
    assert!(!Rc::ptr_eq(&old, new));
    assert!(heap.title_views.borrow().values().all(|v| v.2 == Some(*signature)));
    drop(aliases);
    h.memory.unmapped(VA + PAGE, PAGE);
    h.follow_title_pages();
    assert!(heap.heap.borrow().values().all(|v| v.1 == 0), "a wrap with a gap is entirely null, not a partial span");
    assert_eq!(heap.title_views.borrow().len(), 3, "the null copies still follow their pages");
    assert!(heap.spanned.borrow().is_empty());
    walks();
    assert_eq!(h.alias_signature(VA as u64, (2 * PAGE) as u64), None);
    assert_eq!(h.alias_signature(VA as u64, (2 * PAGE) as u64), None);
    assert_eq!(walks(), 2, "standalone signature reads are fresh, including gaps");
    h.memory.mapped(VA + PAGE, PAGE, &pool, PAGE);
    assert!(h.alias_signature(VA as u64, (2 * PAGE) as u64).is_some());
    assert_eq!(walks(), 1);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert!(heap.heap.borrow().values().all(|v| v.1 == 256), "all copies recover when the gap is mapped");
    // One contiguous pool run holds the recovered range: followed too, as a wrap is (PR 513).
    assert!(heap.title_views.borrow().values().all(|v| v.2.is_some()), "the recovered range is now one contiguous pool run, still followed");
}

/// The title VA is a page-table key, not necessarily a pointer this process can dereference.
/// The merge tried wrap_title_memory(VA, ...) here before trying the pool-backed span.
#[test]
fn a_two_pool_view_uses_pool_pointers_and_keeps_replaced_spans() {
    let h = NativeHost::new(Recording::default());
    const PAGE: usize = 65536;
    const VA: usize = 0x23_4560_0000;
    let pools: Vec<_> = (0..3).map(|i| PoolInfo {
        id: i + 11, base: 0x1000_4000 + i * 2 * PAGE, len: 2 * PAGE, memory_entry: None,
    }).collect();
    h.memory.mapped(VA, PAGE, &pools[0], PAGE);
    h.memory.mapped(VA + PAGE, PAGE, &pools[1], 0);
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    let at = (VA + PAGE - 8) as u64;
    (view[1], view[4], view[8], view[10], view[11]) = (1, 64, 1, at as u32, (at >> 32) as u32);
    crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, start);
    assert!(h.mirror(start).unwrap());
    assert!(h.backend.wraps.borrow().is_empty(), "a logical title VA must never reach the no-copy wrapper");
    assert_eq!(h.backend.spans.borrow()[0], [(0, pools[0].base + PAGE, PAGE), (PAGE, pools[1].base, PAGE)]);
    let key = (VA, 2 * PAGE);
    let old = Rc::downgrade(&h.spans.borrow()[&key].buf);
    assert_eq!(heap.heap.borrow()[&0], (h.spans.borrow()[&key].base as usize + PAGE - 8, 256));
    assert!(heap.title_views.borrow().is_empty());

    // No descriptor rewrite or bound table: the next submission must refresh the span itself.
    h.memory.mapped(VA + PAGE, PAGE, &pools[2], 0);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(h.backend.spans.borrow()[1], [(0, pools[0].base + PAGE, PAGE), (PAGE, pools[2].base, PAGE)]);
    assert_eq!(heap.spanned.borrow()[&0], (key, h.spans.borrow()[&key].version));
    assert!(old.upgrade().is_some(), "the old buffer outlives the heap rewrite");

    h.memory.unmapped(VA + PAGE, PAGE);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(h.backend.spans.borrow()[2], [(0, pools[0].base + PAGE, PAGE)]);
    assert_eq!(heap.heap.borrow()[&0], (h.spans.borrow()[&key].base as usize + PAGE - 8, 256));
    h.memory.unmapped(VA, PAGE);
    h.execute(&(), &[], &[], &[]).unwrap();
    assert_eq!(heap.heap.borrow()[&0].1, 0, "a span without its starting page is null");
    assert!(heap.spanned.borrow().is_empty());
}

/// The cullers' buffer address includes dword 1's low 16 bits. The root SRV must point to
/// this table, and a later mirrored descriptor write must update it in place.
#[test]
fn view_addresses_bind_and_refresh_nonzero_buffer_high_words() {
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 16];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 2).unwrap();
    let mut texture = [0u32; 14];
    (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 64, 32, 1 | 1 << 16, 1);
    let mut image = [0u32; 16];
    (image[0], image[1], image[5]) = (28, 4, 1);
    crate::descriptor::create_placed_srv(std::ptr::null(), &texture, &image, start);
    let mut buffer = [0u32; 14];
    buffer[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[6], view[8], view[10], view[11]) = (1, 64, 1, 0x3450_0000, 0x12);
    crate::descriptor::create_placed_srv(std::ptr::null(), &buffer, &view, start + 32);
    let mut p = Prep {
        root: RootArgs::default(), ps_root: None, bufs: vec![], vertex: vec![],
        index: None, blit: vec![], indirect: None, count: None, heap: None,
    };
    h.view_address_args(Some(&heap), &mut p).unwrap();
    let table = heap.addresses.borrow().as_ref().unwrap().clone();
    assert_eq!(p.root.0, (table.ptr() as u64).to_le_bytes());
    assert!(Rc::ptr_eq(&p.bufs[0], &table.buf), "the submission retains the address table");
    // SAFETY: the table owns at least eight dwords and no backend work reads or writes it.
    let words = || unsafe { std::slice::from_raw_parts(table.ptr().cast::<u32>(), 8).to_vec() };
    assert_eq!(words(), [0, 0xC000_0000, 0x0007_C00F, 0x9000_0FAC, 0x3450_0000, 0x12, 0, 0]);
    view[11] = 0x23;
    h.memory.mapped(0x23_3450_0000, 65536, &PoolInfo {
        id: 1, base: 0x1000_0000, len: 65536, memory_entry: None,
    }, 0);
    crate::descriptor::create_placed_srv(std::ptr::null(), &buffer, &view, start + 32);
    assert!(h.mirror(start + 32).unwrap());
    assert_eq!(&words()[4..], [0x3450_0000, 0x23, 0, 0]);
    crate::descriptor::create_cbv(std::ptr::null(), &[0, 0, 256, 0], start + 32);
    assert!(h.mirror(start + 32).unwrap());
    assert_eq!(&words()[4..], [0; 4]);
}

/// Native registrations accept opaque keys, including placed hashes, not just Resource objects.
/// Preparing the address table walks even entries outside the dispatch's bound root tables.
#[test]
fn view_addresses_never_dereference_native_resource_keys() {
    use crate::descriptor::View;
    let h = NativeHost::new(Recording::default());
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = h.heap(HeapKind::Resource, start, 1).unwrap();
    let mut resource = [0u32; 14];
    (resource[0], resource[4], resource[6], resource[7], resource[9]) = (3, 64, 32, 1 | 1 << 16, 1);
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[5]) = (28, 4, 1);
    let mut placed = [0u32; 16];
    placed[..10].copy_from_slice(&srv);
    let image_words = [0, 0xC000_0000, 0x0007_C00F, 0x9000_0FAC];
    let pso = Rc::new(ComputePso {
        pso: (), threads: [1; 3],
        binding: Some(Binding { view_addresses: true, ..Binding::default() }),
    });
    let mut list = NativeList::default();
    list.set_heaps(Some(&heap), None);
    list.set_compute_pipeline(&pso);
    list.dispatch(1, 1, 1).unwrap();

    for key in [0x4EA9_0000, super::super::textures::placed_key(0x1234_0000, &resource)] {
        assert!(!crate::resource::is_resource(key as _));
        h.register_texture(key, &Rc::new(11));
        for desc in [Some(srv), None] {
            for view in [View::Shader(key, desc), View::Unordered(key, 0, desc)] {
                // Force the initial heap walk through dispatch preparation, before any mirror.
                *heap.addresses.borrow_mut() = None;
                crate::descriptor::record(start, view.clone());
                let prepared = h.prep(&list.cmds[0]).unwrap();
                let table = heap.addresses.borrow().as_ref().unwrap().clone();
                assert_eq!(prepared.root.0, (table.ptr() as u64).to_le_bytes());
                // SAFETY: the CPU backend owns four initialized dwords; no GPU work is run.
                let words = || unsafe { std::slice::from_raw_parts(table.ptr().cast::<u32>(), 4).to_vec() };
                assert_eq!(words(), [0; 4], "an opaque key has no Resource description");
                crate::descriptor::record(start, View::Placed(false, resource, placed));
                let _ = h.mirror(start); // No native texture registered for this placed record.
                assert_eq!(words(), image_words, "placed records supply their own description");
                crate::descriptor::record(start, view);
                let _ = h.mirror(start);
                assert_eq!(words(), [0; 4], "mirrored writes use the same guard");
            }
        }
        h.forget_resource(key);
        assert_eq!(h.address_words(Some(&View::Shader(key, None))), [0; 4], "stale record");
        h.register_buffer(key, 0x23_3450_0000, 256);
        for view in [View::Shader(key, None), View::Unordered(key, 0, None)] {
            assert_eq!(h.address_words(Some(&view)), [0x3450_0000, 0x23, 0, 0]);
        }
        h.forget_resource(key);
        crate::descriptor::forget(key);
    }
    // Real objects still supply their title size (not the possibly downscaled host texture).
    let pixels = [0u32; 64 * 32];
    let object = crate::resource::test_texture_at(pixels.as_ptr() as u64, (64, 32), 1, 28, 0, 0);
    assert!(crate::resource::is_resource(object));
    for view in [View::Shader(object as usize, Some(srv)), View::Unordered(object as usize, 0, None)] {
        assert_eq!(h.address_words(Some(&view)), image_words);
    }
    let release: extern "system" fn(crate::object::This) -> u32 = crate::object::slot(object, 0x10);
    assert_eq!(release(object), 0);
}

/// Exercise upstream's covered-placement and byte-budget ownership on the CPU. The heap
/// protects a sampled copy until its entry is replaced; eviction drops the last host Rc.
#[test]
fn covered_and_budget_eviction_drop_unshown_copies() {
    use crate::native::title::Title;
    use victory_formats::texture::Swizzle;
    let mut t = Title::new(Recording::default(), Default::default());
    t.covered_clear = false; // Pin upstream's wait; the default clearing path is tested below.
    let memory = vec![0x44u32; 16384];
    let base = memory.as_ptr() as usize;
    let pool = PoolInfo { id: 1, base, len: memory.len() * 4, memory_entry: None };
    t.host.memory.mapped(base, pool.len, &pool, 0);
    let mut placed = |at: usize, size: u32| {
        let address = (base + at) as u64;
        let object = crate::resource::test_texture_at(address, (size.into(), size), 1, 28, Swizzle::LINEAR.0, 0);
        let words = crate::resource::texture_desc(object).unwrap();
        let info = crate::resource::placed_info(address, &words).unwrap();
        let key = super::super::textures::placed_key(address, &words);
        t.texture_placed(address, key, 1, info.size);
        t.texture_created(key, info);
        assert!(t.realize(key));
        key
    };
    let covered = placed(0, 32);
    let idle = placed(0x4000, 16);
    let covered_weak = Rc::downgrade(&t.host.texture_at(covered).unwrap());
    let idle_weak = Rc::downgrade(&t.host.texture_at(idle).unwrap());
    let records = [0u32; 8];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 1).unwrap();
    crate::descriptor::create_srv(std::ptr::null(), covered as _, std::ptr::null(), start);
    assert!(t.host.mirror(start).unwrap());
    // A placement at a different start covers the sampled copy.
    t.texture_placed((base + 0x800) as u64, 123, 1, 1024);
    assert!(t.covered.contains_key(&covered));
    assert!(t.unloaded.contains(&covered) && !t.texture_watched(covered));
    t.sync_dirty();
    assert!(!t.unloaded.contains(&covered) && t.texture_watched(covered));
    assert_eq!(t.uploads_of(covered), 1);
    assert!(covered_weak.upgrade().is_some() && heap.shows_texture(covered));
    crate::descriptor::create_srv(std::ptr::null(), idle as _, std::ptr::null(), start);
    assert!(t.host.mirror(start).unwrap());
    t.sync_dirty();
    assert!(covered_weak.upgrade().is_none(), "covered eviction drops the copy");
    assert!(!heap.shows_texture(covered) && t.host.is_evicted(covered));
    assert!(t.pending.contains_key(&covered));
    let bytes = t.budget.live;
    t.set_texture_budget(1);
    t.keep_to_budget();
    assert!(idle_weak.upgrade().is_some(), "a shown copy stays over budget");
    // A null buffer replaces the last texture entry.
    crate::descriptor::record(start, crate::descriptor::View::Constant(0, 0));
    assert!(t.host.mirror(start).unwrap());
    for _ in 0..=super::super::textures::RECHECK { t.keep_to_budget(); }
    assert!(idle_weak.upgrade().is_none(), "budget eviction drops the copy");
    assert!(bytes > 0 && t.budget.live == 0);
    assert!(!heap.shows_texture(idle) && t.host.is_evicted(idle));
    assert_eq!(heap.heap.borrow()[&0].1, 0);
}

/// Synthetic title memory and descriptors; Recording never asks for a Metal device.
pub(in crate::native) struct CoveredCopies {
    pub(in crate::native) title: crate::native::title::Title<Recording>,
    memory: Vec<u32>,
    records: [u32; 8],
}

impl CoveredCopies {
    pub(in crate::native) fn new() -> Self {
        let title = crate::native::title::Title::new(Recording::default(), Default::default());
        let memory = vec![0x44u32; 16384];
        let base = memory.as_ptr() as usize;
        let pool = PoolInfo { id: 1, base, len: memory.len() * 4, memory_entry: None };
        title.host.memory.mapped(base, pool.len, &pool, 0);
        Self { title, memory, records: [0; 8] }
    }

    pub(in crate::native) fn place(&mut self, at: usize, size: u32) -> usize {
        use crate::native::textures::{TextureInfo, placed_key};
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let address = self.memory.as_ptr() as u64 + at as u64;
        let key = placed_key(address, &[size; 14]);
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d, width: size, height: size, depth: 1,
            mips: 1, element_bytes: 4, block: 1, samples: 1, swizzle: Swizzle::LINEAR,
        }).unwrap();
        let info = TextureInfo {
            dimension: Dimension::Tex2d, format: 28, width: size, height: size,
            depth_or_layers: 1, flags: 0, size: layout.size as usize, layout,
            memory: address as *const u8,
        };
        self.title.placed_view(address, key, 1, info, true);
        assert!(self.title.host.knows(key));
        key
    }

    pub(in crate::native) fn show(&mut self, key: usize) -> Rc<NativeHeap<Recording>> {
        let start = self.records.as_ptr() as u64;
        let heap = self.title.host.heap(HeapKind::Resource, start, 1).unwrap();
        let mut srv = [0u32; 10];
        (srv[0], srv[1], srv[5]) = (28, 4, 1);
        crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, start);
        assert!(self.title.host.mirror(start).unwrap());
        heap
    }

    fn list(heap: &Rc<NativeHeap<Recording>>) -> NativeList<Recording> {
        let mut list = NativeList::default();
        let pso = Rc::new(ComputePso { pso: (), threads: [1; 3], binding: None });
        list.set_compute_pipeline(&pso);
        list.set_compute_root_table(0, heap, 0);
        list.dispatch(1, 1, 1).unwrap();
        list
    }
}

impl Drop for CoveredCopies {
    fn drop(&mut self) {
        let keys: Vec<_> = self.title.placed.values().map(|&(k, _)| k).collect();
        for key in keys {
            self.title.host.release_resource(key);
            self.title.texture_released(key);
            crate::descriptor::forget(key);
        }
    }
}

#[test]
fn covered_clear_keeps_stale_records_null_until_a_title_placement() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let weak = Rc::downgrade(&f.title.host.texture_at(old).unwrap());
    let record = f.records;
    let new = f.place(0x800, 16);
    // Both retry caches must be discarded, including a pending attempt on a shown entry.
    heap.pending_mirror.borrow_mut().insert(0, ([7; 12], 0, 0));
    heap.typed_pending.borrow_mut().set(0, true, true);
    f.title.sync_dirty();
    assert!(weak.upgrade().is_none());
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
    assert!(!heap.shows_texture(old));
    assert!(heap.mirrored.borrow().is_empty() && heap.pending_mirror.borrow().is_empty());
    assert!(!heap.typed_pending.borrow().has_explicit());
    assert!(f.title.host.is_covered(old) && !f.title.host.is_evicted(old));
    assert!(f.title.pending.contains_key(&old));
    let bytes = f.title.budget.live;
    // Rebinding, internal realization, and a descriptor copy are not title placements.
    let roots = [Some(Root::Table(heap.clone(), 0))];
    let list = CoveredCopies::list(&heap);
    for _ in 0..3 {
        f.title.host.remirror_tables(&roots, None);
        assert!(f.title.host.take_wanted().is_empty());
        assert!(!f.title.realize(old));
        let info = f.title.pending[&old].clone();
        f.title.placed_view(info.memory as u64, old, 1, info, false);
        let named = f.title.keep_named(std::iter::once(&list));
        f.title.sync_dirty();
        f.title.let_named_go(named);
        assert_eq!(heap.heap.borrow()[&0], (0, 0));
        assert!(!heap.shows_texture(old) && f.title.host.knows(new));
        assert_eq!(f.title.budget.live, bytes);
        assert_eq!(f.title.covered_history.ping_pong, 0);
    }
    // Even a fresh slot holding the same stale record mirrors as null without a request.
    let copied = record;
    let copy_at = copied.as_ptr() as u64;
    let copy_heap = f.title.host.heap(HeapKind::Resource, copy_at, 1).unwrap();
    copy_heap.typed_pending.borrow_mut().set(0, true, true);
    copy_heap.pending_mirror.borrow_mut().insert(0, ([7; 12], 0, 0));
    assert!(f.title.host.mirror(copy_at).unwrap());
    assert_eq!(copy_heap.heap.borrow()[&0], (0, 0));
    assert!(!copy_heap.typed_pending.borrow().has_explicit());
    assert!(copy_heap.pending_mirror.borrow().is_empty());
    assert!(f.title.host.take_wanted().is_empty());
    assert_eq!(f.place(0, 32), old); // An explicit title placement can revive it.
    f.title.host.remirror_tables(&roots, None);
    f.title.sync_dirty();
    assert!(heap.shows_texture(old));
    assert_eq!(heap.heap.borrow()[&0].1, u32::MAX);
    assert_eq!(f.title.uploads_of(old), 1);
    assert_eq!(f.records, record, "the title never rewrote its record");
    assert_eq!(f.title.covered_history.ping_pong, 1);
}

#[test]
fn covered_clear_keeps_copies_named_by_submissions_in_flight() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    f.place(0x800, 16);
    assert!(f.title.names_wanted(), "covered copies require names even with the budget off");
    let list = CoveredCopies::list(&heap);
    let first = f.title.keep_named(std::iter::once(&list));
    let second = f.title.keep_named(std::iter::once(&list));
    assert!(first.contains(&old) && second.contains(&old));
    f.title.sync_dirty();
    assert!(heap.shows_texture(old) && f.title.host.knows(old));
    f.title.let_named_go(first);
    f.title.sync_dirty();
    assert!(heap.shows_texture(old) && f.title.host.knows(old));
    f.title.let_named_go(second);
    f.title.sync_dirty();
    assert!(!f.title.host.knows(old));
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
}

#[test]
fn covered_clear_keeps_a_copy_held_by_a_recorded_list() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    let mut list = NativeList::<Recording>::default();
    list.copy_texture_region(
        &f.title.host.texture_at(old).unwrap(), 0, 0, [0; 3], [1; 3],
        &f.title.host.texture_at(new).unwrap(), 0, 0, [0; 3],
    );
    f.title.sync_dirty();
    assert!(f.title.host.knows(old) && heap.shows_texture(old));
    drop(list);
    f.title.sync_dirty();
    assert!(!f.title.host.knows(old));
    assert_eq!(heap.heap.borrow()[&0], (0, 0));
}

#[test]
fn covered_clear_reverse_displacement_keeps_both_copies() {
    let mut f = CoveredCopies::new();
    let old = f.place(0, 32);
    let heap = f.show(old);
    let new = f.place(0x800, 16);
    f.title.sync_dirty();
    assert!(f.title.host.is_covered(old));
    assert!(!f.title.realize(old));
    assert_eq!(f.title.covered_history.ping_pong, 0);
    assert_eq!(f.place(0, 32), old);
    assert_eq!(f.title.covered_history.ping_pong, 1);
    // Explicit placements in both directions must preserve the exemption as well.
    for _ in 0..3 {
        f.place(0x800, 16);
        f.place(0, 32);
        f.title.sync_dirty();
        assert!(f.title.host.knows(old) && f.title.host.knows(new));
        assert!(f.title.covered.is_empty());
    }
    f.title.host.remirror_tables(&[Some(Root::Table(heap.clone(), 0))], None);
    assert!(heap.shows_texture(old));
    assert_eq!(f.title.covered_history.ping_pong, 1);
}

#[test]
fn covered_clear_switch_off_preserves_the_shown_copy() {
    let mut f = CoveredCopies::new();
    f.title.covered_clear = false;
    let old = f.place(0, 32);
    let heap = f.show(old);
    f.place(0x800, 16);
    f.title.sync_dirty();
    assert!(f.title.host.knows(old) && heap.shows_texture(old));
    assert_eq!(f.title.host.cleared(), 0);
    assert!(heap.mirrored.borrow().contains_key(&0));
    crate::descriptor::record(f.records.as_ptr() as u64, crate::descriptor::View::Constant(0, 0));
    assert!(f.title.host.mirror(f.records.as_ptr() as u64).unwrap());
    f.title.sync_dirty();
    assert!(f.title.host.is_evicted(old));
}

#[test]
fn covered_clear_keeps_gpu_written_and_unordered_copies() {
    for unordered in [false, true] {
        let mut f = CoveredCopies::new();
        let old = f.place(0, 32);
        let heap = f.show(old);
        f.place(0x800, 16);
        assert!(f.title.covered.contains_key(&old));
        if unordered {
            let mut uav = [0u32; 10];
            (uav[0], uav[1]) = (28, 4);
            crate::descriptor::record(f.records.as_ptr() as u64, crate::descriptor::View::Unordered(old, 0, Some(uav)));
            assert!(f.title.host.mirror(f.records.as_ptr() as u64).unwrap());
        } else {
            f.title.gpu_written(old);
        }
        f.title.sync_dirty();
        assert!(f.title.host.knows(old) && heap.shows_texture(old));
        assert_eq!(f.title.host.cleared(), 0);
    }
}

#[test]
fn covered_clear_reads_the_environment_switch() {
    let f = CoveredCopies::new();
    assert_eq!(f.title.covered_clear, std::env::var("VICTORY_COVERED_CLEAR").as_deref() != Ok("0"));
}
