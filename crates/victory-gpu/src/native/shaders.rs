//! Title shaders on the native host: the console container translated to DXIL for a 32-lane
//! host (`Options { wave32_host: true, .. }`: the console's wave64 compute programs run as two host
//! waves), DXIL to a Metal library through the converter (`create_compute_pipeline` and
//! `create_render_pipeline` on the backend: every library is its own file name, because Metal
//! keeps libraries by URL), pipelines cached by the content of their DXIL, and what the host
//! cannot run refused, logged and counted instead of failing the submission.
//!
//! Title memory at addresses the shader computes is found through the page table of
//! `victory_shader`'s "Title memory", which `super::addr` fills with views of the pools
//! themselves (the pool-offset contract). What a translated shader needs
//! from the host that this one does not give yet, and is refused for: buffer views read from the
//! heap by index, the write marks and view addresses of the shadow-copy design, and the page
//! table for graphics pipelines outside the guessed path. See docs/metal-backend.md.

use super::list::{Binding, ComputePso};
use super::state::{self, Graphics};
use super::title::{MeshRefusal, Pipe, Title};
use std::collections::HashMap;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_shader::emit::{HostShader, Options};

/// One policy for dispatch and translation, fixed for the process before prewarm workers run.
/// Opting in restores the existing builder path; it does not make refused stages supported.
pub(crate) fn builder_runs() -> bool {
    static RUNS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RUNS.get_or_init(|| {
        let runs = builder_requested(std::env::var_os("VICTORY_BVH_BUILDER").as_deref());
        if !runs {
            tracing::info!("ray-tracing structure builder off (VICTORY_BVH_BUILDER=1 runs it)");
        }
        runs
    })
}

fn builder_requested(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// The translation settings of the native host.
pub fn options() -> Options {
    Options {
        wave32_host: true,
        zero_copy_memory: true,
        bvh_no_hit: !builder_runs(),
        ..Options::default()
    }
}

/// A console shader container as DXIL for the native host.
pub fn translate(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, options())
}

/// Whether [`translate`]'s translation is in the disk cache already.
pub fn cached(container: &[u8]) -> bool {
    crate::shader_cache::native_present(container, &format!("{:?}", options()))
}

/// Keeps `container`, as the run is about to translate it, for later warm-ups
/// (`crate::prewarm::keep`); nothing with the shader cache off.
#[cfg(target_os = "macos")]
pub fn keep(container: &[u8]) {
    if let Some(dir) = crate::prewarm::kept_dir() {
        crate::prewarm::keep(&dir, &blake3::hash(container).to_hex()[..16], container);
    }
}

/// The files of the containers earlier runs kept ([`keep`]).
#[cfg(target_os = "macos")]
pub fn kept() -> Vec<std::path::PathBuf> {
    crate::prewarm::kept_dir().map(|dir| crate::prewarm::kept(&dir)).unwrap_or_default()
}

/// The container in one of [`kept`]'s files; `None` if it is damaged.
#[cfg(target_os = "macos")]
pub fn read_kept(file: &std::path::Path) -> Option<Vec<u8>> {
    crate::prewarm::read_kept(file)
}

/// [`victory_shader::emit::translate_with`] through the on-disk cache (`shader_cache::native_shader`):
/// `o` is part of the key.
fn translate_cached(container: &[u8], o: Options) -> Result<HostShader, String> {
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_shader(container, &format!("{o:?}"), |c| {
        victory_shader::emit::translate_with(c, o).map_err(|e| e.to_string())
    });
    #[cfg(not(any(windows, target_os = "macos")))]
    return victory_shader::emit::translate_with(container, o).map_err(|e| e.to_string());
}

/// A compute shader's container translated for dispatches of thread counts
/// (`Options::thread_counts`).
pub fn translate_threads(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, Options { thread_counts: true, ..options() })
}

/// A vertex shader's rectangle variant: guessed attribute strips or described output expansion.
pub fn translate_rect(container: &[u8]) -> Result<HostShader, String> {
    translate_cached(container, Options { rect_list: true, ..options() })
}

/// A guessed vertex shader's container translated for rectangle lists or not, writing the
/// `TEXCOORD` parameters in `extra` (bit n) too: those its pixel shader reads and it does not write;
/// with `pixel_reads`, exporting what its pixel shader reads (`Options::pixel_reads`).
pub fn translate_extra(container: &[u8], rect: bool, extra: u32, copy_params: [u8; 32], pixel_reads: Option<(u32, u64)>) -> Result<HostShader, String> {
    translate_cached(container, Options { rect_list: rect, extra_params: extra, copy_params, pixel_reads, ..options() })
}

/// Whether [`translate_extra`]'s translation is in the disk cache already.
pub fn extra_cached(container: &[u8], rect: bool, extra: u32, copy_params: [u8; 32], pixel_reads: Option<(u32, u64)>) -> bool {
    let o = Options { rect_list: rect, extra_params: extra, copy_params, pixel_reads, ..options() };
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_present(container, &format!("{o:?}"));
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (container, rect, extra, copy_params, pixel_reads, o);
        false
    }
}

/// A pixel shader's container translated for colour targets of integer formats (bit n: target n).
pub fn translate_targets(container: &[u8], integer: u8) -> Result<HostShader, String> {
    translate_cached(container, Options { integer_targets: integer, ..options() })
}

/// Whether [`translate_targets`]'s translation is in the disk cache already.
pub fn targets_cached(container: &[u8], integer: u8) -> bool {
    let o = Options { integer_targets: integer, ..options() };
    #[cfg(any(windows, target_os = "macos"))]
    return crate::shader_cache::native_present(container, &format!("{o:?}"));
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = (container, o);
        false
    }
}

/// Bit n set: colour target n has an integer format.
pub fn integer_targets(color: &[Format]) -> u8 {
    color.iter().enumerate().filter(|(n, f)| *n < 8 && f.is_integer()).fold(0, |m, (n, _)| m | 1 << n)
}

/// What the host cannot give a translated shader (the reason is the refusal's); `memory`: it can
/// give the page table (compute and graphics shaders, guessed or described). A shader that indexes the resource
/// heap itself is not refused here: the bound heaps are at the converter's heap slots for every
/// stage (`bind_root_render`, `bind_root` in `victory-metal`).
pub fn unsupported(shader: &HostShader, memory: bool) -> Option<&'static str> {
    if shader.memory_writes {
        return Some("marks the pages it writes (translated without zero_copy_memory)");
    }
    if shader.memory && !memory {
        return Some("reads title memory through the page table in a graphics pipeline");
    }
    if shader.view_marks {
        return Some("needs write marks");
    }
    None
}

/// Why a vertex shader and pixel shader cannot be a pipeline of the host, if they cannot.
fn refusal(vs: &HostShader, ps: Option<&HostShader>, memory: bool) -> Option<&'static str> {
    for s in [Some(vs), ps].into_iter().flatten() {
        if let Some(why) = unsupported(s, memory) {
            return Some(why);
        }
    }
    None
}

/// The thread-group size a compute container declares (`PSV0`'s runtime info).
pub fn threads_of(dxil: &[u8]) -> Option<[usize; 3]> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let part = container.part(FourCC::PSV0)?;
    // The runtime info is 48 bytes from version 2 on; the group size is its last three words.
    if u32::from_le_bytes(part.data.get(..4)?.try_into().ok()?) < 48 {
        return None;
    }
    let d = part.data.get(40..52)?;
    let n = |k: usize| u32::from_le_bytes(d[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
    Some([n(0), n(1), n(2)]).filter(|t| t.iter().all(|&n| n > 0))
}

/// Metal's per-threadgroup mesh output limit. The Wine/D3DMetal measure
/// includes alignment between signature elements; the packed component count
/// alone misses e.g. a 34,816-byte layout that hangs the queue.
const METAL_MESH_OUTPUT_BYTES: u32 = 32_768;

/// Port of `pipeline/host.rs::metal_mesh_bytes` from the Wine build, using
/// the container parser for bounds checks. Keep the larger of the aligned
/// layout and packed component count, as the Wine pipeline guard does.
pub(crate) fn metal_mesh_bytes(dxil: &[u8]) -> Option<u32> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let u32_at = |b: &[u8], at: usize| {
        Some(u32::from_le_bytes(
            b.get(at..at.checked_add(4)?)?.try_into().ok()?,
        ))
    };
    let psv = container.part(FourCC::PSV0)?.data;
    let info = psv.get(4..4usize.checked_add(u32_at(psv, 0)? as usize)?)?;
    // Shader kind 13 is mesh.
    if *info.get(24)? != 13 {
        return None;
    }
    let maximum = |at: usize| {
        Some(u32::from(u16::from_le_bytes(
            info.get(at..at + 2)?.try_into().ok()?,
        )))
    };
    let elements = |name| -> Option<Vec<(u32, u32, bool)>> {
        let Some(part) = container.part(name) else {
            return Some(Vec::new());
        };
        let data = part.data;
        let count = u32_at(data, 0)? as usize;
        let base = u32_at(data, 4)? as usize;
        // Validate the declared records before allocating or iterating.
        data.get(base..base.checked_add(count.checked_mul(32)?)?)?;
        (0..count)
            .map(|i| {
                let at = base + 32 * i;
                Some((
                    u32_at(data, at + 12)?,
                    data[at + 24].count_ones(),
                    u32_at(data, at + 20)? != u32::MAX,
                ))
            })
            .collect()
    };
    let layout = |sizes: &mut dyn Iterator<Item = u32>| {
        sizes.filter(|&size| size > 0).try_fold(0u32, |end, size| {
            end.checked_next_multiple_of(size)?.checked_add(size)
        })
    };
    let vertex = elements(FourCC::OSG1)?;
    let primitive = elements(FourCC::PSG1)?;
    let distances = vertex
        .iter()
        .filter(|e| matches!(e.0, 2 | 3))
        .try_fold(0u32, |sum, e| sum.checked_add(e.1))?;
    let vertex_bytes = layout(
        &mut std::iter::repeat_n(4, distances as usize)
            .chain([16])
            .chain(vertex.iter().filter(|e| e.0 != 1).map(|e| 4 * e.1)),
    )?;
    let primitive_bytes = layout(&mut primitive.iter().map(|e| 4 * e.1))?;
    let (vertices, primitives) = (maximum(12)?, maximum(14)?);
    let aligned = vertices
        .checked_mul(vertex_bytes)?
        .checked_add(primitives.checked_mul(primitive_bytes)?)?;
    let packed = |elements: &[(u32, u32, bool)]| {
        elements
            .iter()
            .filter(|e| e.2)
            .try_fold(0u32, |sum, e| sum.checked_add(4 * e.1))
    };
    let packed = vertices
        .checked_mul(packed(&vertex)?)?
        .checked_add(primitives.checked_mul(packed(&primitive)?)?)?;
    Some(aligned.max(packed))
}

/// Default on for the native Metal host. The override only bypasses this
/// size guard; it does not add the missing native mesh dispatch support.
fn refuse_over_limit_mesh() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MESH_OVER_LIMIT").as_deref() != Ok("1"))
}

/// Whether a mesh program declaring `bytes` of outputs is refused for size: over Metal's
/// documented 32 KiB, unless the guard is off (`refuse`: [`refuse_over_limit_mesh`]). Both checks
/// ask this one (before conversion, `refuse_mesh`, and when the pipeline is made, `make_render`),
/// so `VICTORY_MESH_OVER_LIMIT=1` lets such a program through both.
pub(super) fn over_limit(bytes: u32, refuse: bool) -> bool {
    refuse && bytes > METAL_MESH_OUTPUT_BYTES
}

/// Counts of what the shader path did, and why it refused what it did.
#[derive(Default, Debug, Clone)]
pub struct ShaderStats {
    /// Pipelines made.
    pub pipelines: u64,
    /// Pipelines served from the library cache.
    pub cached: u64,
    /// Times a pipeline was not compiled within `Title::compile_wait`: its work is dropped that
    /// time and nothing is cached, so a later use picks the pipeline up.
    pub put_off: u64,
    /// Refusals, by reason, with how often.
    pub refused: Vec<(String, u64)>,
    /// Mesh source shaders warned for output storage, independently of the bounded detail list.
    mesh_limits: std::collections::HashSet<String>,
    /// Pipeline variants refused for mesh output storage.
    pub mesh_over_limit: u64,
}

impl ShaderStats {
    /// All refusals.
    pub fn refusals(&self) -> u64 {
        self.refused.iter().map(|r| r.1).sum::<u64>() + self.mesh_over_limit
    }

    fn mesh_limit(&mut self, shader: &str, bytes: u32) {
        self.mesh_over_limit += 1;
        if self.mesh_limits.insert(shader.to_owned()) {
            tracing::warn!(shader, bytes, limit = victory_shader::emit::METAL_MESH_OUTPUT_BYTES,
                "mesh outputs exceed Metal's limit; pipeline refused");
        }
    }

    fn refuse(&mut self, what: &str, reason: String) {
        let key = format!("{what}: {reason}");
        match self.refused.iter_mut().find(|r| r.0 == key) {
            Some(r) => r.1 += 1,
            None => {
                tracing::warn!(shader = what, %reason, "title shader refused; its work is dropped");
                if self.refused.len() < 256 {
                    self.refused.push((key, 1));
                }
            }
        }
    }
}

/// The pipelines made, by the content of their DXIL (and, for graphics, their state).
pub struct Libraries<B: GraphicsBackend> {
    compute: HashMap<[u8; 32], Pipe<B>>,
    render: HashMap<[u8; 32], Pipe<B>>,
}

impl<B: GraphicsBackend> Default for Libraries<B> {
    fn default() -> Self {
        Libraries {
            compute: HashMap::new(),
            render: HashMap::new(),
        }
    }
}

impl<B: PresentBackend> Title<B> {
    /// The compute pipeline of a translated shader: from the cache when its DXIL was converted
    /// before, else converted now. [`Pipe::Refused`] (counted) when the host cannot run it.
    pub fn compute_pipe(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        // Shaders indexing the heap themselves run: the heaps are bound for compute as for render
        // (`heap_tests::a_compute_shader_loads_a_texture_by_heap_index`).
        if let Some(why) = unsupported(shader, true) {
            self.shaders.refuse(name, why.into());
            return Pipe::Refused;
        }
        let key = *blake3::hash(&shader.dxil).as_bytes();
        if let Some(p) = self.libraries.compute.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let put_off = self.shaders.put_off;
        let made = self.make_compute(shader, name);
        if self.shaders.put_off == put_off {
            self.libraries.compute.insert(key, made.clone());
        }
        made
    }

    fn make_compute(&mut self, shader: &HostShader, name: &str) -> Pipe<B> {
        let Some(threads) = threads_of(&shader.dxil) else {
            self.shaders.refuse(name, "no thread group size".into());
            return Pipe::Refused;
        };
        // Translator output carries no root signature: the argument buffer is laid out by the
        // binding the host fills it by.
        let binding = Binding { name: name.into(), ..Binding::of(shader) };
        let made = self.host.backend.create_compute_pipeline_within(&shader.dxil, binding.layout().as_ref(), self.compile_wait);
        match made {
            Ok(None) => self.put_off(),
            Ok(Some(pso)) => {
                self.shaders.pipelines += 1;
                self.host.backend.name_pipeline(&pso, name);
                Pipe::Compute(Rc::new(ComputePso {
                    pso,
                    threads,
                    binding: Some(binding),
                }))
            }
            Err(e) => {
                #[cfg(target_os = "macos")]
                self.compile_failed(name, &e);
                self.shaders.refuse(name, format!("conversion: {e}"));
                Pipe::Refused
            }
        }
    }

    /// The render pipeline of a translated vertex/mesh producer, pixel shader and graphics state.
    /// A mesh producer (`HwStage::Ms`: a described mesh stage of a stream pipeline, or a guessed
    /// vertex program that exports primitives) makes a mesh pipeline, also [`Pipe::Render`].
    /// For mesh pipelines `name` identifies the source mesh, independent of the pixel pairing.
    pub fn render_pipe(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        graphics: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        if self.refuse_mesh(vs, name) {
            return Pipe::MeshRefused(MeshRefusal::OverLimit);
        }
        // The draw binds the page table for any render pipeline whose binding asks (`memory`).
        if let Some(why) = refusal(vs, ps, true) {
            self.shaders.refuse(name, why.into());
            return if vs.stage == victory_shader::HwStage::Ms { Pipe::MeshRefused(MeshRefusal::NoPipeline) } else { Pipe::Refused };
        }
        let mut h = blake3::Hasher::new();
        h.update(&vs.dxil);
        h.update(ps.map_or(&[][..], |p| &p.dxil));
        h.update(format!("{graphics:?}").as_bytes());
        // The arguments are part of the pipeline's binding.
        h.update(format!("{:?}{:?}", vs.user_data, ps.map(|p| &p.user_data)).as_bytes());
        let key = *h.finalize().as_bytes();
        if let Some(p) = self.libraries.render.get(&key) {
            self.shaders.cached += 1;
            return p.clone();
        }
        let put_off = self.shaders.put_off;
        let made = self.make_render(vs, ps, graphics, name);
        if self.shaders.put_off == put_off {
            self.libraries.render.insert(key, made.clone());
        }
        made
    }

    /// A pipeline Metal is still compiling: no pipeline this time, and the caller caches nothing.
    fn put_off(&mut self) -> Pipe<B> {
        self.shaders.put_off += 1;
        if self.shaders.put_off.is_power_of_two() {
            tracing::info!(put_off = self.shaders.put_off, "pipelines put off for their compiles so far");
        }
        Pipe::Refused
    }

    fn make_render(
        &mut self,
        vs: &HostShader,
        ps: Option<&HostShader>,
        g: &Graphics,
        name: &str,
    ) -> Pipe<B> {
        let mesh = vs.stage == victory_shader::HwStage::Ms;
        let failed = || if mesh { Pipe::MeshRefused(MeshRefusal::ConversionFailed) } else { Pipe::Refused };
        if mesh {
            match victory_shader::emit::metal_mesh_bytes(&vs.dxil) {
                Some(bytes) if over_limit(bytes, refuse_over_limit_mesh()) => {
                    // A mesh can occur in many target-format variants. Warn once by source shader,
                    // but keep the refusal reason in every cached pipeline for per-dispatch counts.
                    self.shaders.mesh_limit(name, bytes);
                    return Pipe::MeshRefused(MeshRefusal::OverLimit);
                }
                None => {
                    self.shaders.refuse(name, "mesh output footprint is unknown".into());
                    return failed();
                }
                _ => {}
            }
        }
        let (attributes, streams) = if mesh { (Vec::new(), Vec::new()) } else {
            let Some(reads) = state::vertex_inputs(&vs.dxil) else {
                self.shaders.refuse(name, "no input signature".into());
                return failed();
            };
            let Some((mut attributes, streams)) = g.vertex_layout(&reads) else {
                self.shaders
                    .refuse(name, "the vertex shader reads an input the layout lacks".into());
                return failed();
            };
            // A system value ahead of the attributes takes an attribute number: described as the
            // attribute after it, which the converter's function ignores.
            for at in state::system_gaps(&vs.dxil) {
                if let Some(next) = attributes.get(at).copied() {
                    attributes.insert(at, next);
                }
            }
            (attributes, streams)
        };
        // Resolve colour and depth formats, reporting unknown formats as refusals.
        let (color, depth) = match (g.color_formats(), g.depth_format()) {
            (Ok(c), Ok(d)) => (c, d),
            (Err(dxgi), _) => {
                self.shaders
                    .refuse(name, format!("unknown target format {}", dxgi));
                return failed();
            }
            (_, Err(dxgi)) => {
                self.shaders
                    .refuse(name, format!("unknown depth format {}", dxgi));
                return failed();
            }
        };
        if g.samples > 1 {
            self.shaders.refuse(name, "multisampled targets".into());
            return failed();
        }
        let color_state = g.color_state();
        // A pixel shader with one output more than there are targets feeds target 0's second
        // blend source; the converter must be told, or Metal refuses the pipeline.
        let targets = ps.map_or_else(Vec::new, |p| state::pixel_targets(&p.dxil));
        let dual_source = state::dual_source(color.len(), &targets, &color_state);
        if env_set!("VICTORY_TRACE_PIPES") {
            tracing::info!(name, stencil = ?g.stencil(), depth = ?g.depth_test(), ?color_state, "guessed pipeline state");
        }
        let ps_full = ps.map(|ps| Binding { name: name.into(), ..Binding::of(ps) });
        let mut vs_binding = Binding { name: name.into(), ..Binding::of(vs) };
        if let Some(p) = &ps_full {
            vs_binding.typed_buffers |= p.typed_buffers;
            vs_binding.heap_typed_buffers |= p.heap_typed_buffers;
        }
        // As for compute: without the host's layout the converter makes the user-data block a
        // root CBV (a pointer), while the host writes its dwords inline, so the shader reads its
        // user data (descriptor table offsets, root constants) from address 0.
        let vs_layout = vs_binding.layout();
        let ps_layout = ps_full.as_ref().and_then(Binding::layout);
        let ps_binding = ps_full
            .filter(|b| b.memory || b.view_addresses || b.dwords > 0 || !b.views.is_empty());
        let desc = RenderPipelineDesc {
            vs: if mesh { &[] } else { &vs.dxil },
            mesh: mesh.then_some(MeshPipelineDesc { mesh: &vs.dxil, layout: vs_layout.as_ref(), amplification: None, amplification_layout: None }),
            ps: ps.map(|p| &p.dxil[..]),
            vs_layout: vs_layout.as_ref(),
            ps_layout: ps_layout.as_ref(),
            attributes: &attributes,
            streams: &streams,
            color: &color,
            depth,
            depth_test: depth.and(g.depth_test()),
            cull: g.cull(),
            color_state: &color_state,
            dual_source,
            stencil: depth.and(g.stencil()),
            raster: g.raster(),
        };
        match self.host.backend.create_render_pipeline_within(&desc, self.compile_wait) {
            Ok(None) => self.put_off(),
            Ok(Some(p)) => {
                self.shaders.pipelines += 1;
                self.host.backend.name_render_pipeline(&p, name);
                Pipe::Render(Rc::new(p), Some(Rc::new(Binding {
                    ps: ps_binding.map(Rc::new),
                    ..vs_binding
                })))
            }
            Err(e) => {
                #[cfg(target_os = "macos")]
                self.compile_failed(name, &e);
                self.shaders.refuse(name, format!("conversion: {e}"));
                failed()
            }
        }
    }

    /// Refuse before converting any stages of an over-limit mesh pipeline.
    pub(crate) fn refuse_mesh(&mut self, shader: &HostShader, name: &str) -> bool {
        if shader.stage == victory_shader::HwStage::Ms
            && let Some(bytes) = metal_mesh_bytes(&shader.dxil)
            && over_limit(bytes, refuse_over_limit_mesh())
        {
            self.shaders.refuse(name, format!(
                "mesh outputs {bytes} bytes exceed Metal's 32,768-byte limit (VICTORY_MESH_OVER_LIMIT=1 bypasses this guard)"
            ));
            return true;
        }
        false
    }

    /// A translation that failed, counted.
    pub fn refused_translation(&mut self, name: &str, reason: String) {
        self.shaders.refuse(name, format!("translation: {reason}"));
    }

    /// A pipeline compile that ended in `error`: if Metal's compiler died on it rather than
    /// refusing a shader the host cannot run, the frame spends the rest of its pipeline budget
    /// (`glue::compile_failed`), so the frame's other new pipelines are put off instead of each
    /// waiting about a second for a compiler that crashes (issue #181).
    #[cfg(target_os = "macos")]
    pub(crate) fn compile_failed(&mut self, name: &str, error: &str) {
        super::glue::compile_failed(name, error);
    }
}

#[cfg(test)]
mod mesh_tests {
    use super::*;

    /// Synthetic DXBC metadata only; no game data or executable shader code.
    fn mesh(vertices: u16, primitives: u16, outputs: &[(u32, u8)], prims: &[(u32, u8)]) -> Vec<u8> {
        let signature = |elements: &[(u32, u8)]| {
            let mut data = vec![0; 8 + elements.len() * 32];
            data[..4].copy_from_slice(&(elements.len() as u32).to_le_bytes());
            data[4..8].copy_from_slice(&8u32.to_le_bytes());
            for (i, (kind, mask)) in elements.iter().enumerate() {
                let at = 8 + i * 32;
                data[at + 12..at + 16].copy_from_slice(&kind.to_le_bytes());
                data[at + 20..at + 24].copy_from_slice(&(i as u32).to_le_bytes());
                data[at + 24] = *mask;
            }
            data
        };
        let mut psv = vec![0; 52];
        psv[..4].copy_from_slice(&48u32.to_le_bytes());
        psv[16..18].copy_from_slice(&vertices.to_le_bytes());
        psv[18..20].copy_from_slice(&primitives.to_le_bytes());
        psv[28] = 13;
        let mut dxil = vec![0; 44];
        dxil[..4].copy_from_slice(b"DXBC");
        dxil[28..32].copy_from_slice(&3u32.to_le_bytes());
        for (i, (name, data)) in [
            (*b"PSV0", psv),
            (*b"OSG1", signature(outputs)),
            (*b"PSG1", signature(prims)),
        ]
        .into_iter()
        .enumerate()
        {
            let offset = dxil.len() as u32;
            dxil[32 + 4 * i..36 + 4 * i].copy_from_slice(&offset.to_le_bytes());
            dxil.extend(name);
            dxil.extend((data.len() as u32).to_le_bytes());
            dxil.extend(data);
        }
        let size = dxil.len() as u32;
        dxil[24..28].copy_from_slice(&size.to_le_bytes());
        dxil
    }

    #[test]
    fn mesh_limit_includes_alignment_and_declared_counts() {
        let mut outputs = vec![(1, 15)]; // position
        outputs.extend([(0, 15); 15]);
        assert_eq!(
            metal_mesh_bytes(&mesh(128, 0, &outputs, &[])),
            Some(METAL_MESH_OUTPUT_BYTES)
        );
        assert_eq!(metal_mesh_bytes(&mesh(129, 0, &outputs, &[])), Some(33_024));
        let mut aligned = vec![(1, 15)];
        for _ in 0..8 {
            aligned.extend([(0, 1), (0, 15)]);
        }
        // Packed components fit (22,528), aligned Metal output does not (34,816).
        assert_eq!(metal_mesh_bytes(&mesh(128, 0, &aligned, &[])), Some(34_816));
        assert_eq!(
            metal_mesh_bytes(&mesh(128, 64, &[(1, 15)], &[(0, 15)])),
            Some(3_072)
        );
        assert_eq!(metal_mesh_bytes(b"not DXBC"), None);
        let mut not_mesh = mesh(128, 0, &outputs, &[]);
        not_mesh[44 + 8 + 28] = 1; // PSV stage is vertex
        assert_eq!(metal_mesh_bytes(&not_mesh), None);
        let malformed = mesh(128, 0, &outputs, &[]);
        assert_eq!(metal_mesh_bytes(&malformed[..malformed.len() - 1]), None);
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn builder_switch_requires_exact_opt_in() {
        assert!(!builder_requested(None));
        for value in ["", "0", "true", "yes", "01", "1 ", " 1", "2"] {
            assert!(!builder_requested(Some(OsStr::new(value))), "{value:?}");
        }
        assert!(builder_requested(Some(OsStr::new("1"))));
    }

    // Run this in separate processes with the variable absent and set to 1. No process-wide
    // environment mutation races the prewarm workers or other tests.
    #[test]
    fn builder_switch_and_traversal_options_agree() {
        let runs = builder_requested(std::env::var_os("VICTORY_BVH_BUILDER").as_deref());
        assert_eq!(builder_runs(), runs);
        assert_eq!(options().bvh_no_hit, !runs);
    }
}
