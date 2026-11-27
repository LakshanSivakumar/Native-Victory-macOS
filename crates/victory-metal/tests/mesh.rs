//! Synthetic DXIL mesh/fragment pipelines; no title data. Same converter gate as render.rs.
//! Use the universal MSC 4.0 beta 3 `VICTORY_MSC_DYLIB` from docs/native-host.md on both ABIs.
#![cfg(target_os = "macos")]
mod common;
use common::*;
use victory_metal::backend::{ComputeBackend, Heaps, RootArgs};
use victory_metal::metal::Metal;
use victory_metal::resources::*;

#[test]
fn mesh_triangle_direct_and_indirect_read_title_memory() {
    triangle_direct_and_indirect(false);
}

#[test]
fn mesh_triangle_amplified_direct_and_indirect_read_title_memory() {
    triangle_direct_and_indirect(true);
}

fn triangle_direct_and_indirect(amplified: bool) {
    for m in backends() {
        let ms = fixture(if amplified {
            "tri_ms_payload"
        } else {
            "tri_ms"
        });
        let amplification = amplified.then(|| fixture("tri_as"));
        let ps = fixture("mesh_ps");
        let (lib, reflection) = victory_metal::msc::Msc::open_default()
            .convert_mesh_with(&ms, None, false)
            .unwrap();
        assert!(!lib.is_empty());
        assert_eq!(reflection.info.threads, [32, 1, 1]);
        assert_eq!(reflection.info.max_vertices, 3);
        assert_eq!(reflection.info.max_primitives, 1);
        let pipe = m
            .create_render_pipeline(&RenderPipelineDesc {
                mesh: Some(MeshPipelineDesc {
                    mesh: &ms,
                    layout: None,
                    amplification: amplification.as_deref(),
                    amplification_layout: None,
                }),
                ps: Some(&ps),
                color: &[Format::Rgba8Unorm],
                // tri_ms emits a CCW face; the default is back culling with CW fronts.
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap();
        let mut td = TextureDesc::d2(Format::Rgba8Unorm, 16, 16);
        td.render_target = true;
        let color = m.create_texture(&td).unwrap();
        let arena = Arena::new(&m, 1);
        arena.write(64, &[0u32; 3]);
        arena.write(128, &[1u32; 3]);
        for indirect in [false, true] {
            let tint = if indirect {
                [0f32, 1., 0., 1.]
            } else {
                [1., 0., 0., 1.]
            };
            let draw = Draw {
                pipeline: &pipe,
                topology: Topology::Triangles,
                state: DrawState::default(),
                root: RootArgs::default()
                    .constants(&if amplified { [0f32; 4] } else { tint }.map(f32::to_bits)),
                fragment_root: None,
                object_root: amplified
                    .then(|| RootArgs::default().constants(&tint.map(f32::to_bits))),
                heaps: Heaps {
                    resource: None,
                    sampler: None,
                },
                resources: vec![],
                vertex_buffers: vec![],
                index_buffer: None,
                call: DrawCall::Mesh { groups: [1; 3] },
                count: None,
                indirect: indirect.then_some((&arena.buf, arena.at(64) as *const u8)),
            };
            let mut passes = Vec::new();
            if indirect {
                passes.push(Pass::Blit(vec![BlitOp::Copy {
                    src: (&arena.buf, arena.at(128)),
                    dst: (&arena.buf, arena.at(64)),
                    len: 12,
                }]));
            }
            passes.push(Pass::Render(RenderPass {
                color: vec![ColorTarget {
                    texture: &color,
                    clear: Some([0.; 4]),
                }],
                depth: None,
                draws: vec![draw],
            }));
            let q = m.create_queue().unwrap();
            m.make_resident(&q, &[&arena.buf]).unwrap();
            let f = m.create_fence(0).unwrap();
            m.submit_passes(
                &q,
                &PassSubmission {
                    waits: &[],
                    passes: &passes,
                    signals: &[(&f, 1)],
                },
            )
            .unwrap();
            assert!(m.wait(&f, 1, 10_000), "mesh GPU work did not finish");
            let mut pixels = vec![0u8; 16 * 16 * 4];
            m.read_texture(&color, 0, 0, &mut pixels, 64, 0);
            let expected = if indirect {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            };
            assert!(
                pixels.chunks_exact(4).all(|p| p == expected),
                "amplified={amplified} indirect={indirect} residency_sets={}: center {:?}",
                m.residency_sets(),
                &pixels[544..548]
            );
        }
    }
}

fn color_target(m: &Metal) -> victory_metal::gfx::Texture {
    let mut td = TextureDesc::d2(Format::Rgba8Unorm, 16, 16);
    td.render_target = true;
    m.create_texture(&td).unwrap()
}

fn mesh_draw(pipe: &victory_metal::gfx::RenderPipeline) -> Draw<'_, Metal> {
    Draw {
        pipeline: pipe,
        topology: Topology::Triangles,
        state: DrawState::default(),
        root: RootArgs::default(),
        fragment_root: None,
        object_root: None,
        heaps: Heaps {
            resource: None,
            sampler: None,
        },
        resources: vec![],
        vertex_buffers: vec![],
        index_buffer: None,
        call: DrawCall::Mesh { groups: [1; 3] },
        indirect: None,
        count: None,
    }
}

/// A blue clear distinguishes no fragments from a shader returning zero/transparent black.
fn submit_mesh<'a>(
    m: &Metal,
    color: &'a victory_metal::gfx::Texture,
    draw: Draw<'a, Metal>,
    mut passes: Vec<Pass<'a, Metal>>,
) -> Vec<u8> {
    let q = m.create_queue().unwrap();
    // Exercise queue residency and inter-encoder fences as well as useResource fallback.
    if !draw.resources.is_empty() {
        m.make_resident(&q, &draw.resources).unwrap();
    }
    passes.push(Pass::Render(RenderPass {
        color: vec![ColorTarget {
            texture: color,
            clear: Some([0., 0., 1., 1.]),
        }],
        depth: None,
        draws: vec![draw],
    }));
    let f = m.create_fence(0).unwrap();
    m.submit_passes(
        &q,
        &PassSubmission {
            waits: &[],
            passes: &passes,
            signals: &[(&f, 1)],
        },
    )
    .unwrap();
    assert!(m.wait(&f, 1, 10_000), "mesh GPU work did not finish");
    let mut pixels = vec![0u8; 16 * 16 * 4];
    m.read_texture(color, 0, 0, &mut pixels, 64, 0);
    pixels
}

#[test]
fn mesh_uav_proves_execution_without_rasterization() {
    for m in backends() {
        let ms = fixture("mesh_exec_ms");
        let pipe = m
            .create_render_pipeline(&RenderPipelineDesc {
                mesh: Some(MeshPipelineDesc {
                    mesh: &ms,
                    layout: None,
                    amplification: None,
                    amplification_layout: None,
                }),
                color: &[Format::Rgba8Unorm],
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap();
        let color = color_target(&m);
        let arena = Arena::new(&m, 1);
        // Zero groups is a negative control; then prove direct and GPU-written indirect grids.
        for (indirect, groups) in [(false, [0, 2, 2]), (false, [2; 3]), (true, [2; 3])] {
            arena.write(64, &[0u32; 3]);
            arena.write(128, &groups);
            arena.write(256, &[0u32; 256]);
            let mut draw = mesh_draw(&pipe);
            draw.root = RootArgs::default().address(m.gpu_address(&arena.buf, arena.at(256)));
            draw.resources.push(&arena.buf);
            draw.call = DrawCall::Mesh { groups };
            draw.indirect = indirect.then_some((&arena.buf, arena.at(64) as *const u8));
            let passes = if indirect {
                vec![Pass::Blit(vec![BlitOp::Copy {
                    src: (&arena.buf, arena.at(128)),
                    dst: (&arena.buf, arena.at(64)),
                    len: 12,
                }])]
            } else {
                vec![]
            };
            let pixels = submit_mesh(&m, &color, draw, passes);
            let words = arena.read::<u32>(256, 256);
            eprintln!(
                "mesh UAV: indirect={indirect} groups={groups:?} residency_sets={} first={:#x} last={:#x}",
                m.residency_sets(),
                words[0],
                words[255]
            );
            for (i, word) in words.iter().enumerate() {
                let expected = if groups[0] == 0 {
                    0
                } else {
                    0x4d53_0000 + i as u32
                };
                assert_eq!(
                    *word, expected,
                    "mesh lane/group word {i}: indirect={indirect} groups={groups:?}"
                );
            }
            assert!(
                pixels.chunks_exact(4).all(|p| p == [0, 0, 255, 255]),
                "zero-output mesh must preserve the blue clear"
            );
        }
    }
}

/// A count written by earlier GPU work gates a three-dword mesh record.
#[test]
fn a_gpu_count_buffer_limits_indirect_mesh_dispatches() {
    for m in backends() {
        let ms = fixture("mesh_exec_ms");
        let pipe = m.create_render_pipeline(&RenderPipelineDesc {
            mesh: Some(MeshPipelineDesc {
                mesh: &ms, layout: None, amplification: None, amplification_layout: None,
            }),
            color: &[Format::Rgba8Unorm],
            cull: Cull::None,
            ..Default::default()
        }).unwrap();
        let color = color_target(&m);
        let arena = Arena::new(&m, 1);
        for (count, index, runs) in [(0u32, 0, false), (1, 0, true), (1, 1, false), (2, 1, true)] {
            arena.write(64, &[2u32; 3]);
            arena.write(128, &[count]);
            arena.write(132, &[u32::MAX]);
            arena.write(256, &[0u32; 256]);
            let mut draw = mesh_draw(&pipe);
            draw.root = RootArgs::default().address(m.gpu_address(&arena.buf, arena.at(256)));
            draw.resources.push(&arena.buf);
            draw.indirect = Some((&arena.buf, arena.at(64)));
            draw.count = Some((&arena.buf, arena.at(132), index));
            submit_mesh(&m, &color, draw, vec![Pass::Blit(vec![BlitOp::Copy {
                src: (&arena.buf, arena.at(128)), dst: (&arena.buf, arena.at(132)), len: 4,
            }])]);
            for (i, word) in arena.read::<u32>(256, 256).iter().enumerate() {
                assert_eq!(*word, if runs { 0x4d53_0000 + i as u32 } else { 0 },
                    "count={count} index={index} word={i}");
            }
        }
    }
}

#[test]
fn mesh_fullscreen_constant_fragment_without_stage_in() {
    for m in backends() {
        let ms = fixture("mesh_fullscreen_ms");
        let ps = fixture("mesh_solid_ps");
        let pipe = m
            .create_render_pipeline(&RenderPipelineDesc {
                mesh: Some(MeshPipelineDesc {
                    mesh: &ms,
                    layout: None,
                    amplification: None,
                    amplification_layout: None,
                }),
                ps: Some(&ps),
                color: &[Format::Rgba8Unorm],
                cull: Cull::None,
                ..Default::default()
            })
            .unwrap();
        let color = color_target(&m);
        // Explicit first, so a broken default viewport cannot hide the rasterization check.
        for viewport in [
            Some(Viewport {
                x: 0.,
                y: 0.,
                width: 16.,
                height: 16.,
                min_depth: 0.,
                max_depth: 1.,
            }),
            None,
        ] {
            let mut draw = mesh_draw(&pipe);
            draw.state.viewport = viewport;
            let pixels = submit_mesh(&m, &color, draw, vec![]);
            assert!(
                pixels.chunks_exact(4).all(|p| p == [255, 0, 0, 255]),
                "constant fragment: viewport={viewport:?} residency_sets={} center={:?}",
                m.residency_sets(),
                &pixels[544..548]
            );
        }
    }
}

/// Reproduce the original failure with its unchanged shaders, then vary only culling/winding.
#[test]
fn mesh_culling_matches_winding() {
    for m in backends() {
        let ms = fixture("tri_ms");
        let ps = fixture("mesh_ps");
        let color = color_target(&m);
        let mut outcomes = Vec::new();
        for (cull, front_ccw, visible) in [
            (Cull::None, false, true),
            (Cull::None, true, true),
            (Cull::Back, false, false),
            (Cull::Back, true, true),
            (Cull::Front, false, true),
            (Cull::Front, true, false),
        ] {
            let pipe = m
                .create_render_pipeline(&RenderPipelineDesc {
                    mesh: Some(MeshPipelineDesc {
                        mesh: &ms,
                        layout: None,
                        amplification: None,
                        amplification_layout: None,
                    }),
                    ps: Some(&ps),
                    color: &[Format::Rgba8Unorm],
                    cull,
                    raster: Raster {
                        front_ccw,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
            let mut draw = mesh_draw(&pipe);
            draw.root = RootArgs::default().constants(&[1f32, 0., 0., 1.].map(f32::to_bits));
            let pixels = submit_mesh(&m, &color, draw, vec![]);
            let expected = if visible {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            eprintln!(
                "mesh culling: {cull:?} front_ccw={front_ccw} residency_sets={} center={:?} expected={expected:?}",
                m.residency_sets(),
                &pixels[544..548]
            );
            outcomes.push(pixels.chunks_exact(4).all(|p| p == expected));
        }
        assert!(
            outcomes.iter().all(|v| *v),
            "mesh culling matrix: {outcomes:?}"
        );
    }
}

#[test]
fn converter_reflects_mesh_threadgroups_and_outputs() {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return;
    }
    let cache = tempfile::tempdir().unwrap();
    victory_metal::disk::use_dir(cache.path().to_owned());
    let msc = victory_metal::msc::Msc::open_default();
    assert_eq!(
        victory_shader::emit::metal_mesh_bytes(&fixture("tri_ms")),
        Some(96)
    );
    for _ in 0..2 {
        let (lib, reflected) = msc
            .convert_mesh_with(&fixture("tri_ms"), None, false)
            .unwrap();
        assert!(!lib.is_empty());
        assert_eq!(reflected.entry, "main");
        assert_eq!(reflected.info.threads, [32, 1, 1]);
        assert_eq!(reflected.info.max_vertices, 3);
        assert_eq!(reflected.info.max_primitives, 1);
        assert_eq!(reflected.info.topology, 2); // IRMeshShaderPrimitiveTopologyTriangle
        let (_, object) = msc
            .convert_mesh_with(&fixture("tri_as"), None, true)
            .unwrap();
        assert_eq!(object.info.threads, [1, 1, 1]);
        assert_eq!(object.info.payload_bytes, 16);
        let (_, mesh) = msc
            .convert_mesh_with(&fixture("tri_ms_payload"), None, false)
            .unwrap();
        assert_eq!(mesh.info.payload_bytes, 16);
        for name in ["mesh_exec_ms", "mesh_fullscreen_ms"] {
            let dxil = fixture(name);
            assert_eq!(victory_shader::emit::metal_mesh_bytes(&dxil), Some(48));
            let (_, reflected) = msc.convert_mesh_with(&dxil, None, false).unwrap();
            assert_eq!(reflected.info.threads, [32, 1, 1]);
            assert_eq!(reflected.info.max_vertices, 3);
            assert_eq!(reflected.info.max_primitives, 1);
            assert_eq!(reflected.info.topology, 2);
        }
        assert_eq!(msc.convert(&fixture("mesh_solid_ps"), None).unwrap().0, 2);
    }
}
