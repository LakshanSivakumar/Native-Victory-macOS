//! Vertex-ID-only rectangles must complete outputs after evaluating the program.
use super::{Options, lower};
use crate::{Container, FourCC, io, ir, resource, structure};

mod program {
    use crate as victory_shader;
    include!("../../tests/programs/rect.rs");
}

#[test]
fn vertex_id_rectangles_complete_position_and_interpolants() {
    let bytes = program::vertex_container();
    let c = Container::parse(&bytes).unwrap();
    assert!(!super::graphics::is_guessed(&c));
    let native = crate::native_programs(c.part(FourCC::DXIL).unwrap().data)
        .unwrap()
        .remove(0);
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let parts = lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: super::signature(c.part(FourCC::OSG1).unwrap().data).unwrap(),
        prims: vec![],
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: false,
        zero_copy_memory: false,
        rect_list: true,
        copy_params: [0; 32],
        bvh_no_hit: false,
    };
    let expanded = lower::lower(&parts).unwrap();
    // Both xy position and uv are completed, as are the remaining float components. This
    // inspects the live graph feeding the final DXIL stores, not just an unused arithmetic op.
    assert_eq!(
        expanded.completed_outputs,
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3)
        ]
    );
    let ordinary = lower::lower(&lower::Parts {
        rect_list: false,
        ..parts
    })
    .unwrap();
    assert_ne!(ordinary.dxil, expanded.dxil);
    let inputs = super::signature(
        Container::parse(&expanded.dxil)
            .unwrap()
            .part(FourCC::ISG1)
            .unwrap()
            .data,
    )
    .unwrap();
    assert!(inputs.iter().any(|e| e.kind == 1), "vertex ID");
    assert!(inputs.iter().any(|e| e.kind == 2), "instance ID");
    assert!(inputs.iter().all(|e| e.kind != 0), "no vertex buffers");
    // Exercise the public container path used by described native pipelines as well.
    let translated = super::translate_with(
        &bytes,
        Options {
            rect_list: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(translated.stage, crate::HwStage::Vs);
    super::translate(&program::pixel_container()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut translated.dxil.clone()).unwrap();
    }
}
