//! The title's acceleration-structure builder (`XboxBvhBuilder.dll`): its compute pipelines are
//! made past the frame's budget for new pipelines, or all held back. Skipped without the Metal
//! Shader Converter.

use super::addr_tests::title;

/// The acceleration-structure builder's compute pipelines (bytecode in `XboxBvhBuilder.dll`'s
/// image) are made even once the frame has spent its budget for new pipelines, where another
/// pipeline whose shader is not translated yet is put off (its dispatches dropped); the same for
/// dispatches of thread counts. Once one of the builder's pipelines cannot be made, all of them
/// are held back.
#[test]
fn the_bvh_builders_pipelines_are_made_past_the_frames_budget_or_held_back_together() {
    use crate::native::glue::{self, with_title};
    use crate::native::title::Pipe;
    use crate::pipeline::{builder_pipeline, note_title_dll, test_compute, test_compute_container, test_forget_builder};
    if title().is_none() || with_title(|_| ()).is_none() {
        return;
    }
    // `v_mov_b32 v1, <n>; s_endpgm`: a shader of its own for each n.
    let code = |n: u32| test_compute_container(&[0x7E02_02FF, 0x5EED_0000 | n, 0xBF81_0000], &[], 5);
    // The "builder's image": two shaders and bytes that are no container.
    let (a, b) = (code(1), code(2));
    let mut image = a.clone();
    image.extend(&b);
    image.extend(b"not a container at all, but in the builder's image");
    note_title_dll("XboxBvhBuilder.dll", image.as_ptr() as usize, image.len());
    let inside = test_compute(&image[..a.len()]);
    let threads = test_compute(&image[a.len()..a.len() + b.len()]);
    let broken = test_compute(&image[a.len() + b.len()..]);
    let elsewhere = code(3);
    let outside = test_compute(&elsewhere);
    assert!(builder_pipeline(inside) && builder_pipeline(threads) && builder_pipeline(broken));
    assert!(!builder_pipeline(outside), "bytecode outside the builder's image");

    // The frame's budget spent, on this thread only (other tests share the budget).
    glue::test_no_budget(true);
    with_title(|t| {
        assert!(matches!(glue::test_pipe(t, outside, false), Pipe::Refused), "an ordinary pipeline is put off");
        assert!(matches!(glue::test_pipe(t, inside, false), Pipe::Compute(_)), "made at once: {:?}", t.shaders);
        assert!(matches!(glue::test_pipe(t, threads, true), Pipe::Compute(_)), "by thread counts too");
        // One that cannot be made holds every builder pipeline back, made ones too.
        assert!(matches!(glue::test_pipe(t, broken, false), Pipe::Refused));
        assert!(matches!(glue::test_pipe(t, inside, false), Pipe::Refused), "held back");
        assert!(matches!(glue::test_pipe(t, threads, true), Pipe::Refused), "held back");
        glue::test_builder_unheld();
        assert!(matches!(glue::test_pipe(t, inside, false), Pipe::Compute(_)));
    });
    glue::test_no_budget(false);
    test_forget_builder();
    // `image` must outlive the pipelines' containers' last reads.
    drop(image);
}
