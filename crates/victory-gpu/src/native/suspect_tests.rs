//! The frame~800 GPU-stall suspects (issue #404): the six compute containers
//! `VICTORY_TRACE_GPU_FAILS` blamed during the Prestream stall, dispatched on
//! this stack with a fence timeout, so a hang names its shader. Plus two
//! controls of our own (a barrier-`9` butterfly and its wave-op companion,
//! compiled from `dxil-harness` HLSL) to separate the constructs from the
//! game's shader bodies. Ordered least to most suspected: a hang wedges the
//! queue, and everything after it would be noise.

use super::*;
use crate::native::shaders::threads_of;

/// (name under `victory-metal/tests/suspects/`, dispatch x/y/z). Game dims
/// from the attribution log where known, else one group.
const CASES: &[(&str, u32, u32, u32)] = &[
    ("control_barrier9", 5, 5, 33),
    ("control_combined", 5, 5, 33),
    ("compute_67a8e058461086ef", 1, 1, 1),
    ("compute_0c8e4446f3c150f1", 1, 1, 1),
    ("compute_8f9ab72f8c418630", 1, 1, 1),
    ("compute_1e84848f38c78615", 1, 1, 1),
    ("compute_9355b4a50f2628a4", 5, 5, 33),
    ("compute_d19718795242d0fe", 5, 5, 33),
];

fn case(name: &str) -> Vec<u8> {
    let dir = format!("{}/../victory-metal/tests/suspects", env!("CARGO_MANIFEST_DIR"));
    let p = format!("{dir}/{name}.host.dxbc");
    match std::fs::read(&p) {
        Ok(b) => b,
        Err(_) => std::fs::read(format!("{dir}/{name}.dxbc")).unwrap_or_else(|e| panic!("{p}: {e}")),
    }
}

/// One dispatch, one fence, 60 s. Reports rather than asserts: the caller
/// prints and decides.
fn dispatch_and_wait(h: &Host, container: &[u8], gx: u32, gy: u32, gz: u32) -> bool {
    let threads = threads_of(container).unwrap_or([64, 1, 1]);
    let pso = h
        .compute_pipeline(container, threads)
        .unwrap_or_else(|e| panic!("pipeline: {e}"));
    let pool = Pool::new(h, 4);
    let mut l = NativeList::default();
    l.fill(pool.va(0), 1 << 20, 0);
    l.set_compute_pipeline(&pso);
    l.dispatch(gx, gy, gz).unwrap();
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    h.execute(&q, &[&l], &[], &[(&f, 1)]).unwrap();
    h.backend.wait(&f, 1, 60_000)
}

#[test]
fn stall_suspects_finish_or_name_the_hang() {
    let Some(h) = host() else { return };
    let mut hung = Vec::new();
    for (name, gx, gy, gz) in CASES {
        let c = case(name);
        let threads = threads_of(&c).unwrap_or([64, 1, 1]);
        let ok = dispatch_and_wait(&h, &c, *gx, *gy, *gz);
        println!(
            "{name}: threads={threads:?} dims=({gx},{gy},{gz}) {}",
            if ok { "finished" } else { "HUNG (60s)" }
        );
        if !ok {
            hung.push(*name);
        }
    }
    assert!(hung.is_empty(), "GPU hangs: {hung:?}");
}

/// The game's shape, not one dispatch: 64 lists back to back, each dispatching
/// the suspect like a frame does, all awaited on one fence. The stall in #404
/// came from inter-list bookkeeping, so isolation alone may not show it.
#[test]
fn the_prime_suspect_survives_game_shaped_concurrency() {
    let Some(h) = host() else { return };
    let c = case("compute_d19718795242d0fe");
    let threads = threads_of(&c).unwrap_or([64, 1, 1]);
    let pso = h
        .compute_pipeline(&c, threads)
        .unwrap_or_else(|e| panic!("pipeline: {e}"));
    let pool = Pool::new(&h, 8);
    let mut lists = Vec::new();
    for _ in 0..64 {
        let mut l = NativeList::default();
        l.fill(pool.va(0), 1 << 20, 0);
        l.set_compute_pipeline(&pso);
        l.dispatch(5, 5, 33).unwrap();
        lists.push(l);
    }
    let refs: Vec<&NativeList<Metal>> = lists.iter().collect();
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    h.execute(&q, &refs, &[], &[(&f, 1)]).unwrap();
    let ok = h.backend.wait(&f, 1, 120_000);
    println!("64x d19718795242d0fe: {}", if ok { "finished" } else { "HUNG (120s)" });
    assert!(ok, "concurrent dispatch of the prime suspect hung the GPU");
}

/// The #404 shape: a root command buffer that never completes (a spin kernel on
/// purpose), then a trivial follower on the same queue. With residency sets the
/// follower waits on the queue's order event — a wait the trace does not count —
/// so it should hang with the `waits=0` signature the attribution saw. If macOS'
/// GPU watchdog instead kills the root, `Owed` pays the order event and the
/// follower finishes: then a persistent stall needs a root that waits (never
/// runs), not one that spins.
#[test]
fn a_never_finishing_dispatch_cascades_to_waits_zero_followers() {
    let Some(h) = host() else { return };
    let spin = case("control_spin");
    let pso = h
        .compute_pipeline(&spin, [64, 1, 1])
        .unwrap_or_else(|e| panic!("pipeline: {e}"));
    let pool = Pool::new(&h, 4);
    pool.write(0, &[0u32; 64]);

    let mut root = NativeList::default();
    root.fill(pool.va(0), 256, 0);
    root.set_compute_pipeline(&pso);
    root.set_compute_root_view(0, pool.va(0));
    root.set_compute_root_view(1, pool.va(1024));
    root.dispatch(1, 1, 1).unwrap();
    let q = h.backend.create_queue().unwrap();
    let f = h.backend.create_fence(0).unwrap();
    h.execute(&q, &[&root], &[], &[(&f, 1)]).unwrap();
    let root_ok = h.backend.wait(&f, 1, 8_000);
    println!("root (spin): {}", if root_ok { "finished" } else { "hung as intended" });
    if root_ok {
        let spins: u32 = pool.read(1024, 1)[0];
        let seen: u32 = pool.read(1024 + 64 * 4, 1)[0];
        println!("spin diagnostics: spins={spins} saw g_in[0]={seen}");
    }

    let mut follow = NativeList::default();
    follow.fill(pool.va(0), 64, 1);
    let f2 = h.backend.create_fence(0).unwrap();
    h.execute(&q, &[&follow], &[], &[(&f2, 1)]).unwrap();
    let follow_ok = h.backend.wait(&f2, 1, 8_000);
    println!("follower: {}", if follow_ok { "finished (watchdog killed the root; Owed paid the order event)" } else { "HUNG — waits=0 cascade confirmed" });
    println!("order_state: {:?}", q.order_state());
    assert!(!root_ok, "spin kernel finished — the root is not a hang");
    assert!(!follow_ok, "follower finished: no cascade");
}
