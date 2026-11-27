//! What the shader prewarm pool (`pipeline::prewarm`) works on first: the containers earlier runs
//! bound (the list of first binds, `shader_cache::first_binds_file`), then the rest in the order
//! the title registers them. A cold first world frame binds about 1,300 shaders; registered in
//! order, they wait behind most of the title's ~21,600.

use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Entries the list keeps: far more than a world frame binds.
const LIMIT: usize = 16_384;
/// New first binds between two writes of the list: a run the watchdog ends loses at most these.
const WRITE_EVERY: usize = 64;

/// Workers in the pool: all cores but two (the title's render thread and one more).
pub(crate) fn pool_size(cores: usize) -> usize {
    cores.saturating_sub(2).max(1)
}

/// How long a frame may run, from its first new host pipeline, before pipelines whose shaders
/// are not translated yet are put off ([`FrameBudget`]): well under the title's 30 s watchdog.
pub(crate) const FIRST_BIND_BUDGET: Duration = Duration::from_secs(10);

/// When the current frame made its first new host pipeline. Wall time, not time spent making
/// pipelines: the frame's thread also translates inline elsewhere (rectangle lists, extra
/// parameters, thread counts), and the watchdog counts all of it.
pub(crate) struct FrameBudget {
    frame: u64,
    started: Option<Instant>,
    /// [`FrameBudget::spend`] was called for this frame: none of its budget is left, whatever
    /// time has passed, and a compile crashed Metal's compiler to take it.
    crashed: bool,
}

impl FrameBudget {
    pub(crate) const fn new() -> Self {
        FrameBudget { frame: 0, started: None, crashed: false }
    }

    /// What frame `frame`, making a new pipeline at `now`, has left of `budget` since its first
    /// (zero: spent).
    pub(crate) fn left(&mut self, frame: u64, now: Instant, budget: Duration) -> Duration {
        if frame != self.frame || self.started.is_none() {
            *self = FrameBudget { frame, started: Some(now), crashed: false };
        }
        if self.crashed {
            return Duration::ZERO;
        }
        self.started.map_or(budget, |t| budget.saturating_sub(now.saturating_duration_since(t)))
    }

    /// Spends what frame `frame` has left of its budget: [`Self::left`] is zero for it from `now`
    /// on, and the next frame starts with the full budget again, as one that never spends. A
    /// compile that crashed Metal's compiler spends its frame: the frame's other new pipelines
    /// are put off rather than each paying a crash of its own (issue #181).
    pub(crate) fn spend(&mut self, frame: u64, now: Instant) {
        if frame != self.frame || self.started.is_none() {
            *self = FrameBudget { frame, started: Some(now), crashed: false };
        }
        self.crashed = true;
    }

    /// Whether a compile that crashed Metal's compiler spent this frame, rather than the frame
    /// running out of time. The compiler service may be down for the rest of such a frame, so it
    /// gets no new pipeline at all; a frame merely past its time still gets the short wait a
    /// pipeline the archive holds takes. False for the next frame.
    pub(crate) fn spent_by_crash(&self) -> bool {
        self.crashed
    }
}

/// Two queues: the shaders earlier runs bound first, then the others.
pub(crate) struct Queue<T> {
    lists: Mutex<(VecDeque<T>, VecDeque<T>)>,
    ready: Condvar,
}

impl<T> Queue<T> {
    pub(crate) const fn new() -> Self {
        Queue { lists: Mutex::new((VecDeque::new(), VecDeque::new())), ready: Condvar::new() }
    }

    pub(crate) fn push(&self, item: T, bound_before: bool) {
        let mut l = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        if bound_before { l.0.push_back(item) } else { l.1.push_back(item) }
        self.ready.notify_one();
    }

    /// The next shader, waiting for one.
    pub(crate) fn pop(&self) -> T {
        let mut l = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(item) = l.0.pop_front().or_else(|| l.1.pop_front()) {
                return item;
            }
            l = self.ready.wait(l).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// The list of first binds: what the last runs bound, then what this one binds that they did not.
#[derive(Default)]
pub(crate) struct FirstBinds {
    known: HashSet<String>,
    order: Vec<String>,
    /// Entries in the file as last written (or read).
    written: usize,
}

impl FirstBinds {
    /// The list in `text` (one short hash per line; anything else is skipped).
    pub(crate) fn parse(text: &str) -> Self {
        let mut list = FirstBinds::default();
        for line in text.lines().map(str::trim) {
            if line.len() == 16 && line.bytes().all(|b| b.is_ascii_hexdigit()) {
                list.add(line);
            }
        }
        list.written = list.order.len();
        list
    }

    pub(crate) fn len(&self) -> usize {
        self.order.len()
    }

    pub(crate) fn contains(&self, hash: &str) -> bool {
        self.known.contains(hash)
    }

    /// Adds `hash`; whether it was new.
    pub(crate) fn add(&mut self, hash: &str) -> bool {
        if self.order.len() >= LIMIT || !self.known.insert(hash.to_owned()) {
            return false;
        }
        self.order.push(hash.to_owned());
        true
    }

    /// The text to write, if enough is new since the last write.
    pub(crate) fn due(&mut self) -> Option<String> {
        if self.order.len() < self.written + WRITE_EVERY {
            return None;
        }
        self.written = self.order.len();
        Some(self.order.iter().map(|h| format!("{h}\n")).collect())
    }
}

/// Writes `text` to `path` through a temporary name. Failures are not fatal: the next run
/// prewarms in the title's order.
pub(crate) fn write(path: &Path, text: &str) {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path)).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Runs the calling thread at the lowest priority of the default QoS class: below the title's
/// threads, so its frame work goes first when cores are short. Not the utility class, which
/// keeps a thread on the efficiency cores: there a pool of 16 translated each shader about twice
/// as slowly as six at the default (220 against 100 ms in a run), with the title's thread waiting
/// on the shaders it needed.
#[cfg(target_os = "macos")]
pub(crate) fn lower_priority() {
    // `QOS_CLASS_DEFAULT` in <sys/qos.h>; -15 is the lowest relative priority it allows.
    const QOS_CLASS_DEFAULT: u32 = 0x15;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: libSystem's documented call, for the calling thread only.
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, -15) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_bound_before_come_first() {
        let q = Queue::new();
        q.push("a", false);
        q.push("b", true);
        q.push("c", false);
        q.push("d", true);
        assert_eq!([q.pop(), q.pop(), q.pop(), q.pop()], ["b", "d", "a", "c"]);
    }

    #[test]
    fn each_frame_has_its_own_budget() {
        let mut b = FrameBudget::new();
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let budget = Duration::from_millis(100);
        assert_eq!(b.left(5, at(0), budget), budget);
        assert_eq!(b.left(5, at(99), budget), Duration::from_millis(1));
        assert!(b.left(5, at(100), budget).is_zero());
        // The next frame starts from its own first pipeline.
        assert_eq!(b.left(6, at(150), budget), budget);
        assert!(!b.left(6, at(249), budget).is_zero());
        assert!(b.left(6, at(250), budget).is_zero());
    }

    #[test]
    fn a_spent_frame_starts_the_next_one_whole() {
        // A compile that crashed Metal's compiler spends the frame's budget (issue #181: nine
        // crashes held one frame for 7.3 s): the rest of that frame's new pipelines are put off,
        // and the next frame tries again with the full budget, as one that never spent. A frame
        // that is merely past its time is told apart: it still gets the short wait a pipeline the
        // archive holds takes.
        let mut b = FrameBudget::new();
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let budget = Duration::from_millis(100);
        // A frame that has made no pipeline yet can spend too.
        b.spend(4, at(0));
        assert!(b.left(4, at(0), budget).is_zero());
        assert!(b.spent_by_crash());
        assert_eq!(b.left(5, at(10), budget), budget);
        assert!(!b.spent_by_crash(), "the next frame is not spent by the crash");
        assert_eq!(b.left(5, at(99), budget), Duration::from_millis(11));
        b.spend(5, at(99));
        assert!(b.left(5, at(99), budget).is_zero());
        assert!(b.left(5, at(150), budget).is_zero(), "spent outlasts the frame's remaining time");
        assert!(b.spent_by_crash());
        assert_eq!(b.left(6, at(200), budget), budget);
        assert!(!b.spent_by_crash());
        // A frame that runs out of time is spent too, but not by a crash.
        assert_eq!(b.left(6, at(249), budget), Duration::from_millis(51));
        assert!(b.left(6, at(300), budget).is_zero());
        assert!(!b.spent_by_crash(), "a frame out of time did not crash");
    }

    #[test]
    fn the_pool_leaves_two_cores() {
        assert_eq!(pool_size(18), 16);
        assert_eq!(pool_size(8), 6);
        assert_eq!(pool_size(2), 1);
        assert_eq!(pool_size(1), 1);
    }

    #[test]
    fn the_list_keeps_earlier_runs_and_adds_this_ones() {
        let mut list = FirstBinds::parse("00000000000000aa\nnot a hash\n00000000000000bb\n");
        assert!(list.contains("00000000000000aa") && list.contains("00000000000000bb"));
        assert!(!list.add("00000000000000aa"));
        assert_eq!(list.due(), None);
        for n in 0..WRITE_EVERY {
            assert!(list.add(&format!("{n:016x}")));
        }
        let text = list.due().expect("64 new binds");
        assert!(text.starts_with("00000000000000aa\n00000000000000bb\n0000000000000000\n"));
        assert_eq!(list.due(), None);
        // A second run reads what the first wrote.
        let again = FirstBinds::parse(&text);
        assert_eq!(again.order.len(), 2 + WRITE_EVERY);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("first-binds.txt");
        write(&path, &text);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
