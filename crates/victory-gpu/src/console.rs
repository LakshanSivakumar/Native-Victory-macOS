//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); Grand Theft Auto VI sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    /// Compute units of the GPU.
    pub const fn compute_units(self) -> u32 {
        match self {
            Console::SeriesX => 52,
            Console::SeriesS => 20,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_x_declares_its_memory_and_compute_units() {
        assert_eq!(Console::SeriesX.title_memory(), (10 * GIB, 7 * GIB / 2));
        assert_eq!(Console::SeriesX.compute_units(), 52);
        assert_eq!(Console::SeriesX.name(), "Xbox Series X");
    }

    #[test]
    fn series_s_declares_its_single_memory_range_and_compute_units() {
        assert_eq!(Console::SeriesS.title_memory(), (0, 7_776 << 20));
        assert_eq!(Console::SeriesS.compute_units(), 20);
        assert_eq!(Console::SeriesS.name(), "Xbox Series S");
    }

    #[test]
    fn set_then_current_round_trips() {
        // The one stateful test: the console flag is process-global, so keep it in one test.
        set(Console::SeriesS);
        assert_eq!(current(), Console::SeriesS);
        set(Console::SeriesX);
        assert_eq!(current(), Console::SeriesX);
    }
}
