//! Cache invalidation on config change. PR 1: wipe-all; PR 7 may add
//! field-aware partial invalidation if profiling shows it's needed.
//!
//! `Invalidator` is a pure revision detector — it returns whether the
//! caller should perform a wipe this frame. The actual wipe is performed
//! by the caller (via `World::wipe()`), which gives the caller a single
//! point of control over the cache AND the in-flight set (the former
//! `maybe_wipe(&mut ChunkCache)` API forgot the in-flight set, causing
//! stale geometry to land 1-2 frames after an edit).

pub struct Invalidator {
    last_config_revision: u64,
    current: u64,
}

impl Invalidator {
    pub fn new() -> Self {
        Self { last_config_revision: 0, current: 0 }
    }

    /// Call when the active config changes (e.g., slider edit, preset
    /// load, file-watcher swap). Bumps the revision counter.
    pub fn bump(&mut self) {
        self.current = self.current.wrapping_add(1);
    }

    /// Call once per frame. Returns `true` exactly once per `bump()`:
    /// when the revision has advanced since the last call. The caller
    /// is responsible for performing the wipe (typically
    /// `world.wipe(); scene.clear();`).
    pub fn take_pending(&mut self) -> bool {
        if self.current != self.last_config_revision {
            self.last_config_revision = self.current;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bump_means_not_pending() {
        let mut inv = Invalidator::new();
        assert!(!inv.take_pending());
    }

    #[test]
    fn bump_then_take_returns_true_once() {
        let mut inv = Invalidator::new();
        inv.bump();
        assert!(inv.take_pending());
        // Second take is a no-op for the same bump.
        assert!(!inv.take_pending());
    }

    #[test]
    fn multiple_bumps_collapse_to_one_take() {
        let mut inv = Invalidator::new();
        inv.bump();
        inv.bump();
        inv.bump();
        assert!(inv.take_pending());
        assert!(!inv.take_pending());
    }
}
