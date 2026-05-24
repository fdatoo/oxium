//! Cache invalidation on config change. PR 1: wipe-all; PR 7 may add
//! field-aware partial invalidation if profiling shows it's needed.
//!
//! `Invalidator` is a pure revision detector — it returns whether the
//! caller should perform a wipe this frame. The actual wipe is performed
//! by the caller (via `World::wipe()`), which gives the caller a single
//! point of control over the cache AND the in-flight set.
//!
//! Edits that come in bursts (slider drags, spline-knot drags) are
//! routed through `queue()` instead of `bump()`. Each `queue()` call
//! restarts a debounce timer; `tick()` promotes a queued change to an
//! actual revision bump only after `DEBOUNCE` has elapsed without a
//! new edit. The net effect: dragging a slider through 200 values
//! triggers one wipe-and-refill at release, not 200.

use std::time::{Duration, Instant};

/// How long the user must pause before a queued change actually
/// bumps the revision counter. 200 ms matches the old viz binary's
/// `REGEN_DEBOUNCE`; long enough to avoid mid-drag wipes, short
/// enough that release-to-refresh feels immediate.
pub const DEBOUNCE: Duration = Duration::from_millis(200);

pub struct Invalidator {
    last_config_revision: u64,
    current: u64,
    /// `Some(t)` while a queued change is waiting out the debounce
    /// window. Reset to `Some(now)` on every `queue()` call; cleared
    /// when `tick()` promotes it to a `bump()`.
    pending_since: Option<Instant>,
}

impl Invalidator {
    pub fn new() -> Self {
        Self {
            last_config_revision: 0,
            current: 0,
            pending_since: None,
        }
    }

    /// Immediately bump the revision counter. Use for changes that
    /// should refresh right away — paint-mode toggles, the R key,
    /// preset reload.
    pub fn bump(&mut self) {
        self.current = self.current.wrapping_add(1);
        self.pending_since = None;
    }

    /// Queue a change for a deferred bump. Each call restarts the
    /// debounce window, so a continuous drag triggers exactly one
    /// bump once the user pauses for `DEBOUNCE`.
    pub fn queue(&mut self) {
        self.pending_since = Some(Instant::now());
    }

    /// Call once per frame. If a queued change's debounce window has
    /// elapsed, promote it to a `bump()`.
    pub fn tick(&mut self) {
        if let Some(t) = self.pending_since
            && t.elapsed() >= DEBOUNCE
        {
            self.bump();
        }
    }

    /// True if a queued change is still waiting for its debounce
    /// window to elapse. Used by the status bar to show a "pending
    /// (Nms)" indicator.
    pub fn pending_remaining(&self) -> Option<Duration> {
        let t = self.pending_since?;
        let elapsed = t.elapsed();
        if elapsed >= DEBOUNCE {
            None
        } else {
            Some(DEBOUNCE - elapsed)
        }
    }

    /// Current revision counter. Bumped on `bump()`; consumed by
    /// `take_pending`. Used as a cache key by overlays that want to
    /// re-render exactly once per config change.
    pub fn revision(&self) -> u64 {
        self.current
    }

    /// Call once per frame after `tick()`. Returns `true` exactly
    /// once per `bump()`: when the revision has advanced since the
    /// last call. The caller is responsible for performing the wipe
    /// (typically `world.wipe(); scene refill`).
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

    #[test]
    fn queue_without_tick_does_not_promote() {
        let mut inv = Invalidator::new();
        inv.queue();
        // tick() not called yet → no revision bump.
        assert!(!inv.take_pending());
    }

    #[test]
    fn queue_promotes_after_debounce_elapses() {
        let mut inv = Invalidator::new();
        inv.queue();
        // Simulate elapsed time by manually rewinding pending_since.
        inv.pending_since = Some(Instant::now() - DEBOUNCE - Duration::from_millis(10));
        inv.tick();
        assert!(inv.take_pending());
    }

    #[test]
    fn queue_restart_within_debounce_window_does_not_promote() {
        let mut inv = Invalidator::new();
        inv.queue();
        // 100ms elapsed (less than 200ms debounce)
        inv.pending_since = Some(Instant::now() - Duration::from_millis(100));
        // Restart the timer
        inv.queue();
        // Even though the original queue was 100ms ago, the restart resets it.
        inv.tick();
        assert!(!inv.take_pending());
    }
}
