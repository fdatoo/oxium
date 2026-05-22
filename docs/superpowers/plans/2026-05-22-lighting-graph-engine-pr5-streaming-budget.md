# Lighting Graph Engine PR5 — Streaming-Mode Budget Bump

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the ~45 fps dip at launch by temporarily boosting the light-engine tick budget to 500k ops/tick during initial chunk stream-in, falling back to 50k once the backlog clears.

**Architecture:** Two detection signals (chunks_pending > 100 OR engine queue > 10k ops) gate a dynamic budget in `AppState::step`. A new `LightEngine::pending_ops_count()` method (built on a promoted `BucketQueue::len()`) feeds both the budget decision and a new HUD counter. The legacy `LQ` HUD label is replaced by `LO` (light ops) under the default graph-engine build.

**Tech Stack:** Rust 2024, `cargo test`, `cargo clippy`, `cargo fmt`

---

## File Map

| File | Change |
|---|---|
| `src/lighting/queue.rs` | Promote `BucketQueue::len()` out of `#[cfg(test)]` |
| `src/lighting/engine.rs` | Add `LightEngine::pending_ops_count()` + unit test |
| `src/app.rs` | Add `PerfSnapshot::light_ops_pending`; dynamic budget in `step` |
| `src/render/hud.rs` | `LQ` → `LO`, source from `perf.light_ops_pending` |
| `src/profiler.rs` | `FrameCounters::light_queue` → `light_ops`; CSV header update |

---

### Task 1: Promote `BucketQueue::len()` and add `LightEngine::pending_ops_count()`

**Goal:** Expose queue depth from `LightEngine` so the call site in `app.rs` can read it without `#[cfg(test)]` restrictions.

**Files:**
- Modify: `src/lighting/queue.rs` (line 104 — remove `#[cfg(test)]`)
- Modify: `src/lighting/engine.rs` (add method + unit test)

**Acceptance Criteria:**
- [ ] `BucketQueue::len()` compiles in non-test builds (no `#[cfg(test)]` gate)
- [ ] `LightEngine::pending_ops_count()` returns the sum of all pending work across all channels
- [ ] New unit test `pending_ops_count_reports_queue_depth` passes: non-zero after enqueue, zero after idle tick
- [ ] `cargo test` passes (256 tests + new test)
- [ ] `cargo clippy` clean, `cargo fmt` applied

**Verify:** `cargo test pending_ops_count` → 1 test passing

**Steps:**

- [ ] **Step 1: Remove `#[cfg(test)]` from `BucketQueue::len()`**

  In `src/lighting/queue.rs`, change:
  ```rust
  // BEFORE (line 102-107):
  /// Total entries across all buckets. O(16) — fine for tests, avoid in
  /// hot paths.
  #[cfg(test)]
  pub fn len(&self) -> usize {
      self.buckets.iter().map(|b| b.len()).sum()
  }
  ```
  To:
  ```rust
  // AFTER:
  /// Total entries across all buckets. O(16) — fine for debug/HUD, avoid in
  /// hot propagation paths.
  pub fn len(&self) -> usize {
      self.buckets.iter().map(|b| b.len()).sum()
  }
  ```

- [ ] **Step 2: Add `LightEngine::pending_ops_count()`**

  In `src/lighting/engine.rs`, add this method to the `impl LightEngine` block (after `is_idle`):
  ```rust
  /// Total pending work across all channels: pending block-change records,
  /// plus every entry in every increase/decrease queue for sky and RGB.
  /// O(constant) — sums 13 lengths (1 HashMap + 4 channels × 3 fields).
  /// Intended for budget decisions and HUD display; avoid in the hot tick path.
  pub fn pending_ops_count(&self) -> usize {
      let ch = |c: &ChannelEngine| {
          c.block_nodes_to_check.len() + c.increase.len() + c.decrease.len()
      };
      self.pending_block_changes.len()
          + ch(&self.sky)
          + self.block_rgb.iter().map(ch).sum::<usize>()
  }
  ```

- [ ] **Step 3: Write the unit test**

  In `src/lighting/engine.rs`, inside the `#[cfg(test)]` module (after the existing tests), add:
  ```rust
  #[test]
  fn pending_ops_count_reports_queue_depth() {
      use crate::voxel::coords::BlockPos;
      use glam::IVec3;
      use crate::lighting::queue::QueueEntry;

      let mut e = LightEngine::default();
      assert_eq!(e.pending_ops_count(), 0, "fresh engine has no pending ops");

      // Push one entry into the sky increase queue directly.
      e.sky.increase.push(QueueEntry {
          pos: BlockPos(IVec3::ZERO),
          from_level: 15,
          propagation_mask: 0,
      });
      assert!(e.pending_ops_count() > 0, "count should be non-zero after enqueue");

      // Drain it.
      let _ = e.sky.increase.pop_highest();
      assert_eq!(e.pending_ops_count(), 0, "count should be zero after drain");
  }
  ```

- [ ] **Step 4: Run tests and lint**

  ```bash
  cargo test pending_ops_count -- --nocapture
  cargo clippy
  cargo fmt
  ```
  Expected: test passes, no warnings.

- [ ] **Step 5: Commit**

  ```bash
  git add src/lighting/queue.rs src/lighting/engine.rs
  git commit -m "feat(lighting): pending_ops_count — promote BucketQueue::len + engine queue depth (PR5, part 1/3)"
  ```

---

### Task 2: Dynamic budget in `app.rs` + `PerfSnapshot::light_ops_pending`

**Goal:** Replace the hardcoded 50k budget with a dynamic choice (500k during stream-in, 50k at rest) and track queue depth in `PerfSnapshot` for the HUD.

**Files:**
- Modify: `src/app.rs`

**Acceptance Criteria:**
- [ ] `PerfSnapshot` has a `light_ops_pending: usize` field
- [ ] Budget is 500 000 when `chunks_pending > 100 || light_pending > 10_000`; 50 000 otherwise
- [ ] `perf.light_ops_pending` is set each frame to the pre-tick queue depth
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean, `cargo fmt` applied

**Verify:** `cargo test` → all tests pass; `cargo clippy` → no warnings

**Steps:**

- [ ] **Step 1: Add `light_ops_pending` to `PerfSnapshot`**

  In `src/app.rs`, find the `PerfSnapshot` struct. After the `light_queue` field (line ~115), add:
  ```rust
  /// Graph-engine op queue depth entering this frame (pre-tick snapshot).
  /// Non-zero during initial stream-in; approaches zero as light converges.
  /// Always 0 under `legacy-lighting` (use `light_queue` there instead).
  pub light_ops_pending: usize,
  ```

- [ ] **Step 2: Replace the hardcoded budget in `AppState::step`**

  Find the existing light tick call (around line 414):
  ```rust
  time(prof, "light_engine_tick", || {
      self.world.light_engine_tick(50_000);
  });
  ```

  Replace with:
  ```rust
  let light_pending = self.world.light_engine.pending_ops_count();
  let light_budget = if self.perf.chunks_pending > 100 || light_pending > 10_000 {
      500_000
  } else {
      50_000
  };
  self.perf.light_ops_pending = light_pending;
  time(prof, "light_engine_tick", || {
      self.world.light_engine_tick(light_budget);
  });
  ```

  Note: `self.perf.chunks_pending` is computed **after** the tick in `step()` (lines 447-455), so at the budget decision point it holds the *previous* frame's value (0 on frame 0). That one-frame lag is fine for a budget heuristic — `light_pending > 10_000` fires on frame 1 regardless, because the engine's queue is already large when the first tick runs.

- [ ] **Step 3: Run tests and lint**

  ```bash
  cargo test
  cargo clippy
  cargo fmt
  ```
  Expected: all tests pass, no warnings.

- [ ] **Step 4: Commit**

  ```bash
  git add src/app.rs
  git commit -m "perf(lighting): dynamic tick budget — 500k streaming / 50k normal (PR5, part 2/3)"
  ```

---

### Task 3: HUD `LO` rename + profiler `light_ops` rename

**Goal:** Replace the always-zero legacy `LQ` counter in the HUD (and profiler CSV) with `LO` (light ops pending), sourced from `perf.light_ops_pending`.

**Files:**
- Modify: `src/render/hud.rs`
- Modify: `src/profiler.rs`
- Modify: `src/app.rs` (update the profiler call site)

**Acceptance Criteria:**
- [ ] HUD perf line shows `LO:` (not `LQ:`) on non-legacy builds; value is `perf.light_ops_pending`
- [ ] `FrameCounters::light_queue` field renamed to `light_ops` throughout `profiler.rs`
- [ ] CSV header contains `light_ops` (not `light_queue`)
- [ ] The `AppState::step` profiler flush uses `light_ops: self.perf.light_ops_pending as u32`
- [ ] `cargo test` passes, `cargo clippy` clean

**Verify:** `cargo test` → all tests pass; `cargo clippy` → no warnings; `grep -r "light_queue" src/` → only appears inside `#[cfg(feature = "legacy-lighting")]` blocks

**Steps:**

- [ ] **Step 1: Update `FrameCounters` in `profiler.rs`**

  In `src/profiler.rs`, rename the field and update the CSV header and row format:
  ```rust
  // In FrameCounters struct — change:
  pub light_queue: u32,
  // To:
  pub light_ops: u32,
  ```

  Change the CSV header string (line ~132) from:
  ```
  "frame_id,t_session_ms,fps,work_ms,draw_calls,light_queue,chunks_rendered,chunks_loaded,chunks_pending,edits"
  ```
  To:
  ```
  "frame_id,t_session_ms,fps,work_ms,draw_calls,light_ops,chunks_rendered,chunks_loaded,chunks_pending,edits"
  ```

  Change the row format reference (line ~151) from:
  ```rust
  counters.light_queue,
  ```
  To:
  ```rust
  counters.light_ops,
  ```

- [ ] **Step 2: Update the profiler flush in `app.rs`**

  Find the `p.finish_frame(...)` call (around line 541). Change:
  ```rust
  light_queue: self.perf.light_queue as u32,
  ```
  To:
  ```rust
  light_ops: self.perf.light_ops_pending as u32,
  ```

  The existing `#[cfg(feature = "legacy-lighting")]` block that sets `self.perf.light_queue` remains unchanged — it's dead in non-legacy builds and that's fine (the field still exists on `PerfSnapshot` as a no-op).

- [ ] **Step 3: Update `hud.rs`**

  In `src/render/hud.rs`, change the comment and format string (lines ~246-258):
  ```rust
  // BEFORE:
  // Perf line: live counters for the debug HUD. "LQ" = light
  // queue depth (chunks waiting for the relight pump); steady
  // non-zero means cascade isn't terminating. "CH" = chunk meshes
  // currently held by the renderer.
  let perf_str = format!(
      "LQ: {} LD: {} PE: {} CH: {} DC: {} WMS: {:.1}",
      perf.light_queue,
      perf.chunks_loaded,
      perf.chunks_pending,
      perf.chunks_rendered,
      perf.draw_calls,
      perf.work_ms,
  );
  ```
  ```rust
  // AFTER:
  // Perf line: live counters for the debug HUD. "LO" = light-engine
  // op queue depth entering this frame; large during initial stream-in,
  // approaches zero as lighting converges. "CH" = chunk mesh count.
  let perf_str = format!(
      "LO: {} LD: {} PE: {} CH: {} DC: {} WMS: {:.1}",
      perf.light_ops_pending,
      perf.chunks_loaded,
      perf.chunks_pending,
      perf.chunks_rendered,
      perf.draw_calls,
      perf.work_ms,
  );
  ```

- [ ] **Step 4: Run final checks**

  ```bash
  cargo test
  cargo clippy
  cargo fmt
  grep -r "light_queue" src/
  ```
  Expected: all tests pass; no warnings; `grep` output shows `light_queue` only inside `#[cfg(feature = "legacy-lighting")]` guards in `app.rs`.

- [ ] **Step 5: Commit**

  ```bash
  git add src/render/hud.rs src/profiler.rs src/app.rs
  git commit -m "feat(hud): replace LQ with LO (light-ops pending) for graph-engine builds (PR5, part 3/3)"
  ```
