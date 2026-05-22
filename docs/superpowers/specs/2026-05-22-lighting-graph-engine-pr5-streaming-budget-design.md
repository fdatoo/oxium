# Lighting Graph Engine PR5 — Streaming-Mode Budget Bump

**Date:** 2026-05-22
**Status:** Approved, ready for implementation planning
**Parent spec:** [`2026-05-21-lighting-graph-engine-design.md`](2026-05-21-lighting-graph-engine-design.md) — PR B5

## Problem

After PR4 landed, the graph engine starts each session by flooding sky-source and emissive ops for every loaded chunk. With ~12k chunks at radius 16, this produces hundreds of thousands of queued ops. At the current hardcoded 50k-op/tick budget the queue drains over ~100 frames (~1.7 s at 60 fps), during which the lighting tick consumes a measurable share of frame time and the world looks dark until convergence. The FPS during this window measured ~45 rather than 60+.

## Goal

Reduce the convergence window by temporarily boosting the per-tick op budget when the engine is known to be in a "catch-up" phase (initial stream-in or any large burst). Once the backlog clears, revert to the normal budget so interactive edits stay cheap.

## Streaming-Mode Detection

Two signals OR'd together gate the high-budget mode:

| Signal | Threshold | Rationale |
|---|---|---|
| `perf.chunks_pending` | > 100 | Chunks still being generated or loaded from disk — the world is actively streaming in |
| `engine.pending_ops_count()` | > 10 000 | Engine has a large backlog even if gen/load has finished (the actual bottleneck post-PR4) |

Either signal alone is sufficient. Together they cover: (a) initial gen where chunks arrive over many frames, and (b) post-gen convergence where all slots are `Stored` but hundreds of thousands of sky-source ops are still queued.

## Budget Values

| Mode | Budget (ops/tick) | Rationale |
|---|---|---|
| Normal | 50 000 | Current value; keeps lighting ticks under ~0.5 ms during normal play |
| Streaming | 500 000 | 10× boost; at ~5 ns/op costs ~2.5 ms but converges 12k-chunk queue in ~10 frames (~170 ms) instead of ~100 frames (~1.7 s) |

## Changes

### `src/lighting/queue.rs`

Remove `#[cfg(test)]` from `BucketQueue::len()`. It becomes a public non-test method so production code can read queue depth. The doc comment notes it is O(16) (sums 16 `VecDeque::len()` calls) and should not be called in the hot propagation path.

### `src/lighting/engine.rs`

Add `LightEngine::pending_ops_count() -> usize`:

```rust
pub fn pending_ops_count(&self) -> usize {
    let ch = |c: &ChannelEngine| {
        c.block_nodes_to_check.len() + c.increase.len() + c.decrease.len()
    };
    self.pending_block_changes.len()
        + ch(&self.sky)
        + self.block_rgb.iter().map(ch).sum::<usize>()
}
```

Add a unit test `pending_ops_count_reports_queue_depth`: enqueue a sky increase op on a synthetic engine, assert count > 0; tick to convergence in a minimal synthetic world, assert count == 0.

### `src/app.rs`

**`PerfSnapshot`:**
- Add `light_ops_pending: usize`. Field replaces the semantic role of `light_queue` for graph-engine builds.

**`AppState::step`:**
- Compute `light_pending = self.world.light_engine.pending_ops_count()` before the tick.
- Derive `light_budget`: 500 000 if streaming, 50 000 otherwise.
- Pass `light_budget` to `self.world.light_engine_tick(...)`.
- Assign `self.perf.light_ops_pending = light_pending` (pre-tick snapshot — the relevant figure is "how much was queued entering this frame").

The existing `perf.light_queue` field is retained for the `legacy-lighting` feature path.

### `src/render/hud.rs`

In the perf string, change the `LQ` label to `LO` and source it from `perf.light_ops_pending` instead of `perf.light_queue` for default (graph-engine) builds. Under `#[cfg(feature = "legacy-lighting")]` keep `LQ` with the old meaning.

Updated format string (non-legacy):
```
LO: {} LD: {} PE: {} CH: {} DC: {} WMS: {:.1}
```

### `src/profiler.rs`

Rename `FrameCounters::light_queue` → `light_ops`. Update the CSV header accordingly (`light_queue` → `light_ops`). Update the one call site in `AppState::step`.

## Testing

- **Unit:** `pending_ops_count_reports_queue_depth` in `engine.rs`.
- **Smoke:** `cargo run --profile dev` — at launch, HUD `LO` should show a large number (~tens of thousands) declining to zero over a few seconds, then stay at zero. FPS should remain ≥ 60 throughout rather than dropping to ~45.
- **Regression:** `cargo test` — 256 tests must continue to pass.
- **Screenshot diff:** No visual change expected (same light values, just faster convergence). Existing baselines should remain within noise threshold.

## Non-Goals

- Multithreaded propagation.
- Tuning the water-cost constant.
- Any visual pipeline changes.
