# Worldgen editable DAG — follow-up artifact

**Date:** 2026-05-19
**Status:** Blocked. Resume after the worldgen overhaul lands.
**Parent:** `docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md`
**Promotion criteria:** Worldgen overhaul session has merged; viz redesign PRs 1–7 are stable on `main`.

## Why this is deferred

The viz redesign (Approach 1) ships a **read-only** DAG view. The user has explicitly confirmed they want an **editable** DAG eventually — dragging nodes, rewiring edges, watching terrain reflow live — but only after the in-flight worldgen overhaul completes. Refactoring the worldgen pipeline into a runtime graph while the module is being heavily edited by another session would produce constant merge churn and a fragile foundation.

This artifact captures the shape of that work so that, when the overhaul lands, a future session can pick it up without re-brainstorming.

## What changes

### `worldgen/`

The density composition (`density = bias + relief - cave_sdf`, the spline-correction chain, the climate-driven blends) currently hardcodes its topology in Rust functions. The refactor replaces this with a runtime `DensityGraph`:

```rust
pub struct DensityGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,   // (src_node, src_socket, dst_node, dst_socket)
    pub root: NodeId,       // the node that produces the final density value
}

pub trait Node: Send + Sync {
    fn evaluate(&self, ctx: &EvalCtx) -> f32;
    fn kind(&self) -> NodeKind;
}

pub enum NodeKind {
    Noise(NoiseParams),
    Spline(CubicSpline),
    NestedSpline(NestedSpline),
    Blend { op: BlendOp },        // add, mul, lerp, min, max
    Clamp { lo: f32, hi: f32 },
    SampleFromRegion(RegionField),
    Constant(f32),
    Sink,                         // designates root
}
```

`WorldgenConfig` becomes a graph topology serialised in RON (current typed fields become a degenerate graph that round-trips into the new format). The `Generator::fill_chunk` path replaces direct function calls with a single `graph.evaluate(&ctx)` call per voxel.

### `worldgen_viz/`

The PR-5 read-only DAG panel becomes interactive:

- Drag a node → reposition.
- Drag from output socket → input socket → new edge.
- Right-click → delete node / insert blend / change op.
- Edits to splines stay in their existing widgets — the spline node now opens the spline editor when clicked.
- Live trace continues to work; probing a column lights up every node's current value.

The streaming cache invalidates on every graph edit, same conservative path as v1.

## Estimated work

| Area | Estimate |
|---|---|
| `worldgen` graph evaluator + node trait | 1 week |
| `WorldgenConfig` graph topology + RON migration | 3 days |
| Determinism re-baseline (all golden hashes change) | 2 days |
| Performance regression hunt (likely ~2× slower per voxel without inlining) | 1 week |
| `worldgen_viz` editable canvas | 1 week |
| Tests for graph evaluator + viz editor | 4 days |

Total: ~3 weeks worldgen + ~1 week viz, sequentially.

## Test impact

- New graph-evaluator test suite (round-trip RON, edge validation, cycle detection, determinism).
- Re-baseline of all `worldgen` golden / fingerprint tests (the underlying values won't change if the topology matches today's, but the byte representation will).
- Performance benchmark for chunk fill in release. Target: stay within 1.5× of pre-refactor.

## Open questions to revisit at promotion time

- Whether the graph evaluator should be JIT-compiled (Cranelift / `rune`) for performance.
- Whether undo/redo of graph edits is a v1 requirement or a follow-up.
- Whether the editor should support copy/paste of subgraphs ("preset fragments").
- Whether RON is still the right serialisation format, or whether graph topology benefits from a more graph-native format (e.g., GraphML).

This artifact is a placeholder, not a plan. Promote it to a full design spec when the overhaul lands.
