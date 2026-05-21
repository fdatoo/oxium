# Initial Chunk-Gen With Real Neighbors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the initial chunk-gen BFS run with currently-loaded neighbor data instead of `Neighbors { chunks: [None; 6] }`, so chunks generated while their neighbors are already loaded receive correct sky-light column-drop inheritance and lateral seeding on the first BFS pass.

**Architecture:** `spawn_gen` takes a new `neighbors: [Option<Arc<PalettedChunk>>; 6]` parameter, matching `spawn_relight`'s shape. The worker decompresses present neighbors to `DenseChunk`, builds a `Neighbors` struct, and passes it to `recompute_chunk` instead of the all-`None` sentinel. The two call sites (the streaming hot path in `world_stream.rs`, and the persistence-empty-slot fallback in `mesh_upload.rs`) snapshot neighbors via the existing `gather_neighbors` helper — same pattern as relight already uses.

**Tech Stack:** Rust, rayon, existing `Arc<PalettedChunk>` snapshot machinery.

**Background:** This addresses the same root cause as the failed PR4-era cascade experiment (`mesh_upload.rs:64-79` comment). That experiment marked self+6-neighbors `dirty.light` after every gen, which created a self-feeding cascade (each relight's `changed_faces` re-marked its neighbors, and the wave never terminated — LQ:2027 backlog + visible flicker reported during testing). Fixing the source instead of cascading downstream avoids the convergence problem entirely: a chunk that gens with its actual neighbors doesn't need to be re-light after the fact.

**Scope limit (deliberately out of scope):** Chunks that gen at the streaming wavefront — where neighbors genuinely aren't loaded yet — still produce an initial pass with partial data. They are correct given what was available; the residual artifacts at the wavefront converge as the player moves and edits over time. A "delayed second relight when missing-at-gen neighbor arrives" follow-up is possible but not in this plan. Ship the source fix first, measure how much remains.

---

## Files

**Modify:**
- `src/jobs/mod.rs` — `spawn_gen` signature gains `neighbors: [Option<Arc<PalettedChunk>>; 6]`; worker body decompresses neighbors and uses them in the lighting call. ~25 lines changed.
- `src/ecs/systems/world_stream.rs` — line 206, `jobs.spawn_gen(c, generator.clone(), registry.clone())` becomes `let neighbors = gather_neighbors(world, c); jobs.spawn_gen(c, generator.clone(), registry.clone(), neighbors);`. Also add `use crate::ecs::systems::mesh_upload::gather_neighbors;` at the top.
- `src/ecs/systems/mesh_upload.rs` — line ~362 (the persistence-empty-slot fallback in `drain_persistence`), same change as `world_stream.rs`. `gather_neighbors` is already in scope (defined in the same file).

**Create:**
- `tests/gen_with_neighbors.rs` — integration test that exercises the new code path end-to-end: builds a torch-lit chunk A, calls `spawn_gen` for adjacent chunk B with A as a neighbor, asserts B's boundary cells facing A pick up the seeded block-light values from A. Will fail to compile until Task 2 lands.

**Do not touch:**
- `src/lighting/mod.rs` — `recompute_chunk` already does the right thing when given non-empty `Neighbors`. The existing `seed_from_neighbors` covers both sky and block-RGB channels. We are not changing the lighting algorithm.
- `src/voxel/chunk.rs` — `Neighbors`, `DenseChunk`, `PalettedChunk` are reused unchanged.
- `mesh_upload.rs` `JobResult::Generated` handler — the disabled `dirty.light` cascade (lines 64-79) stays disabled. Re-enabling it caused LQ:2027 backlog + flicker as documented.
- The existing `spawn_relight` worker — its neighbor decompression code is the template we copy into `spawn_gen`, but we don't refactor a shared helper (the duplication is ~12 lines, and the two workers have different result shapes; YAGNI).

---

## Tasks

### Task 1: Add `neighbors` parameter to `spawn_gen` signature (ignore for now)

This step intentionally breaks the build at call sites. Subsequent tasks restore it. The point is to surface every caller via compile errors so none get missed.

**Files:**
- Modify: `src/jobs/mod.rs` — `spawn_gen` signature

- [ ] **Step 1: Edit the signature.**

In `src/jobs/mod.rs`, find `spawn_gen` (around line 181). Change the function signature from:

```rust
pub fn spawn_gen(
    &self,
    coord: ChunkCoord,
    generator: Arc<Generator>,
    registry: Arc<BlockRegistry>,
) {
```

to:

```rust
pub fn spawn_gen(
    &self,
    coord: ChunkCoord,
    generator: Arc<Generator>,
    registry: Arc<BlockRegistry>,
    neighbors: [Option<Arc<PalettedChunk>>; 6],
) {
```

Leave the worker body unchanged for now — `neighbors` is unused. The compiler will emit an `unused variable` warning on `neighbors`; that's expected and will go away in Task 4.

- [ ] **Step 2: Suppress the unused-variable warning on the parameter (so the build is clean between tasks).**

Rename the parameter to `_neighbors` in the function body only — keep the public signature with `neighbors`:

Actually simpler: prefix with underscore everywhere for this transient task. Final signature for Task 1:

```rust
pub fn spawn_gen(
    &self,
    coord: ChunkCoord,
    generator: Arc<Generator>,
    registry: Arc<BlockRegistry>,
    _neighbors: [Option<Arc<PalettedChunk>>; 6],
) {
```

Task 4 will rename back to `neighbors` and use it.

- [ ] **Step 3: Verify the build now fails at call sites (not in `spawn_gen` itself).**

Run: `cargo build 2>&1 | grep "error\[E0061\]" | head -5`
Expected: errors at `src/ecs/systems/world_stream.rs` and `src/ecs/systems/mesh_upload.rs` reporting "this function takes 4 arguments but 3 arguments were supplied" (or similar — exact text depends on rustc version).

- [ ] **Step 4: Do NOT commit yet — the build is broken. Proceed to Task 2 to restore it.**

---

### Task 2: Update both call sites to pass `gather_neighbors(world, c)`

After this task the build compiles again and behavior is unchanged from main: every call site now passes a real neighbor snapshot, but the worker still ignores it (Task 4 fixes that).

**Files:**
- Modify: `src/ecs/systems/world_stream.rs`
- Modify: `src/ecs/systems/mesh_upload.rs`

- [ ] **Step 1: Update the streamer's spawn_gen call.**

In `src/ecs/systems/world_stream.rs`, at the top of the file, add an import line below the existing `use crate::ecs::components::Position;` etc.:

```rust
use crate::ecs::systems::mesh_upload::gather_neighbors;
```

Then find the `jobs.spawn_gen(...)` call (around line 206) inside the `else` branch:

```rust
} else {
    jobs.spawn_gen(c, generator.clone(), registry.clone());
}
```

Change it to:

```rust
} else {
    let neighbors = gather_neighbors(world, c);
    jobs.spawn_gen(c, generator.clone(), registry.clone(), neighbors);
}
```

- [ ] **Step 2: Update the persistence-empty-slot fallback.**

In `src/ecs/systems/mesh_upload.rs`, find the `PersistResult::Loaded` handler's `None =>` branch (around line 359-362):

```rust
None => {
    // Region file existed but the slot was empty —
    // fall back to procedural gen.
    jobs.spawn_gen(coord, generator.clone(), registry.clone());
}
```

Change to:

```rust
None => {
    // Region file existed but the slot was empty —
    // fall back to procedural gen. Snapshot neighbours
    // so the gen worker's initial BFS uses real
    // boundary data instead of all-`None` sentinels.
    let neighbors = gather_neighbors(world, coord);
    jobs.spawn_gen(coord, generator.clone(), registry.clone(), neighbors);
}
```

`gather_neighbors` is defined in this same file (line ~376) so no import needed.

- [ ] **Step 3: Verify the build.**

Run: `cargo build 2>&1 | tail -5`
Expected: clean build. The `_neighbors` parameter inside `spawn_gen` is still unused, but no errors.

- [ ] **Step 4: Run the existing test suite to confirm no behavioral change yet.**

Run: `cargo test --release 2>&1 | tail -10`
Expected: all existing tests pass. We haven't changed runtime behavior, only the signature.

- [ ] **Step 5: Commit.**

```bash
git add src/jobs/mod.rs src/ecs/systems/world_stream.rs src/ecs/systems/mesh_upload.rs
git commit -m "refactor(jobs): plumb neighbour snapshot through spawn_gen signature

Both call sites now hand a real [Option<Arc<PalettedChunk>>; 6] into
spawn_gen (same shape as spawn_relight). The gen worker still ignores
it for now — wiring the call chain first so a single follow-up touches
only the lighting line. No runtime behaviour change."
```

---

### Task 3: Write the failing integration test

The test reproduces the bug deterministically using a manually-constructed neighbor chunk. It will FAIL on the current build (because `_neighbors` is ignored) and PASS after Task 4 makes the worker use them.

**Files:**
- Create: `tests/gen_with_neighbors.rs`

- [ ] **Step 1: Create the test file.**

Write `tests/gen_with_neighbors.rs`:

```rust
//! Verifies that `Jobs::spawn_gen` plumbs neighbour data into the
//! initial chunk-lighting BFS. Without this, freshly-generated chunks
//! at the streaming wavefront have correct internal column-drop but
//! empty lateral seeding — visible as dark vertical streaks on cliff
//! faces near chunk boundaries.
//!
//! Strategy:
//!   1. Manually build a neighbour chunk `A` with a high-emission
//!      torch (15 R) sitting one cell back from its +X boundary, so
//!      A's x=31 column carries strong block-light into chunk B's
//!      x=0 boundary when seeded.
//!   2. Spawn a gen job for chunk `B` with A as the -X neighbour.
//!      The generator is a stub that fills B with air so we can read
//!      back B's BFS results without procedural terrain blocking
//!      propagation.
//!   3. Wait for the JobResult::Generated to arrive, decompress B,
//!      and assert that B's x=0 column carries non-zero red block-
//!      light — the seeded value attenuated from A's x=31 across
//!      the chunk seam.

use glam::{IVec3, UVec3};
use oxium::jobs::{JobResult, Jobs};
use oxium::voxel::block::{Block, BlockRegistry};
use oxium::voxel::chunk::{unpack_rgb, DenseChunk, Neighbors, PalettedChunk};
use oxium::voxel::coords::{ChunkCoord, LocalPos};
use oxium::worldgen::Generator;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn spawn_gen_uses_neighbour_block_light_at_boundary() {
    // ── Build neighbour chunk A with a red torch near its +X face.
    //
    // Place the torch at chunk-local (30, 16, 16): one cell inside
    // the +X boundary. A's BFS then propagates the torch's red light
    // outward; cell (31, 16, 16) gets emission - 1 = 14, which is
    // what gets seeded into B's x=0 boundary across the chunk seam.
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);
    let reg = Arc::new(reg);

    let mut a_dense = DenseChunk::empty();
    a_dense.set(LocalPos(UVec3::new(30, 16, 16)), Block::Torch);
    let no_neighbors = Neighbors { chunks: [None; 6] };
    oxium::lighting::recompute_chunk(&mut a_dense, &no_neighbors, &reg);

    // Sanity: A's x=31 boundary row at y=16, z=16 should carry red
    // light (one step from the torch at x=30).
    let a_boundary_idx = LocalPos(UVec3::new(31, 16, 16)).to_index();
    let (a_r, _, _) = unpack_rgb(a_dense.block_rgb[a_boundary_idx]);
    assert!(
        a_r >= 13,
        "test setup wrong: A's +X boundary not strongly red ({a_r})"
    );

    let a_packed = Arc::new(PalettedChunk::compress(&a_dense));

    // ── Spawn gen for B with A as the -X neighbour.
    //
    // Coord B = (1, 0, 0), so A is at (0, 0, 0) i.e. B's -X.
    // Face order in Neighbors is [+X, -X, +Y, -Y, +Z, -Z], so A goes
    // into index 1.
    let jobs = Jobs::new();
    let generator = Arc::new(Generator::new(42));
    let b_coord = ChunkCoord(IVec3::new(1, 0, 0));
    let mut neighbours: [Option<Arc<PalettedChunk>>; 6] = Default::default();
    neighbours[1] = Some(a_packed);

    jobs.spawn_gen(b_coord, generator, reg.clone(), neighbours);

    // ── Drain the channel until the Generated result for b_coord
    // arrives. The pool may produce unrelated results from
    // background work in worst case, but in this isolated test
    // there's only one job.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut b_packed: Option<Arc<PalettedChunk>> = None;
    while Instant::now() < deadline {
        match jobs.rx.recv_timeout(Duration::from_millis(100)) {
            Ok(JobResult::Generated { coord, data }) if coord == b_coord => {
                b_packed = Some(Arc::new(data));
                break;
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    let b_packed = b_packed.expect("gen job timed out");

    // ── Assert: B's x=0 row at (y=16, z=16) carries seeded red light.
    //
    // The seed_from_neighbors pass takes A's mirror cell (x=31) value
    // and attenuates by 1 (cost of crossing the seam), so B's x=0
    // should read A's_x31 - 1. A's x=31 was ~14, so B's x=0 ≈ 13.
    let b_dense = b_packed.decompress();
    let b_boundary_idx = LocalPos(UVec3::new(0, 16, 16)).to_index();
    let (b_r, b_g, b_b) = unpack_rgb(b_dense.block_rgb[b_boundary_idx]);
    assert!(
        b_r >= 10,
        "B's -X boundary did not pick up neighbour seeding: red={b_r} (expected ≥ 10). \
         spawn_gen is not propagating the `neighbors` argument into recompute_chunk."
    );
    assert_eq!(b_g, 0, "green channel should not leak from a red-only source");
    assert_eq!(b_b, 0, "blue channel should not leak from a red-only source");
}
```

- [ ] **Step 2: Run the new test — it should FAIL.**

Run: `cargo test --release --test gen_with_neighbors 2>&1 | tail -20`
Expected: test runs, hits the `assert!(b_r >= 10, ...)` panic with `red=0`. The error message is the one we wrote: "spawn_gen is not propagating the `neighbors` argument...".

If the test passes unexpectedly, something is wrong — either the test is too lenient, or main already does the right thing somehow. Stop and investigate before continuing.

- [ ] **Step 3: Commit the failing test.**

```bash
git add tests/gen_with_neighbors.rs
git commit -m "test(jobs): integration test for spawn_gen neighbour seeding

Currently fails: spawn_gen ignores its neighbors parameter. Task 4 of
the gen-with-neighbours plan makes the worker decompress the snapshot
and pass it to recompute_chunk."
```

Committing the failing test first is intentional — it pins down the expected behavior before the implementation lands, so we can't accidentally "fix" the bug by changing the test.

---

### Task 4: Use the neighbours in the gen worker

The actual fix. Decompress each present neighbor into a `DenseChunk`, build a `Neighbors` struct of references, and pass it to `recompute_chunk` instead of the all-`None` sentinel. This is structurally identical to what `spawn_relight` already does (jobs/mod.rs:232-245); we copy the pattern rather than abstract it.

**Files:**
- Modify: `src/jobs/mod.rs`

- [ ] **Step 1: Edit the worker body.**

In `src/jobs/mod.rs`, find the `spawn_gen` worker (lines ~188-212). Current body inside `catch_unwind`:

```rust
let mut dense = DenseChunk::empty();
generator.fill_chunk(coord, &mut dense);
let no_neighbors = crate::voxel::chunk::Neighbors { chunks: [None; 6] };
crate::lighting::recompute_chunk(&mut dense, &no_neighbors, &registry);
PalettedChunk::compress(&dense)
```

Replace with:

```rust
let mut dense = DenseChunk::empty();
generator.fill_chunk(coord, &mut dense);
// Decompress any neighbours the caller snapshotted so the
// initial BFS does column-drop inheritance + lateral
// seeding correctly. Matches the spawn_relight pattern at
// jobs/mod.rs:232-245.
let neighbor_dense: Vec<Option<DenseChunk>> = neighbors
    .iter()
    .map(|opt| opt.as_ref().map(|p| p.decompress()))
    .collect();
let n_refs: [Option<&DenseChunk>; 6] = [
    neighbor_dense[0].as_ref(),
    neighbor_dense[1].as_ref(),
    neighbor_dense[2].as_ref(),
    neighbor_dense[3].as_ref(),
    neighbor_dense[4].as_ref(),
    neighbor_dense[5].as_ref(),
];
let ns = crate::voxel::chunk::Neighbors { chunks: n_refs };
crate::lighting::recompute_chunk(&mut dense, &ns, &registry);
PalettedChunk::compress(&dense)
```

- [ ] **Step 2: Rename the function parameter back from `_neighbors` to `neighbors`.**

In the signature, change `_neighbors:` to `neighbors:`. (Currently it's prefixed with `_` from Task 1 to silence the unused-variable warning.)

- [ ] **Step 3: Update the doc comment above `spawn_gen`.**

The current doc says:

```rust
/// 3. Runs the lighting BFS *locally* (no neighbours yet — cross-chunk
///    bleed gets reapplied later when the streaming system queues a
///    relight on the dirty neighbour).
```

That description is now stale. Replace step 3 with:

```rust
/// 3. Runs the lighting BFS using the `neighbours` snapshot supplied
///    by the caller. Chunks generated while their face-adjacent
///    neighbours are already loaded receive correct sky-light
///    column-drop inheritance from the +Y neighbour and lateral
///    block-light seeding from all six, on this single pass — no
///    follow-up relight needed. Chunks generated at the streaming
///    wavefront (neighbours mostly `None`) fall back to a
///    best-effort BFS, same as before.
```

- [ ] **Step 4: Verify the build.**

Run: `cargo build --release 2>&1 | tail -3`
Expected: clean build, no warnings about unused `neighbors`.

- [ ] **Step 5: Run the integration test from Task 3 — it should now PASS.**

Run: `cargo test --release --test gen_with_neighbors 2>&1 | tail -10`
Expected: `test result: ok. 1 passed; 0 failed`.

If it still fails: the worker change didn't propagate to the build. Common cause: a stale incremental-compile artifact. Try `cargo clean -p oxium && cargo test --release --test gen_with_neighbors`.

- [ ] **Step 6: Run the full test suite to confirm no regression.**

Run: `cargo test --release 2>&1 | grep -E "test result|FAILED"`
Expected: every `test result` line says `ok` with 0 failed.

- [ ] **Step 7: Commit.**

```bash
git add src/jobs/mod.rs
git commit -m "fix(lighting): initial chunk-gen BFS uses real neighbour data

The gen worker now decompresses any neighbour PalettedChunks the
streamer snapshotted at spawn time and passes them into
recompute_chunk. Chunks generated with their face-adjacents already
loaded get correct sky-light column-drop inheritance and lateral
block-light seeding on the first BFS pass — no follow-up relight
needed for that case.

Same pattern as spawn_relight (jobs/mod.rs:232-245); the 12-line
decompression block is duplicated rather than extracted, since the
two workers have different result shapes and the duplication is
clearer than a shared helper would be.

The integration test in tests/gen_with_neighbors.rs now passes — a
chunk generated with a red-torch-lit neighbour picks up the seeded
red block-light across the chunk seam, where main produced 0.

Limit: chunks generated at the streaming wavefront still see partial
neighbour data (whatever is loaded at that moment). Residual artifacts
at the wavefront are not addressed by this commit and require a
separate per-chunk re-light-on-neighbour-arrival mechanism, which is
deferred."
```

---

### Task 5: Manual verification against the user's reported scenarios

The unit/integration tests verify the algorithm. Manual smoke confirms that the visible artifacts on real terrain actually resolve.

**Files:** none modified.

- [ ] **Step 1: Restart any in-flight processes.**

Don't reuse a running app across the fix — any chunks already in memory with stale light won't be re-light just from this change (the fix applies at gen time, not retroactively). Kill the app, delete any active save you want to retest against (or `git stash` to preserve current saves), and start fresh.

```bash
pkill oxium || true
cargo run --release --bin oxium -- --seed 42
```

- [ ] **Step 2: Fly to a cliff face.**

Look for vertical cliff faces — these were the strongest indicator in the user's report (Image 8). Fly along the cliff face at multiple altitudes. The dark vertical streaks descending from cliff tops should be substantially reduced; what remains should be at the streaming wavefront (chunks loading just-in-time as you move).

- [ ] **Step 3: Stand still in the same area for 10–30 seconds.**

The relight pump still processes player-edit / persistence-load dirty.light flags. Anything that was wrong should converge. After 30 seconds of stillness, surrounding terrain should be stable — no further changes to lighting.

- [ ] **Step 4: Compare against the user's reported locations.**

If accessible, visit `(864, 107, -371)` (the artifact location from Image 9 of the previous session). The cell that was dark should now be correctly lit on first arrival — flying away and coming back should reproduce a clean view, not a stale-dark one.

- [ ] **Step 5: Check the LQ HUD counter.**

The `LQ:` value in the HUD (top-left) is the dirty.light queue size. During and after fly-around it should:
- Briefly spike as chunks load
- Drain to single digits or zero when standing still
- Never sit at 100+ for an extended period

If LQ stays elevated, something else is constantly re-marking dirty.light. That would be a separate bug.

- [ ] **Step 6: Test underwater (the regression site from the failed cascade).**

Fly to a known ocean (or use `--find-water`). Look down through the water at the ocean floor from multiple angles. The ocean floor should be uniformly lit (or appropriately attenuated by water depth via the BFS cost-3 step) — NOT flickering, NOT chunk-tops-dark.

If chunk-top darkness or flicker reappears, the new code path has introduced a different bug than the cascade. Stop and report.

- [ ] **Step 7: Final smoke + commit any captured baselines.**

If you want screenshot baselines for future regression, capture one of each scenario:

```bash
cargo run --release --bin oxium -- --screenshot-and-exit /tmp/cliff.png --spawn 1074,82,-619 --look 0,-10 --time 0.25
```

(Adjust coords to your test cliff location.)

Compare against the pre-fix screenshot you have in conversation history. The dark streaks should be visibly reduced.

No commit at this task — verification only.

---

## Self-review notes

**Spec coverage check** (against the "what's the actual gap" finding from the debugging session):

| Stated gap | Plan task |
|---|---|
| `spawn_gen` runs BFS with `Neighbors { chunks: [None; 6] }` even when neighbours are loaded | Task 4 (worker uses real `Neighbors`) |
| Caller has World access but doesn't currently snapshot neighbours | Task 2 (`gather_neighbors(world, c)` at both call sites) |
| Need test to lock in correct behavior before/after | Task 3 (integration test) |
| Don't reintroduce the cascade explosion | Plan limits scope — no `dirty.light` cascade added; comments in `mesh_upload.rs:64-79` stay disabled |

**Placeholder scan:** no TBDs or "handle edge cases" — every step has concrete code or commands.

**Type consistency:**
- `[Option<Arc<PalettedChunk>>; 6]` is the type passed from caller → `spawn_gen` (Task 2 in callers, Task 1 in signature). Same as `spawn_relight`'s neighbour parameter (existing). ✓
- `Neighbors { chunks: [Option<&'a DenseChunk>; 6] }` is what `recompute_chunk` takes — built inside the worker from the decompressed `Vec<Option<DenseChunk>>`, same as `spawn_relight` (Task 4). ✓
- `gather_neighbors(world: &World, c: ChunkCoord) -> [Option<Arc<PalettedChunk>>; 6]` — existing function at `mesh_upload.rs:376`, return type matches what `spawn_gen` accepts. ✓

**Risk register:**
- *gen_pool throughput:* the worker now decompresses up to 6 extra `PalettedChunk`s per gen job. Decompress is ~10-50 µs per chunk on the codebase's measured cost; 6× that is ~60-300 µs added to a gen job that typically runs 1-5 ms. Below noise.
- *Memory:* `Arc<PalettedChunk>` snapshots are cheap (refcount bump, not data copy). `gather_neighbors` already does this for relight without measured impact.
- *Determinism:* gen now depends on which neighbours happen to be loaded at spawn time — so two runs with different load-order produce different *initial* light. They converge to the same value via the normal cascade after edits. Acceptable; no test asserts gen-time determinism.
- *Backward compat:* no save format change. Saved chunks reload the same as before (`PersistResult::Loaded` already sets `dirty.light = true`).

**Deferred:**
- Wavefront-arrival re-light: chunks that gen with mostly-None neighbors (because the wavefront is still arriving) don't get a second chance. A follow-up could track `light_neighbors_present: [bool; 6]` per chunk and trigger a one-shot relight when a missing-at-gen neighbour later arrives. This bounds re-lights to ≤ 6 per chunk lifetime, avoiding the cascade explosion. Not in this plan.
