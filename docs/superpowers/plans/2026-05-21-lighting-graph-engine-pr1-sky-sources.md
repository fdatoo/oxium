# Lighting Graph Engine PR 1 — ChunkSkyLightSources Heightmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-chunk sky-source heightmap (`ChunkSkyLightSources`) that records, for each `(x, z)` column in the chunk, the world-Y of the lowest cell that is a sky source — i.e., the lowest non-opaque cell with nothing opaque above it in this chunk. Populate it whenever a chunk is installed in the `World`. **Do not wire it into the existing BFS** — it just sits in `ChunkMeta` ready for PR2/PR3 to consume.

**Architecture:** New module `src/lighting/sky_sources.rs` owns the struct and its build function. `ChunkMeta` gains a `pub sky_sources: ChunkSkyLightSources` field with a `Default` impl. `World::insert` becomes the single chokepoint that builds the heightmap from the chunk's `PalettedChunk` data — every chunk-load path (`JobResult::Generated`, `JobResult::LoadedFromDisk`) already funnels through `World::insert`, so we don't have to touch the streaming systems.

**Tech stack:** Rust. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`

**Risk:** Low. Pure-additive change. No existing behaviour modified. Existing BFS continues to run unchanged.

---

## Files

**Create:**
- `src/lighting/sky_sources.rs` — `ChunkSkyLightSources` struct, `build_from_dense`, `lowest_source_y`, unit tests

**Modify:**
- `src/lighting/mod.rs` — `pub mod sky_sources;` + re-export `ChunkSkyLightSources`
- `src/voxel/chunk.rs` — add `pub sky_sources: ChunkSkyLightSources` field to `ChunkMeta`
- `src/voxel/world.rs` — `World::insert` builds the heightmap from the decompressed chunk

**Do not touch in this PR:**
- The existing BFS in `src/lighting/mod.rs` (`recompute_chunk`, `sky_light`, `block_rgb`, `seed_from_neighbors`, `snapshot_face_boundaries`)
- Anything in `src/render/`, `src/mesher/`, `src/ecs/systems/`
- `World::set_block`, `World::mark_below_dirty` (deferred to PR3)
- `mesh_upload.rs` (the only `World::insert` callsite — passing through `insert` is enough)

---

## Semantics

For a column `(x, z)` in a chunk:

- Scan top-down within the chunk (`y = 31` down to `y = 0`).
- The **first opaque block** found at local y = `Y_op` defines the floor of the column's "sky zone". Cells at local `y > Y_op` are sources.
- `lowest_source_y` (returned in world-Y space) = `chunk.world_y_of(Y_op + 1)`. I.e. the world-Y of the cell directly above the topmost opaque block.
- If **no opaque block** is found in this chunk's column, `lowest_source_y = i32::MIN` (sentinel meaning "this whole column passed through the chunk transparent; some chunk above or below has the real source floor").

This intentionally ignores opacity in `+Y` neighbour chunks for PR1. PR2 wires the lookup to also walk above-chunks when computing `lowest_source_y`. The PR1 simplification is the same one today's BFS already makes (`sky_light()` defaults to `light = 15` when `+Y` is `None`), so it's not a regression.

---

## Tasks

### Task 1: Create the empty `sky_sources` module

**Files:**
- Create: `src/lighting/sky_sources.rs`
- Modify: `src/lighting/mod.rs`

- [ ] **Step 1: Create `src/lighting/sky_sources.rs` with module skeleton.**

```rust
//! Per-chunk sky-source heightmap. For each `(x, z)` column in the chunk,
//! records the world-Y of the lowest cell that is a sky light source —
//! i.e., a non-opaque cell with nothing opaque above it (within this
//! chunk; +Y neighbour lookup is deferred to PR2).
//!
//! The graph-engine sky channel reads this heightmap to know which cells
//! are sources (`y >= lowest_source_y` → source at level 15) and to emit
//! the right add/remove-source ops when a column's heightmap changes.
//!
//! See `docs/superpowers/specs/2026-05-21-lighting-graph-engine-design.md`
//! — "Sky source heightmap" section.

use crate::voxel::block::BlockRegistry;
use crate::voxel::chunk::DenseChunk;
use crate::voxel::coords::{ChunkCoord, CHUNK_DIM, CHUNK_DIM_U};
use glam::UVec3;

/// Sentinel meaning "no opaque block in this column inside this chunk."
/// The whole column passed through as transparent; the real source floor
/// (if any) lives in a chunk above or below.
pub const NO_SOURCE_FLOOR: i32 = i32::MIN;

/// Per-chunk record of where the sky-source floor sits for each `(x, z)`
/// column. Each entry is a **world-Y coordinate**: the y of the lowest
/// cell that is itself a sky source (level 15 from sky).
///
/// Cells at world-Y `>= lowest_source_y(x, z)` within this chunk are
/// sources. Cells below propagate via the graph engine.
#[derive(Debug, Clone)]
pub struct ChunkSkyLightSources {
    lowest_source_y: Box<[i32; (CHUNK_DIM_U * CHUNK_DIM_U) as usize]>,
}

impl Default for ChunkSkyLightSources {
    /// An empty heightmap — every column reports `NO_SOURCE_FLOOR`.
    /// Used for chunks that haven't had `build_from_dense` called yet.
    fn default() -> Self {
        Self {
            lowest_source_y: Box::new(
                [NO_SOURCE_FLOOR; (CHUNK_DIM_U * CHUNK_DIM_U) as usize],
            ),
        }
    }
}

impl ChunkSkyLightSources {
    /// World-Y of the lowest sky-source cell in column `(lx, lz)`. Returns
    /// `NO_SOURCE_FLOOR` if the column has no opaque block in this chunk.
    pub fn lowest_source_y(&self, lx: u32, lz: u32) -> i32 {
        debug_assert!(lx < CHUNK_DIM_U && lz < CHUNK_DIM_U);
        self.lowest_source_y[Self::idx(lx, lz)]
    }

    #[inline]
    fn idx(lx: u32, lz: u32) -> usize {
        (lx + lz * CHUNK_DIM_U) as usize
    }

    /// Scan a chunk's blocks top-down per column. For each column, the
    /// first opaque cell defines the source floor; everything above is a
    /// source. If no opaque cell is found in the column within this
    /// chunk, the entry is `NO_SOURCE_FLOOR`.
    ///
    /// `chunk_coord` is needed to translate the local-Y of the topmost
    /// opaque cell into a world-Y.
    pub fn build_from_dense(
        dense: &DenseChunk,
        chunk_coord: ChunkCoord,
        registry: &BlockRegistry,
    ) -> Self {
        let mut out = Self::default();
        let chunk_bottom_y = chunk_coord.0.y * CHUNK_DIM;
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                let mut floor: i32 = NO_SOURCE_FLOOR;
                for ly in (0..CHUNK_DIM_U).rev() {
                    let pos = crate::voxel::coords::LocalPos(UVec3::new(lx, ly, lz));
                    let block = dense.blocks[pos.to_index()];
                    if registry.info(block).opaque {
                        // First opaque cell from the top — source floor
                        // is the cell immediately above it.
                        floor = chunk_bottom_y + ly as i32 + 1;
                        break;
                    }
                }
                out.lowest_source_y[Self::idx(lx, lz)] = floor;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::Block;
    use crate::voxel::chunk::DenseChunk;
    use crate::voxel::coords::LocalPos;
    use glam::IVec3;

    #[test]
    fn default_is_all_no_source_floor() {
        let s = ChunkSkyLightSources::default();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), NO_SOURCE_FLOOR);
            }
        }
    }

    #[test]
    fn all_air_chunk_has_no_source_floor_anywhere() {
        let dense = DenseChunk::empty();
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(
                    s.lowest_source_y(lx, lz), NO_SOURCE_FLOOR,
                    "column ({lx},{lz}) should have no floor in all-air chunk",
                );
            }
        }
    }

    #[test]
    fn all_stone_chunk_floor_is_just_above_chunk_top() {
        // A chunk at chunk_coord (0, 0, 0) spans world-Y 0..32. Topmost
        // opaque cell in every column is at local y=31, world y=31. The
        // source floor is the cell above, world y=32.
        let dense = DenseChunk::new_filled(Block::Stone);
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), 32);
            }
        }
    }

    #[test]
    fn stone_layer_at_local_y_20_gives_floor_at_world_y_21() {
        // 1-block stone layer at y=20 across the whole chunk;
        // everything else air. Floor = local y+1 = 21 in world space
        // (chunk_coord.y = 0).
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(s.lowest_source_y(lx, lz), 21);
            }
        }
    }

    #[test]
    fn floor_uses_topmost_opaque_when_multiple_layers_exist() {
        // Two opaque layers at y=10 and y=20. Topmost (y=20) defines
        // the floor: world y=21.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        assert_eq!(s.lowest_source_y(5, 5), 21);
    }

    #[test]
    fn floor_is_per_column_independent() {
        // Stone at y=15 only in column (0,0); rest of chunk is air.
        let mut dense = DenseChunk::empty();
        dense.set(LocalPos(UVec3::new(0, 15, 0)), Block::Stone);
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        assert_eq!(s.lowest_source_y(0, 0), 16, "column (0,0) has floor at world y=16");
        assert_eq!(
            s.lowest_source_y(1, 0), NO_SOURCE_FLOOR,
            "column (1,0) has no floor — should be NO_SOURCE_FLOOR",
        );
        assert_eq!(s.lowest_source_y(0, 1), NO_SOURCE_FLOOR);
    }

    #[test]
    fn world_y_translation_respects_chunk_coord() {
        // Same stone layer at local y=10, but chunk is at chunk_coord
        // (0, 3, 0) which spans world-Y 96..128. Local y=10 → world y=106.
        // Source floor = world y=107.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::new(0, 3, 0)), &reg,
        );
        assert_eq!(s.lowest_source_y(5, 5), 107);
    }

    #[test]
    fn negative_chunk_y_translates_correctly() {
        // Chunk at chunk_coord (0, -1, 0) spans world-Y -32..0.
        // Stone at local y=5 → world y=-27. Floor = world y=-26.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 5, lz)), Block::Stone);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::new(0, -1, 0)), &reg,
        );
        assert_eq!(s.lowest_source_y(5, 5), -26);
    }

    #[test]
    fn non_opaque_non_air_does_not_create_floor() {
        // Water and leaves are non-opaque; they should NOT define a
        // source floor (sky still propagates through them, attenuated).
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 20, lz)), Block::Water);
            }
        }
        let reg = BlockRegistry::new();
        let s = ChunkSkyLightSources::build_from_dense(
            &dense, ChunkCoord(IVec3::ZERO), &reg,
        );
        assert_eq!(
            s.lowest_source_y(5, 5), NO_SOURCE_FLOOR,
            "water is non-opaque; should not create a source floor",
        );
    }
}
```

- [ ] **Step 2: Wire the module into `src/lighting/mod.rs`.**

Find the top of `src/lighting/mod.rs` (the file currently containing `recompute_chunk` etc). Add `pub mod sky_sources;` and a re-export immediately after the existing `use` block at the top, before the `const D: i32` line.

```rust
pub mod sky_sources;

pub use sky_sources::{ChunkSkyLightSources, NO_SOURCE_FLOOR};
```

- [ ] **Step 3: Run the unit tests in the new module.**

Run: `cargo test --lib lighting::sky_sources::tests -- --nocapture`
Expected: all 9 tests pass.

- [ ] **Step 4: Verify nothing else broke.**

Run: `cargo test --lib`
Expected: full test suite passes; no other tests touched.

- [ ] **Step 5: Commit.**

```bash
git add src/lighting/sky_sources.rs src/lighting/mod.rs
git commit -m "feat(lighting): ChunkSkyLightSources heightmap (graph-engine PR1, part 1/3)

Per-chunk per-column world-Y of the lowest sky-source cell. Built from
a DenseChunk + ChunkCoord + BlockRegistry. Returns NO_SOURCE_FLOOR for
columns with no opaque block in the chunk. +Y neighbour lookup deferred
to PR2; PR1's column scan is single-chunk only, matching the
simplification today's BFS already makes (light=15 when +Y is None).

Not wired to anything yet — PR1 part 2 adds the ChunkMeta field;
PR1 part 3 populates it at chunk-install."
```

---

### Task 2: Add `sky_sources` to `ChunkMeta`

**Files:**
- Modify: `src/voxel/chunk.rs:393` (the `ChunkMeta` struct definition)

- [ ] **Step 1: Add the field to `ChunkMeta`.**

Find the `ChunkMeta` struct around `src/voxel/chunk.rs:393`. It currently looks like:

```rust
#[derive(Debug, Default)]
pub struct ChunkMeta {
    pub state: ChunkState,
    pub dirty: ChunkDirty,
    pub modified: bool,
    pub mesh_version: u64,
}
```

Replace with:

```rust
#[derive(Debug, Default)]
pub struct ChunkMeta {
    pub state: ChunkState,
    pub dirty: ChunkDirty,
    pub modified: bool,
    pub mesh_version: u64,
    /// Per-column world-Y of the lowest sky-source cell. Built by
    /// `crate::lighting::ChunkSkyLightSources::build_from_dense` at
    /// `World::insert` time and rebuilt whenever the chunk's blocks
    /// change. Consumed by the graph-engine sky channel (PR3).
    ///
    /// Defaults to a heightmap full of `NO_SOURCE_FLOOR`, which is
    /// the safe value for a freshly-defaulted `ChunkMeta` — no
    /// floor means "treat every cell as a potential source"
    /// (matches today's BFS column-drop default of `light = 15`).
    pub sky_sources: crate::lighting::ChunkSkyLightSources,
}
```

- [ ] **Step 2: Confirm `Default` still derives.**

`ChunkSkyLightSources` already has a manual `Default` impl (returns all-`NO_SOURCE_FLOOR`). `ChunkMeta` is `#[derive(Debug, Default)]`, so `Default::default()` will call `ChunkSkyLightSources::default()`. No additional code needed.

Run: `cargo build --lib`
Expected: clean build. If `Default` doesn't derive, ensure `ChunkSkyLightSources` has `Default` impl and that the lighting module re-exports it correctly.

- [ ] **Step 3: Run the test suite.**

Run: `cargo test --lib`
Expected: all tests pass. Some tests construct `ChunkMeta::default()` indirectly via `World::insert`; the new field doesn't change behaviour because nothing reads it yet.

- [ ] **Step 4: Commit.**

```bash
git add src/voxel/chunk.rs
git commit -m "feat(lighting): ChunkMeta gains sky_sources field (graph-engine PR1, part 2/3)

Adds pub sky_sources: ChunkSkyLightSources to ChunkMeta, defaulting to
the all-NO_SOURCE_FLOOR heightmap so legacy code paths see safe values.
Not populated yet — part 3 wires World::insert to build the real
heightmap at chunk-install time."
```

---

### Task 3: Populate the heightmap at `World::insert`

**Files:**
- Modify: `src/voxel/world.rs:71` (the `World::insert` function)

- [ ] **Step 1: Write the failing integration test.**

Append to the `tests` module at the bottom of `src/voxel/world.rs`:

```rust
    /// World::insert must populate the chunk's sky_sources heightmap
    /// so PR2/PR3 can consume it without further plumbing.
    #[test]
    fn insert_populates_sky_sources_from_chunk_data() {
        use crate::lighting::NO_SOURCE_FLOOR;
        use crate::voxel::chunk::DenseChunk;
        use crate::voxel::coords::LocalPos;
        use glam::UVec3;

        let mut w = World::new(42);

        // Build a chunk with a single stone layer at local y=10
        // across the whole footprint; everything else is air.
        let mut dense = DenseChunk::empty();
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                dense.set(LocalPos(UVec3::new(lx, 10, lz)), Block::Stone);
            }
        }
        let chunk = PalettedChunk::compress(&dense);

        let coord = ChunkCoord(IVec3::ZERO);
        w.insert(coord, chunk);

        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after insert");
        };

        // Every column should report world-y=11 (one above the stone).
        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(
                    meta.sky_sources.lowest_source_y(lx, lz), 11,
                    "column ({lx},{lz}) should have floor at world y=11",
                );
            }
        }
    }

    /// All-air chunk: heightmap should be entirely NO_SOURCE_FLOOR
    /// (matches the default), but populated rather than default-stub.
    #[test]
    fn insert_populates_sky_sources_even_for_all_air() {
        use crate::lighting::NO_SOURCE_FLOOR;
        use crate::voxel::chunk::DenseChunk;

        let mut w = World::new(42);
        let chunk = PalettedChunk::compress(&DenseChunk::empty());
        let coord = ChunkCoord(IVec3::new(2, 1, -3));

        w.insert(coord, chunk);

        let ChunkSlot::Stored { meta, .. } = w.chunks.get(&coord).unwrap() else {
            panic!("chunk should be Stored after insert");
        };

        for lz in 0..CHUNK_DIM_U {
            for lx in 0..CHUNK_DIM_U {
                assert_eq!(meta.sky_sources.lowest_source_y(lx, lz), NO_SOURCE_FLOOR);
            }
        }
    }
```

The existing test imports at the top of the `tests` module already cover `Block`, `World`, `ChunkCoord`, `PalettedChunk`, `ChunkSlot`, `CHUNK_DIM_U`, `IVec3`. If any of these are missing in your version, add them — but the existing test file already brings them in via the broader `use` block.

- [ ] **Step 2: Run the failing test.**

Run: `cargo test --lib voxel::world::tests::insert_populates_sky_sources_from_chunk_data -- --nocapture`
Expected: FAIL — the assertion is `lowest_source_y(lx, lz), 11` but `World::insert` doesn't build the heightmap yet, so the field is the default `NO_SOURCE_FLOOR` and the test fails on the first assertion.

- [ ] **Step 3: Implement the population in `World::insert`.**

Find `World::insert` around `src/voxel/world.rs:71`:

```rust
    pub fn insert(&mut self, c: ChunkCoord, data: PalettedChunk) {
        let meta = ChunkMeta {
            state: ChunkState::Generated,
            dirty: ChunkDirty {
                mesh: true,
                light: false,
            },
            ..Default::default()
        };
        self.chunks.insert(
            c,
            ChunkSlot::Stored {
                data: std::sync::Arc::new(data),
                meta,
            },
        );
    }
```

Replace with:

```rust
    pub fn insert(&mut self, c: ChunkCoord, data: PalettedChunk) {
        // Build the sky-source heightmap once at install time. Decompressing
        // the chunk is ~50 µs (palette + 4-bit unpack) and the scan is
        // 32×32×32 opacity lookups (~50 µs more) — small enough to do
        // inline; PR2/PR3 will move heightmap updates onto the per-edit
        // path so this only runs on initial install.
        let dense = data.decompress();
        let sky_sources = crate::lighting::ChunkSkyLightSources::build_from_dense(
            &dense, c, &self.registry,
        );
        let meta = ChunkMeta {
            state: ChunkState::Generated,
            dirty: ChunkDirty {
                mesh: true,
                light: false,
            },
            sky_sources,
            ..Default::default()
        };
        self.chunks.insert(
            c,
            ChunkSlot::Stored {
                data: std::sync::Arc::new(data),
                meta,
            },
        );
    }
```

- [ ] **Step 4: Run the previously-failing test.**

Run: `cargo test --lib voxel::world::tests::insert_populates_sky_sources_from_chunk_data -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run the other new test and the full suite.**

Run: `cargo test --lib`
Expected: all tests pass, including `insert_populates_sky_sources_even_for_all_air`. No existing test fails — the new field is only read by the new tests; everything else is unchanged.

- [ ] **Step 6: Commit.**

```bash
git add src/voxel/world.rs
git commit -m "feat(lighting): populate sky_sources at World::insert (graph-engine PR1, part 3/3)

Every chunk install (Generated + LoadedFromDisk both funnel through
World::insert) now builds the ChunkSkyLightSources heightmap inline
from the chunk's blocks. Cost is ~100 µs per install (50 µs decompress +
50 µs scan); fine on the chunk-load critical path.

The heightmap is populated but not yet consumed — PR2 introduces the
ChannelEngine that reads it. Existing BFS continues to run unchanged.
This concludes PR1."
```

---

## Verification

After all three tasks:

- [ ] **Run the full lib test suite once more.**

Run: `cargo test --lib`
Expected: clean pass. New tests added: 9 in `lighting::sky_sources::tests` + 2 in `voxel::world::tests` = 11 new.

- [ ] **Confirm zero behaviour change to the existing BFS / streaming / rendering.**

Run: `cargo run --release` and walk around for ~30 seconds. Sky lighting, block lighting, chunk streaming, and edits should be visually and behaviourally indistinguishable from main.

Reason: PR1 only **adds** data. Nothing reads `sky_sources` yet; the existing `lighting::recompute_chunk` BFS still runs and still populates `DenseChunk::sky_light` / `DenseChunk::block_rgb` the same way. The only observable change is that `cargo build` takes one more `.rs` file to compile and each `World::insert` does ~100 µs more work (well below the streaming budget).

- [ ] **Check `git log` shows three focused commits.**

Run: `git log --oneline -5`
Expected:
```
<sha> feat(lighting): populate sky_sources at World::insert (graph-engine PR1, part 3/3)
<sha> feat(lighting): ChunkMeta gains sky_sources field (graph-engine PR1, part 2/3)
<sha> feat(lighting): ChunkSkyLightSources heightmap (graph-engine PR1, part 1/3)
```

If you want to flatten into a single commit before opening the PR, that's fine — but three focused commits also reads cleanly in review.

---

## What PR2 will do

For context — not part of this PR's work:

- Introduce `BucketQueue` (16 magnitude buckets) in `src/lighting/queue.rs`.
- Introduce the empty `ChannelEngine` and `LightEngine` skeleton in `src/lighting/engine.rs`.
- Add a `LightEngine` field to `World`.
- Engine `tick()` is a no-op (returns immediately). Nothing enqueues anything yet.
- Unit tests for `BucketQueue`'s push/pop ordering and chunk-purge.

PR2 is also low-risk and pure-additive. PR3 is the cutover and is where the existing BFS gets deleted.
