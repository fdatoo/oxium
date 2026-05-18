# oxium — Design Spec

**Date:** 2026-05-18
**Status:** Approved, ready for implementation planning

## Summary

A single-player voxel sandbox built from scratch in Rust on `wgpu` + `winit`. The MVP gives the player an infinite procedurally-generated world with caves, walkable terrain, dynamic lighting with a day/night cycle, place/destroy interaction, and persistent saves. Stylized smooth aesthetic — no texture pipeline, vertex colors with baked ambient occlusion.

## Scope

**In scope (v0):**

- Infinite world streamed in 32³-voxel chunks
- 3D-noise terrain with caves and overhangs
- Sun and torch light propagation (BFS flood fill)
- Day/night cycle (procedural sky)
- World save/load to disk
- Stylized smooth visuals — vertex tints + baked AO, no textures
- Walking with gravity + jumping + AABB-vs-voxel collision; `F` toggles creative flight
- Place/destroy via voxel raycast; small keyboard-cycled block palette
- Vast render distance (16+ chunks) with 3-level LOD

**Out of scope (v0):**

- Multiplayer / networking
- Mobs, items, inventory, crafting, hunger, combat
- Biomes (single biome only — biome system is a follow-on)
- Survival mechanics
- Modding API, asset hot-reload
- Incremental light propagation (recompute-on-dirty for v0)
- Pixel-art textures, item models, particle systems
- World versioning / save migration

## Architecture

### Tech stack

- **Language:** Rust (edition 2024, `rustc` 1.95+)
- **Graphics:** `wgpu`
- **Windowing/input:** `winit`
- **Math:** `glam`
- **ECS:** `hecs` (used for game entities only — chunks live outside the ECS)
- **Threading:** `rayon` worker pool + `crossbeam` channels for results
- **Noise:** `noise` crate (Simplex, FBM, Ridged)
- **Serialization:** `bincode`
- **Compression:** `zstd`
- **Errors:** `anyhow` at boundaries, `thiserror` for domain errors

Single binary, internal modules. No workspace split — promote to a library later only if a second consumer appears.

### Module layout

```
src/
├── main.rs            # winit event loop, wires App
├── app.rs             # App: ecs, world, renderer, jobs, time
│
├── ecs/               # game entities (player, camera, sun, cursor, future mobs)
│   ├── mod.rs         # schedule: input → time → movement → physics → interact → stream → mesh-upload → render
│   ├── components.rs  # Position, Velocity, Aabb, Movement, Camera, PlayerInput, TimeOfDay, ...
│   └── systems/       # one file per system function
│
├── voxel/             # voxel world (resource, NOT an ECS entity)
│   ├── block.rs       # Block enum + BlockRegistry
│   ├── chunk.rs       # Chunk: dense [Block; 32³] + light arrays + dirty flags + LOD mesh handles
│   ├── world.rs       # World: HashMap<ChunkCoord, ChunkSlot> + streaming
│   ├── coords.rs      # ChunkCoord, BlockPos, LocalPos
│   └── raycast.rs     # Amanatides–Woo DDA voxel raycast
│
├── worldgen/          # noise → blocks (pure: (seed, coord) → chunk)
├── lighting/          # sun + torch flood-fill, recompute-on-dirty
├── mesher/            # greedy mesher + AO baker; LOD downsampling
├── render/            # wgpu setup, pipelines, shaders, frustum cull, draw
├── physics/           # AABB swept collision
├── persistence/       # region files (16×16 chunks), zstd
└── jobs/              # rayon-backed job system + result channels

assets/shaders/        # *.wgsl, embedded via include_str!
```

### Threading model

- **Main thread**: winit event loop, ECS schedule, render submission, GPU uploads
- **Rayon pool**: chunk generation, light recompute, greedy meshing
- **Persistence thread**: dedicated single thread for blocking I/O (separate from rayon pool to avoid starving compute jobs)
- **Channels**: `crossbeam` MPSC; main thread drains completed jobs each frame

The ECS schedule is an explicit per-frame function:

```
input → time → movement → physics → interaction → world_stream → mesh_upload → render
```

Manual ordering, not a macro-driven scheduler — easier to debug.

### Coordinate spaces

- `BlockPos: IVec3` — world-space block coordinates
- `ChunkCoord: IVec3` — `BlockPos.floor_div(32)`
- `LocalPos: UVec3` — block position within a chunk, `0..32`
- World units = blocks (1 block = 1 unit); Y-up

## Voxel data model

```rust
#[repr(u16)]
pub enum Block {
    Air = 0, Stone, Dirt, Grass, Sand, Water, Wood, Leaves, Torch, /* ~16 total */
}

pub struct BlockInfo {
    pub solid: bool,
    pub opaque: bool,
    pub emission: u8,                  // 0..15
    pub color: [f32; 4],
    pub top_color: Option<[f32; 4]>,   // grass top vs side
}

pub const CHUNK_DIM: usize = 32;
pub const CHUNK_VOL: usize = 32_768;

/// Hot, transient view used by meshers, lighters, and edit paths.
/// Never stored long-term — produced by `PalettedChunk::decompress()`.
pub struct DenseChunk {
    pub blocks:      Box<[Block; CHUNK_VOL]>,   // 64 KB
    pub sky_light:   Box<[u8; CHUNK_VOL]>,      // 32 KB (4 bits used)
    pub block_light: Box<[u8; CHUNK_VOL]>,      // 32 KB (4 bits used)
}

/// Canonical in-RAM and on-disk form. ~50 KB.
pub struct PalettedChunk {
    pub palette:     Vec<Block>,
    pub indices:     BitPackedArray,            // 4..8 bits/voxel
    pub sky_light:   Packed4Bit<CHUNK_VOL>,
    pub block_light: Packed4Bit<CHUNK_VOL>,
}

/// Per-chunk metadata kept alongside the PalettedChunk.
pub struct ChunkMeta {
    pub state:        ChunkState,               // Empty | Generating | Generated | Meshing(lod) | Ready
    pub mesh_handles: [Option<MeshHandle>; 3],  // L0/L1/L2 GPU meshes
    pub dirty:        ChunkDirty,
    pub modified:     bool,
}

pub struct World {
    pub chunks:   HashMap<ChunkCoord, ChunkSlot>,
    pub registry: BlockRegistry,
    pub seed:     u64,
}

pub enum ChunkSlot {
    Pending,                                     // job in flight
    Stored { data: PalettedChunk, meta: ChunkMeta },
}
```

### Memory and compression

A dense `Chunk` is ~128 KB (64 KB blocks + 32 KB sky_light + 32 KB block_light). At horizontal radius 16, ~17k chunks loaded → ~2.2 GB worst case. Infeasible without compression.

**Approach: PalettedChunk is the canonical in-RAM and on-disk form.** Every loaded chunk lives as:

```
PalettedChunk {
    palette:     Vec<Block>,           // typically <16 entries
    indices:     BitPackedArray,       // 4 bits/voxel = 16 KB (autoexpands if palette > 16)
    sky_light:   Packed4Bit<32_768>,   // 16 KB
    block_light: Packed4Bit<32_768>,   // 16 KB
}
// ~50 KB total, vs 128 KB dense
```

At radius 16: ~17k chunks × ~50 KB ≈ ~850 MB. Acceptable.

`ChunkSlot` becomes simpler — `Stored(PalettedChunk)` or `Pending`. Hot operations (meshing, lighting recompute, edits) get a temporary `DenseChunk` view via a short-lived `decompress() → DenseChunk` borrow; the result is recompressed into the slot when done. An LRU of recently-decompressed dense views avoids thrashing on hot edits.

Palette/4-bit-light packing is the one upfront optimization — without it the vast-render-distance requirement is infeasible.

## Rendering

### wgpu setup

- One adapter (prefer discrete), one device + queue.
- Three pipelines: opaque voxels, translucent (water), sky.
- Depth buffer, MSAA off for v0.

### Vertex format (16 bytes, packed)

```rust
#[repr(C)]
struct Vertex {
    pos:         [u8; 3],   // local chunk coords 0..32
    ao:          u8,        // 0..3 baked
    color:       [u8; 4],
    normal_face: u8,        // enum: +X -X +Y -Y +Z -Z
    light:       u8,        // 4 bits sky, 4 bits block
    _pad:        [u8; 2],
}
```

A per-chunk uniform supplies the chunk's world-space origin so the vertex shader reconstructs world positions.

### Greedy meshing

For each of 6 face directions, sweep slice-by-slice:

1. Build a 2D mask of visible faces of block X (visible = neighbor non-opaque AND different appearance).
2. Greedy-merge adjacent equal mask cells into rectangles.
3. Emit one quad per rectangle with per-corner AO + light.

**AO bake**: for each quad corner, sample the 3 neighbor blocks diagonally adjacent on that face; `ao = saturating_sub(3, occupied_count)`. Smooth shading is automatic via vertex interpolation.

**Neighbor data**: meshing needs read access to the 6 chunk neighbors' edge slices. The mesher takes a `ChunkView { center: &DenseChunk, neighbors: [&DenseChunk; 6] }`. The worker thread that owns the mesh job is responsible for decompressing center + neighbors from their `PalettedChunk` form for the duration of the job.

### LOD

Three resolution levels, with chunk radii tuned at runtime. Defaults at render distance 16:

- **L0** (radius 0..6 chunks): full-resolution greedy mesh
- **L1** (6..12): 2× downsampled — collapse each 2³ to one representative non-air block, greedy-mesh the 16³ result
- **L2** (12..16): 4× downsampled, 8³ result

LOD chosen per-chunk by camera distance. Transitions are popping in v0 (no geomorphing or screen-space dithering).

### Render loop

Each frame, in `ecs::systems::render`:

1. Compute view + projection from camera entity → frustum planes.
2. Iterate loaded chunks, frustum-cull by chunk AABB, bucket by LOD.
3. Sort opaque front-to-back, translucent back-to-front.
4. Single render pass: sky → opaque → translucent.
5. Present.

### Sky and day/night

Procedural in the sky shader. A `TimeOfDay` uniform `t ∈ [0,1)` drives:

- Sun direction
- Sky gradient (zenith/horizon palettes lerped day → dusk → night)
- Sun-direction uniform shared with the opaque pipeline

Shaders live in `assets/shaders/*.wgsl`, embedded via `include_str!`.

## World generation

Pure function `(seed, coord) → chunk`, runs on worker threads.

```rust
pub struct Generator {
    height_noise: Fbm<Simplex>,
    detail_noise: Fbm<Simplex>,
    cave_noise:   Ridged<Simplex>,
    seed:         u64,
}
impl Generator {
    pub fn fill_chunk(&self, coord: ChunkCoord, out: &mut DenseChunk);
    // Caller compresses the result into a PalettedChunk before storing in the World.
}
```

**Pipeline (single pass, no neighbor dependencies):**

1. For each `(x, z)` column: `height = base + height_noise(x, z) * amplitude`.
2. For each `y` in chunk:
   - `y > height + detail` → `Air`
   - else if `cave_noise(x, y, z)` above threshold → `Air`
   - else: surface = `Grass`, next 3 = `Dirt`, deeper = `Stone`; below sea-level surface = `Sand`
3. Sea-level fill: any `Air` below `SEA_LEVEL` becomes `Water`.

Single biome for v0. Deterministic — tested with golden fixtures.

## Lighting

Two BFS flood fills, both on worker threads, both **recompute-on-dirty** (not incremental).

**Sky light:**

1. For each `(x, z)` column from the top of the loaded vertical range: drop `sky_light = 15` down until hitting an opaque block. Each air cell on the way: `sky_light = 15`.
2. BFS outward from those cells: `neighbor.sky_light = max(neighbor.sky_light, self - 1 - opacity_cost)`. Water has extra cost 3 instead of 1.

**Block light:**

1. Seed every block with `info.emission > 0`.
2. Same BFS rule, independent of sky.

Vertex shader picks `max(sky_light * sun_intensity, block_light)`. `sun_intensity` is a time-of-day uniform → 0 at midnight, so torches naturally dominate at night.

**Dirty model:** edits mark the chunk + cross-boundary neighbors as `light_dirty`. A worker recomputes (debounced one frame) then triggers a remesh.

**Neighbor gating:** a chunk can't enter `Meshing` or `Lighting` until all 6 neighbors are at least `Generated`. The streaming policy generates an outer "skin" one ring beyond the render radius to guarantee forward progress.

### Why recompute over incremental

| Approach | Per-edit light latency | Per-edit total latency (with mesh) | Code cost |
|---|---|---|---|
| Incremental BFS | ~0.05 ms | 10–30 ms | ~150 LOC, several easy bugs |
| Recompute-on-dirty | 2–4 ms × up to 4 chunks ≈ 12 ms | 25–45 ms | ~40 LOC, hard to get wrong |

At single-edit-per-click human rates the user-visible difference is ~1 frame of extra delay, all on worker threads. Incremental wins decisively for bulk edits (e.g. explosions across many chunks); v0 has no bulk edits, so we ship recompute. The `lighting::recompute_chunk(&mut DenseChunk, &Neighbors)` signature abstracts the strategy and can be swapped later without touching callers.

## Player and physics

### Components

```rust
struct Position(Vec3);                                  // feet position
struct Velocity(Vec3);
struct Aabb { half: Vec3 }                              // 0.3 × 0.9 × 0.3
struct Movement { mode: Walk | Fly, speed: f32, jump_v: f32 }
struct Camera { yaw: f32, pitch: f32, fov: f32, eye_offset: Vec3 }
struct PlayerInput { wishdir: Vec3, jump: bool, sprint: bool, place: bool, break_: bool }
```

Single player entity with all of the above. Sun and cursor highlight are separate entities.

### Input

`winit` events buffered into `App` per frame, drained by `systems::input`. Mouse delta → camera yaw/pitch (pitch clamped ±89°). WASD/Space/Shift → `wishdir`. LMB/RMB → edge-triggered `break_`/`place`. `F` → toggle movement mode. `1`–`9` → select block.

### Movement

Walking: quake-style ground accel + friction. Gravity −28 m/s², jump 8.4 m/s ≈ 1.25-block jump height. Flight: direct velocity = wish × speed, no gravity.

### Physics

AABB swept collision against the voxel world, **axis-by-axis** (resolve X, then Y, then Z — avoids corner-snag bugs from simultaneous resolution):

```rust
fn sweep_aabb(world: &World, aabb: &Aabb, pos: Vec3, vel: Vec3, dt: f32)
    -> (Vec3, Vec3, Grounded);
```

For each axis: compute swept AABB along that axis only; walk overlapping integer voxel cells; on first solid hit, snap to contact plane minus epsilon and zero that velocity component. `Grounded` tracked when downward-Y resolution snaps from above.

Water is non-solid; player AABB overlap applies drag + buoyancy. No swimming controls in v0 — player sinks slowly and walks the bottom.

`dt` from the event loop, clamped to ≤100 ms to absorb hitches. At max walk speed ≤5 m/s and clamped dt, max displacement per frame is ½ block — well within swept handling.

### Interaction

DDA voxel raycast (Amanatides–Woo) from camera every frame; result drives the cursor highlight (wireframe quad overlay on the targeted face).

- `break_`: target → `Air`; mark chunk + face-adjacent neighbors dirty.
- `place`: neighbor cell on hit face → selected block; reject if it would overlap player AABB.

## Persistence

### Layout

```
saves/<world-name>/
├── world.toml          # seed, sea level, spawn pos, version (= 1)
└── regions/
    ├── r.0.0.bin       # one file per 16×16 chunks
    ├── r.-1.0.bin
    └── ...
```

### Region file format

```
[ 0..4096 ]    header: 256 × u32, packed (offset_in_sectors: u24, len_sectors: u8)
[ 4096..  ]    chunk blobs (zstd-compressed bincode of PalettedChunk)
```

Slot index = `(local_cx << 4) | local_cz`. Sector size = 4096 bytes. Resized chunks get appended to EOF with header updated; v0 accepts fragmentation (later: `compact` subcommand).

Chunk blob = `bincode(PalettedChunk)` then `zstd`. Sky light is stored, not recomputed on load — fast loads.

### When we save

- **On modify**: chunk's `ChunkMeta::modified = true`, no immediate write.
- **On unload**: when a chunk leaves the load radius and is `modified`, write before evicting. Pure-generated unmodified chunks are not saved (we can regenerate from seed).
- **Clean shutdown**: flush all `modified` chunks.
- **Autosave**: every 60 s, drain modified chunks on the persistence thread.

### Failure handling

- File missing → fresh generate from seed (same code path as "never modified").
- Corrupt blob → log warning, regenerate from seed, mark `modified` so next save overwrites. Don't crash.
- Version mismatch in `world.toml` → refuse to load with clear error. v0 is version 1, no migrations.

### Threading

One dedicated persistence thread (not in the rayon pool). Main thread enqueues `Save(coord, snapshot)` / `Load(coord)`; results return on a channel. Snapshots use the palette-compressed form, which is also the on-disk shape — save is memcpy + zstd.

## Testing strategy

**Unit-tested exhaustively (pure modules):**

| Module | Approach |
|---|---|
| `voxel/coords.rs` | Round-trip tests; negative coords; chunk boundaries |
| `voxel/raycast.rs` | Table-driven: edges, inside-solid, miss, max-distance |
| `worldgen/` | Golden-file: hash of `(seed=42, coord=…)` chunks; ~5 fixtures |
| `lighting/` | Fixture chunks: torch falloff, sky through hole, cross-chunk |
| `mesher/greedy.rs` | Quad/vertex counts on known fixtures (e.g. solid 32³ → 6 quads) |
| `physics/sweep.rs` | Scenarios: fall onto floor, walk into wall, wall slide, jump |
| `persistence/region.rs` | Round-trip; double-open cache; deliberate corruption → regen flag |
| `jobs/` | Scheduling priority, backpressure |
| `render/cull.rs` | Frustum math (pure) |

**Not unit-tested:** `render/` (wgpu setup), ECS-glue systems (5–20 LOC of tested function calls), `main.rs`, `app.rs`.

**Integration smoke** (`tests/smoke.rs`): boots headless (null/llvmpipe wgpu backend), runs 60 sim frames on a fixed-seed world with scripted inputs, asserts no panics + expected player position.

**Posture:** TDD for pure modules. Render/ECS glue is test-after with smoke test as safety net.

## Implementation milestones

Ten milestones, each ending in a runnable binary with visibly more capability than the last. Stop-and-evaluate after every one.

| # | Name | Days | Outcome |
|---|---|---|---|
| M0 | Skeleton | 1 | Window opens; clear color; deps pinned |
| M1 | Render one hardcoded chunk | 2–3 | Naive culled mesh of a stone slab on screen |
| M2 | Free camera + player entity | 1 | hecs + scheduler; fly around the slab |
| M3 | World + worldgen | 2–3 | Streaming chunks generated on rayon; walk on terrain |
| M4 | Greedy meshing + AO | 2 | Stylized shading; mesh tests; FPS jumps |
| M5 | Lighting | 2–3 | Sky + torch BFS; day/night sky shader |
| M6 | Physics + walking | 2 | Swept AABB; jumping; `F` toggles flight |
| M7 | Block interaction | 1 | DDA raycast; cursor highlight; click to break/place |
| M8 | LOD | 2–3 | L1/L2 meshes; vast render distance feasible |
| M9 | Persistence | 2 | Region files; autosave; world survives restart |
| M10 | Polish | 1–2 | Flamegraph; HUD; v0.1 tag |

**Total: ~18–25 working days** solo.

**Ordering rationale:** each milestone makes the binary more fun to run (keeps motivation high); nothing is built before its first consumer (lighting after terrain, persistence last so iteration uses fresh worlds).

## Key design decisions and rationale

| Decision | Chosen | Alternative considered | Why |
|---|---|---|---|
| Engine layer | Custom `wgpu` + `winit` | Bevy, hybrid | User wants control + learning depth |
| ECS | `hecs` (game entities only) | Bevy ECS standalone, no ECS | Lightweight, no hidden scheduling; chunks deliberately excluded |
| Chunk size | 32³ | 16² × 128 (Minecraft) | Cubic is friendlier for 3D noise + caves + LOD downsampling |
| Meshing | Greedy + baked AO | Naive culled, binary greedy | 3× more code than naive, but pays off massively at vast render distance |
| LOD | 2× / 4× voxel downsampling | Marching cubes far field, none | Visual consistency; simpler than MC; required by vast distance |
| Lighting | Recompute-on-dirty | Incremental BFS | ~10× less code; user-visible delay is ~1 frame for single edits |
| Worldgen biomes | Single biome v0 | Multiple from day 1 | Architecture supports adding later without reshaping |
| Persistence | Minecraft-style region files (.mca-ish) | Per-chunk files, sled/sqlite | Sequential I/O; battle-tested shape; ~256 files vs ~22k |
| Save policy | Modified-only + autosave | Save everything | Worldgen deterministic — saving unmodified chunks is pure waste |
| Storage format | Palette-compressed (in RAM and on disk, same shape) | Dense everywhere | Vast render distance is infeasible without compression |
| Movement | Walking + flight toggle | Flight only, walking only | Walking is the game-feel; flight is necessary dev tool |
| Collision | Axis-by-axis sweep | Simultaneous resolution | Avoids well-known corner-snag bugs |
| Threading | rayon pool + dedicated persistence thread | All-rayon, single-threaded | Blocking I/O off the compute pool; main thread keeps render budget |

## Open questions

None blocking implementation. Items consciously deferred:

- LOD popping mitigation (geomorphing / screen-space dither) — wait until M8 reveals whether it's actually annoying
- Block selection UI (hotbar render vs corner text) — corner text in v0, hotbar with M10 polish if time
- Texture support — design accommodates it (vertex format has color + face), but no asset pipeline in v0
