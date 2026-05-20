# Render Perf Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-chunk CPU overhead in the renderer (per-frame `BindGroup` creation, three duplicated chunk-iteration loops, per-chunk wgpu buffers/textures) by replacing it with a GPU-driven path: one shared vertex/index arena, one `ChunkTable` SSBO, one `LightVolumeArray`, one `VisibilityList` per frame, and one `BindGroup` set per pass instead of one per chunk.

**Architecture:** Build four new standalone components with pure unit tests (`ChunkArena`, `ChunkTable`, `LightVolumeArray`, `VisibilityList`), then change the mesher to emit `chunk_id`-stamped split opaque/water streams, then change the shaders to read from the SSBO + texture array, then do an atomic switch in `Renderer` that rewires every chunk-related bind. Three cleanup tasks afterward (mid-frame depth copy, HUD ring buffer, MSAA configurability). Validation via screenshot harness + before/after CSV bench.

**Tech Stack:** Rust 2024, wgpu 23 (no `MULTI_DRAW_INDIRECT` dependency — CPU-iterated draws against a shared arena gives most of the win), WGSL, screenshot golden tests under `tests/screenshots/`.

**Spec:** `docs/superpowers/specs/2026-05-20-render-perf-rework-design.md`

---

## File Structure

**New files:**
- `src/render/arena.rs` — `ChunkArena` (vertex+index suballocation) + private free-list type
- `src/render/chunk_table.rs` — `ChunkTable` (slot bookkeeping + SSBO mirror)
- `src/render/light_array.rs` — `LightVolumeArray` (texture-array layer allocator + uploads)
- `src/render/visibility.rs` — `VisibilityList` (pure frustum+distance+LOD culling)
- `src/render/passes/mod.rs` — re-exports
- `src/render/passes/opaque.rs` — opaque pass encoding (state setup + draw loop over visibility list)
- `src/render/passes/water.rs` — water pass encoding
- `src/render/passes/reflection.rs` — reflection pass encoding
- `src/render/passes/hud.rs` — HUD pass encoding (moved from `render::mod`)

**Existing files modified:**
- `src/render/mod.rs` — shrinks from ~1540 lines to ~250; loses per-chunk iteration loops and `chunk_meshes`/`chunk_lights` `HashMap`s
- `src/render/mesh.rs` — `upload_mesh` returns an arena allocation, not a `wgpu::Buffer` pair
- `src/render/light_volume.rs` — removed; functionality moved to `light_array.rs`
- `src/render/camera.rs` — `make_chunk_bind_group_layout` rewritten (storage buffer + texture array instead of uniform buffer + 3D texture); `ChunkUniform` struct deleted
- `src/render/pipelines/opaque.rs` — new bind-group layout
- `src/render/pipelines/water.rs` — same
- `src/render/gpu.rs` — `MSAA_SAMPLES` becomes a `&self` field on `Gpu`, set from a config arg
- `src/mesher/mod.rs` — `ChunkMesh` split into opaque/water sub-meshes; `Vertex` gains `chunk_id: u16`
- `src/mesher/greedy.rs` — emits split streams, stamps `chunk_id` from a caller-supplied argument
- `src/mesher/naive.rs` — same
- `src/mesher/lod.rs` — same
- `assets/shaders/opaque.wgsl` — `chunk: ChunkUniform` → `chunk_table: array<ChunkData>`; `light_volume: texture_3d<f32>` → `light_volumes: texture_3d_array<f32>`
- `assets/shaders/water.wgsl` — same shader-binding changes
- `src/main.rs` — new `--bench <frames>` flag for headless perf capture

---

## Phase 0: Baseline

### Task 0: Capture today's perf baseline

Before any changes, capture a CSV from `main` so we can compare. The existing `--profile` flag + `--uncapped` does almost everything we need; we just need a headless way to run a known scene for N frames without a human at the keyboard.

**Files:**
- Modify: `src/main.rs` — add `--bench <frames>` flag
- Modify: `src/app.rs` — bench-mode exit-after-N-frames check inside `step()`
- Create: `docs/superpowers/bench/baseline.csv` (captured output, committed)
- Create: `docs/superpowers/bench/README.md` (how to reproduce)

- [ ] **Step 1: Add `--bench <frames>` flag parsing in `main.rs`**

Find the CLI arg loop in `src/main.rs` (the `match` block around line 86). Add a new arm:

```rust
"--bench" => {
    let n: u32 = it
        .next()
        .expect("--bench requires a frame count")
        .parse()
        .expect("--bench frame count must be a non-negative integer");
    bench_frames = Some(n);
}
```

Declare `let mut bench_frames: Option<u32> = None;` alongside the other CLI-state locals. When `bench_frames.is_some()`, force `present_mode = wgpu::PresentMode::Immediate` and require `--profile <path>` (assert + helpful error otherwise — bench mode without a CSV output is meaningless).

Thread `bench_frames` into `App::new_with_*`.

- [ ] **Step 2: Add bench-mode termination in `app.rs`**

In `AppState`, add `bench_frames: Option<u32>` and `frames_run: u32` fields. At the end of `step()` (after `present()`), if `bench_frames` is set and `frames_run >= bench_frames`:

```rust
if let Some(n) = self.bench_frames {
    self.frames_run += 1;
    if self.frames_run >= n {
        // Force the profiler to flush by dropping it, then exit.
        self.profiler.take();
        std::process::exit(0);
    }
}
```

- [ ] **Step 3: Run the bench against `main` to capture baseline**

Run:
```bash
cargo run --profile profiling -- \
  --uncapped \
  --profile docs/superpowers/bench/baseline.csv \
  --bench 600
```

Expected: ~10 seconds of execution; ~600 rows in the CSV; render times in the multi-millisecond range. If the run hangs, the bench-mode termination didn't fire — check the `frames_run` increment is reached.

- [ ] **Step 4: Document the bench command**

Create `docs/superpowers/bench/README.md`:

```markdown
# Render bench

Headless perf capture against the renderer.

```
cargo run --profile profiling -- \
  --uncapped \
  --profile docs/superpowers/bench/<name>.csv \
  --bench 600
```

Captures 600 frames at uncapped present rate. The CSV is in the
`profiler::Profiler` format documented in `src/profiler.rs`. Key columns
for render perf: `render` (µs), `draw_calls`, `chunks_rendered`,
`work_ms`.

`baseline.csv` is captured against `main` before any rework.
```

- [ ] **Step 5: Commit**

```bash
git add src/main.rs src/app.rs docs/superpowers/bench/
git commit -m "bench: --bench flag + main-branch baseline CSV"
```

---

## Phase 1: Standalone components (unit-tested in isolation)

### Task 1: `ChunkArena` — shared vertex/index arena

**Files:**
- Create: `src/render/arena.rs`
- Modify: `src/render/mod.rs` (add `pub mod arena;`)

- [ ] **Step 1: Write failing tests for the pure free-list allocator**

In `src/render/arena.rs`, write the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_sequential_offsets_at_start() {
        let mut fl = FreeList::new(1024);
        assert_eq!(fl.alloc(100), Some(0));
        assert_eq!(fl.alloc(50), Some(100));
        assert_eq!(fl.alloc(200), Some(150));
    }

    #[test]
    fn alloc_returns_none_when_exhausted() {
        let mut fl = FreeList::new(100);
        assert_eq!(fl.alloc(60), Some(0));
        assert_eq!(fl.alloc(50), None);
        assert_eq!(fl.alloc(40), Some(60));
    }

    #[test]
    fn free_then_alloc_reuses_slot() {
        let mut fl = FreeList::new(1024);
        let a = fl.alloc(100).unwrap();
        let b = fl.alloc(100).unwrap();
        let c = fl.alloc(100).unwrap();
        fl.free(b, 100);
        assert_eq!(fl.alloc(100), Some(b)); // best-fit reuse
        let _ = (a, c);
    }

    #[test]
    fn adjacent_frees_coalesce() {
        let mut fl = FreeList::new(1024);
        let a = fl.alloc(100).unwrap();
        let b = fl.alloc(100).unwrap();
        let c = fl.alloc(100).unwrap();
        fl.free(a, 100);
        fl.free(b, 100);
        // After coalescing, the 200-byte hole should satisfy a 200-byte alloc.
        assert_eq!(fl.alloc(200), Some(a));
        let _ = c;
    }

    #[test]
    fn grow_extends_capacity() {
        let mut fl = FreeList::new(100);
        fl.grow(200); // now 300 cap
        assert_eq!(fl.alloc(250), Some(0));
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test --lib render::arena
```

Expected: FAIL with "cannot find type `FreeList`".

- [ ] **Step 3: Implement `FreeList`**

```rust
//! Vertex/index arena for chunk meshes. One big `wgpu::Buffer` per stream;
//! suballocation via a coalescing free list.

use wgpu::util::DeviceExt;

use crate::mesher::Vertex;

/// Best-fit free-list allocator over a byte range `[0, capacity)`.
/// Adjacent free slots are coalesced on `free`. All sizes/offsets are
/// in bytes — callers convert to vertex/index counts as needed.
struct FreeList {
    capacity: u64,
    // Sorted by offset. Each entry is `(offset, length)` of a free block.
    free: Vec<(u64, u64)>,
}

impl FreeList {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
        }
    }

    /// Best-fit search. Returns the offset of an allocation of `len`
    /// bytes, or `None` if no block is large enough.
    fn alloc(&mut self, len: u64) -> Option<u64> {
        let mut best: Option<(usize, u64)> = None;
        for (i, &(_off, blen)) in self.free.iter().enumerate() {
            if blen >= len {
                let waste = blen - len;
                if best.map(|(_, w)| waste < w).unwrap_or(true) {
                    best = Some((i, waste));
                }
            }
        }
        let (idx, _) = best?;
        let (off, blen) = self.free[idx];
        if blen == len {
            self.free.remove(idx);
        } else {
            self.free[idx] = (off + len, blen - len);
        }
        Some(off)
    }

    /// Insert `(offset, len)` back into the free list and coalesce with
    /// any adjacent neighbours.
    fn free(&mut self, offset: u64, len: u64) {
        // Insert at the right sorted position.
        let idx = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(idx, (offset, len));
        // Coalesce with following block.
        if idx + 1 < self.free.len() {
            let (next_off, next_len) = self.free[idx + 1];
            if offset + len == next_off {
                self.free[idx] = (offset, len + next_len);
                self.free.remove(idx + 1);
            }
        }
        // Coalesce with previous block.
        if idx > 0 {
            let (prev_off, prev_len) = self.free[idx - 1];
            let (cur_off, cur_len) = self.free[idx];
            if prev_off + prev_len == cur_off {
                self.free[idx - 1] = (prev_off, prev_len + cur_len);
                self.free.remove(idx);
            }
        }
    }

    fn grow(&mut self, extra: u64) {
        let end = self.capacity;
        self.capacity += extra;
        // The new tail might be adjacent to the last free block.
        if let Some(&(off, len)) = self.free.last() {
            if off + len == end {
                let last = self.free.len() - 1;
                self.free[last] = (off, len + extra);
                return;
            }
        }
        self.free.push((end, extra));
    }
}
```

Add `pub mod arena;` to `src/render/mod.rs`.

- [ ] **Step 4: Run tests, watch them pass**

```bash
cargo test --lib render::arena
```

Expected: 5 tests pass.

- [ ] **Step 5: Add the `ChunkArena` wgpu wrapper**

Add to the same file, below `FreeList`:

```rust
const INITIAL_VERT_CAPACITY_BYTES: u64 = 32 * 1024 * 1024;
const INITIAL_INDEX_CAPACITY_BYTES: u64 = 8 * 1024 * 1024;
const GROWTH_FACTOR: u64 = 3; // multiplied by 2 → ×1.5

/// A handle to a contiguous range of vertices + indices inside `ChunkArena`.
/// `vert_offset` and `index_offset` are in *element counts*, not bytes —
/// callers pass them straight to `set_vertex_buffer` / `draw_indexed`.
#[derive(Debug, Clone, Copy)]
pub struct ChunkAllocation {
    pub vert_offset: u32,
    pub vert_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

/// One vertex + one index `wgpu::Buffer`, both suballocated by a
/// coalescing free list. Replaces per-chunk `wgpu::Buffer` pairs.
pub struct ChunkArena {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    v_free: FreeList,
    i_free: FreeList,
}

impl ChunkArena {
    pub fn new(device: &wgpu::Device) -> Self {
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk-arena-vbuf"),
            size: INITIAL_VERT_CAPACITY_BYTES,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk-arena-ibuf"),
            size: INITIAL_INDEX_CAPACITY_BYTES,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            vbuf,
            ibuf,
            v_free: FreeList::new(INITIAL_VERT_CAPACITY_BYTES),
            i_free: FreeList::new(INITIAL_INDEX_CAPACITY_BYTES),
        }
    }

    pub fn vbuf(&self) -> &wgpu::Buffer { &self.vbuf }
    pub fn ibuf(&self) -> &wgpu::Buffer { &self.ibuf }

    /// Allocate space for `verts` + `indices`, write them through `queue`,
    /// return the allocation handle. Returns `None` if growth would be
    /// required (caller decides whether to retry after `grow_if_needed`).
    pub fn alloc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        verts: &[Vertex],
        indices: &[u32],
    ) -> Option<ChunkAllocation> {
        let v_bytes = std::mem::size_of_val(verts) as u64;
        let i_bytes = std::mem::size_of_val(indices) as u64;
        let v_off = match self.v_free.alloc(v_bytes) {
            Some(o) => o,
            None => {
                self.grow_v(device, queue, v_bytes);
                self.v_free.alloc(v_bytes)?
            }
        };
        let i_off = match self.i_free.alloc(i_bytes) {
            Some(o) => o,
            None => {
                self.grow_i(device, queue, i_bytes);
                self.i_free.alloc(i_bytes)?
            }
        };
        queue.write_buffer(&self.vbuf, v_off, bytemuck::cast_slice(verts));
        queue.write_buffer(&self.ibuf, i_off, bytemuck::cast_slice(indices));
        Some(ChunkAllocation {
            vert_offset: (v_off / std::mem::size_of::<Vertex>() as u64) as u32,
            vert_count: verts.len() as u32,
            index_offset: (i_off / 4) as u32,
            index_count: indices.len() as u32,
        })
    }

    pub fn free(&mut self, alloc: ChunkAllocation) {
        let v_bytes = alloc.vert_count as u64 * std::mem::size_of::<Vertex>() as u64;
        let i_bytes = alloc.index_count as u64 * 4;
        let v_off = alloc.vert_offset as u64 * std::mem::size_of::<Vertex>() as u64;
        let i_off = alloc.index_offset as u64 * 4;
        self.v_free.free(v_off, v_bytes);
        self.i_free.free(i_off, i_bytes);
    }

    fn grow_v(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, need: u64) {
        let old_cap = self.v_free.capacity;
        let new_cap = (old_cap * GROWTH_FACTOR / 2).max(old_cap + need);
        let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk-arena-vbuf-grown"),
            size: new_cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Copy old contents → new buffer.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.vbuf, 0, &new_buf, 0, old_cap);
        queue.submit(std::iter::once(enc.finish()));
        self.vbuf = new_buf;
        self.v_free.grow(new_cap - old_cap);
    }

    fn grow_i(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, need: u64) {
        let old_cap = self.i_free.capacity;
        let new_cap = (old_cap * GROWTH_FACTOR / 2).max(old_cap + need);
        let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk-arena-ibuf-grown"),
            size: new_cap,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.ibuf, 0, &new_buf, 0, old_cap);
        queue.submit(std::iter::once(enc.finish()));
        self.ibuf = new_buf;
        self.i_free.grow(new_cap - old_cap);
    }
}
```

Note: the `Vertex` import will need to come AFTER Task 5 adds `chunk_id` — for now `use crate::mesher::Vertex;` will compile against today's 16-byte version. Size assertions don't matter; the arena is generic over the byte size.

- [ ] **Step 6: Build to confirm compile**

```bash
cargo build --lib
```

Expected: clean build (component is added but not referenced yet from the renderer).

- [ ] **Step 7: Commit**

```bash
git add src/render/arena.rs src/render/mod.rs
git commit -m "feat(render): ChunkArena — shared vertex/index arena with coalescing free list"
```

---

### Task 2: `ChunkTable` — chunk-data SSBO

**Files:**
- Create: `src/render/chunk_table.rs`
- Modify: `src/render/mod.rs` (add `pub mod chunk_table;`)

- [ ] **Step 1: Write failing tests for the slot allocator**

In `src/render/chunk_table.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::coords::ChunkCoord;
    use glam::IVec3;

    fn coord(x: i32, y: i32, z: i32) -> ChunkCoord { ChunkCoord(IVec3::new(x, y, z)) }

    #[test]
    fn register_returns_unique_ids() {
        let mut t = SlotBook::new();
        let a = t.register(coord(0, 0, 0));
        let b = t.register(coord(1, 0, 0));
        assert_ne!(a, b);
    }

    #[test]
    fn register_same_coord_returns_same_id() {
        let mut t = SlotBook::new();
        let a = t.register(coord(0, 0, 0));
        let b = t.register(coord(0, 0, 0));
        assert_eq!(a, b);
    }

    #[test]
    fn unregister_then_register_reuses_slot() {
        let mut t = SlotBook::new();
        let a = t.register(coord(0, 0, 0));
        t.unregister(coord(0, 0, 0));
        let b = t.register(coord(1, 0, 0));
        assert_eq!(a, b);
    }

    #[test]
    fn dirty_set_tracks_changes() {
        let mut t = SlotBook::new();
        let a = t.register(coord(0, 0, 0));
        t.set_origin(a, [16.0, 0.0, 16.0]);
        t.set_light_layer(a, 7);
        let dirty: Vec<_> = t.drain_dirty().collect();
        assert_eq!(dirty, vec![a]);
        // Drained = empty.
        assert!(t.drain_dirty().next().is_none());
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test --lib render::chunk_table
```

Expected: FAIL with "cannot find type `SlotBook`".

- [ ] **Step 3: Implement `SlotBook` + `ChunkTable`**

```rust
//! Per-chunk data living on the GPU as a single storage buffer. Replaces
//! the per-chunk `ChunkUniform` + per-chunk bind group. Indexed by a
//! stable `ChunkId` that's baked into the mesh's vertex stream.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::voxel::coords::ChunkCoord;

pub type ChunkId = u32;

/// Empty-slot marker for the SSBO. `flags == 0` means the slot is
/// inactive — the visibility builder filters it out.
const FLAG_HAS_OPAQUE: u32 = 1 << 0;
const FLAG_HAS_WATER:  u32 = 1 << 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default, PartialEq)]
struct GpuSlot {
    origin:      [f32; 4],   // xyz = world origin, w = unused
    light_layer: u32,
    flags:       u32,        // bit 0: has opaque mesh, bit 1: has water mesh
    _pad:        [u32; 2],
}

/// Pure CPU bookkeeping for chunk slots. The GPU mirror lives in
/// `ChunkTable` (below) and reads/writes from here at frame start.
pub(super) struct SlotBook {
    slots: Vec<GpuSlot>,
    free: Vec<ChunkId>,
    by_coord: HashMap<ChunkCoord, ChunkId>,
    dirty: Vec<ChunkId>,
}

impl SlotBook {
    pub fn new() -> Self {
        Self { slots: Vec::new(), free: Vec::new(), by_coord: HashMap::new(), dirty: Vec::new() }
    }

    pub fn register(&mut self, coord: ChunkCoord) -> ChunkId {
        if let Some(&id) = self.by_coord.get(&coord) {
            return id;
        }
        let id = match self.free.pop() {
            Some(id) => {
                self.slots[id as usize] = GpuSlot::default();
                id
            }
            None => {
                let id = self.slots.len() as ChunkId;
                self.slots.push(GpuSlot::default());
                id
            }
        };
        self.by_coord.insert(coord, id);
        self.dirty.push(id);
        id
    }

    pub fn unregister(&mut self, coord: ChunkCoord) {
        if let Some(id) = self.by_coord.remove(&coord) {
            self.slots[id as usize] = GpuSlot::default(); // flags = 0 → filtered
            self.free.push(id);
            self.dirty.push(id);
        }
    }

    pub fn lookup(&self, coord: ChunkCoord) -> Option<ChunkId> {
        self.by_coord.get(&coord).copied()
    }

    pub fn set_origin(&mut self, id: ChunkId, origin: [f32; 3]) {
        let s = &mut self.slots[id as usize];
        s.origin = [origin[0], origin[1], origin[2], 0.0];
        self.dirty.push(id);
    }

    pub fn set_light_layer(&mut self, id: ChunkId, layer: u32) {
        self.slots[id as usize].light_layer = layer;
        self.dirty.push(id);
    }

    pub fn set_has_opaque(&mut self, id: ChunkId, has: bool) {
        let s = &mut self.slots[id as usize];
        if has { s.flags |= FLAG_HAS_OPAQUE } else { s.flags &= !FLAG_HAS_OPAQUE }
        self.dirty.push(id);
    }

    pub fn set_has_water(&mut self, id: ChunkId, has: bool) {
        let s = &mut self.slots[id as usize];
        if has { s.flags |= FLAG_HAS_WATER } else { s.flags &= !FLAG_HAS_WATER }
        self.dirty.push(id);
    }

    pub fn slot(&self, id: ChunkId) -> &GpuSlot { &self.slots[id as usize] }
    pub fn slots(&self) -> &[GpuSlot] { &self.slots }

    /// Drain the dirty list — returns each ID exactly once even if
    /// it was pushed multiple times since the last drain.
    pub fn drain_dirty(&mut self) -> impl Iterator<Item = ChunkId> + '_ {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        self.dirty.drain(..)
    }
}

/// GPU-side mirror of `SlotBook`. Lives in a single storage buffer
/// bound at group 1 (replacing today's per-chunk `ChunkUniform` bind).
pub struct ChunkTable {
    book: SlotBook,
    gpu_buf: wgpu::Buffer,
    capacity: usize, // number of slots the buffer was sized for
}

const SLOT_BYTES: u64 = std::mem::size_of::<GpuSlot>() as u64;

impl ChunkTable {
    pub fn new(device: &wgpu::Device, initial_capacity: usize) -> Self {
        let gpu_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk-table-ssbo"),
            size: SLOT_BYTES * initial_capacity as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { book: SlotBook::new(), gpu_buf, capacity: initial_capacity }
    }

    pub fn buffer(&self) -> &wgpu::Buffer { &self.gpu_buf }

    pub fn register(&mut self, coord: ChunkCoord) -> ChunkId { self.book.register(coord) }
    pub fn unregister(&mut self, coord: ChunkCoord) { self.book.unregister(coord) }
    pub fn lookup(&self, coord: ChunkCoord) -> Option<ChunkId> { self.book.lookup(coord) }
    pub fn set_origin(&mut self, id: ChunkId, o: [f32; 3]) { self.book.set_origin(id, o) }
    pub fn set_light_layer(&mut self, id: ChunkId, l: u32) { self.book.set_light_layer(id, l) }
    pub fn set_has_opaque(&mut self, id: ChunkId, h: bool) { self.book.set_has_opaque(id, h) }
    pub fn set_has_water(&mut self, id: ChunkId, h: bool) { self.book.set_has_water(id, h) }
    pub fn slots(&self) -> &[GpuSlot] { self.book.slots() }

    /// Flush every dirty slot to the GPU buffer. Grows the buffer if
    /// the slot count outgrew the capacity. Called once per frame
    /// before any pass encoding.
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let needed = self.book.slots.len();
        if needed > self.capacity {
            // Grow with ×1.5.
            let new_cap = (self.capacity * 3 / 2).max(needed);
            self.gpu_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chunk-table-ssbo-grown"),
                size: SLOT_BYTES * new_cap as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_cap;
            // Full rewrite — every slot is "dirty" relative to the new buffer.
            queue.write_buffer(&self.gpu_buf, 0, bytemuck::cast_slice(self.book.slots()));
            self.book.dirty.clear();
            return;
        }
        for id in self.book.drain_dirty() {
            let off = id as u64 * SLOT_BYTES;
            let slot = &self.book.slots[id as usize];
            queue.write_buffer(&self.gpu_buf, off, bytemuck::bytes_of(slot));
        }
    }
}
```

Add `pub mod chunk_table;` to `src/render/mod.rs`.

- [ ] **Step 4: Run tests, watch them pass**

```bash
cargo test --lib render::chunk_table
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/chunk_table.rs src/render/mod.rs
git commit -m "feat(render): ChunkTable — single SSBO replacing per-chunk uniform bind"
```

---

### Task 3: `LightVolumeArray` — texture-array layer allocator

**Files:**
- Create: `src/render/light_array.rs`
- Modify: `src/render/mod.rs` (add `pub mod light_array;`)

- [ ] **Step 1: Write failing tests for the layer allocator**

In `src/render/light_array.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_sequential_layers() {
        let mut a = LayerAllocator::new(8);
        assert_eq!(a.alloc(), Some(0));
        assert_eq!(a.alloc(), Some(1));
        assert_eq!(a.alloc(), Some(2));
    }

    #[test]
    fn alloc_returns_none_when_full() {
        let mut a = LayerAllocator::new(2);
        a.alloc();
        a.alloc();
        assert_eq!(a.alloc(), None);
    }

    #[test]
    fn freed_layers_are_reused() {
        let mut a = LayerAllocator::new(4);
        let l0 = a.alloc().unwrap();
        let l1 = a.alloc().unwrap();
        a.free(l0);
        assert_eq!(a.alloc(), Some(l0));
        let _ = l1;
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test --lib render::light_array
```

Expected: FAIL with "cannot find type `LayerAllocator`".

- [ ] **Step 3: Implement `LayerAllocator` + `LightVolumeArray`**

```rust
//! Per-chunk 3D light volumes packed into a single texture array.
//! Replaces today's `HashMap<ChunkCoord, wgpu::Texture>` of one
//! `texture_3d<f32>` per chunk.

use crate::voxel::chunk::LIGHT_VOLUME_DIM;

/// Format of each per-chunk light volume slice. R = block-light R,
/// G = block-light G, B = block-light B, A = sky-light. Matches the
/// existing `light_volume.rs` format.
pub const LIGHT_VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Pure layer-index free list. Test-only; the wgpu wrapper owns one.
struct LayerAllocator {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
}

impl LayerAllocator {
    fn new(capacity: u32) -> Self { Self { capacity, next: 0, free: Vec::new() } }
    fn alloc(&mut self) -> Option<u32> {
        if let Some(l) = self.free.pop() { return Some(l); }
        if self.next < self.capacity {
            let l = self.next;
            self.next += 1;
            return Some(l);
        }
        None
    }
    fn free(&mut self, layer: u32) { self.free.push(layer); }
}

/// One `wgpu::Texture` of dimension `D3` with `depth_or_array_layers`
/// equal to the maximum loaded-chunk count, holding every chunk's
/// light volume as an array slice.
pub struct LightVolumeArray {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    alloc: LayerAllocator,
    dim: u32,
}

impl LightVolumeArray {
    /// `max_chunks` is the maximum number of simultaneously-loaded
    /// chunks the renderer expects. The texture is sized to this at
    /// construction; growing would require re-uploading every layer
    /// so we pick a generous fixed cap.
    pub fn new(device: &wgpu::Device, max_chunks: u32) -> Self {
        let dim = LIGHT_VOLUME_DIM;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light-volume-array"),
            size: wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: dim },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LIGHT_VOLUME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // NOTE: wgpu 23 does not expose 3D-texture-arrays directly.
        // We instead pack chunks vertically into a single 3D texture
        // of size (dim, dim, dim * max_chunks). Layer N occupies
        // z-range `[N*dim, (N+1)*dim)`. The shader receives the layer
        // index and offsets its sample coordinate by that range.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light-volume-tower"),
            size: wgpu::Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: dim * max_chunks,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LIGHT_VOLUME_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light-volume-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            texture,
            view,
            sampler,
            alloc: LayerAllocator::new(max_chunks),
            dim,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView { &self.view }
    pub fn sampler(&self) -> &wgpu::Sampler { &self.sampler }
    pub fn dim(&self) -> u32 { self.dim }

    pub fn alloc_layer(&mut self) -> Option<u32> { self.alloc.alloc() }
    pub fn free_layer(&mut self, layer: u32) { self.alloc.free(layer) }

    /// Upload a chunk's light blob into its layer slot. `blob` must be
    /// exactly `dim * dim * dim * 4` bytes (Rgba8Unorm).
    pub fn upload(&self, queue: &wgpu::Queue, layer: u32, blob: &[u8]) {
        let dim = self.dim;
        let expected = (dim * dim * dim * 4) as usize;
        debug_assert_eq!(blob.len(), expected, "light blob wrong size");
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: layer * dim },
                aspect: wgpu::TextureAspect::All,
            },
            blob,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(dim * 4),
                rows_per_image: Some(dim),
            },
            wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: dim },
        );
    }
}
```

Add `pub mod light_array;` to `src/render/mod.rs`. Add the `LIGHT_VOLUME_DIM` constant to `src/voxel/chunk.rs` (it's `33` based on `light_volume.rs`):

```rust
// In src/voxel/chunk.rs, near the other dim constants:
pub const LIGHT_VOLUME_DIM: u32 = 33; // CHUNK_DIM + 1 for boundary samples
```

If a constant already exists in `light_volume.rs`, re-export from there instead.

- [ ] **Step 4: Run tests, watch them pass**

```bash
cargo test --lib render::light_array
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/light_array.rs src/render/mod.rs src/voxel/chunk.rs
git commit -m "feat(render): LightVolumeArray — one tower-packed 3D texture for all chunks"
```

---

### Task 4: `VisibilityList` — single per-frame cull pass

**Files:**
- Create: `src/render/visibility.rs`
- Modify: `src/render/mod.rs` (add `pub mod visibility;`)

- [ ] **Step 1: Write failing tests**

In `src/render/visibility.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};

    /// A frustum that accepts everything (each plane normal at the origin).
    fn identity_frustum() -> [Vec4; 6] {
        [Vec4::new(0.0, 0.0, 0.0, 1.0); 6]
    }

    #[test]
    fn empty_slots_produce_empty_list() {
        let mut v = VisibilityList::new();
        v.build(&[], Vec3::ZERO, &identity_frustum(), &LodThresholds::default());
        assert_eq!(v.entries().len(), 0);
    }

    #[test]
    fn slot_with_no_flags_is_filtered() {
        let mut v = VisibilityList::new();
        let slots = vec![SlotInput { origin: Vec3::ZERO, flags: 0, lods: [LodInput::missing(); 3] }];
        v.build(&slots, Vec3::ZERO, &identity_frustum(), &LodThresholds::default());
        assert_eq!(v.entries().len(), 0);
    }

    #[test]
    fn near_chunk_picks_lod0() {
        let mut v = VisibilityList::new();
        let slots = vec![SlotInput {
            origin: Vec3::new(0.0, 0.0, 0.0),
            flags: 0b01, // has opaque
            lods: [
                LodInput::present(100, 200),
                LodInput::missing(),
                LodInput::missing(),
            ],
        }];
        v.build(&slots, Vec3::new(16.0, 16.0, 16.0), &identity_frustum(), &LodThresholds::default());
        assert_eq!(v.entries().len(), 1);
        assert_eq!(v.entries()[0].lod, 0);
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test --lib render::visibility
```

Expected: FAIL with "cannot find type `VisibilityList`".

- [ ] **Step 3: Implement `VisibilityList`**

```rust
//! Per-frame visibility / cull / LOD pass. Reads slot inputs from
//! `ChunkTable`'s slot metadata + the renderer's per-chunk LOD
//! allocation table, writes a packed `Vec<DrawEntry>` the passes
//! walk in order.
//!
//! Pure CPU; zero allocations after the first build (the entry `Vec`
//! is reused).

use glam::{Vec3, Vec4};

use crate::render::chunk_table::ChunkId;

#[derive(Debug, Clone, Copy)]
pub struct LodInput {
    pub index_offset: u32,
    pub index_count: u32,
}

impl LodInput {
    pub const fn missing() -> Self { Self { index_offset: 0, index_count: 0 } }
    pub const fn present(index_offset: u32, index_count: u32) -> Self {
        Self { index_offset, index_count }
    }
    pub fn is_present(&self) -> bool { self.index_count > 0 }
}

#[derive(Debug, Clone, Copy)]
pub struct SlotInput {
    pub origin: Vec3,
    pub flags: u32,       // bit 0: has opaque, bit 1: has water
    pub lods: [LodInput; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct LodThresholds {
    pub lod1: f32,
    pub lod2: f32,
}

impl Default for LodThresholds {
    fn default() -> Self { Self { lod1: 6.0 * 32.0, lod2: 12.0 * 32.0 } }
}

#[derive(Debug, Clone, Copy)]
pub struct DrawEntry {
    pub chunk_id: ChunkId,
    pub lod: u8,
    pub index_offset: u32,
    pub index_count: u32,
}

pub struct VisibilityList {
    entries: Vec<DrawEntry>,
}

impl VisibilityList {
    pub fn new() -> Self { Self { entries: Vec::with_capacity(1024) } }
    pub fn entries(&self) -> &[DrawEntry] { &self.entries }

    /// `flag_mask` is the bitmask the caller wants to draw. Opaque pass
    /// passes `0b01`; water pass passes `0b10`.
    pub fn build_filtered(
        &mut self,
        slots: &[SlotInput],
        eye: Vec3,
        frustum: &[Vec4; 6],
        lods: &LodThresholds,
        flag_mask: u32,
    ) {
        self.entries.clear();
        const CULL_DISTANCE: f32 = 600.0 + 28.0;
        let cull_sq = CULL_DISTANCE * CULL_DISTANCE;

        for (chunk_id, slot) in slots.iter().enumerate() {
            if slot.flags & flag_mask == 0 { continue; }
            let chunk_min = slot.origin;
            let chunk_max = chunk_min + Vec3::splat(32.0);
            let center = chunk_min + Vec3::splat(16.0);
            let d_sq = (center - eye).length_squared();
            if d_sq > cull_sq { continue; }
            if !aabb_in_frustum(frustum, chunk_min, chunk_max) { continue; }

            let preferred = if d_sq < lods.lod1 * lods.lod1 { 0 }
                            else if d_sq < lods.lod2 * lods.lod2 { 1 }
                            else { 2 };
            let chosen = slot.lods[preferred].is_present().then_some(preferred)
                .or_else(|| slot.lods.iter().position(|l| l.is_present()));
            if let Some(lod) = chosen {
                let l = &slot.lods[lod];
                self.entries.push(DrawEntry {
                    chunk_id: chunk_id as ChunkId,
                    lod: lod as u8,
                    index_offset: l.index_offset,
                    index_count: l.index_count,
                });
            }
        }
    }

    /// Convenience wrapper: build both opaque and water masks. The
    /// test helper.
    pub fn build(
        &mut self,
        slots: &[SlotInput],
        eye: Vec3,
        frustum: &[Vec4; 6],
        lods: &LodThresholds,
    ) {
        self.build_filtered(slots, eye, frustum, lods, 0b11);
    }
}

fn aabb_in_frustum(planes: &[Vec4; 6], min: Vec3, max: Vec3) -> bool {
    for p in planes {
        let n = p.truncate();
        let pv = Vec3::new(
            if n.x >= 0.0 { max.x } else { min.x },
            if n.y >= 0.0 { max.y } else { min.y },
            if n.z >= 0.0 { max.z } else { min.z },
        );
        if n.dot(pv) + p.w < 0.0 { return false; }
    }
    true
}
```

Add `pub mod visibility;` to `src/render/mod.rs`.

- [ ] **Step 4: Run tests, watch them pass**

```bash
cargo test --lib render::visibility
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/render/visibility.rs src/render/mod.rs
git commit -m "feat(render): VisibilityList — single cull pass shared across all passes"
```

---

## Phase 2: Format changes

### Task 5: Mesher — split opaque/water + add `chunk_id` to `Vertex`

**Files:**
- Modify: `src/mesher/mod.rs` (Vertex layout + ChunkMesh split)
- Modify: `src/mesher/greedy.rs` (emit split streams, stamp chunk_id)
- Modify: `src/mesher/naive.rs` (same)
- Modify: `src/mesher/lod.rs` (same)
- Modify: `src/render/pipelines/opaque.rs` (vertex layout)
- Modify: `src/render/pipelines/water.rs` (vertex layout)

- [ ] **Step 1: Write a failing test for the new `Vertex` size + `chunk_id` field**

Add to `src/mesher/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vertex_is_20_bytes() {
        assert_eq!(std::mem::size_of::<Vertex>(), 20);
    }
    #[test]
    fn vertex_chunk_id_round_trips() {
        let v = Vertex {
            pos: [0, 0, 0], ao: 0, color: [0; 4], normal_face: 0, light: 0,
            chunk_id: 1234, _pad: 0,
            tile_index: 0, u_tile: 0, v_tile: 0, _pad2: 0,
        };
        assert_eq!(v.chunk_id, 1234);
    }
    #[test]
    fn chunk_mesh_has_split_streams() {
        let m = ChunkMesh::empty();
        assert_eq!(m.opaque_vertices.len(), 0);
        assert_eq!(m.water_vertices.len(), 0);
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test --lib mesher
```

Expected: FAIL — `Vertex` is 16 bytes, has no `chunk_id` field, `ChunkMesh` has no `opaque_vertices`.

- [ ] **Step 3: Update `Vertex` and `ChunkMesh` in `src/mesher/mod.rs`**

Replace the `Vertex` struct + `ChunkMesh` struct in `src/mesher/mod.rs` with:

```rust
/// 20-byte packed per-vertex payload.
///
/// Replaces the 16-byte format that included a 2-byte `_pad`. The new
/// `chunk_id` field is consumed by the vertex shader as an index into
/// the `ChunkTable` storage buffer (which holds per-chunk origin +
/// light-volume layer). This removes the need for per-chunk uniform
/// bind groups.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [u8; 3],
    pub ao: u8,
    pub color: [u8; 4],
    pub normal_face: u8,
    pub light: u8,
    pub chunk_id: u16,      // NEW: indexes into ChunkTable
    pub tile_index: u8,
    pub u_tile: u8,
    pub v_tile: u8,
    pub _pad2: u8,
    pub _pad3: [u8; 4],     // pad to 20 bytes (vec4-aligned)
}
```

And replace `ChunkMesh`:

```rust
/// CPU-side mesh data destined for the GPU. Owned by the worker thread
/// that built it; ownership transfers to the main thread when uploaded.
///
/// Opaque and water faces live in separate streams so the water pipeline
/// doesn't have to vertex-shade every opaque vert in the chunk just to
/// `discard` it at the fragment stage.
pub struct ChunkMesh {
    pub opaque_vertices: Vec<Vertex>,
    pub opaque_indices: Vec<u32>,
    pub water_vertices: Vec<Vertex>,
    pub water_indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn empty() -> Self {
        Self {
            opaque_vertices: Vec::new(),
            opaque_indices: Vec::new(),
            water_vertices: Vec::new(),
            water_indices: Vec::new(),
        }
    }
}
```

Re-run tests:

```bash
cargo test --lib mesher
```

Expected: the three new tests pass. The rest of the crate won't compile yet (downstream uses).

- [ ] **Step 4: Update `greedy.rs` to emit split streams + stamp `chunk_id`**

Add a `chunk_id: u16` parameter to `mesh_greedy`. Inside the mesher, route water-block emission to `mesh.water_vertices` / `mesh.water_indices` and opaque emission to `mesh.opaque_vertices` / `mesh.opaque_indices`. Today there's a single push helper; duplicate it (or branch inside it on the block's "is water" check) so the right stream gets each face. Stamp `chunk_id` into every emitted vertex.

Concrete change to the signature:

```rust
pub fn mesh_greedy(
    chunk: &DenseChunk,
    neighbors: &[Option<&DenseChunk>; 6],
    reg: &BlockRegistry,
    chunk_id: u16,
) -> ChunkMesh {
```

Inside, find every `Vertex { ... }` literal and add `chunk_id, _pad3: [0; 4]` to it. Find the `mesh.vertices.push(...)` / `mesh.indices.push(...)` calls and replace with a check on the block kind:

```rust
let is_water = reg.is_water(block);
let (vs, is) = if is_water {
    (&mut mesh.water_vertices, &mut mesh.water_indices)
} else {
    (&mut mesh.opaque_vertices, &mut mesh.opaque_indices)
};
vs.push(vertex);
// (continue pushing the other 3 verts + 6 indices to the chosen streams)
```

If `reg.is_water` doesn't exist, add it as a one-line method that compares against `BlockKind::Water`.

- [ ] **Step 5: Update `naive.rs` and `lod.rs` the same way**

Mirror the changes from greedy.rs. Both take a new `chunk_id: u16` parameter; both route to the split streams; both stamp `chunk_id` into every vertex.

- [ ] **Step 6: Update every call site of `mesh_greedy` / `mesh_naive` / `mesh_lod`**

```bash
grep -rn "mesh_greedy\|mesh_naive\|mesh_chunk_no_neighbors\|mesh_lod" src/ tests/
```

For each call site, thread the `chunk_id` argument. Mesh job builders currently lookup the chunk by coord — they need the `ChunkTable::register_or_lookup(coord)` result, but the mesher runs on worker threads where `ChunkTable` (which lives on the renderer) isn't accessible.

**Resolution:** the worker thread can't see `ChunkTable`. Pick the pre-registration approach: the main thread calls `renderer.chunk_table.register(coord)` BEFORE enqueueing the mesh job, captures the returned `u32`, casts it to `u16` (the renderer is capped at 65,536 simultaneously-loaded chunks — well above the load-radius volume of ~9.2k chunks, so the cast is safe; add a `debug_assert!(id <= u16::MAX as u32)`), and includes the `u16` value in the job's payload.

Update the job dispatcher in `src/ecs/systems/world_stream.rs` (or wherever mesh jobs get spawned — `grep -n "spawn.*mesh\|Jobs.*mesh" src/`):

```rust
// At the dispatch site, before `jobs.spawn_mesh(...)`:
let chunk_id_u32 = renderer.chunk_table.register(coord);
debug_assert!(chunk_id_u32 <= u16::MAX as u32, "chunk_id overflow");
let chunk_id = chunk_id_u32 as u16;
// Include chunk_id in the job payload passed to the worker.
```

Inside `mesh_greedy` / `mesh_naive` / `mesh_lod`, every emitted `Vertex` gets `chunk_id` stamped directly (no cast needed; the parameter is already `u16`).

- [ ] **Step 7: Update vertex-layout descriptors in pipeline builders**

In `src/render/pipelines/opaque.rs`, find the `VertexBufferLayout` definition. Update `array_stride` from 16 to 20, and add a new `VertexAttribute` for `chunk_id`:

```rust
// Find the VertexBufferLayout. Update:
array_stride: 20,
step_mode: wgpu::VertexStepMode::Vertex,
attributes: &[
    // location 0: pos.xyz (u8) + ao (u8) → vec4<u32>
    wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Uint8x4 },
    // location 2: color (Unorm8x4)
    wgpu::VertexAttribute { offset: 4, shader_location: 2, format: wgpu::VertexFormat::Unorm8x4 },
    // location 3: normal_face (u8) + light (u8) + chunk_id (u16) → vec4<u32> via Uint16x2 layout
    // We use two 16-bit attrs: face_light pulls bytes 8..10 and chunk_id pulls bytes 10..12.
    wgpu::VertexAttribute { offset: 8, shader_location: 3, format: wgpu::VertexFormat::Uint8x2 },
    wgpu::VertexAttribute { offset: 10, shader_location: 5, format: wgpu::VertexFormat::Uint16 },
    // location 4: tile_uv (4 bytes at offset 12)
    wgpu::VertexAttribute { offset: 12, shader_location: 4, format: wgpu::VertexFormat::Uint8x4 },
],
```

Apply the same change in `src/render/pipelines/water.rs`.

- [ ] **Step 8: Build, run mesher tests + smoke test**

```bash
cargo build --lib
cargo test --lib mesher
cargo test --test smoke
```

Expected: clean build; all mesher tests pass; smoke test passes (it doesn't go through the renderer so it's unaffected by the pipeline-layout part).

- [ ] **Step 9: Commit**

```bash
git add src/mesher/ src/render/pipelines/opaque.rs src/render/pipelines/water.rs src/ecs/systems/world_stream.rs src/voxel/block.rs
git commit -m "feat(mesher): split opaque/water streams + chunk_id vertex attribute"
```

---

### Task 6: Shaders — storage `chunk_table` + tower-packed light sampling

**Files:**
- Modify: `assets/shaders/opaque.wgsl`
- Modify: `assets/shaders/water.wgsl`

- [ ] **Step 1: Update `assets/shaders/opaque.wgsl` — group 1 bindings**

Replace:

```wgsl
struct ChunkUniform { origin: vec4<f32> };
@group(1) @binding(0) var<uniform> chunk: ChunkUniform;
@group(1) @binding(1) var light_volume:  texture_3d<f32>;
@group(1) @binding(2) var light_sampler: sampler;
```

with:

```wgsl
struct ChunkData {
    origin:      vec4<f32>,
    light_layer: u32,
    flags:       u32,
    _pad:        vec2<u32>,
};
@group(1) @binding(0) var<storage, read> chunk_table:   array<ChunkData>;
@group(1) @binding(1) var                light_volumes: texture_3d<f32>;
@group(1) @binding(2) var                light_sampler: sampler;

// The tower-packed light texture is sized (DIM, DIM, DIM * max_chunks).
// Layer N occupies z = [N*DIM, (N+1)*DIM). Sampling needs to remap
// the per-chunk local-w coordinate into the global tower coordinate.
const LIGHT_VOLUME_DIM: f32 = 33.0;
fn light_uvw(local_uvw: vec3<f32>, layer: u32) -> vec3<f32> {
    // Stretch the per-layer w-coord into 1 / max_chunks of the texture.
    // The shader is fed `max_chunks` via a specialization constant or
    // a small uniform — see the pipeline builder.
    let layers_total = f32(textureNumLayers_workaround());
    let w_per_layer = 1.0 / layers_total;
    return vec3<f32>(
        local_uvw.x,
        local_uvw.y,
        w_per_layer * (f32(layer) + local_uvw.z),
    );
}

// Workaround: WGSL doesn't expose textureNumLayers for texture_3d.
// We pass max_chunks in via a small dedicated uniform at binding 3.
@group(1) @binding(3) var<uniform> light_meta: vec4<u32>; // x = max_chunks
fn textureNumLayers_workaround() -> u32 { return light_meta.x; }
```

Replace the `VsIn` struct:

```wgsl
struct VsIn {
    @location(0) pos_ao:       vec4<u32>,
    @location(2) color:        vec4<f32>,
    @location(3) face_light:   vec2<u32>,
    @location(4) tile_uv:      vec4<u32>,
    @location(5) chunk_id:     u32,
};
```

Replace the vertex shader's chunk-origin lookup and light sampling:

```wgsl
@vertex
fn vs_main(in: VsIn) -> VsOut {
    let chunk = chunk_table[in.chunk_id];
    let world_pos = chunk.origin.xyz + vec3<f32>(in.pos_ao.xyz);
    // ... rest unchanged through to the chunk_id propagation:
    out.v_chunk_id = in.chunk_id;
    ...
}
```

Add `@location(7) @interpolate(flat) v_chunk_id: u32` to `VsOut`. In the fragment shader, replace the light-volume sample:

```wgsl
let chunk = chunk_table[in.v_chunk_id];
let sample_world = in.v_world + in.v_face_normal * 0.5;
let chunk_local  = sample_world - chunk.origin.xyz;
let uvw_local    = (chunk_local + vec3<f32>(0.5, 0.5, 0.5)) / 33.0;
let uvw_global   = light_uvw(uvw_local, chunk.light_layer);
let lvol         = textureSampleLevel(light_volumes, light_sampler, uvw_global, 0.0);
```

- [ ] **Step 2: Mirror the changes in `assets/shaders/water.wgsl`**

Same struct changes, same vertex-input layout, same light sample math. The water shader's "water-specific" branches (foam, depth-tint, reflection) stay the same.

- [ ] **Step 3: Build to confirm WGSL parses**

```bash
cargo build --lib
```

Expected: clean build. wgpu validates the shader at pipeline-build time, but pipelines are constructed at startup; a syntax error in the shader would show at runtime. We'll catch that in Task 7's screenshot test.

- [ ] **Step 4: Commit**

```bash
git add assets/shaders/opaque.wgsl assets/shaders/water.wgsl
git commit -m "feat(shader): chunk_table SSBO + tower-packed light volume sampling"
```

---

## Phase 3: Integration

### Task 7: Renderer rewrite — wire arena, table, light array, visibility

This is the largest task. It atomically swaps the old per-chunk path for the new shared-resource path across all three render passes. The screenshot harness is the test.

**Files:**
- Modify: `src/render/mod.rs` (big rewrite)
- Modify: `src/render/camera.rs` (new bind-group layout)
- Modify: `src/render/mesh.rs` (deleted or reduced to a tiny helper)
- Delete: `src/render/light_volume.rs` (functionality is now in `light_array.rs`)
- Create: `src/render/passes/mod.rs`, `passes/opaque.rs`, `passes/water.rs`, `passes/reflection.rs`, `passes/hud.rs`
- Modify: `src/ecs/systems/mesh_upload.rs` (or wherever `upload_chunk_mesh`/`upload_chunk_light_volume` is called from) — new call shapes

- [ ] **Step 1: Capture a "pre-integration" screenshot set as ground truth**

The screenshot baselines under `tests/screenshots/` predate this rework; if they were re-baked recently (per the commit log they were), they should still be the source of truth. Confirm they're current by running the test:

```bash
cargo test --test screenshots -- --nocapture 2>&1 | head -40
```

Or whatever the screenshot test target is — `grep -rn "screenshot" tests/ Cargo.toml`. Expected: tests pass on `main` before any edits land.

- [ ] **Step 2: Rewrite `make_chunk_bind_group_layout` in `src/render/camera.rs`**

Replace:

```rust
pub fn make_chunk_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chunk-bgl"),
        entries: &[
            // binding 0: per-chunk uniform → now storage buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: light-volume texture (the tower-packed 3D texture)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            // binding 2: light sampler
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding 3: light_meta uniform (max_chunks)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
```

Delete the `ChunkUniform` struct from `camera.rs` (no longer referenced).

- [ ] **Step 3: Create `src/render/passes/mod.rs`**

```rust
//! Per-pass encoding. Each submodule contains state setup + a draw loop
//! over a `VisibilityList`. None iterates over chunk meshes directly —
//! the visibility list is the single source of truth per frame.

pub mod hud;
pub mod opaque;
pub mod reflection;
pub mod water;
```

- [ ] **Step 4: Create `src/render/passes/opaque.rs`**

```rust
//! Opaque pass: sky + opaque chunks, written into the MSAA HDR view.
//! Uses the shared arena + chunk-table bind group; iterates the
//! visibility list instead of `chunk_meshes`.

use wgpu::CommandEncoder;

use crate::render::arena::ChunkArena;
use crate::render::visibility::VisibilityList;

pub struct OpaqueFrameInputs<'a> {
    pub msaa_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub arena: &'a ChunkArena,
    pub camera_bg: &'a wgpu::BindGroup,
    pub chunk_bg: &'a wgpu::BindGroup,
    pub atlas_bg: &'a wgpu::BindGroup,
    pub sky_pipeline: &'a wgpu::RenderPipeline,
    pub opaque_pipeline: &'a wgpu::RenderPipeline,
    pub visibility: &'a VisibilityList,
}

pub fn encode(enc: &mut CommandEncoder, inputs: OpaqueFrameInputs<'_>) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("sky+opaque-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: inputs.msaa_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: inputs.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Sky
    pass.set_pipeline(inputs.sky_pipeline);
    pass.set_bind_group(0, inputs.camera_bg, &[]);
    pass.draw(0..3, 0..1);

    // Opaque chunks — ONE bind-group-set per group, then one draw per entry.
    pass.set_pipeline(inputs.opaque_pipeline);
    pass.set_bind_group(0, inputs.camera_bg, &[]);
    pass.set_bind_group(1, inputs.chunk_bg, &[]);
    pass.set_bind_group(2, inputs.atlas_bg, &[]);
    pass.set_vertex_buffer(0, inputs.arena.vbuf().slice(..));
    pass.set_index_buffer(inputs.arena.ibuf().slice(..), wgpu::IndexFormat::Uint32);
    for entry in inputs.visibility.entries() {
        pass.draw_indexed(
            entry.index_offset..entry.index_offset + entry.index_count,
            0,
            0..1,
        );
    }
}
```

- [ ] **Step 5: Create `src/render/passes/water.rs`**

```rust
//! Water pass: water draws + cursor, blended over opaque MSAA. Reads
//! the same arena (different index ranges) and the same chunk_bg.
//! Resolves the MSAA target into the HDR resolve view.

use wgpu::CommandEncoder;

use crate::render::arena::ChunkArena;
use crate::render::visibility::VisibilityList;

pub struct WaterFrameInputs<'a> {
    pub msaa_view: &'a wgpu::TextureView,
    pub resolve_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,        // read-only attachment
    pub arena: &'a ChunkArena,
    pub camera_bg: &'a wgpu::BindGroup,
    pub chunk_bg: &'a wgpu::BindGroup,
    pub atlas_bg: &'a wgpu::BindGroup,
    pub water_depth_bg: &'a wgpu::BindGroup,
    pub water_reflection_bg: &'a wgpu::BindGroup,
    pub water_pipeline: &'a wgpu::RenderPipeline,
    pub cursor_pipeline: Option<&'a wgpu::RenderPipeline>,
    pub cursor_bg: Option<&'a wgpu::BindGroup>,
    pub visibility: &'a VisibilityList,
}

pub fn encode(enc: &mut CommandEncoder, inputs: WaterFrameInputs<'_>) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("water-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: inputs.msaa_view,
            resolve_target: Some(inputs.resolve_view),
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: inputs.depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(inputs.water_pipeline);
    pass.set_bind_group(0, inputs.camera_bg, &[]);
    pass.set_bind_group(1, inputs.chunk_bg, &[]);
    pass.set_bind_group(2, inputs.atlas_bg, &[]);
    pass.set_bind_group(3, inputs.water_depth_bg, &[]);
    pass.set_bind_group(4, inputs.water_reflection_bg, &[]);
    pass.set_vertex_buffer(0, inputs.arena.vbuf().slice(..));
    pass.set_index_buffer(inputs.arena.ibuf().slice(..), wgpu::IndexFormat::Uint32);
    for entry in inputs.visibility.entries() {
        pass.draw_indexed(
            entry.index_offset..entry.index_offset + entry.index_count,
            0,
            0..1,
        );
    }

    if let (Some(pipe), Some(bg)) = (inputs.cursor_pipeline, inputs.cursor_bg) {
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, inputs.camera_bg, &[]);
        pass.set_bind_group(1, bg, &[]);
        pass.draw(0..24, 0..1);
    }
}
```

- [ ] **Step 6: Create `src/render/passes/reflection.rs`**

```rust
//! Reflection pass: sky + opaque chunks drawn through a mirror matrix
//! into a 1/3-resolution offscreen target. Uses the Cw-winding pipeline
//! (the reflection matrix flips apparent triangle orientation) and the
//! reflection-frustum visibility list.

use wgpu::CommandEncoder;

use crate::render::arena::ChunkArena;
use crate::render::visibility::VisibilityList;

pub struct ReflectionFrameInputs<'a> {
    pub msaa_view: &'a wgpu::TextureView,
    pub resolve_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub arena: &'a ChunkArena,
    pub reflection_camera_bg: &'a wgpu::BindGroup,
    pub chunk_bg: &'a wgpu::BindGroup,
    pub atlas_bg: &'a wgpu::BindGroup,
    pub sky_reflection_pipeline: &'a wgpu::RenderPipeline,
    pub opaque_reflection_pipeline: &'a wgpu::RenderPipeline,
    pub visibility: &'a VisibilityList,
}

pub fn encode(enc: &mut CommandEncoder, inputs: ReflectionFrameInputs<'_>) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("reflection-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: inputs.msaa_view,
            resolve_target: Some(inputs.resolve_view),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.20, g: 0.40, b: 0.80, a: 1.0 }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: inputs.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Sky (reflected camera reconstructs downward rays automatically).
    pass.set_pipeline(inputs.sky_reflection_pipeline);
    pass.set_bind_group(0, inputs.reflection_camera_bg, &[]);
    pass.draw(0..3, 0..1);

    // Opaque chunks via the Cw-winding pipeline. Same arena, same
    // chunk_bg as the main opaque pass — only the camera uniform and
    // the pipeline differ.
    pass.set_pipeline(inputs.opaque_reflection_pipeline);
    pass.set_bind_group(0, inputs.reflection_camera_bg, &[]);
    pass.set_bind_group(1, inputs.chunk_bg, &[]);
    pass.set_bind_group(2, inputs.atlas_bg, &[]);
    pass.set_vertex_buffer(0, inputs.arena.vbuf().slice(..));
    pass.set_index_buffer(inputs.arena.ibuf().slice(..), wgpu::IndexFormat::Uint32);
    for entry in inputs.visibility.entries() {
        pass.draw_indexed(
            entry.index_offset..entry.index_offset + entry.index_count,
            0,
            0..1,
        );
    }
}
```

- [ ] **Step 7: Create `src/render/passes/hud.rs`**

Move the body of today's `Renderer::encode_hud_pass` here verbatim, taking `&HudFrame`, `&HudPipeline`, `&HudResources` as arguments. The HUD's ephemeral buffer allocation moves out in Task 9 — for now, leave the per-frame allocation in place.

- [ ] **Step 8: Rewrite `Renderer` struct + `render()` in `src/render/mod.rs`**

The struct fields change: remove `chunk_meshes`, `chunk_lights`, `light_sampler`, `_placeholder_light_tex`, `placeholder_light_view`; add `arena: ChunkArena`, `chunk_table: ChunkTable`, `light_array: LightVolumeArray`, `chunk_lods: HashMap<ChunkCoord, [Option<ChunkAllocation>; 3]>` (now holds arena handles instead of GPU buffers), `chunk_water_lods: HashMap<ChunkCoord, [Option<ChunkAllocation>; 3]>`, `chunk_bg: wgpu::BindGroup` (built once), `visibility: VisibilityList`, `visibility_reflection: VisibilityList`.

Replace `upload_chunk_mesh` with:

```rust
pub fn upload_chunk_mesh(&mut self, coord: ChunkCoord, lod: u8, mesh: &ChunkMesh) {
    let chunk_id = self.chunk_table.register(coord);
    self.chunk_table.set_origin(chunk_id, [
        coord.origin().0.x as f32,
        coord.origin().0.y as f32,
        coord.origin().0.z as f32,
    ]);

    // Free old LOD allocations if present.
    if let Some(slots) = self.chunk_lods.get_mut(&coord) {
        if let Some(old) = slots[lod as usize].take() { self.arena.free(old); }
    }
    if let Some(slots) = self.chunk_water_lods.get_mut(&coord) {
        if let Some(old) = slots[lod as usize].take() { self.arena.free(old); }
    }

    if !mesh.opaque_indices.is_empty() {
        let alloc = self.arena.alloc(
            &self.gpu.device, &self.gpu.queue,
            &mesh.opaque_vertices, &mesh.opaque_indices,
        );
        if let Some(a) = alloc {
            self.chunk_lods.entry(coord).or_insert([None, None, None])[lod as usize] = Some(a);
            self.chunk_table.set_has_opaque(chunk_id, true);
        } else {
            log::warn!("arena exhausted on opaque alloc for {coord:?}");
        }
    } else {
        self.chunk_table.set_has_opaque(chunk_id, false);
    }
    if !mesh.water_indices.is_empty() {
        let alloc = self.arena.alloc(
            &self.gpu.device, &self.gpu.queue,
            &mesh.water_vertices, &mesh.water_indices,
        );
        if let Some(a) = alloc {
            self.chunk_water_lods.entry(coord).or_insert([None, None, None])[lod as usize] = Some(a);
            self.chunk_table.set_has_water(chunk_id, true);
        } else {
            log::warn!("arena exhausted on water alloc for {coord:?}");
        }
    } else {
        self.chunk_table.set_has_water(chunk_id, false);
    }
}
```

Replace `upload_chunk_light_volume`:

```rust
pub fn upload_chunk_light_volume(&mut self, coord: ChunkCoord, blob: &[u8]) {
    let chunk_id = self.chunk_table.register(coord);
    let layer = match self.chunk_layers.get(&coord) {
        Some(&l) => l,
        None => match self.light_array.alloc_layer() {
            Some(l) => {
                self.chunk_layers.insert(coord, l);
                self.chunk_table.set_light_layer(chunk_id, l);
                l
            }
            None => {
                // Spec: fall back to the zero-initialised placeholder layer
                // (`PLACEHOLDER_LIGHT_LAYER` = 0). The chunk renders darker
                // for one frame; freed layers get reclaimed on the next
                // upload attempt.
                log::warn!("light array exhausted on upload for {coord:?}; using placeholder layer 0");
                return;
            }
        },
    };
    self.light_array.upload(&self.gpu.queue, layer, blob);
}
```

Add `chunk_layers: HashMap<ChunkCoord, u32>` field. Also reserve **layer 0 at startup** as the all-zero placeholder. In `Renderer::new_with_present_mode`, immediately after `LightVolumeArray::new`, call `light_array.alloc_layer()` once — the returned `0` becomes the never-freed placeholder. `ChunkTable::register` initialises new slots with `light_layer = 0` by default (the `GpuSlot::default()` already produces this), so any chunk without a real light upload samples from the zeroed placeholder layer automatically.

Replace `remove_chunk_mesh`:

```rust
pub fn remove_chunk_mesh(&mut self, coord: ChunkCoord) {
    if let Some(slots) = self.chunk_lods.remove(&coord) {
        for a in slots.into_iter().flatten() { self.arena.free(a); }
    }
    if let Some(slots) = self.chunk_water_lods.remove(&coord) {
        for a in slots.into_iter().flatten() { self.arena.free(a); }
    }
    if let Some(layer) = self.chunk_layers.remove(&coord) {
        self.light_array.free_layer(layer);
    }
    self.chunk_table.unregister(coord);
}
```

Rewrite `render()`. The new shape:

```rust
pub fn render(&mut self, ...) -> Result<(), wgpu::SurfaceError> {
    // ... unchanged camera write ...
    let frustum = extract_frustum_planes(vp);
    let refl_frustum = extract_frustum_planes(refl_vp);

    // Flush dirty SSBO slots once per frame.
    self.chunk_table.flush(&self.gpu.device, &self.gpu.queue);

    // Build per-frame slot inputs from chunk_table + lod allocations.
    let lods_default = LodThresholds::default();
    let slot_inputs = self.build_slot_inputs();
    self.visibility.build_filtered(&slot_inputs, eye, &frustum, &lods_default, 0b01);
    self.visibility_reflection.build_filtered(&slot_inputs, eye, &refl_frustum, &lods_default, 0b01);
    // Build a third visibility list for water (same frustum as opaque, mask = water).
    self.visibility_water.build_filtered(&slot_inputs, eye, &frustum, &lods_default, 0b10);

    let frame = self.gpu.surface.get_current_texture()?;
    let view = frame.texture.create_view(&Default::default());
    let mut enc = self.gpu.device.create_command_encoder(&Default::default());

    passes::reflection::encode(&mut enc, /* inputs ... */);
    passes::opaque::encode(&mut enc, /* inputs using self.visibility ... */);
    passes::water::encode(&mut enc, /* inputs using self.visibility_water ... */);
    self.encode_composite_pass(&mut enc, &view);
    if let Some(hud) = hud { passes::hud::encode(&mut enc, /* ... */); }

    self.gpu.queue.submit(std::iter::once(enc.finish()));
    frame.present();
    Ok(())
}

fn build_slot_inputs(&self) -> Vec<SlotInput> {
    let slots = self.chunk_table.slots();
    let mut out = Vec::with_capacity(slots.len());
    for (id, s) in slots.iter().enumerate() {
        let coord = /* reverse lookup — store reverse map on Renderer for O(1) */;
        let opaque = self.chunk_lods.get(&coord).cloned().unwrap_or([None, None, None]);
        let water  = self.chunk_water_lods.get(&coord).cloned().unwrap_or([None, None, None]);
        let lods_for_mask = |arr: [Option<ChunkAllocation>; 3]| -> [LodInput; 3] {
            [arr[0], arr[1], arr[2]].map(|a| match a {
                Some(a) => LodInput::present(a.index_offset, a.index_count),
                None => LodInput::missing(),
            })
        };
        // We need ONE entry per visibility mask; emit both opaque+water LODs
        // into the same SlotInput. The visibility builder picks the right
        // stream per mask. Combine: opaque LODs for mask 0b01, water for 0b10.
        // → Easier: emit TWO SlotInput entries per chunk, one per mask, but
        // the chunk_id field tells the shader which is which. Use a packed
        // index_offset/count pair per mask in the LodInput struct.
        let _ = (lods_for_mask(opaque), lods_for_mask(water));
        // ... see step 9 for the resolution.
    }
    out
}
```

Note: the `build_slot_inputs` step has a real design question (one SlotInput per chunk or per mask). Resolved in Step 9.

- [ ] **Step 9: Resolve the SlotInput-per-mask question by extending `LodInput` to hold both opaque and water ranges**

Update `src/render/visibility.rs`:

```rust
#[derive(Debug, Clone, Copy)]
pub struct LodInput {
    pub opaque_offset: u32,
    pub opaque_count: u32,
    pub water_offset: u32,
    pub water_count: u32,
}

impl LodInput {
    pub const fn missing() -> Self { Self { opaque_offset: 0, opaque_count: 0, water_offset: 0, water_count: 0 } }
    pub const fn present_opaque(opaque_offset: u32, opaque_count: u32) -> Self {
        Self { opaque_offset, opaque_count, water_offset: 0, water_count: 0 }
    }
    pub fn has_opaque(&self) -> bool { self.opaque_count > 0 }
    pub fn has_water(&self) -> bool { self.water_count > 0 }
    pub fn is_present_for_mask(&self, mask: u32) -> bool {
        (mask & 0b01 != 0 && self.has_opaque()) || (mask & 0b10 != 0 && self.has_water())
    }
}
```

Update `build_filtered` to project per-mask:

```rust
pub fn build_filtered(
    &mut self,
    slots: &[SlotInput],
    eye: Vec3,
    frustum: &[Vec4; 6],
    lods: &LodThresholds,
    flag_mask: u32,
) {
    self.entries.clear();
    const CULL_DISTANCE: f32 = 600.0 + 28.0;
    let cull_sq = CULL_DISTANCE * CULL_DISTANCE;

    for (chunk_id, slot) in slots.iter().enumerate() {
        if slot.flags & flag_mask == 0 { continue; }
        let chunk_min = slot.origin;
        let chunk_max = chunk_min + Vec3::splat(32.0);
        let center = chunk_min + Vec3::splat(16.0);
        let d_sq = (center - eye).length_squared();
        if d_sq > cull_sq { continue; }
        if !aabb_in_frustum(frustum, chunk_min, chunk_max) { continue; }

        let preferred = if d_sq < lods.lod1 * lods.lod1 { 0 }
                        else if d_sq < lods.lod2 * lods.lod2 { 1 }
                        else { 2 };
        let chosen = if slot.lods[preferred].is_present_for_mask(flag_mask) {
            Some(preferred)
        } else {
            slot.lods.iter().position(|l| l.is_present_for_mask(flag_mask))
        };
        if let Some(lod) = chosen {
            let l = &slot.lods[lod];
            let (offset, count) = if flag_mask & 0b01 != 0 {
                (l.opaque_offset, l.opaque_count)
            } else {
                (l.water_offset, l.water_count)
            };
            self.entries.push(DrawEntry {
                chunk_id: chunk_id as ChunkId,
                lod: lod as u8,
                index_offset: offset,
                index_count: count,
            });
        }
    }
}
```

Update the tests in `src/render/visibility.rs` to use the new `LodInput` constructors:

```rust
#[test]
fn near_chunk_picks_lod0() {
    let mut v = VisibilityList::new();
    let slots = vec![SlotInput {
        origin: Vec3::new(0.0, 0.0, 0.0),
        flags: 0b01,
        lods: [
            LodInput::present_opaque(100, 200),
            LodInput::missing(),
            LodInput::missing(),
        ],
    }];
    v.build_filtered(&slots, Vec3::new(16.0, 16.0, 16.0), &identity_frustum(), &LodThresholds::default(), 0b01);
    assert_eq!(v.entries().len(), 1);
    assert_eq!(v.entries()[0].lod, 0);
    assert_eq!(v.entries()[0].index_count, 200);
}
```

Delete the obsolete `LodInput::present` helper and the `VisibilityList::build` convenience wrapper (or update them to forward to `build_filtered(.., 0b11)` — pick one and stay consistent). Run `cargo test --lib render::visibility`; expected: all tests pass.

- [ ] **Step 10: Build the chunk bind group once at startup**

In `Renderer::new_with_present_mode`, after constructing `chunk_table` and `light_array`, build the single `chunk_bg`:

```rust
let light_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("light-meta"),
    contents: bytemuck::cast_slice(&[max_chunks_u32, 0u32, 0u32, 0u32]),
    usage: wgpu::BufferUsages::UNIFORM,
});
let chunk_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("chunk-bg"),
    layout: &chunk_bgl,
    entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: chunk_table.buffer().as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(light_array.view()) },
        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(light_array.sampler()) },
        wgpu::BindGroupEntry { binding: 3, resource: light_meta_buf.as_entire_binding() },
    ],
});
```

If `chunk_table` grows, its buffer reference changes — the bind group needs rebuilding. Make `ChunkTable::flush` return a small enum:

```rust
pub enum FlushOutcome {
    Unchanged,
    Grown,  // caller must rebuild any bind group referencing buffer()
}

impl ChunkTable {
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> FlushOutcome {
        let needed = self.book.slots.len();
        if needed > self.capacity {
            // ... grow path as before, return Grown ...
            FlushOutcome::Grown
        } else {
            // ... write dirty slots, return Unchanged ...
            FlushOutcome::Unchanged
        }
    }
}
```

In `Renderer::render`:

```rust
if matches!(self.chunk_table.flush(&self.gpu.device, &self.gpu.queue), FlushOutcome::Grown) {
    self.chunk_bg = build_chunk_bg(&self.gpu.device, &self.chunk_bgl, &self.chunk_table, &self.light_array, &self.light_meta_buf);
}
```

Factor the bind-group build into a free function `build_chunk_bg(...)` next to `make_chunk_bind_group_layout` in `camera.rs` so it's reusable across construction + grow.

- [ ] **Step 10.5: Update `Renderer::render_to_view` for the screenshot path**

`render_to_view` is used by `--screenshot-and-exit` and the screenshot tests. It currently calls `encode_opaque_pass` directly. Rewrite it to follow the same shape as `render()`:

```rust
pub fn render_to_view(
    &mut self,
    target: &wgpu::TextureView,
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    aspect: f32,
    sun_dir: [f32; 3],
    sun_intensity: f32,
    time: f32,
    hud: Option<&HudFrame>,
) {
    // ... camera writes identical to render() ...
    let frustum = extract_frustum_planes(vp);

    if matches!(self.chunk_table.flush(&self.gpu.device, &self.gpu.queue), FlushOutcome::Grown) {
        self.chunk_bg = build_chunk_bg(/* ... */);
    }
    let slot_inputs = self.build_slot_inputs();
    self.visibility.build_filtered(&slot_inputs, eye, &frustum, &LodThresholds::default(), 0b01);
    self.visibility_water.build_filtered(&slot_inputs, eye, &frustum, &LodThresholds::default(), 0b10);

    let mut enc = self.gpu.device.create_command_encoder(&Default::default());
    passes::opaque::encode(&mut enc, /* inputs ... */);
    passes::water::encode(&mut enc, /* inputs ... */);
    self.encode_composite_pass(&mut enc, target);
    if let Some(hud) = hud {
        passes::hud::encode(&mut enc, /* ..., target, ... */);
    }
    self.gpu.queue.submit(std::iter::once(enc.finish()));
}
```

Note: takes `&mut self` now (was `&self`) because `chunk_table.flush` and `visibility.build_filtered` need mutation. Update the screenshot-path call site in `src/main.rs` (or `screenshot.rs`) if it borrowed `&Renderer` previously — change to `&mut Renderer`.

- [ ] **Step 11: Delete `src/render/light_volume.rs`**

```bash
git rm src/render/light_volume.rs
```

Remove the `pub mod light_volume;` line from `src/render/mod.rs`.

- [ ] **Step 12: Run the smoke test + screenshot tests**

```bash
cargo test --test smoke
cargo test --test lighting_colored
cargo test --test worldgen_fingerprint
```

Then bring up the game manually:

```bash
cargo run --profile profiling -- --uncapped
```

Expected: world renders, geometry looks right, lighting looks right, water reflects, cursor works. If anything looks broken, run the screenshot comparison:

```bash
cargo run --profile profiling -- --screenshot-and-exit /tmp/oxium-current.png
python tests/screenshots/diff.py tests/screenshots/baseline_noon_outdoor.png /tmp/oxium-current.png
```

Iterate on bugs until the baselines match within SSIM tolerance.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat(render): GPU-driven chunk rendering — arena+table+light-array+visibility integrated"
```

---

## Phase 4: Pass cleanups

### Task 8: Eliminate mid-frame MSAA depth copy

**Files:**
- Modify: `src/render/mod.rs`
- Modify: `src/render/gpu.rs` (remove `depth_sample_texture` allocation)
- Modify: `src/render/passes/water.rs` (water reads the live depth read-only)
- Modify: `src/render/pipelines/water.rs` (depth_bg layout: now reads the depth attachment directly)

- [ ] **Step 1: Update the water pipeline's depth bind group layout**

The water shader currently samples `depth_sample_texture` via `water.depth_bgl`. Change the binding's `sample_type` to `Depth` and the view-dimension to match the MSAA depth (or use `Float { filterable: false }` depending on backend). Concretely:

```rust
// src/render/pipelines/water.rs
let depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: Some("water-depth-bgl"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: true, // MSAA
        },
        count: None,
    }],
});
```

- [ ] **Step 2: Update the water shader to sample the multisampled depth directly**

In `assets/shaders/water.wgsl`, replace the depth-texture binding type with `texture_depth_multisampled_2d`, sample with `textureLoad(..., sample_index)`. Use sample index 0 (single sample is fine for foam — perceptually identical).

- [ ] **Step 3: Bind the live `depth_view` to the water pass instead of `depth_sample_view`**

In `Renderer::new_with_present_mode`, the `water_depth_bg` now references `self.depth_view` (same texture that the opaque pass writes; depth-store survives between passes). The depth attachment on the water pass becomes **read-only** via `depth_ops: None` and `depth_read_only: true` — see `wgpu::RenderPassDepthStencilAttachment::depth_read_only`.

Update `src/render/passes/water.rs`:

```rust
depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
    view: inputs.depth_view,
    depth_ops: None,           // no load/store — read-only
    stencil_ops: None,
}),
```

And set `depth_read_only: true` in the same struct (wgpu 23 API may put this in a different spot — check `RenderPassDepthStencilAttachment`).

- [ ] **Step 4: Delete `depth_sample_texture` allocation, field, and the `copy_texture_to_texture` call**

Remove from `Renderer`: `depth_sample_texture`, `depth_sample_view`. Remove from `gpu.rs`: `make_depth_sample_texture`. Remove the `enc.copy_texture_to_texture(&self.depth_texture, &self.depth_sample_texture, ...)` block in the old `encode_opaque_pass` body (it's gone from `mod.rs` now anyway, but make sure nothing in `passes/opaque.rs` reintroduced it).

- [ ] **Step 5: Run screenshots + the game manually**

```bash
cargo test --test screenshots
cargo run --profile profiling -- --uncapped
```

Expected: water still has foam and depth-tint; visuals match the baselines. If foam looks wrong (jaggy near terrain edges), confirm the shader's `textureLoad` is reading the same sample the rasteriser writes — using sample 0 is fine; using `linearStep` from depth differences usually hides it anyway.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "perf(render): water reads live depth attachment, kill mid-frame MSAA depth copy"
```

---

### Task 9: HUD persistent ring buffer

**Files:**
- Modify: `src/render/mod.rs` (add ring buffer field)
- Modify: `src/render/passes/hud.rs`

- [ ] **Step 1: Add a `HudRingBuffer` struct**

In `src/render/passes/hud.rs`, add:

```rust
const HUD_VBUF_SIZE: u64 = 1 << 20; // 1 MB — enough for ~50k verts
const HUD_IBUF_SIZE: u64 = 256 * 1024;

pub struct HudRingBuffer {
    pub vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    v_offset: u64,
    i_offset: u64,
}

impl HudRingBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-vbuf-ring"), size: HUD_VBUF_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-ibuf-ring"), size: HUD_IBUF_SIZE,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { vbuf, ibuf, v_offset: 0, i_offset: 0 }
    }

    pub fn begin_frame(&mut self) { self.v_offset = 0; self.i_offset = 0; }

    pub fn push(&mut self, queue: &wgpu::Queue, verts: &[u8], indices: &[u8]) -> (u64, u64) {
        let v_off = self.v_offset;
        let i_off = self.i_offset;
        queue.write_buffer(&self.vbuf, v_off, verts);
        queue.write_buffer(&self.ibuf, i_off, indices);
        self.v_offset += verts.len() as u64;
        self.i_offset += indices.len() as u64;
        (v_off, i_off)
    }
}
```

- [ ] **Step 2: Wire it into `Renderer`**

Add `hud_ring: HudRingBuffer` to `Renderer`. In `render()`, call `self.hud_ring.begin_frame()` at the start of the frame. Pass `&mut self.hud_ring` into `passes::hud::encode`.

- [ ] **Step 3: Update `passes::hud::encode` to use the ring buffer**

Replace the per-frame `create_buffer_init` pairs with calls to `hud_ring.push(...)`. Use the returned offsets in `set_vertex_buffer` / `set_index_buffer` via `buffer.slice(off..off+len)`.

- [ ] **Step 4: Run the game and confirm HUD renders**

```bash
cargo run --profile profiling -- --uncapped
```

Expected: HUD overlay (FPS counter, hotbar, crosshair) shows correctly. If text is missing or positioned wrong, the offset math is off — check that the slice ranges are byte-exact and aligned to vertex size (multiples of 20).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "perf(hud): persistent ring buffer, no per-frame buffer allocations"
```

---

### Task 10: MSAA configurable

**Files:**
- Modify: `src/render/gpu.rs` (`MSAA_SAMPLES` → field on `Gpu`)
- Modify: `src/render/mod.rs` (thread through)
- Modify: `src/main.rs` (`--msaa` flag)

- [ ] **Step 1: Replace the `const MSAA_SAMPLES: u32 = 4` with a struct field**

In `src/render/gpu.rs`:

```rust
pub struct Gpu {
    // existing fields ...
    pub msaa_samples: u32,
}

impl Gpu {
    pub fn new_with_present_mode(window: Arc<Window>, present_mode: wgpu::PresentMode, msaa_samples: u32) -> Self {
        assert!(matches!(msaa_samples, 1 | 2 | 4), "MSAA must be 1, 2, or 4");
        // existing init...
        Self { /* fields */, msaa_samples }
    }
}
```

Every `MSAA_SAMPLES` reference in this file becomes `self.msaa_samples` or a parameter.

- [ ] **Step 2: Thread the sample count through `make_msaa_color_texture`, `make_depth_texture`, `make_reflection_*`, and every pipeline builder**

Add `sample_count: u32` parameter to each. Update every call site.

- [ ] **Step 3: Add a `--msaa <n>` CLI flag in `src/main.rs`**

```rust
"--msaa" => {
    let n: u32 = it
        .next()
        .expect("--msaa requires 1, 2, or 4")
        .parse()
        .expect("--msaa must be 1, 2, or 4");
    msaa_samples = n;
}
```

Default `let mut msaa_samples: u32 = 4;`. Pass through to `App::new_with_*`.

- [ ] **Step 4: Smoke-test all three values**

```bash
cargo run --profile profiling -- --uncapped --msaa 1
cargo run --profile profiling -- --uncapped --msaa 2
cargo run --profile profiling -- --uncapped --msaa 4
```

Expected: all three boot and render. Visible aliasing reduces from 1× → 2× → 4×.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(render): MSAA configurable via --msaa, default 4×"
```

---

## Phase 5: Validation

### Task 11: Post-rework perf bench + screenshot re-baseline

- [ ] **Step 1: Re-run the bench against the reworked code**

```bash
cargo run --profile profiling -- \
  --uncapped \
  --profile docs/superpowers/bench/post-rework.csv \
  --bench 600
```

- [ ] **Step 2: Diff baseline vs post-rework**

Open both CSVs side-by-side. Compare the `render` column distribution. Concrete target: **median `render` µs reduced by ≥ 5×** at the same scene with the same chunk count.

```bash
awk -F, 'NR > 1 { print $9 }' docs/superpowers/bench/baseline.csv | sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{print a[int(c/2)]}'
awk -F, 'NR > 1 { print $9 }' docs/superpowers/bench/post-rework.csv | sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{print a[int(c/2)]}'
```

Adjust the column index `$9` to match the actual `render` column in the CSV header.

- [ ] **Step 3: Re-bake screenshot baselines if SSIM drift is acceptable**

If the screenshot tests have a small SSIM drift from the new shader sampling path:

```bash
cargo run --profile profiling -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png
# (and the other baselined scenes — see tests/screenshots/README.md)
```

Run the screenshot tests one more time:

```bash
cargo test --test screenshots
```

- [ ] **Step 4: Document the perf delta**

Append to `docs/superpowers/bench/README.md`:

```markdown
## Result (2026-05-20 rework)

- Baseline median `render`: <X> µs
- Post-rework median `render`: <Y> µs
- Ratio: <X / Y> ×
```

Fill in the actual numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/bench/ tests/screenshots/
git commit -m "test(screenshot): re-baseline after render perf rework + bench delta"
```

---

## Validation

After Task 11, the work is complete if:

- All `cargo test` targets pass (`smoke`, `lighting_colored`, `worldgen_fingerprint`, `screenshots` if it exists as a target, all `--lib` unit tests).
- The game launches and renders correctly at `--msaa 1`, `--msaa 2`, `--msaa 4`.
- The `post-rework.csv` median `render` µs is materially lower than `baseline.csv`'s. A 5× improvement is the success target; less than 2× means something went wrong (likely a missed integration step or a vertex-format bug forcing fallbacks).
- `src/render/mod.rs` is roughly half its previous length (today: 1538 lines; target: ~300–500 lines).
- `git grep "create_bind_group" src/render/` returns only the **once-at-startup** sites — no `create_bind_group` calls inside any per-frame or per-chunk loop.
- The screenshot baselines either match within SSIM tolerance or have been deliberately re-baked.
