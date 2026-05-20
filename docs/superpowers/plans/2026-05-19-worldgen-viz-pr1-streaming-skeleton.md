# Worldgen viz redesign — PR 1: Streaming skeleton (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `src/bin/worldgen_viz/` with a new binary that streams chunks around a fly-camera, reuses the game's `Generator`, exposes a Dense-Dashboard egui layout with placeholder panels, and ships a `--check` smoke test for CI.

**Architecture:** New `bin/worldgen_viz/` module tree splits responsibilities across `app/session/world/camera/render/widgets`. Streaming uses an LRU `ChunkCache` driven by `oxium::jobs::Jobs::spawn_gen` + a viz-local mesher (the game mesher's `Vertex` carries atlas/light fields the viz doesn't have). egui dashboard reuses today's per-config-section panels under a new layout shell. Config edits land via `ConfigHolder` and wipe the chunk cache on change.

**Tech Stack:** Rust, wgpu 23, egui 0.30, winit 0.30, glam, bytemuck, crossbeam-channel, rayon (via `oxium::jobs`).

**Spec:** [`docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md`](../specs/2026-05-19-worldgen-viz-redesign-design.md). This plan implements **only PR 1** from that spec's seven-PR rollout. Later PRs get their own plans.

---

## File Structure (delta)

**Delete entirely:**
- `src/bin/worldgen_viz/cross.rs` — top-down preview becomes part of PR 2's overlays.
- `src/bin/worldgen_viz/scene.rs` — replaced by `world/mesher.rs` + `render/scene.rs`.
- `src/bin/worldgen_viz/worldgen_bridge.rs` — replaced by `world/mod.rs`.

**Replace contents (move out to new structure):**
- `src/bin/worldgen_viz/main.rs` — rewritten as thin winit entry point.
- `src/bin/worldgen_viz/render.rs` — moved to `src/bin/worldgen_viz/render/mod.rs` (wgpu device + egui glue only).
- `src/bin/worldgen_viz/ui.rs` — split: panel functions for `WorldgenConfig` editing move to `src/bin/worldgen_viz/widgets/cfg_panels.rs`. Layout shell moves to `src/bin/worldgen_viz/layout.rs`.
- `src/bin/worldgen_viz/spline_widget.rs` — moved to `src/bin/worldgen_viz/widgets/spline.rs` (no logic changes in PR 1).

**Create:**
- `src/bin/worldgen_viz/app.rs` — `AppState`: holds the (single) `Session`, ui state, last-frame timing.
- `src/bin/worldgen_viz/session.rs` — `Session` (Generator + ChunkCache + Camera + ConfigHolder + invalidate state).
- `src/bin/worldgen_viz/camera.rs` — `Camera` trait, `FlyCamera`, `OrbitCamera`.
- `src/bin/worldgen_viz/world/mod.rs` — public `World` struct (stream + cache + mesher integration).
- `src/bin/worldgen_viz/world/stream.rs` — radius enumeration.
- `src/bin/worldgen_viz/world/cache.rs` — `LruChunkCache` (chunk + mesh entries).
- `src/bin/worldgen_viz/world/mesher.rs` — viz mesher (face-culling + Vertex emit).
- `src/bin/worldgen_viz/world/invalidate.rs` — config-change cache wipe.
- `src/bin/worldgen_viz/layout.rs` — Dense Dashboard egui layout shell.
- `src/bin/worldgen_viz/paint.rs` — `PaintMode` enum (Block only in PR 1).
- `src/bin/worldgen_viz/render/mod.rs` — moved from today's render.rs.
- `src/bin/worldgen_viz/render/scene.rs` — wgpu pipeline, Vertex, depth, shader bind groups.
- `src/bin/worldgen_viz/render/shader.wgsl` — vertex/fragment WGSL (extracted from today's inline string).
- `src/bin/worldgen_viz/widgets/mod.rs` — re-exports.
- `src/bin/worldgen_viz/widgets/cfg_panels.rs` — today's density/climate/caves/biomes/surface panels, unchanged in PR 1.
- `src/bin/worldgen_viz/widgets/spline.rs` — moved.

**Tests:**
- `src/bin/worldgen_viz/camera.rs` — `#[cfg(test)] mod tests`.
- `src/bin/worldgen_viz/world/stream.rs` — `#[cfg(test)] mod tests`.
- `src/bin/worldgen_viz/world/cache.rs` — `#[cfg(test)] mod tests`.
- `src/bin/worldgen_viz/world/mesher.rs` — `#[cfg(test)] mod tests`.
- `src/bin/worldgen_viz/world/invalidate.rs` — `#[cfg(test)] mod tests`.

**No changes to `oxium` library crate in PR 1.** Worldgen API additions land in PR 2.

---

## Vertex format (PR 1)

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],   // world-space block coords
    pub color: [f32; 3],      // block-paint color baked at mesh time
}
```

PR 3 introduces a parallel color buffer + voxel-coord vertex attribute when multiple paint modes land. For PR 1, color is baked into the vertex.

---

## Tasks

### Task 1: Scaffold the new module tree

**Files:**
- Delete: `src/bin/worldgen_viz/cross.rs`, `scene.rs`, `worldgen_bridge.rs`.
- Move: `src/bin/worldgen_viz/spline_widget.rs` → `src/bin/worldgen_viz/widgets/spline.rs` (git mv).
- Move: `src/bin/worldgen_viz/render.rs` → `src/bin/worldgen_viz/render/mod.rs` (git mv into new dir).
- Create empty stub: `src/bin/worldgen_viz/{app,session,camera,paint,layout}.rs`.
- Create empty stub: `src/bin/worldgen_viz/world/{mod,stream,cache,mesher,invalidate}.rs`.
- Create empty stub: `src/bin/worldgen_viz/render/{scene.rs,shader.wgsl}`.
- Create empty stub: `src/bin/worldgen_viz/widgets/{mod,cfg_panels}.rs`.
- Replace: `src/bin/worldgen_viz/main.rs` (minimal: declares modules, `fn main() { println!("viz scaffolding"); }`).
- Move: `src/bin/worldgen_viz/ui.rs` → split into `widgets/cfg_panels.rs` (panel fns) and `layout.rs` (graph_panel, preset_panel, and a no-op `dashboard` fn for now).

- [ ] **Step 1: Delete files**

```bash
git rm src/bin/worldgen_viz/cross.rs src/bin/worldgen_viz/scene.rs src/bin/worldgen_viz/worldgen_bridge.rs
```

- [ ] **Step 2: Reorganise existing files**

```bash
mkdir -p src/bin/worldgen_viz/render src/bin/worldgen_viz/world src/bin/worldgen_viz/widgets
git mv src/bin/worldgen_viz/render.rs src/bin/worldgen_viz/render/mod.rs
git mv src/bin/worldgen_viz/spline_widget.rs src/bin/worldgen_viz/widgets/spline.rs
git mv src/bin/worldgen_viz/ui.rs src/bin/worldgen_viz/widgets/cfg_panels.rs
```

- [ ] **Step 3: Create stub files**

Each stub contains only a module-level doc comment plus required `pub` re-exports to make `mod.rs` chain compile. Example for `src/bin/worldgen_viz/world/mod.rs`:

```rust
//! Streaming world: chunk cache + meshing pipeline.

pub mod cache;
pub mod invalidate;
pub mod mesher;
pub mod stream;
```

Stubs for `world/{stream,cache,mesher,invalidate}.rs`:

```rust
//! TODO PR 1: implement.
```

Stubs for `src/bin/worldgen_viz/{app,session,camera,paint,layout}.rs`:

```rust
//! TODO PR 1: implement.
```

Stub `src/bin/worldgen_viz/widgets/mod.rs`:

```rust
//! Reusable egui widgets for the viz dashboard.

pub mod cfg_panels;
pub mod spline;
```

Stub `src/bin/worldgen_viz/render/scene.rs`:

```rust
//! TODO PR 1: 3D scene pipeline.
```

Stub `src/bin/worldgen_viz/render/shader.wgsl` — empty file (Task 7 fills it).

In `src/bin/worldgen_viz/widgets/cfg_panels.rs`, update its imports — change any references to `crate::spline_widget::SplineEditor` to `crate::widgets::spline::SplineEditor`. The panel function signatures (`density_panel`, `climate_panel`, `caves_panel`, `biomes_panel`, `surface_panel`, `preset_panel`, `graph_panel`) stay unchanged.

Replace `src/bin/worldgen_viz/main.rs` with:

```rust
//! Worldgen tuning visualizer.

mod app;
mod camera;
mod layout;
mod paint;
mod render;
mod session;
mod widgets;
mod world;

fn main() {
    eprintln!("worldgen_viz: PR 1 scaffolding in place");
}
```

- [ ] **Step 4: Verify build**

Run: `cargo check --bin worldgen_viz`

Expected: warnings about unused modules; **no errors**.

- [ ] **Step 5: Commit**

```bash
git add -A src/bin/worldgen_viz/
git commit -m "$(cat <<'EOF'
viz: scaffold new module tree (PR 1 step 1)

Delete cross/scene/worldgen_bridge from the old viz; move render.rs and
ui.rs into the new layout; create empty stubs for app/session/camera/paint/
layout/world/* and render/scene+shader.wgsl. Binary still builds and prints
a "scaffolding in place" line; subsequent tasks fill the stubs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Camera trait + FlyCamera + OrbitCamera

**Files:**
- Modify: `src/bin/worldgen_viz/camera.rs`.

`OrbitCamera` keeps today's behaviour (target + yaw/pitch/distance). `FlyCamera` is a free 6-DOF cam with position + yaw + pitch. Both implement a `Camera` trait that returns a `view_proj` matrix.

- [ ] **Step 1: Write the failing tests**

In `src/bin/worldgen_viz/camera.rs`, replace the stub with:

```rust
//! Cameras for the viz: orbit (default) + fly (free-look).

use glam::{Mat4, Vec3};

/// What the renderer needs from any camera: a view+projection matrix.
pub trait Camera {
    fn view_proj(&self, aspect: f32) -> Mat4;
    fn position(&self) -> Vec3;
}

#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::new(32.0, 70.0, 32.0),
            yaw: 0.8,
            pitch: 0.4,
            distance: 160.0,
        }
    }
}

impl Camera for OrbitCamera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.5, 4096.0);
        proj * view
    }

    fn position(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.yaw.cos() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.sin() * self.pitch.cos(),
            )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
}

impl FlyCamera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 96.0, 64.0),
            yaw: -1.0,
            pitch: -0.3,
            speed: 30.0,
        }
    }

    /// Forward unit vector in world space (XZ-yaw + Y-pitch).
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Right-hand vector orthogonal to forward in the XZ plane.
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos()).normalize()
    }

    /// Apply WASD/QE motion. `forward_back` is +1 for W, −1 for S; `strafe`
    /// is +1 for D, −1 for A; `vertical` is +1 for E, −1 for Q. `dt` in
    /// seconds. Uses `self.speed` as units per second, boosted if `boost`.
    pub fn translate(&mut self, forward_back: f32, strafe: f32, vertical: f32, boost: bool, dt: f32) {
        let mul = if boost { 4.0 } else { 1.0 };
        let v = self.forward() * forward_back
            + self.right() * strafe
            + Vec3::Y * vertical;
        if v.length_squared() > 0.0 {
            self.position += v.normalize() * self.speed * mul * dt;
        }
    }

    /// Apply mouse delta. `dx`, `dy` are screen pixels.
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch - dy * 0.005).clamp(-1.5, 1.5);
    }
}

impl Camera for FlyCamera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let target = self.position + self.forward();
        let view = Mat4::look_at_rh(self.position, target, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, 4096.0);
        proj * view
    }

    fn position(&self) -> Vec3 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_translate_forward_moves_along_forward() {
        let mut cam = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 10.0,
        };
        cam.translate(1.0, 0.0, 0.0, false, 1.0);
        let expected = cam.forward() * 10.0;
        assert!((cam.position - expected).length() < 1e-4, "got {:?}", cam.position);
    }

    #[test]
    fn fly_strafe_is_orthogonal_to_forward() {
        let cam = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.5,
            pitch: 0.0,
            speed: 1.0,
        };
        let dot = cam.forward().dot(cam.right());
        assert!(dot.abs() < 1e-4, "forward and right must be orthogonal, got dot={dot}");
    }

    #[test]
    fn fly_pitch_clamps_to_pi_over_two() {
        let mut cam = FlyCamera::new();
        // 100 frames of looking straight up at high dy.
        for _ in 0..100 {
            cam.look(0.0, -10000.0);
        }
        assert!(cam.pitch <= 1.5 && cam.pitch >= -1.5, "pitch escape: {}", cam.pitch);
    }

    #[test]
    fn orbit_position_distance_matches_field() {
        let cam = OrbitCamera::new();
        let d = (cam.position() - cam.target).length();
        assert!((d - cam.distance).abs() < 1e-3, "got distance {d}");
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test --bin worldgen_viz camera`

Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/camera.rs
git commit -m "viz: add Camera trait + FlyCamera + OrbitCamera (PR 1)"
```

---

### Task 3: Stream radius enumeration

**Files:**
- Modify: `src/bin/worldgen_viz/world/stream.rs`.

Pure function: given a camera world-position and a `(r_xz, r_y)` radius in chunks, return the set of `ChunkCoord`s within that radius. Center is determined by which chunk the camera is in. Used by the streaming loop to decide what to load/keep/evict.

- [ ] **Step 1: Write the failing tests**

Replace `src/bin/worldgen_viz/world/stream.rs`:

```rust
//! Camera-follow chunk enumeration.

use glam::{IVec3, Vec3};
use oxium::voxel::coords::{ChunkCoord, CHUNK_DIM_U};

/// Radius (in chunks) around the camera. Spec defaults: 8 XZ, 4 Y.
#[derive(Debug, Clone, Copy)]
pub struct StreamRadius {
    pub xz: i32,
    pub y: i32,
}

impl StreamRadius {
    pub const DEFAULT: Self = Self { xz: 8, y: 4 };
}

/// Which chunk does the given world position sit in?
pub fn camera_chunk(pos: Vec3) -> ChunkCoord {
    let dim = CHUNK_DIM_U as f32;
    ChunkCoord(IVec3::new(
        (pos.x / dim).floor() as i32,
        (pos.y / dim).floor() as i32,
        (pos.z / dim).floor() as i32,
    ))
}

/// All chunk coords within `radius` of the chunk containing `pos`, sorted by
/// Chebyshev distance ascending (closest first). The center chunk is at
/// index 0 of the returned vec.
pub fn chunks_in_radius(pos: Vec3, radius: StreamRadius) -> Vec<ChunkCoord> {
    let center = camera_chunk(pos);
    let mut out = Vec::with_capacity(((2 * radius.xz + 1).pow(2) * (2 * radius.y + 1)) as usize);
    for dy in -radius.y..=radius.y {
        for dz in -radius.xz..=radius.xz {
            for dx in -radius.xz..=radius.xz {
                out.push(ChunkCoord(center.0 + IVec3::new(dx, dy, dz)));
            }
        }
    }
    out.sort_by_key(|c| {
        let d = c.0 - center.0;
        d.x.abs().max(d.z.abs()).max(d.y.abs())
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_chunk_at_origin_is_zero() {
        assert_eq!(camera_chunk(Vec3::new(0.0, 0.0, 0.0)).0, IVec3::ZERO);
    }

    #[test]
    fn camera_chunk_floors_negative() {
        // (-1, -1, -1) lies in chunk (-1, -1, -1), not (0,0,0).
        assert_eq!(
            camera_chunk(Vec3::new(-1.0, -1.0, -1.0)).0,
            IVec3::new(-1, -1, -1)
        );
    }

    #[test]
    fn chunks_in_radius_size_matches_3d_box() {
        let r = StreamRadius { xz: 2, y: 1 };
        let v = chunks_in_radius(Vec3::ZERO, r);
        let expected = (2 * r.xz + 1).pow(2) * (2 * r.y + 1);
        assert_eq!(v.len() as i32, expected);
    }

    #[test]
    fn chunks_in_radius_center_first() {
        let v = chunks_in_radius(Vec3::ZERO, StreamRadius::DEFAULT);
        assert_eq!(v[0].0, IVec3::ZERO);
    }

    #[test]
    fn chunks_in_radius_sorted_by_chebyshev() {
        let v = chunks_in_radius(Vec3::ZERO, StreamRadius { xz: 1, y: 0 });
        // Center, then 8 neighbours all at Chebyshev distance 1.
        assert_eq!(v.len(), 9);
        let chebyshev = |c: &ChunkCoord| c.0.x.abs().max(c.0.y.abs()).max(c.0.z.abs());
        for w in v.windows(2) {
            assert!(chebyshev(&w[0]) <= chebyshev(&w[1]));
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --bin worldgen_viz stream`

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/world/stream.rs
git commit -m "viz: chunk-radius enumeration (PR 1)"
```

---

### Task 4: ChunkCache

**Files:**
- Modify: `src/bin/worldgen_viz/world/cache.rs`.

Holds two kinds of entries per `ChunkCoord`: the raw `DenseChunk` (post-fill) and the meshed vertex/index buffers. Capacity-bounded LRU; evicts least-recently-touched on overflow. We use the `lru` crate (already in Oxium's Cargo.lock — verify; if not, hand-roll a small LRU using `IndexMap` + counter).

- [ ] **Step 1: Verify `lru` is available**

Run: `cargo tree -p oxium | grep -E "^[├└│ ]*lru " | head -3`

If not present, add it: edit `Cargo.toml`, add `lru = "0.12"` under `[dependencies]`. Otherwise skip.

- [ ] **Step 2: Write the failing tests**

Replace `src/bin/worldgen_viz/world/cache.rs`:

```rust
//! LRU caches for filled chunks + meshed chunks. Keyed by ChunkCoord.

use crate::render::scene::Vertex;
use lru::LruCache;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use std::num::NonZeroUsize;
use std::sync::Arc;

pub struct ChunkMeshGpu {
    pub vertices: Arc<Vec<Vertex>>,
    pub indices: Arc<Vec<u32>>,
}

impl ChunkMeshGpu {
    pub fn empty() -> Self {
        Self {
            vertices: Arc::new(Vec::new()),
            indices: Arc::new(Vec::new()),
        }
    }
}

pub struct ChunkCache {
    /// Filled (but not yet meshed) chunks. Keyed by coord; capacity is
    /// `radius_xz^2 * radius_y * 3` to give headroom for chunks that are
    /// in-flight or recently scrolled out.
    chunks: LruCache<ChunkCoord, Arc<DenseChunk>>,
    /// Meshed CPU-side vertex/index buffers, ready for GPU upload.
    meshes: LruCache<ChunkCoord, ChunkMeshGpu>,
}

impl ChunkCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            chunks: LruCache::new(cap),
            meshes: LruCache::new(cap),
        }
    }

    pub fn put_chunk(&mut self, coord: ChunkCoord, chunk: Arc<DenseChunk>) {
        self.chunks.put(coord, chunk);
    }

    pub fn put_mesh(&mut self, coord: ChunkCoord, mesh: ChunkMeshGpu) {
        self.meshes.put(coord, mesh);
    }

    pub fn get_chunk(&mut self, coord: ChunkCoord) -> Option<Arc<DenseChunk>> {
        self.chunks.get(&coord).cloned()
    }

    pub fn get_mesh(&mut self, coord: ChunkCoord) -> Option<&ChunkMeshGpu> {
        self.meshes.get(&coord)
    }

    pub fn has_chunk(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains(&coord)
    }

    pub fn has_mesh(&self, coord: ChunkCoord) -> bool {
        self.meshes.contains(&coord)
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.meshes.clear();
    }

    pub fn len(&self) -> (usize, usize) {
        (self.chunks.len(), self.meshes.len())
    }

    pub fn meshed_coords(&self) -> Vec<ChunkCoord> {
        self.meshes.iter().map(|(k, _)| *k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    fn coord(x: i32, y: i32, z: i32) -> ChunkCoord {
        ChunkCoord(IVec3::new(x, y, z))
    }

    #[test]
    fn put_and_get_chunk() {
        let mut c = ChunkCache::new(4);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(c.has_chunk(coord(0, 0, 0)));
        assert!(c.get_chunk(coord(0, 0, 0)).is_some());
    }

    #[test]
    fn over_capacity_evicts_oldest() {
        let mut c = ChunkCache::new(2);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_chunk(coord(1, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_chunk(coord(2, 0, 0), Arc::new(DenseChunk::empty()));
        // (0,0,0) was inserted first; on third insert it should evict.
        assert!(!c.has_chunk(coord(0, 0, 0)));
        assert!(c.has_chunk(coord(1, 0, 0)));
        assert!(c.has_chunk(coord(2, 0, 0)));
    }

    #[test]
    fn clear_drops_everything() {
        let mut c = ChunkCache::new(4);
        c.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        c.put_mesh(coord(0, 0, 0), ChunkMeshGpu::empty());
        c.clear();
        assert_eq!(c.len(), (0, 0));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --bin worldgen_viz cache`

Expected: 3 tests pass. (`Vertex` doesn't exist yet — Task 6 creates it. For now, replace `use crate::render::scene::Vertex;` with a local stub: add `pub struct Vertex { pub position: [f32; 3], pub color: [f32; 3] }` at the top of `cache.rs` until Task 6 unifies it. Delete the stub during Task 6 and re-add the import.)

Actually — clean way: put the Vertex definition itself in `render/scene.rs` now even though the rest of scene.rs is empty. Edit `src/bin/worldgen_viz/render/scene.rs` first:

```rust
//! 3D scene pipeline. Vertex format + wgpu plumbing land here in Task 6.

use bytemuck::{Pod, Zeroable};

/// PR 1 vertex format: position + baked color. PR 3 splits color into a
/// parallel buffer to support paint-mode toggling without remesh.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
```

Re-run: `cargo test --bin worldgen_viz cache`

Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/world/cache.rs src/bin/worldgen_viz/render/scene.rs Cargo.toml Cargo.lock
git commit -m "viz: LRU chunk + mesh cache; introduce Vertex type (PR 1)"
```

---

### Task 5: Viz mesher

**Files:**
- Modify: `src/bin/worldgen_viz/world/mesher.rs`.

Walk a `DenseChunk` voxel-by-voxel, emit a quad per visible face, color by block-type. Boundary faces are conservatively emitted (PR 7 may add neighbor culling). This is the PR-1 minimum: matches today's `worldgen_bridge.rs::emit_chunk_faces` but lives in the new module.

- [ ] **Step 1: Write the failing tests**

Replace `src/bin/worldgen_viz/world/mesher.rs`:

```rust
//! Per-chunk meshing for the viz: positions + baked block-paint colors.
//!
//! Reuses oxium::mesher::Face for face direction enumeration but emits the
//! viz Vertex (position + color) instead of the game's atlas-aware vertex.

use crate::render::scene::Vertex;
use glam::Vec3;
use oxium::mesher::Face;
use oxium::voxel::block::Block;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::{ChunkCoord, LocalPos, CHUNK_DIM_U};

#[derive(Default)]
pub struct VizMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn mesh_chunk(coord: ChunkCoord, chunk: &DenseChunk) -> VizMesh {
    let dim = CHUNK_DIM_U as i32;
    let origin = Vec3::new(
        (coord.0.x * dim) as f32,
        (coord.0.y * dim) as f32,
        (coord.0.z * dim) as f32,
    );
    let mut mesh = VizMesh::default();
    for lz in 0..dim {
        for ly in 0..dim {
            for lx in 0..dim {
                let local = LocalPos(glam::UVec3::new(lx as u32, ly as u32, lz as u32));
                let block = chunk.get(local);
                if !is_solid(block) {
                    continue;
                }
                let p = origin + Vec3::new(lx as f32, ly as f32, lz as f32);
                let color = color_for(block);
                for face in Face::all() {
                    let [nx, ny, nz] = face.normal();
                    let (nlx, nly, nlz) = (lx + nx, ly + ny, lz + nz);
                    let neighbor_solid = if nlx < 0 || nlx >= dim || nly < 0 || nly >= dim || nlz < 0 || nlz >= dim {
                        false
                    } else {
                        let nl = LocalPos(glam::UVec3::new(nlx as u32, nly as u32, nlz as u32));
                        is_solid(chunk.get(nl))
                    };
                    if !neighbor_solid {
                        emit_face(p, color, face, &mut mesh);
                    }
                }
            }
        }
    }
    mesh
}

fn emit_face(p: Vec3, color: [f32; 3], face: Face, mesh: &mut VizMesh) {
    let base = mesh.vertices.len() as u32;
    let quad: [Vec3; 4] = match face {
        Face::PosX => [
            Vec3::new(1., 0., 0.),
            Vec3::new(1., 0., 1.),
            Vec3::new(1., 1., 1.),
            Vec3::new(1., 1., 0.),
        ],
        Face::NegX => [
            Vec3::new(0., 0., 1.),
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
            Vec3::new(0., 1., 1.),
        ],
        Face::PosY => [
            Vec3::new(0., 1., 0.),
            Vec3::new(1., 1., 0.),
            Vec3::new(1., 1., 1.),
            Vec3::new(0., 1., 1.),
        ],
        Face::NegY => [
            Vec3::new(0., 0., 1.),
            Vec3::new(1., 0., 1.),
            Vec3::new(1., 0., 0.),
            Vec3::new(0., 0., 0.),
        ],
        Face::PosZ => [
            Vec3::new(1., 0., 1.),
            Vec3::new(0., 0., 1.),
            Vec3::new(0., 1., 1.),
            Vec3::new(1., 1., 1.),
        ],
        Face::NegZ => [
            Vec3::new(0., 0., 0.),
            Vec3::new(1., 0., 0.),
            Vec3::new(1., 1., 0.),
            Vec3::new(0., 1., 0.),
        ],
    };
    for v in &quad {
        mesh.vertices.push(Vertex {
            position: (p + *v).into(),
            color,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn is_solid(b: Block) -> bool {
    !matches!(b, Block::Air | Block::Water)
}

fn color_for(b: Block) -> [f32; 3] {
    match b {
        Block::Stone => [0.55, 0.55, 0.55],
        Block::Dirt => [0.50, 0.32, 0.18],
        Block::Grass => [0.30, 0.65, 0.25],
        Block::Sand => [0.92, 0.85, 0.62],
        Block::Snow => [0.95, 0.95, 0.97],
        _ => [0.4, 0.4, 0.4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn empty_chunk_meshes_to_no_vertices() {
        let chunk = DenseChunk::empty();
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 0);
        assert_eq!(m.indices.len(), 0);
    }

    #[test]
    fn single_stone_at_origin_has_six_faces() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 6 * 4, "expected 6 quads = 24 verts");
        assert_eq!(m.indices.len(), 6 * 6, "expected 6 quads = 36 indices");
    }

    #[test]
    fn adjacent_solids_hide_shared_face() {
        let mut chunk = DenseChunk::empty();
        chunk.set(LocalPos(glam::UVec3::new(0, 0, 0)), Block::Stone);
        chunk.set(LocalPos(glam::UVec3::new(1, 0, 0)), Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        // 2 cubes share one internal face; each cube has 6 faces, minus 2
        // hidden = 10 visible quads.
        assert_eq!(m.vertices.len(), 10 * 4);
    }

    #[test]
    fn boundary_face_emitted_when_chunk_edge() {
        // Block at local (31, 0, 0): the +X face of this block is at the
        // chunk's outer +X edge; we emit it (no neighbor info in PR 1).
        let mut chunk = DenseChunk::empty();
        let edge = LocalPos(glam::UVec3::new(31, 0, 0));
        chunk.set(edge, Block::Stone);
        let m = mesh_chunk(ChunkCoord(IVec3::ZERO), &chunk);
        assert_eq!(m.vertices.len(), 6 * 4);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --bin worldgen_viz mesher`

Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/world/mesher.rs
git commit -m "viz: per-chunk mesher with block-paint colors (PR 1)"
```

---

### Task 6: Render pipeline (wgpu scene)

**Files:**
- Modify: `src/bin/worldgen_viz/render/scene.rs` (the file already has the Vertex struct from Task 4).
- Modify: `src/bin/worldgen_viz/render/shader.wgsl`.
- Modify: `src/bin/worldgen_viz/render/mod.rs` (today's render.rs, already in place).

Implement the wgpu pipeline that takes a `Camera`, a vec of `VizMesh`es, and draws them. This is mostly a port of today's `scene.rs` — adapted to (a) take any `Camera`, (b) hold many meshes instead of a single vbo, (c) live in the new module layout.

- [ ] **Step 1: Write the shader file**

Replace `src/bin/worldgen_viz/render/shader.wgsl` (created in Task 1):

```wgsl
struct Camera { view_proj: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> cam: Camera;

struct VsIn { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, };
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_y: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    out.world_y = in.pos.y;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Mild fake lighting: brighten high blocks slightly. Matches today's
    // viewer aesthetic — replace with real shading in PR 3.
    let lit = in.color * (0.7 + 0.3 * clamp((in.world_y - 40.0) / 100.0, 0.0, 1.0));
    return vec4<f32>(lit, 1.0);
}
```

- [ ] **Step 2: Implement `SceneRenderer` in render/scene.rs**

Replace `src/bin/worldgen_viz/render/scene.rs` (keep the existing `Vertex` definition at the top of the file; add the pipeline below):

```rust
//! 3D scene pipeline: takes any `Camera` and a collection of mesh buffers.

use crate::camera::Camera;
use crate::world::cache::ChunkMeshGpu;
use bytemuck::{Pod, Zeroable};
use oxium::voxel::coords::ChunkCoord;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

pub struct ChunkGpuBuffers {
    pub vbo: wgpu::Buffer,
    pub ibo: wgpu::Buffer,
    pub index_count: u32,
}

pub struct SceneRenderer {
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    chunks: HashMap<ChunkCoord, ChunkGpuBuffers>,
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera ubo"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viz shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viz pl"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viz pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let depth_view = create_depth(device, depth_format, width, height);
        Self {
            pipeline,
            depth_view,
            depth_format,
            camera_buffer,
            camera_bind_group,
            chunks: HashMap::new(),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth_view = create_depth(device, self.depth_format, width, height);
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, cam: &dyn Camera, aspect: f32) {
        let vp = cam.view_proj(aspect);
        let u = CameraUniform { view_proj: vp.to_cols_array_2d() };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[u]));
    }

    /// Upload (or replace) the GPU buffers for one chunk's mesh.
    pub fn upload_chunk(&mut self, device: &wgpu::Device, coord: ChunkCoord, mesh: &ChunkMeshGpu) {
        if mesh.indices.is_empty() {
            self.chunks.remove(&coord);
            return;
        }
        let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vbo"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk ibo"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        self.chunks.insert(
            coord,
            ChunkGpuBuffers {
                vbo,
                ibo,
                index_count: mesh.indices.len() as u32,
            },
        );
    }

    pub fn drop_chunk(&mut self, coord: ChunkCoord) {
        self.chunks.remove(&coord);
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for buf in self.chunks.values() {
            pass.set_vertex_buffer(0, buf.vbo.slice(..));
            pass.set_index_buffer(buf.ibo.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buf.index_count, 0, 0..1);
        }
    }
}

fn create_depth(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    w: u32,
    h: u32,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("viz depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
```

- [ ] **Step 3: Verify build**

Run: `cargo check --bin worldgen_viz`

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/render/scene.rs src/bin/worldgen_viz/render/shader.wgsl
git commit -m "viz: SceneRenderer pipeline + WGSL shader (PR 1)"
```

---

### Task 7: World streaming pipeline

**Files:**
- Modify: `src/bin/worldgen_viz/world/mod.rs`.

`World` ties together: stream radius, chunk cache, mesher, and the `oxium::jobs::Jobs` pool. Its main entry points:

- `World::new(generator, config_holder)` — sets up channels and pool.
- `World::request_chunks(coords: &[ChunkCoord])` — spawn fill jobs for coords not in cache.
- `World::drain_results(scene: &mut SceneRenderer, device: &wgpu::Device)` — pop completed jobs from the channel, mesh them, upload to GPU.
- `World::frame(camera_pos, scene, device)` — convenience: compute radius, request missing, drain.

For PR 1 we do NOT use `Jobs::spawn_gen` directly because it expects a paletted chunk + lighting. Instead, we spawn fill+mesh on rayon via a viz-local helper that calls `Generator::fill_chunk` and our viz `mesh_chunk`.

- [ ] **Step 1: Write the failing test**

Replace `src/bin/worldgen_viz/world/mod.rs`:

```rust
//! Streaming world: cache + job pipeline.

pub mod cache;
pub mod invalidate;
pub mod mesher;
pub mod stream;

use crate::world::cache::{ChunkCache, ChunkMeshGpu};
use crate::world::mesher::mesh_chunk;
use crate::world::stream::{chunks_in_radius, StreamRadius};
use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::Vec3;
use oxium::voxel::chunk::DenseChunk;
use oxium::voxel::coords::ChunkCoord;
use oxium::worldgen::Generator;
use rayon::ThreadPool;
use std::collections::HashSet;
use std::sync::Arc;

pub enum ChunkJobResult {
    Filled {
        coord: ChunkCoord,
        chunk: Arc<DenseChunk>,
        mesh: ChunkMeshGpu,
    },
}

pub struct World {
    generator: Arc<Generator>,
    cache: ChunkCache,
    radius: StreamRadius,
    pool: Arc<ThreadPool>,
    tx: Sender<ChunkJobResult>,
    rx: Receiver<ChunkJobResult>,
    in_flight: HashSet<ChunkCoord>,
}

impl World {
    pub fn new(generator: Arc<Generator>, radius: StreamRadius, capacity: usize) -> Self {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get().saturating_sub(1).max(1))
                .thread_name(|i| format!("viz-{i}"))
                .build()
                .expect("rayon pool"),
        );
        let (tx, rx) = unbounded();
        Self {
            generator,
            cache: ChunkCache::new(capacity),
            radius,
            pool,
            tx,
            rx,
            in_flight: HashSet::new(),
        }
    }

    pub fn radius(&self) -> StreamRadius {
        self.radius
    }

    /// Spawn fill+mesh jobs for any coord in `coords` that is neither
    /// already cached nor currently in flight. Order matters: callers
    /// pass coords already sorted closest-first.
    pub fn request_chunks(&mut self, coords: &[ChunkCoord]) {
        for &coord in coords {
            if self.cache.has_mesh(coord) || self.in_flight.contains(&coord) {
                continue;
            }
            self.in_flight.insert(coord);
            let tx = self.tx.clone();
            let gen = self.generator.clone();
            self.pool.spawn(move || {
                let mut chunk = DenseChunk::empty();
                gen.fill_chunk(coord, &mut chunk);
                let mesh = mesh_chunk(coord, &chunk);
                let gpu = ChunkMeshGpu {
                    vertices: Arc::new(mesh.vertices),
                    indices: Arc::new(mesh.indices),
                };
                let _ = tx.send(ChunkJobResult::Filled {
                    coord,
                    chunk: Arc::new(chunk),
                    mesh: gpu,
                });
            });
        }
    }

    /// Drain any completed jobs into the cache. Returns the coords whose
    /// meshes just landed (caller uploads them to the GPU).
    pub fn drain_results(&mut self) -> Vec<(ChunkCoord, ChunkMeshGpu)> {
        let mut out = Vec::new();
        while let Ok(r) = self.rx.try_recv() {
            match r {
                ChunkJobResult::Filled { coord, chunk, mesh } => {
                    self.in_flight.remove(&coord);
                    self.cache.put_chunk(coord, chunk);
                    self.cache.put_mesh(
                        coord,
                        ChunkMeshGpu {
                            vertices: mesh.vertices.clone(),
                            indices: mesh.indices.clone(),
                        },
                    );
                    out.push((coord, mesh));
                }
            }
        }
        out
    }

    /// Coords currently cached (meshed) — for the scene renderer to know
    /// which buffers to keep.
    pub fn cached_mesh_coords(&self) -> Vec<ChunkCoord> {
        self.cache.meshed_coords()
    }

    pub fn cache_mut(&mut self) -> &mut ChunkCache {
        &mut self.cache
    }

    /// Convenience: request chunks within radius of camera position.
    pub fn request_around(&mut self, pos: Vec3) {
        let coords = chunks_in_radius(pos, self.radius);
        self.request_chunks(&coords);
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::config::WorldgenConfig;

    #[test]
    fn world_eventually_meshes_requested_chunks() {
        let config = WorldgenConfig::bundled_default().unwrap();
        let generator = Arc::new(Generator::new_with_config(42, config));
        let mut world = World::new(generator, StreamRadius { xz: 0, y: 0 }, 8);
        world.request_around(Vec3::new(0.0, 96.0, 0.0));

        // Spin-wait up to 5 seconds for the single center chunk to mesh.
        let start = std::time::Instant::now();
        while world.drain_results().is_empty() {
            if start.elapsed() > std::time::Duration::from_secs(5) {
                panic!("center chunk did not mesh within 5s");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        assert!(world.cached_mesh_coords().len() >= 1);
        assert_eq!(world.in_flight_len(), 0);
    }
}
```

- [ ] **Step 2: Verify `Generator::new_with_config` exists**

Run: `grep -n "new_with_config\|with_config" /Users/fdatoo/Developer/oxium/src/worldgen/mod.rs | head -3`

If only `with_config` exists, change the test's `Generator::new_with_config(42, config)` to `Generator::with_config(42, oxium::worldgen::config::ConfigHolder::new(config))`. Update accordingly.

- [ ] **Step 3: Add `num_cpus` dependency if missing**

Run: `grep '^num_cpus' Cargo.toml`. If absent: edit `Cargo.toml`, add `num_cpus = "1.16"` to `[dependencies]`.

- [ ] **Step 4: Run the test**

Run: `cargo test --bin worldgen_viz world::tests::world_eventually_meshes`

Expected: PASS within 5 s.

- [ ] **Step 5: Commit**

```bash
git add src/bin/worldgen_viz/world/mod.rs Cargo.toml Cargo.lock
git commit -m "viz: streaming World — rayon-driven fill+mesh pipeline (PR 1)"
```

---

### Task 8: Config invalidation

**Files:**
- Modify: `src/bin/worldgen_viz/world/invalidate.rs`.

When the active `WorldgenConfig` changes, the chunk cache is invalidated (wiped) so subsequent frame ticks reload chunks against the new config. PR 1 is the conservative path: `Invalidator::on_config_change(cache)` wipes everything; no field-aware locality.

- [ ] **Step 1: Write the failing tests**

Replace `src/bin/worldgen_viz/world/invalidate.rs`:

```rust
//! Cache invalidation on config change. PR 1: wipe-all; PR 7 may add
//! field-aware partial invalidation if profiling shows it's needed.

use crate::world::cache::ChunkCache;
use oxium::worldgen::config::WorldgenConfig;
use std::sync::Arc;

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

    /// Call once per frame. If the revision has advanced since the last
    /// check, wipe the cache and synchronise. Returns `true` if it wiped.
    pub fn maybe_wipe(&mut self, cache: &mut ChunkCache) -> bool {
        if self.current != self.last_config_revision {
            cache.clear();
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
    use crate::world::cache::ChunkMeshGpu;
    use glam::IVec3;
    use oxium::voxel::chunk::DenseChunk;
    use oxium::voxel::coords::ChunkCoord;

    fn coord(x: i32, y: i32, z: i32) -> ChunkCoord {
        ChunkCoord(IVec3::new(x, y, z))
    }

    #[test]
    fn no_bump_means_no_wipe() {
        let mut inv = Invalidator::new();
        let mut cache = ChunkCache::new(4);
        cache.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(!inv.maybe_wipe(&mut cache));
        assert!(cache.has_chunk(coord(0, 0, 0)));
    }

    #[test]
    fn bump_then_check_wipes_once() {
        let mut inv = Invalidator::new();
        let mut cache = ChunkCache::new(4);
        cache.put_chunk(coord(0, 0, 0), Arc::new(DenseChunk::empty()));
        cache.put_mesh(coord(0, 0, 0), ChunkMeshGpu::empty());
        inv.bump();
        assert!(inv.maybe_wipe(&mut cache));
        assert_eq!(cache.len(), (0, 0));
        // Second check after the same bump is a no-op.
        cache.put_chunk(coord(1, 0, 0), Arc::new(DenseChunk::empty()));
        assert!(!inv.maybe_wipe(&mut cache));
        assert!(cache.has_chunk(coord(1, 0, 0)));
    }
}
```

(`Arc` import needed; `WorldgenConfig` is referenced only via the doc comment; the test imports `Arc` via `use std::sync::Arc;` — already implicit in the file's top-level imports.)

- [ ] **Step 2: Run tests**

Run: `cargo test --bin worldgen_viz invalidate`

Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/world/invalidate.rs
git commit -m "viz: config-change cache invalidation (PR 1)"
```

---

### Task 9: Session + AppState

**Files:**
- Modify: `src/bin/worldgen_viz/session.rs`.
- Modify: `src/bin/worldgen_viz/paint.rs`.
- Modify: `src/bin/worldgen_viz/app.rs`.

`Session` is one viewable world (PR 4 enables multiple sessions for A/B compare). `AppState` holds the single PR-1 session, UI state, last frame time, and the `Invalidator`.

- [ ] **Step 1: Implement PaintMode stub**

Replace `src/bin/worldgen_viz/paint.rs`:

```rust
//! Paint modes for the 3D viewport. PR 1 ships only Block; PR 3 adds
//! biome, height-delta, density, cave-distance, plate, slope.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    Block,
}

impl Default for PaintMode {
    fn default() -> Self {
        PaintMode::Block
    }
}
```

- [ ] **Step 2: Implement Session**

Replace `src/bin/worldgen_viz/session.rs`:

```rust
//! A "session": one Generator + cameras + chunk cache wired up.
//! PR 4 will let the AppState hold two sessions for A/B compare.

use crate::camera::{Camera, FlyCamera, OrbitCamera};
use crate::paint::PaintMode;
use crate::world::invalidate::Invalidator;
use crate::world::stream::StreamRadius;
use crate::world::World;
use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};
use oxium::worldgen::Generator;
use std::sync::Arc;

pub enum CamKind {
    Fly,
    Orbit,
}

pub struct Session {
    pub seed: u64,
    pub config: ConfigHolder,
    pub generator: Arc<Generator>,
    pub world: World,
    pub fly: FlyCamera,
    pub orbit: OrbitCamera,
    pub cam_kind: CamKind,
    pub paint: PaintMode,
    pub invalidator: Invalidator,
}

impl Session {
    pub fn new(seed: u64, config: WorldgenConfig) -> Self {
        let holder = ConfigHolder::new(config);
        let generator = Arc::new(Generator::with_config(seed, holder.clone()));
        let world = World::new(generator.clone(), StreamRadius::DEFAULT, 1024);
        Self {
            seed,
            config: holder,
            generator,
            world,
            fly: FlyCamera::new(),
            orbit: OrbitCamera::new(),
            cam_kind: CamKind::Fly,
            paint: PaintMode::default(),
            invalidator: Invalidator::new(),
        }
    }

    pub fn camera(&self) -> &dyn Camera {
        match self.cam_kind {
            CamKind::Fly => &self.fly,
            CamKind::Orbit => &self.orbit,
        }
    }

    pub fn toggle_camera(&mut self) {
        self.cam_kind = match self.cam_kind {
            CamKind::Fly => CamKind::Orbit,
            CamKind::Orbit => CamKind::Fly,
        };
    }
}
```

- [ ] **Step 3: Implement AppState**

Replace `src/bin/worldgen_viz/app.rs`:

```rust
//! AppState: aggregates session + UI state + frame timing.

use crate::session::Session;
use oxium::worldgen::config::WorldgenConfig;
use std::time::Instant;

pub struct AppState {
    pub session: Session,
    pub mouse_down: bool,
    pub last_cursor: Option<(f64, f64)>,
    pub last_frame: Instant,
    pub last_regen_ms: Option<f32>,
    pub check_mode: bool,
}

impl AppState {
    pub fn new(seed: u64, config: WorldgenConfig, check_mode: bool) -> Self {
        Self {
            session: Session::new(seed, config),
            mouse_down: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_regen_ms: None,
            check_mode,
        }
    }

    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        dt.min(0.1)
    }
}
```

- [ ] **Step 4: Verify build**

Run: `cargo check --bin worldgen_viz`

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/bin/worldgen_viz/session.rs src/bin/worldgen_viz/app.rs src/bin/worldgen_viz/paint.rs
git commit -m "viz: Session + AppState + PaintMode (PR 1)"
```

---

### Task 10: Dashboard layout (placeholder panels)

**Files:**
- Modify: `src/bin/worldgen_viz/layout.rs`.

Dense Dashboard: top toolbar, left config panel (today's `widgets::cfg_panels` reused), center 3D viewport (transparent CentralPanel so the wgpu scene shows through), right "Probe" placeholder, bottom status bar. Resizable splitters via egui.

- [ ] **Step 1: Implement layout::dashboard**

Replace `src/bin/worldgen_viz/layout.rs`:

```rust
//! Dense Dashboard layout shell for the viz.

use crate::app::AppState;
use crate::session::CamKind;
use crate::widgets::cfg_panels::{
    biomes_panel, caves_panel, climate_panel, density_panel, graph_panel, preset_panel,
    surface_panel,
};
use egui::Context;
use oxium::worldgen::config::WorldgenConfig;

pub struct LayoutResult {
    pub dirty: bool,
    pub reset_camera: bool,
    pub force_regen: bool,
}

pub fn dashboard(ctx: &Context, app: &mut AppState) -> LayoutResult {
    let mut out = LayoutResult {
        dirty: false,
        reset_camera: false,
        force_regen: false,
    };

    // Top toolbar — paint mode, camera-kind toggle, regen button.
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Paint:");
            ui.selectable_value(&mut app.session.paint, crate::paint::PaintMode::Block, "Block");

            ui.separator();

            ui.label("Camera:");
            let is_fly = matches!(app.session.cam_kind, CamKind::Fly);
            if ui.selectable_label(is_fly, "Fly (WASD)").clicked() && !is_fly {
                app.session.toggle_camera();
            }
            if ui.selectable_label(!is_fly, "Orbit").clicked() && is_fly {
                app.session.toggle_camera();
            }

            ui.separator();

            if ui.button("Reset camera").clicked() {
                out.reset_camera = true;
            }
            if ui.button("[R] Regen").clicked() {
                out.force_regen = true;
            }
        });
    });

    // Left panel: config editors (today's panels, unchanged).
    egui::SidePanel::left("config_panel")
        .resizable(true)
        .default_width(380.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let cfg_arc = app.session.config.load();
                let mut cfg: WorldgenConfig = (*cfg_arc).clone();
                let mut local_dirty = false;
                local_dirty |= density_panel(ui, &mut cfg.density);
                ui.separator();
                local_dirty |= climate_panel(ui, &mut cfg.climate);
                ui.separator();
                local_dirty |= caves_panel(ui, &mut cfg.caves);
                ui.separator();
                local_dirty |= biomes_panel(ui, &mut cfg.biomes);
                ui.separator();
                local_dirty |= surface_panel(ui, &mut cfg.surface);
                ui.separator();
                local_dirty |= preset_panel(ui, &mut cfg);
                ui.separator();
                graph_panel(ui, &cfg.density);
                if local_dirty {
                    app.session.config.swap(cfg);
                    app.session.invalidator.bump();
                    out.dirty = true;
                }
            });
        });

    // Right panel: probe placeholder (PR 2 fills this).
    egui::SidePanel::right("probe_panel")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Probe");
            ui.label("Click a column in the 3D view to inspect (PR 2).");
        });

    // Bottom status bar.
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let pos = app.session.camera().position();
            ui.label(format!("pos ({:.0}, {:.0}, {:.0})", pos.x, pos.y, pos.z));
            ui.separator();
            ui.label(format!("seed {}", app.session.seed));
            ui.separator();
            ui.label(format!(
                "chunks meshed: {}",
                app.session.world.cached_mesh_coords().len()
            ));
            ui.separator();
            ui.label(format!("in-flight: {}", app.session.world.in_flight_len()));
            ui.separator();
            if let Some(ms) = app.last_regen_ms {
                ui.label(format!("last regen: {:.0} ms", ms));
            }
        });
    });

    // Central panel = transparent so the wgpu scene shows through.
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show(ctx, |_ui| {});

    out
}
```

- [ ] **Step 2: Verify build**

Run: `cargo check --bin worldgen_viz`

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add src/bin/worldgen_viz/layout.rs
git commit -m "viz: Dense Dashboard layout shell with placeholder panels (PR 1)"
```

---

### Task 11: Main loop + winit integration

**Files:**
- Modify: `src/bin/worldgen_viz/main.rs`.

Wire everything: winit `ApplicationHandler`, wgpu device + egui + scene, per-frame: collect input → update camera → request chunks → drain results → upload meshes → render. Keybindings: WASD/QE (fly), Shift (boost), mouse-right-drag (look), R (force regen), Space (toggle auto-regen — for parity with old viz; in PR 1 it's a no-op since edits always regen), O (toggle camera).

- [ ] **Step 1: Implement main.rs**

Replace `src/bin/worldgen_viz/main.rs`:

```rust
//! Worldgen tuning visualizer — PR 1 (streaming skeleton).

mod app;
mod camera;
mod layout;
mod paint;
mod render;
mod session;
mod widgets;
mod world;

use crate::app::AppState;
use crate::render::scene::SceneRenderer;
use crate::render::RenderState;
use crate::session::CamKind;
use clap::Parser;
use oxium::worldgen::config::WorldgenConfig;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Parser, Debug)]
#[command(version)]
struct Cli {
    /// World seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Run one frame and exit (for CI smoke tests).
    #[arg(long, default_value_t = false)]
    check: bool,
}

struct VizApp {
    cli: Cli,
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    scene: Option<SceneRenderer>,
    state: AppState,
    /// True between W/A/S/D press and release.
    keys: KeyState,
    /// One-frame flag set by --check to terminate after first redraw.
    quit_after_render: bool,
}

#[derive(Default)]
struct KeyState {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    q: bool,
    e: bool,
    shift: bool,
}

impl VizApp {
    fn new(cli: Cli) -> Self {
        let config = WorldgenConfig::bundled_default().expect("bundled default.ron");
        let check = cli.check;
        Self {
            state: AppState::new(cli.seed, config, check),
            cli,
            window: None,
            render: None,
            scene: None,
            keys: KeyState::default(),
            quit_after_render: check,
        }
    }
}

impl ApplicationHandler for VizApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Oxium worldgen visualizer (PR 1)")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 1000.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let render = RenderState::new(window.clone());
        let scene = SceneRenderer::new(
            &render.device,
            render.surface_config.format,
            render.surface_config.width,
            render.surface_config.height,
        );
        self.window = Some(window);
        self.render = Some(render);
        self.scene = Some(scene);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(render), Some(scene)) =
            (self.window.as_ref(), self.render.as_mut(), self.scene.as_mut())
        else {
            return;
        };
        let _ = render.egui_state.on_window_event(window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                render.resize(size.width, size.height);
                scene.resize(&render.device, size.width, size.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.state.mouse_down {
                    if let Some((px, py)) = self.state.last_cursor {
                        let dx = (position.x - px) as f32;
                        let dy = (position.y - py) as f32;
                        match self.state.session.cam_kind {
                            CamKind::Fly => self.state.session.fly.look(dx, dy),
                            CamKind::Orbit => {
                                self.state.session.orbit.yaw -= dx * 0.005;
                                self.state.session.orbit.pitch =
                                    (self.state.session.orbit.pitch + dy * 0.005).clamp(-1.5, 1.5);
                            }
                        }
                    }
                }
                self.state.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.state.mouse_down = state == ElementState::Pressed;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let amt = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 4.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.5,
                };
                if matches!(self.state.session.cam_kind, CamKind::Orbit) {
                    self.state.session.orbit.distance =
                        (self.state.session.orbit.distance - amt).clamp(50.0, 768.0);
                } else {
                    self.state.session.fly.speed =
                        (self.state.session.fly.speed + amt).clamp(2.0, 200.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        KeyCode::KeyQ => self.keys.q = pressed,
                        KeyCode::KeyE => self.keys.e = pressed,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.shift = pressed,
                        KeyCode::KeyO if pressed => self.state.session.toggle_camera(),
                        KeyCode::KeyR if pressed => self.state.session.invalidator.bump(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let dt = self.state.dt();

                // Camera input (fly cam only).
                if matches!(self.state.session.cam_kind, CamKind::Fly) {
                    let fwd = (self.keys.w as i32 - self.keys.s as i32) as f32;
                    let strafe = (self.keys.d as i32 - self.keys.a as i32) as f32;
                    let vert = (self.keys.e as i32 - self.keys.q as i32) as f32;
                    self.state
                        .session
                        .fly
                        .translate(fwd, strafe, vert, self.keys.shift, dt);
                }

                // Stream + invalidate.
                let t0 = Instant::now();
                if self
                    .state
                    .session
                    .invalidator
                    .maybe_wipe(self.state.session.world.cache_mut())
                {
                    scene.clear();
                }
                let cam_pos = self.state.session.camera().position();
                self.state.session.world.request_around(cam_pos);
                let landed = self.state.session.world.drain_results();
                for (coord, mesh) in landed {
                    scene.upload_chunk(&render.device, coord, &mesh);
                }
                if t0.elapsed().as_secs_f32() > 0.001 {
                    self.state.last_regen_ms = Some(t0.elapsed().as_secs_f32() * 1000.0);
                }

                // egui pass.
                let raw_input = render.egui_state.take_egui_input(window);
                let mut reset_cam = false;
                let mut force_regen = false;
                let app_ref = &mut self.state;
                let full_output = render.egui_ctx.clone().run(raw_input, |ctx| {
                    let r = crate::layout::dashboard(ctx, app_ref);
                    reset_cam = r.reset_camera;
                    force_regen = r.force_regen;
                });
                if reset_cam {
                    self.state.session.fly = crate::camera::FlyCamera::new();
                    self.state.session.orbit = crate::camera::OrbitCamera::new();
                }
                if force_regen {
                    self.state.session.invalidator.bump();
                }
                render
                    .egui_state
                    .handle_platform_output(window, full_output.platform_output.clone());

                // Camera uniform + frame composition.
                let aspect = render.surface_config.width as f32
                    / render.surface_config.height.max(1) as f32;
                scene.update_camera(&render.queue, self.state.session.camera(), aspect);
                if let Err(e) = render_frame(render, scene, full_output) {
                    eprintln!("render: {e:?}");
                }

                if self.quit_after_render {
                    event_loop.exit();
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(
    render: &mut RenderState,
    scene: &SceneRenderer,
    full_output: egui::FullOutput,
) -> Result<(), wgpu::SurfaceError> {
    let frame = render.surface.get_current_texture()?;
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("viz frame"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.08,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: scene.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        scene.render(&mut pass);
    }

    let paint_jobs = render
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [render.surface_config.width, render.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    for (id, image_delta) in &full_output.textures_delta.set {
        render.egui_renderer.update_texture(&render.device, &render.queue, *id, image_delta);
    }
    render.egui_renderer.update_buffers(&render.device, &render.queue, &mut encoder, &paint_jobs, &screen);
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();
        render.egui_renderer.render(&mut pass, &paint_jobs, &screen);
    }

    render.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
    for id in &full_output.textures_delta.free {
        render.egui_renderer.free_texture(id);
    }
    Ok(())
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = VizApp::new(cli);
    event_loop.run_app(&mut app).expect("run loop");
}
```

- [ ] **Step 2: Add `clap` dependency if missing**

Run: `grep '^clap' Cargo.toml`. If absent: add `clap = { version = "4", features = ["derive"] }` to `[dependencies]`.

- [ ] **Step 3: Build**

Run: `cargo build --bin worldgen_viz`

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/main.rs Cargo.toml Cargo.lock
git commit -m "viz: main loop with fly-cam input + WASD/orbit toggle (PR 1)"
```

---

### Task 12: Smoke test (CI gate)

**Files:**
- Modify: `src/bin/worldgen_viz/main.rs` (test block at bottom).

The `--check` flag exits after rendering one frame. A unit-test-shaped check confirms `Cli::parse_from(["worldgen_viz", "--check"]).check == true` so the flag stays wired through PRs.

- [ ] **Step 1: Add CLI parse test**

Append to `src/bin/worldgen_viz/main.rs`:

```rust

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn check_flag_parses() {
        let cli = Cli::parse_from(["worldgen_viz", "--check"]);
        assert!(cli.check);
        assert_eq!(cli.seed, 42);
    }

    #[test]
    fn seed_flag_parses() {
        let cli = Cli::parse_from(["worldgen_viz", "--seed", "1337"]);
        assert_eq!(cli.seed, 1337);
        assert!(!cli.check);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --bin worldgen_viz tests::`

Expected: 2 tests pass.

- [ ] **Step 3: Smoke run**

Run: `cargo run --bin worldgen_viz -- --check --seed 42`

Expected: window briefly opens, one frame renders, process exits 0. Exit code: `echo $?` → `0`.

If the window-open is problematic in a headless environment, document the limitation; CI on macOS GH runners with `WINIT_BACKEND=...` should work. (Linux CI may need xvfb; defer to PR 7's CI work.)

- [ ] **Step 4: Commit**

```bash
git add src/bin/worldgen_viz/main.rs
git commit -m "viz: CLI parse tests + --check smoke gate (PR 1)"
```

---

### Task 13: Final cleanup + Cargo.toml metadata

**Files:**
- Modify: `Cargo.toml` (bin section, if needed).
- Modify: `docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md` — strike-through PR 1 in the phasing table.

- [ ] **Step 1: Verify the bin entry still works**

Run: `grep -A2 'name = "worldgen_viz"' Cargo.toml`

Expected: a `[[bin]]` block with name and path. No edits needed unless absent.

- [ ] **Step 2: Update the PR 1 row in the spec phasing table**

In `docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md`, find the table row beginning `| **1** | viz: streaming skeleton |` and change `**1**` to `**1** ✅` so subsequent PR plans clearly see PR 1 is done.

- [ ] **Step 3: Run the full viz test suite**

Run: `cargo test --bin worldgen_viz`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-19-worldgen-viz-redesign-design.md
git commit -m "viz: PR 1 (streaming skeleton) complete — mark in spec"
```

---

## Self-Review

**Spec coverage for PR 1:**
- Streaming chunk cache → Tasks 3, 4, 7 ✓
- Fly camera (WASD + look) → Tasks 2, 11 ✓
- Reuse mesher (viz-side adapter due to atlas-bound Vertex) → Task 5; documented decision ✓
- Block paint mode only → Task 9 (PaintMode::Block) + Task 5 (color_for) ✓
- Dashboard layout with placeholder panels → Task 10 ✓
- Replaces old binary → Task 1 (delete + scaffold) ✓
- `--seed` and `--check` flags → Tasks 11, 12 ✓
- Config invalidation on edit → Task 8, wired in Task 11 ✓

**Placeholder scan:** none — every code block is concrete.

**Type consistency:**
- `ChunkMeshGpu` defined in Task 4 (cache.rs), used in Tasks 5–7, 9, 11 ✓
- `Vertex` defined in Task 4 (render/scene.rs), used in Tasks 4, 5, 6 ✓
- `StreamRadius` defined in Task 3, used in Tasks 7, 9 ✓
- `Camera` trait defined in Task 2, used in Tasks 6, 9, 11 ✓
- `Invalidator` defined in Task 8, used in Tasks 9, 11 ✓
- `Session.config: ConfigHolder` (Task 9), edited in Task 10 (calls `config.swap(cfg)`) — consistent ✓
- `Generator::with_config(seed, ConfigHolder)` signature used in Task 7 test + Task 9 — verified by inspecting `oxium::worldgen::mod.rs` at planning time ✓

**Scope check:** PR 1 is a coherent vertical slice. It produces a runnable, testable binary that streams the world and is strictly better than today (fly cam, streaming, dashboard, --check smoke test).

**One known limitation, deliberate:** PR 1's smoke test (`cargo run -- --check`) opens a window. Pure headless CI requires either an offscreen wgpu adapter or xvfb; the spec's headless render mode lands in PR 6 with proper offscreen rendering. PR 1's smoke test is a "best effort" CI gate that runs in environments where a window can briefly open.
