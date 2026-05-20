# Lighting PR 2 — Colored Block Light

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen the block-light pipeline from one 4-bit scalar to three 4-bit channels (R/G/B). Block emissions become `[u8; 3]` triples. The BFS, the chunk storage (both dense and paletted), and the on-disk region format all carry RGB. The mesher and shader continue to consume a single scalar brightness derived as `max(R, G, B)` — visual output is byte-stable against PR 1's baselines.

**Architecture:** The data widens; the rendering doesn't (yet). Per-pixel sampling of an RGB light volume lands in PR 3 along with shader changes. PR 2 is pure infrastructure: it gives PR 3 the data to consume. Saves bump format version with a backward-compat read path (old `block_light: Packed4Bit` upgrades to `R = G = B = block_light`).

**Tech Stack:** Rust, bincode 1.x for on-disk chunk serialization, existing `Packed4Bit` helper for 4-bit packing. No new crates.

**Spec:** `docs/superpowers/specs/2026-05-20-lighting-design.md` (Data model + CPU light propagation sections)

---

## Files

**Create:**
- `tests/lighting_colored.rs` — integration test for colored emission via a custom block registry

**Modify:**
- `src/voxel/block.rs` — `BlockInfo::emission: [u8; 3]`, update all 11 block initializers (`Air`, `Stone`, `Dirt`, `Grass`, `Sand`, `Water`, `Wood`, `Leaves`, `Torch`, `Lava`, `Snow`)
- `src/voxel/chunk.rs` — `DenseChunk::block_rgb: Box<[u16; CHUNK_VOL]>` replaces `block_light`. `PalettedChunk::block_red`/`block_green`/`block_blue` replace `block_light`. Update `compress` / `decompress`.
- `src/lighting/mod.rs` — BFS over the packed-u16 RGB cell type; `snapshot_face_boundaries` widens
- `src/mesher/greedy.rs` — `light_at` closure pulls R/G/B and emits `max(R, G, B)` as the legacy scalar light byte
- `src/mesher/lod.rs` — downsampling reads `block_rgb`, derives scalar brightness the same way
- `src/persistence/region.rs` — versioned wire format with a 4-byte magic prefix; v1 (legacy) read path upgrades to v2 by setting `R = G = B = old block_light`

**Do not touch in this PR:**
- Any shader file (`assets/shaders/*.wgsl`) — colored block light becomes visible in PR 3
- The vertex format produced by the mesher — still 1 byte for `(sky << 4) | block_brightness`
- `src/render/mod.rs` (no GPU-side change)
- `src/worldgen/` (worldgen reads/writes blocks but not light)

---

## Backward-compatibility design

Region files written before this PR contain a bincode'd `PalettedChunk` with the old shape (palette + indices + sky_light + block_light, all `Packed4Bit`). Bincode is offset-sensitive, so adding fields would silently misread old data as garbage.

**Approach: magic prefix on new writes.**

```
write: zstd(magic + bincode(new PalettedChunk))
read:  if first 4 bytes == magic: zstd → strip magic → bincode-deser new struct
       else:                       zstd → bincode-deser old struct, upgrade in memory
```

The legacy upgrade rule is `R = G = B = block_light` — old saves' torches load as bright-white-equivalent (since brightness is `max(R,G,B) = block_light`), indistinguishable from today's render until the chunk is re-relit.

Magic value: `*b"OX2\0"` (4 bytes). The leading byte `O` (0x4F) is unambiguous against the old format: bincode encodes a `Vec<Block>` length as a u64-le, whose first byte is the low byte of the palette length (typically 1-16). 0x4F = 79 is well outside that range, so misreading old data as new is impossible.

---

## Tasks

Each task ends with `cargo test` + a screenshot regression check using the PR 1 helper:
```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/<scene>.png <args...>
python3 tests/screenshots/diff.py tests/screenshots/baseline_<scene>.png /tmp/<scene>.png
```

The regression bar is "within noise floor" on all 5 baselines (noon, underwater, cave, sunset, fog_horizon). PR 2 makes no visual change.

### Task 1: Widen `BlockInfo::emission` to `[u8; 3]`

**Files:**
- Modify: `src/voxel/block.rs:114-115` (struct field), and all `infos[X] = BlockInfo {...}` initializers (~11 spots)

- [ ] **Step 1: Change the struct field.**

In `src/voxel/block.rs`, find `pub struct BlockInfo { …  pub emission: u8 }` and change to:

```rust
pub struct BlockInfo {
    // …existing fields…
    /// Per-channel block-light emission, 0..15 each (R, G, B). Non-zero
    /// channels seed that channel of the block-light BFS.
    pub emission: [u8; 3],
}
```

- [ ] **Step 2: Update the default initializer.**

In `BlockRegistry::new` (around line 163), the loop fill uses `emission: 0,`. Change to `emission: [0, 0, 0],`.

- [ ] **Step 3: Update every block initializer.**

For each of `Air`, `Stone`, `Dirt`, `Grass`, `Sand`, `Water`, `Wood`, `Leaves`, `Snow`, change `emission: 0,` → `emission: [0, 0, 0],`. For Torch, change `emission: 13,` → `emission: [13, 13, 13],`. For Lava, change `emission: 15,` → `emission: [15, 15, 15],`. (Use grep first to make sure no initializer is missed.)

```bash
grep -n "emission:" src/voxel/block.rs
```

- [ ] **Step 4: Update existing tests.**

In `src/voxel/block.rs` tests (search for `info(Block::Torch).emission`), change `> 0` checks to:
```rust
let e = r.info(Block::Torch).emission;
assert!(e[0] > 0 || e[1] > 0 || e[2] > 0);
```

- [ ] **Step 5: Compile-check.**

Run: `cargo build`
Expected: errors in `src/lighting/mod.rs` (`info.emission > 0`, `info.emission` value reads). Those are addressed in Task 3. For now `cargo build` should fail there — that's fine.

- [ ] **Step 6: Defer commit.**

Don't commit yet — Task 2 lands the chunk-side widening alongside, and the build needs to come back clean before we lock anything in. Skip to Task 2.

---

### Task 2: Widen `DenseChunk` from `block_light: u8` array to `block_rgb: u16` array

The packed u16 cell is `(R << 8) | (G << 4) | B`, R/G/B each 4 bits. Top 4 bits of the u16 are unused (set to 0).

**Files:**
- Modify: `src/voxel/chunk.rs` — `DenseChunk` struct, `new_filled`/`empty`, tests

- [ ] **Step 1: Update DenseChunk struct.**

In `src/voxel/chunk.rs:28-35`, replace the `block_light` field with `block_rgb`:

```rust
pub struct DenseChunk {
    pub blocks: Box<[Block; CHUNK_VOL]>,
    pub sky_light: Box<[u8; CHUNK_VOL]>,
    /// Per-voxel packed block-light: `(R << 8) | (G << 4) | B`,
    /// each channel 4 bits (0..=15). Top 4 bits unused. Populated by
    /// the colored block-light BFS in `lighting::recompute_chunk`.
    pub block_rgb: Box<[u16; CHUNK_VOL]>,
}
```

Update the doc comment above the struct (around line 22-23): `64 KB blocks + 32 KB sky-light + 64 KB block-rgb`.

- [ ] **Step 2: Update `new_filled`.**

Replace `block_light: Box::new([0u8; CHUNK_VOL])` with `block_rgb: Box::new([0u16; CHUNK_VOL])`.

- [ ] **Step 3: Add packing helpers.**

Add at the top of `src/voxel/chunk.rs` (above `DenseChunk`):
```rust
/// Pack `(R, G, B)` channels (each 0..=15) into the u16 layout used
/// by `DenseChunk::block_rgb`. Out-of-range inputs are masked to 4 bits.
#[inline]
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 & 0x0F) << 8) | ((g as u16 & 0x0F) << 4) | (b as u16 & 0x0F)
}

/// Inverse of `pack_rgb`: extract R/G/B (each 0..=15) from a packed cell.
#[inline]
pub fn unpack_rgb(cell: u16) -> (u8, u8, u8) {
    let r = ((cell >> 8) & 0x0F) as u8;
    let g = ((cell >> 4) & 0x0F) as u8;
    let b = (cell & 0x0F) as u8;
    (r, g, b)
}

/// Scalar brightness for a packed cell, used by the mesher/legacy shader
/// path until PR 3 swaps to a 3D light volume.
#[inline]
pub fn rgb_brightness(cell: u16) -> u8 {
    let (r, g, b) = unpack_rgb(cell);
    r.max(g).max(b)
}
```

- [ ] **Step 4: Update tests in this file.**

In the `tests` module (around line 270), find:
```rust
d.block_light[200] = 0x7;
…
assert_eq!(d2.block_light[200], 0x7);
```
Change to:
```rust
d.block_rgb[200] = pack_rgb(0x7, 0x0, 0x0);
…
assert_eq!(d2.block_rgb[200], pack_rgb(0x7, 0x0, 0x0));
```
The test verifies the packing roundtrips through compress + decompress; the channel choice is arbitrary.

- [ ] **Step 5: Defer compile-check.**

Build will still fail (PalettedChunk and lighting reference `block_light`). Continue to Task 3.

---

### Task 3: Update lighting BFS for RGB propagation

**Files:**
- Modify: `src/lighting/mod.rs` — `block_light` function renamed/rewritten, `seed_from_neighbors` handles u16, `snapshot_face_boundaries` widens

- [ ] **Step 1: Rewrite the `block_light` function as `block_rgb`.**

Find `fn block_light(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, reg: &BlockRegistry)` and replace its body. The new function name is `block_rgb`. Replace at lines 228-262:

```rust
/// Compute RGB block light: seed each emissive block with its three-channel
/// emission, BFS outward, per-channel attenuation by 1 per air step (cost-1
/// extra in water). Boundary cells are also seeded from neighbour chunks so
/// colored sources continue to glow into adjacent chunks rather than
/// hard-cutting at the seam.
fn block_rgb(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, reg: &BlockRegistry) {
    use crate::voxel::chunk::{pack_rgb, unpack_rgb};

    chunk.block_rgb.iter_mut().for_each(|v| *v = 0);

    // Queue entry: (x, y, z, packed u16). Channels propagate together so
    // the BFS visits each cell once for all three.
    let mut q: VecDeque<(i32, i32, i32, u16)> = VecDeque::new();
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let [er, eg, eb] = reg.info(chunk.blocks[idx]).emission;
                if er > 0 || eg > 0 || eb > 0 {
                    let cell = pack_rgb(er, eg, eb);
                    chunk.block_rgb[idx] = cell;
                    q.push_back((x, y, z, cell));
                }
            }
        }
    }
    seed_from_neighbors(chunk, neighbors, BfsChannel::BlockRgb);
    // Re-enqueue any boundary cell the seed-pass bumped to non-zero so
    // the BFS picks them up.
    for z in 0..D {
        for y in 0..D {
            for x in 0..D {
                let idx = LocalPos(UVec3::new(x as u32, y as u32, z as u32)).to_index();
                let on_boundary = x == 0 || y == 0 || z == 0
                    || x == D - 1 || y == D - 1 || z == D - 1;
                let cell = chunk.block_rgb[idx];
                if on_boundary && cell != 0 {
                    let (r, g, b) = unpack_rgb(cell);
                    // Only enqueue if any channel can still propagate (>= 2).
                    if r >= 2 || g >= 2 || b >= 2 {
                        q.push_back((x, y, z, cell));
                    }
                }
            }
        }
    }
    bfs_spread_rgb(&mut q, chunk, reg);
}
```

- [ ] **Step 2: Add an internal `BfsChannel` enum and rewrite `seed_from_neighbors`.**

Replace the existing `seed_from_neighbors` (lines 111-148) with a version that handles all four channels:

```rust
#[derive(Copy, Clone)]
enum BfsChannel {
    Sky,
    BlockRgb,
}

fn seed_from_neighbors(chunk: &mut DenseChunk, neighbors: &Neighbors<'_>, channel: BfsChannel) {
    use crate::mesher::Face;
    use crate::voxel::chunk::{pack_rgb, unpack_rgb};
    for face in Face::all() {
        if matches!(channel, BfsChannel::Sky) && face == Face::PosY {
            continue;
        }
        let Some(n) = neighbors.chunks[face as usize] else {
            continue;
        };
        for v in 0..D {
            for u in 0..D {
                let (our_lp, their_lp) = mirror_boundary(face, u, v);
                let our_idx = our_lp.to_index();
                let their_idx = their_lp.to_index();
                match channel {
                    BfsChannel::Sky => {
                        let seeded = n.sky_light[their_idx].saturating_sub(1);
                        if seeded > chunk.sky_light[our_idx] {
                            chunk.sky_light[our_idx] = seeded;
                        }
                    }
                    BfsChannel::BlockRgb => {
                        let (tr, tg, tb) = unpack_rgb(n.block_rgb[their_idx]);
                        let (or, og, ob) = unpack_rgb(chunk.block_rgb[our_idx]);
                        let new_r = or.max(tr.saturating_sub(1));
                        let new_g = og.max(tg.saturating_sub(1));
                        let new_b = ob.max(tb.saturating_sub(1));
                        if new_r != or || new_g != og || new_b != ob {
                            chunk.block_rgb[our_idx] = pack_rgb(new_r, new_g, new_b);
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 3: Update the sky pass call site.**

Find `seed_from_neighbors(chunk, neighbors, /* is_sky */ true)` in `sky_light` (around line 82) and change to `seed_from_neighbors(chunk, neighbors, BfsChannel::Sky)`. Remove the old `is_sky: bool` parameter from older calls — there were two: the sky-pass one (Task step) and the old block-pass one (now gone, replaced by `block_rgb` above which calls `BfsChannel::BlockRgb`).

- [ ] **Step 4: Add `bfs_spread_rgb` next to the existing `bfs_spread`.**

Add this function after the existing `bfs_spread`:

```rust
/// RGB BFS step. Spreads all three channels simultaneously to neighbours
/// within this chunk only; cross-chunk spread is the streaming system's
/// responsibility (via `light_dirty` cascade).
fn bfs_spread_rgb(
    q: &mut VecDeque<(i32, i32, i32, u16)>,
    chunk: &mut DenseChunk,
    reg: &BlockRegistry,
) {
    use crate::mesher::Face;
    use crate::voxel::chunk::{pack_rgb, unpack_rgb};
    while let Some((x, y, z, cell)) = q.pop_front() {
        let (lr, lg, lb) = unpack_rgb(cell);
        if lr <= 1 && lg <= 1 && lb <= 1 {
            continue;
        }
        for face in Face::all() {
            let [dx, dy, dz] = face.normal();
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= D || ny >= D || nz >= D {
                continue;
            }
            let idx = LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)).to_index();
            let info = reg.info(chunk.blocks[idx]);
            if info.opaque {
                continue;
            }
            let cost: u8 = if chunk.blocks[idx] == Block::Water { 3 } else { 1 };
            let attenuation = cost.saturating_sub(1);
            // Per-channel propagation: each channel attenuates independently.
            let prop_r = lr.saturating_sub(1).saturating_sub(attenuation);
            let prop_g = lg.saturating_sub(1).saturating_sub(attenuation);
            let prop_b = lb.saturating_sub(1).saturating_sub(attenuation);
            let (cur_r, cur_g, cur_b) = unpack_rgb(chunk.block_rgb[idx]);
            let new_r = cur_r.max(prop_r);
            let new_g = cur_g.max(prop_g);
            let new_b = cur_b.max(prop_b);
            if new_r != cur_r || new_g != cur_g || new_b != cur_b {
                chunk.block_rgb[idx] = pack_rgb(new_r, new_g, new_b);
                q.push_back((nx, ny, nz, chunk.block_rgb[idx]));
            }
        }
    }
}
```

- [ ] **Step 5: Update `bfs_spread` (the sky BFS).**

Replace the existing `bfs_spread` body — the `is_sky` bool branch goes away because it now only handles sky. Rename to `bfs_spread_sky`:

```rust
fn bfs_spread_sky(
    q: &mut VecDeque<(i32, i32, i32, u8)>,
    chunk: &mut DenseChunk,
    reg: &BlockRegistry,
) {
    use crate::mesher::Face;
    while let Some((x, y, z, level)) = q.pop_front() {
        if level <= 1 {
            continue;
        }
        let next = level - 1;
        for face in Face::all() {
            let [dx, dy, dz] = face.normal();
            let (nx, ny, nz) = (x + dx, y + dy, z + dz);
            if nx < 0 || ny < 0 || nz < 0 || nx >= D || ny >= D || nz >= D {
                continue;
            }
            let idx = LocalPos(UVec3::new(nx as u32, ny as u32, nz as u32)).to_index();
            let info = reg.info(chunk.blocks[idx]);
            if info.opaque {
                continue;
            }
            let cost: u8 = if chunk.blocks[idx] == Block::Water { 3 } else { 1 };
            let prop = next.saturating_sub(cost.saturating_sub(1));
            if prop > chunk.sky_light[idx] {
                chunk.sky_light[idx] = prop;
                q.push_back((nx, ny, nz, prop));
            }
        }
    }
}
```

Update its caller (in `sky_light` around line 97) from `bfs_spread(&mut q, chunk, reg, /* is_sky */ true)` to `bfs_spread_sky(&mut q, chunk, reg)`.

- [ ] **Step 6: Update `recompute_chunk`.**

In `recompute_chunk` (around line 30), change `block_light(chunk, neighbors, reg)` to `block_rgb(chunk, neighbors, reg)`.

- [ ] **Step 7: Update `snapshot_face_boundaries` for the new channels.**

Replace the function body (around line 161-184). The output array bytes/cell grows from 2 to 4 (sky + R + G + B):

```rust
pub fn snapshot_face_boundaries(chunk: &crate::voxel::chunk::DenseChunk) -> [Vec<u8>; 6] {
    use crate::mesher::Face;
    use crate::voxel::chunk::unpack_rgb;
    std::array::from_fn(|face_i| {
        let face = match face_i {
            0 => Face::PosX,
            1 => Face::NegX,
            2 => Face::PosY,
            3 => Face::NegY,
            4 => Face::PosZ,
            5 => Face::NegZ,
            _ => unreachable!(),
        };
        let mut out = Vec::with_capacity((D * D * 4) as usize);
        for v in 0..D {
            for u in 0..D {
                let (our_lp, _) = mirror_boundary(face, u, v);
                let idx = our_lp.to_index();
                let (r, g, b) = unpack_rgb(chunk.block_rgb[idx]);
                out.push(chunk.sky_light[idx]);
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
        out
    })
}
```

- [ ] **Step 8: Update the existing in-file tests.**

In the `tests` module of `src/lighting/mod.rs`, the `torch_emits_block_light_with_falloff` test reads `chunk.block_light[…]`. Change to use `chunk.block_rgb` + `unpack_rgb`:

```rust
#[test]
fn torch_emits_block_light_with_falloff() {
    use crate::voxel::chunk::unpack_rgb;
    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
    let r = BlockRegistry::new();
    recompute_chunk(&mut c, &empty_neighbors(), &r);
    let center = LocalPos(UVec3::new(16, 16, 16)).to_index();
    let adj    = LocalPos(UVec3::new(17, 16, 16)).to_index();
    let far    = LocalPos(UVec3::new(20, 16, 16)).to_index();
    let (cr, cg, cb) = unpack_rgb(c.block_rgb[center]);
    let (ar, _ag, _ab) = unpack_rgb(c.block_rgb[adj]);
    let (fr, _fg, _fb) = unpack_rgb(c.block_rgb[far]);
    assert_eq!(cr, 13);
    assert_eq!(cg, 13);
    assert_eq!(cb, 13);
    assert!(ar >= 11, "adj R too low: {}", ar);
    assert!(fr < ar);
}
```

- [ ] **Step 9: Compile.**

Run: `cargo build`
Expected: errors only in `mesher/greedy.rs`, `mesher/lod.rs`, `voxel/chunk.rs::PalettedChunk`, and `persistence/region.rs`. Tasks 4, 5, 6, 7 address those.

---

### Task 4: Update mesher to read block_rgb + emit scalar brightness

The vertex format doesn't change in PR 2 — we still pack `(sky << 4) | block_brightness` into one byte. Brightness is derived as `max(R, G, B)`.

**Files:**
- Modify: `src/mesher/greedy.rs:95-128` (the `light_at` closure inside `build_chunk_mesh`)
- Modify: `src/mesher/lod.rs:125-138` (the per-column light sampling in `downsample_to_top_quads`)

- [ ] **Step 1: Update greedy mesher's `light_at`.**

In `src/mesher/greedy.rs`, find the closure at line 95-128 that reads `chunk.sky_light[idx]` and `chunk.block_light[idx]`. Both reads of `chunk.block_light[idx]` and `n.block_light[idx]` become reads of `chunk.block_rgb[idx]` / `n.block_rgb[idx]`, passed through `rgb_brightness`.

Change the inline read:
```rust
return (chunk.sky_light[idx] & 0x0F) << 4 | (chunk.block_light[idx] & 0x0F);
```
to:
```rust
let brightness = crate::voxel::chunk::rgb_brightness(chunk.block_rgb[idx]);
return (chunk.sky_light[idx] & 0x0F) << 4 | (brightness & 0x0F);
```

And the neighbor branch:
```rust
return (n.sky_light[idx] & 0x0F) << 4 | (n.block_light[idx] & 0x0F);
```
to:
```rust
let brightness = crate::voxel::chunk::rgb_brightness(n.block_rgb[idx]);
return (n.sky_light[idx] & 0x0F) << 4 | (brightness & 0x0F);
```

- [ ] **Step 2: Update LOD downsample.**

In `src/mesher/lod.rs` around line 127-128:
```rust
light_sky = src.sky_light[idx];
light_blk = src.block_light[idx];
```
becomes:
```rust
light_sky = src.sky_light[idx];
light_blk = crate::voxel::chunk::rgb_brightness(src.block_rgb[idx]);
```

- [ ] **Step 3: Compile.**

Run: `cargo build`
Expected: only PalettedChunk and persistence errors remain.

---

### Task 5: Update `PalettedChunk` with three Packed4Bit channels

**Files:**
- Modify: `src/voxel/chunk.rs` — `PalettedChunk` struct, `all_air`, `compress`, `decompress`

- [ ] **Step 1: Replace the `block_light` field.**

In `src/voxel/chunk.rs:78-88`, the struct becomes:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalettedChunk {
    pub palette: Vec<Block>,
    pub indices: Packed4Bit,
    pub sky_light: Packed4Bit,
    /// Red channel of per-voxel block light (0..=15).
    pub block_red: Packed4Bit,
    /// Green channel of per-voxel block light (0..=15).
    pub block_green: Packed4Bit,
    /// Blue channel of per-voxel block light (0..=15).
    pub block_blue: Packed4Bit,
}
```

- [ ] **Step 2: Update `all_air`.**

```rust
pub fn all_air() -> Self {
    Self {
        palette: vec![Block::Air],
        indices: Packed4Bit::zeros(CHUNK_VOL),
        sky_light: Packed4Bit::zeros(CHUNK_VOL),
        block_red: Packed4Bit::zeros(CHUNK_VOL),
        block_green: Packed4Bit::zeros(CHUNK_VOL),
        block_blue: Packed4Bit::zeros(CHUNK_VOL),
    }
}
```

- [ ] **Step 3: Update `compress`.**

In the loop that packs light (around line 129-134), replace:
```rust
let mut sky = Packed4Bit::zeros(CHUNK_VOL);
let mut blk = Packed4Bit::zeros(CHUNK_VOL);
for i in 0..CHUNK_VOL {
    sky.set(i, dense.sky_light[i] & 0x0F);
    blk.set(i, dense.block_light[i] & 0x0F);
}
```
with:
```rust
let mut sky = Packed4Bit::zeros(CHUNK_VOL);
let mut r = Packed4Bit::zeros(CHUNK_VOL);
let mut g = Packed4Bit::zeros(CHUNK_VOL);
let mut b = Packed4Bit::zeros(CHUNK_VOL);
for i in 0..CHUNK_VOL {
    sky.set(i, dense.sky_light[i] & 0x0F);
    let (rr, gg, bb) = unpack_rgb(dense.block_rgb[i]);
    r.set(i, rr);
    g.set(i, gg);
    b.set(i, bb);
}
```

And the trailing struct literal becomes:
```rust
Self {
    palette,
    indices,
    sky_light: sky,
    block_red: r,
    block_green: g,
    block_blue: b,
}
```

- [ ] **Step 4: Update `decompress`.**

Replace lines 153-163 with:
```rust
let mut sky = Box::new([0u8; CHUNK_VOL]);
let mut block_rgb = Box::new([0u16; CHUNK_VOL]);
for i in 0..CHUNK_VOL {
    sky[i] = self.sky_light.get(i);
    block_rgb[i] = pack_rgb(
        self.block_red.get(i),
        self.block_green.get(i),
        self.block_blue.get(i),
    );
}
DenseChunk {
    blocks,
    sky_light: sky,
    block_rgb,
}
```

- [ ] **Step 5: Compile.**

Run: `cargo build`
Expected: only `persistence/region.rs` errors remain (it serialises/deserialises the old struct shape).

- [ ] **Step 6: Run the in-file roundtrip test.**

```bash
cargo test --release -p oxium --lib voxel::chunk
```
Expected: PASS for the chunk-roundtrip test (which now writes/reads `block_rgb[200] = pack_rgb(0x7, 0x0, 0x0)`).

---

### Task 6: Versioned region read/write with v1 upgrade path

**Files:**
- Modify: `src/persistence/region.rs` — add magic prefix to writes; on read, branch by magic
- Modify: `src/voxel/chunk.rs` — add a private `PalettedChunkV1` struct and `From<PalettedChunkV1> for PalettedChunk`

- [ ] **Step 1: Define a V1 mirror struct.**

At the bottom of `src/voxel/chunk.rs` (still in the `pub` mod since the persistence layer needs to deser into it), add:

```rust
/// Legacy v1 paletted-chunk layout used by region files written before
/// the colored-block-light upgrade. Only deserialized — never written.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PalettedChunkV1 {
    pub palette: Vec<Block>,
    pub indices: Packed4Bit,
    pub sky_light: Packed4Bit,
    pub block_light: Packed4Bit,
}

impl From<PalettedChunkV1> for PalettedChunk {
    /// Convert v1 (single-channel) to v2 (RGB) by mirroring the brightness
    /// into all three channels. Old saves render the same as today
    /// (max(R,G,B) = old block_light) until chunks are re-relit.
    fn from(v1: PalettedChunkV1) -> Self {
        Self {
            palette: v1.palette,
            indices: v1.indices,
            sky_light: v1.sky_light,
            block_red: v1.block_light.clone(),
            block_green: v1.block_light.clone(),
            block_blue: v1.block_light,
        }
    }
}
```

- [ ] **Step 2: Add the magic constant.**

At the top of `src/persistence/region.rs`:
```rust
/// Magic prefix written before bincode payload to mark the v2 (RGB
/// block light) chunk format. Old saves have no prefix; the read path
/// falls back to deserialising a `PalettedChunkV1` when the prefix is
/// absent. See PR 2 of the lighting overhaul for context.
const CHUNK_V2_MAGIC: &[u8; 4] = b"OX2\0";
```

- [ ] **Step 3: Prefix the write payload.**

Replace the existing serialize line (around line 168):
```rust
let blob_bin = bincode::serialize(data).map_err(|e| RegionError::Decode(e.to_string()))?;
```
with:
```rust
let mut blob_bin = Vec::with_capacity(4 + 64 * 1024);
blob_bin.extend_from_slice(CHUNK_V2_MAGIC);
let payload = bincode::serialize(data).map_err(|e| RegionError::Decode(e.to_string()))?;
blob_bin.extend_from_slice(&payload);
```

- [ ] **Step 4: Branch on the magic in `read_chunk`.**

Replace lines 226-229 with:
```rust
let blob_bin = zstd::stream::decode_all(blob.as_slice())
    .map_err(|e| RegionError::Zstd(e.to_string()))?;
let chunk: PalettedChunk = if blob_bin.starts_with(CHUNK_V2_MAGIC) {
    bincode::deserialize(&blob_bin[CHUNK_V2_MAGIC.len()..])
        .map_err(|e| RegionError::Decode(e.to_string()))?
} else {
    // Legacy v1 path: no magic prefix, single-channel block_light.
    let v1: crate::voxel::chunk::PalettedChunkV1 =
        bincode::deserialize(&blob_bin).map_err(|e| RegionError::Decode(e.to_string()))?;
    v1.into()
};
Ok(chunk)
```

- [ ] **Step 5: Compile.**

Run: `cargo build`
Expected: clean build. All shape mismatches resolved.

- [ ] **Step 6: Add a v1-roundtrip integration test.**

In `src/persistence/region.rs` test module, add:
```rust
#[test]
fn legacy_v1_blob_upgrades_on_read() {
    use crate::voxel::block::Block;
    use crate::voxel::chunk::{Packed4Bit, PalettedChunkV1, CHUNK_VOL};
    let mut block_light = Packed4Bit::zeros(CHUNK_VOL);
    block_light.set(123, 0x9);
    let v1 = PalettedChunkV1 {
        palette: vec![Block::Air, Block::Torch],
        indices: Packed4Bit::zeros(CHUNK_VOL),
        sky_light: Packed4Bit::zeros(CHUNK_VOL),
        block_light,
    };

    // Write a v1-shaped blob to disk by hand (NO magic prefix).
    let tmp = tempfile::tempdir().unwrap();
    let region_path = tmp.path().join("r.0.0.bin");
    {
        let payload = bincode::serialize(&v1).unwrap();
        let zstd = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
        let header = vec![0u8; (super::HEADER_SECTORS * super::SECTOR) as usize];
        let mut header = header;
        // Sector layout: header is 1 sector; payload starts at sector index 1.
        let end_sector = super::HEADER_SECTORS;
        let blob_len = zstd.len() as u32;
        let needed_sectors =
            ((4 + zstd.len()) as u64).div_ceil(super::SECTOR).max(1);
        let slot = super::slot_index(crate::voxel::coords::ChunkCoord(
            glam::IVec3::new(0, 0, 0),
        ));
        let entry = ((end_sector as u32) << 8) | (needed_sectors as u32).min(0xFF);
        header[slot * 4..slot * 4 + 4].copy_from_slice(&entry.to_le_bytes());
        let mut f = std::fs::File::create(&region_path).unwrap();
        use std::io::Write;
        f.write_all(&header).unwrap();
        f.write_all(&blob_len.to_le_bytes()).unwrap();
        f.write_all(&zstd).unwrap();
        let pad = (needed_sectors * super::SECTOR) as usize - (4 + zstd.len());
        if pad > 0 {
            f.write_all(&vec![0u8; pad]).unwrap();
        }
    }

    // Now read it through the regular path and confirm v1→v2 upgrade.
    let chunk = read_chunk(
        &region_path,
        crate::voxel::coords::ChunkCoord(glam::IVec3::new(0, 0, 0)),
    )
    .unwrap();
    // All three channels should equal the original block_light.
    assert_eq!(chunk.block_red.get(123), 0x9);
    assert_eq!(chunk.block_green.get(123), 0x9);
    assert_eq!(chunk.block_blue.get(123), 0x9);
}
```

(The `HEADER_SECTORS`, `SECTOR`, and `slot_index` constants are private in `region.rs`; this test is in the same module so it has access. Adjust `pub(super)` visibility if compilation complains.)

- [ ] **Step 7: Run all tests.**

```bash
cargo test --release
```
Expected: 227+ existing tests pass, plus the new `legacy_v1_blob_upgrades_on_read` test. The chunk roundtrip in `src/voxel/chunk.rs` should also still pass.

- [ ] **Step 8: Commit Tasks 1-6 as one logical change.**

Tasks 1-6 are interdependent (the build doesn't compile cleanly until they're all in). Commit as one unit:
```bash
git add src/voxel/block.rs src/voxel/chunk.rs src/lighting/mod.rs src/mesher/greedy.rs src/mesher/lod.rs src/persistence/region.rs
git commit -m "$(cat <<'EOF'
feat(lighting): RGB block light infrastructure (PR 2)

Block emissions become [u8; 3]. DenseChunk holds packed u16 cells
(R<<8 | G<<4 | B) in block_rgb. PalettedChunk holds three Packed4Bit
channels. The BFS propagates all three channels simultaneously,
per-channel attenuation by one per air step.

The mesher and shader continue to consume max(R, G, B) as the legacy
scalar brightness, so visual output is unchanged. PR 3 introduces
per-pixel sampling of the RGB volume.

Region files gain a 4-byte magic prefix ("OX2\\0") on writes. Old
saves without the prefix deserialize as PalettedChunkV1 and upgrade
in-memory by setting R = G = B = old block_light.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Colored-emission integration test

Adds end-to-end test coverage for the RGB path using a custom registry that overrides `Torch` to red-only emission. Validates that R propagates, G/B do not, and that cross-chunk seam carries color.

**Files:**
- Create: `tests/lighting_colored.rs`

- [ ] **Step 1: Add a custom-emission helper to `BlockRegistry`.**

In `src/voxel/block.rs`, add a public test-only method:
```rust
impl BlockRegistry {
    /// Test-only: override one block's emission. Used by integration
    /// tests for colored block light without introducing new block kinds.
    #[doc(hidden)]
    pub fn set_emission_for_tests(&mut self, b: Block, e: [u8; 3]) {
        self.infos[b as usize].emission = e;
    }
}
```

(The `infos: [BlockInfo; BLOCK_COUNT]` field is currently private. Either keep this method as the only accessor, or change visibility — easiest is to keep the helper.)

- [ ] **Step 2: Write the integration test file.**

Create `tests/lighting_colored.rs`:
```rust
//! Integration test for PR 2's colored block-light propagation. Uses a
//! custom `BlockRegistry` that overrides Torch to red-only emission so
//! we can assert per-channel falloff without needing a new block kind.

use glam::UVec3;
use oxium::lighting::recompute_chunk;
use oxium::voxel::block::{Block, BlockRegistry};
use oxium::voxel::chunk::{unpack_rgb, DenseChunk, Neighbors};
use oxium::voxel::coords::LocalPos;

fn empty_neighbors() -> Neighbors<'static> {
    Neighbors { chunks: [None; 6] }
}

#[test]
fn red_torch_propagates_only_red() {
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);

    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(16, 16, 16)), Block::Torch);
    recompute_chunk(&mut c, &empty_neighbors(), &reg);

    let center = LocalPos(UVec3::new(16, 16, 16)).to_index();
    let adj    = LocalPos(UVec3::new(17, 16, 16)).to_index();
    let (cr, cg, cb) = unpack_rgb(c.block_rgb[center]);
    let (ar, ag, ab) = unpack_rgb(c.block_rgb[adj]);

    assert_eq!(cr, 15);
    assert_eq!(cg, 0);
    assert_eq!(cb, 0);
    assert_eq!(ar, 14);
    assert_eq!(ag, 0, "green should not appear from a red-only source");
    assert_eq!(ab, 0, "blue should not appear from a red-only source");
}

#[test]
fn mixed_emissions_compose_per_channel() {
    // Two adjacent-but-not-touching torches, one red-only and one
    // green-only. The mid-point should pick up both R and G but no B.
    let mut reg = BlockRegistry::new();
    reg.set_emission_for_tests(Block::Torch, [15, 0, 0]);
    reg.set_emission_for_tests(Block::Lava,  [0, 15, 0]);

    let mut c = DenseChunk::empty();
    c.set(LocalPos(UVec3::new(10, 16, 16)), Block::Torch);
    c.set(LocalPos(UVec3::new(20, 16, 16)), Block::Lava);
    recompute_chunk(&mut c, &empty_neighbors(), &reg);

    let mid = LocalPos(UVec3::new(15, 16, 16)).to_index();
    let (r, g, b) = unpack_rgb(c.block_rgb[mid]);
    assert!(r > 0,  "red should reach midpoint from torch");
    assert!(g > 0,  "green should reach midpoint from lava");
    assert_eq!(b, 0, "blue must remain zero");
    // Symmetric distance (5 cells from each), so r == g.
    assert_eq!(r, g, "channels are symmetric");
}
```

- [ ] **Step 3: Run the new tests.**

```bash
cargo test --release --test lighting_colored
```
Expected: 2 passed.

- [ ] **Step 4: Run full test suite for sanity.**

```bash
cargo test --release
```
Expected: 229+ passed, 0 failed, 4 ignored.

- [ ] **Step 5: Commit.**

```bash
git add src/voxel/block.rs tests/lighting_colored.rs
git commit -m "$(cat <<'EOF'
test(lighting): integration test for colored block-light BFS

Covers per-channel propagation (red source produces no green/blue)
and per-channel composition (red + green sources mid-point lights
both channels but not blue). Uses a test-only registry hook to
override block emissions without introducing new block kinds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Screenshot regression sweep

Verify the existing visual baselines from PR 1 still pass under the new RGB infrastructure. Since the mesher emits `max(R, G, B)` as the legacy scalar brightness, and existing emissions are uniform (Torch = [13,13,13], Lava = [15,15,15]), the rendered brightness should be byte-stable (= 13 and 15 respectively, same as today).

**Files:** none modified.

- [ ] **Step 1: Capture all 5 scenes.**

```bash
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/pr2_noon.png --look 45,-15 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/pr2_under.png --spawn 64,60,-12 --look 0,-10 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/pr2_cave.png --spawn 0,30,0 --look 0,-30 --time 0.5
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/pr2_sunset.png --look 90,-10 --time 0.78
OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit /tmp/pr2_fog.png --look 0,0 --time 0.5
```

- [ ] **Step 2: Diff each against its baseline.**

```bash
python3 tests/screenshots/diff.py tests/screenshots/baseline_noon_outdoor.png /tmp/pr2_noon.png
python3 tests/screenshots/diff.py tests/screenshots/baseline_underwater.png  /tmp/pr2_under.png
python3 tests/screenshots/diff.py tests/screenshots/baseline_cave.png        /tmp/pr2_cave.png
python3 tests/screenshots/diff.py tests/screenshots/baseline_sunset.png      /tmp/pr2_sunset.png
python3 tests/screenshots/diff.py tests/screenshots/baseline_fog_horizon.png /tmp/pr2_fog.png
```
Expected: each scene reports "DIFF: within noise floor".

**If a scene flags as regression:** the most likely cause is `max(R, G, B)` diverging from the old `block_light` value for some block. Verify Torch emission is `[13, 13, 13]` and Lava is `[15, 15, 15]` in `block.rs`. If a non-emissive block's emission was accidentally bumped, max would shift.

- [ ] **Step 3: Live smoke test.**

```bash
cargo run --release --bin oxium
```
- Walk around outdoor at noon — verify torches glow the same warmth as before
- Walk into water — verify underwater grade engages (still in composite from PR 1, unaffected by this PR)
- Walk underground near placed torches — verify their brightness profile is unchanged
- Exit cleanly with `Esc → /quit` or window close

- [ ] **Step 4: Final commit (none expected).**

Verify `git status` is clean. No baseline updates expected since this PR is visually byte-stable.

---

### Task 9: Wrap up

- [ ] **Step 1: Branch state check.**

```bash
git log main..HEAD --oneline
git status
```
Expected: 2 commits ahead of main, clean working tree.

- [ ] **Step 2: Hand off to finishing-a-development-branch.**

Per the established pattern, after the implementation is verified the next sub-skill is `superpowers:finishing-a-development-branch` (Option 1: merge to local main, prep a new worktree for PR 3).

---

## Self-review notes

Spec coverage:

| Spec requirement | Plan task |
|------------------|-----------|
| `BlockInfo::emission: [u8; 3]` | Task 1 |
| `DenseChunk::block_rgb: Box<[u16; CHUNK_VOL]>` | Task 2 |
| BFS over packed RGB cells (u16) | Task 3 |
| Cross-chunk seam carries colored light | Task 3 (seed_from_neighbors), Task 7 indirectly |
| Per-channel cost (1 per air, 3 per water) | Task 3 step 4 (`prop_r/g/b`) |
| Mesher continues to emit scalar brightness | Task 4 (`max(R,G,B)`) |
| PalettedChunk gains R/G/B Packed4Bit fields | Task 5 |
| Save format bump + legacy fallback (R=G=B=block_light) | Task 6 |
| Updated tests: red emission, mixed compose | Task 7 |
| Existing tests still pass | Task 6 step 7, Task 8 step 1 |
| **Out of scope for PR 2 (deferred):** ambient bounce, shader-side colored render | — (PR 3 / PR 6) |

Self-review complete. No placeholders. Function/type names consistent across tasks (`pack_rgb`, `unpack_rgb`, `rgb_brightness`, `BfsChannel`, `block_rgb`, `PalettedChunkV1`).
