# Part I — Foundations Content Implementation Plan (Phase 1, Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the seven remaining chapters of *Part I — Foundations* (1.2 Determinism, 1.3 Coherent Noise, 1.5 Voronoi, 1.6 Domain Warping, 1.7 Splines, 1.8 SDFs, 1.9 Trilerp), each with its inline widget where the design specifies one, and each grounded in the actual Oxium source code. Chapters 1.1 and 1.4 already exist from Plan 1.

**Architecture:** One task per chapter. Each task: (a) the implementer reads the relevant engine source to ground the chapter in real types/functions; (b) ports any math the widget needs into TypeScript under `docs/book/src/math/`; (c) adds a new entry to `doc_render parity` and updates `parity.json`; (d) builds the widget under `docs/book/src/widgets/` following the existing `FbmExplorer.tsx` pattern; (e) writes the MDX chapter; (f) builds; (g) commits.

**Tech Stack:** Same as Plan 1 — Docusaurus 3 + TypeScript + React 18; KaTeX for math; existing `simplex-noise@4` + `alea@1` widget deps; `doc_render` Rust binary for engine-authoritative images.

**Out of plan scope:** Part II, III, IV, V chapters (Plans 3–6). Plans 7+ for appendices.

---

## Pre-flight: enter a worktree

All work happens in an isolated git worktree, not on `main`.

Before starting Task 1, run:

```
EnterWorktree(name: "docs-part1")
```

(or `git worktree add .worktrees/docs-part1 -b docs/book-part1 && cd .worktrees/docs-part1` if no native tool is available).

This plan's file paths are relative to the repo root **inside the worktree**.

---

## Reference patterns the chapter tasks reuse

Read these files once at the start; every task uses them as templates and won't be repeated in the per-task instructions.

- **Existing widget pattern:** `docs/book/src/widgets/FbmExplorer.tsx` — React functional component with `useRef`/`useEffect`/`useState`, a canvas, sliders in a CSS grid, `useBookSeed` hook. New widgets follow this shape.
- **Existing TS math pattern:** `docs/book/src/math/fbm.ts` — exports a `make*` factory that returns the sampling function, has a `// parity:` comment naming the Rust source.
- **Existing parity pattern:** `src/bin/doc_render/parity_dump.rs` (the `cases` array and the per-function loop) — new entries follow the same shape; the TS-vs-Rust gap (different gradient permutations) means tolerance can stay loose where exact match is impossible, tight where pure-math algorithms allow it.
- **Existing chapter pattern:** `docs/book/content/part-1-foundations/1.4-fbm.mdx` — the canonical chapter skeleton (hook → intuition → build-it → in-engine → what-you-can-now-do → next), MDX imports at the top, `<LazyWidget>` wrapping the widget.

Every chapter ends with a "Next →" link to the following chapter (or back to Part II if it's 1.9).

---

## Task 1: Chapter 1.2 — Deterministic Randomness

The hash mixer in `src/worldgen/hash.rs` is the foundation of every "random-but-reproducible" decision in the engine. This chapter teaches what an integer hash mixer is, why we use one instead of a stateful PRNG, and how the engine threads `(seed, coord, salt)` through every roll.

**No widget.** Chapter is prose + code-reading.

**Files:**
- Modify: `docs/book/content/part-1-foundations/1.2-determinism.mdx` (currently a `<Stub />`)

### Steps

- [ ] **Step 1: Read the hash module**

Run:

```bash
cat src/worldgen/hash.rs
```

Note: `mix(seed: u64, words: &[i32]) -> u64`, `mix_u32`, `mix_unit` (returns f32 in `[0, 1)`), `mix_range(lo, hi)`. Read the doc comments at the top of the file — they describe the design ("xor-shift / golden-ratio multiply", "per-position multipliers keep `mix(s, &[a, b])` distinct from `mix(s, &[b, a])`", "finaliser — full avalanche").

Also grep for callsites:

```bash
grep -rn "hash::mix" src/worldgen/ | head -20
```

You'll see uses in `plates.rs` (plate seed jitter), `trees.rs` (tree-in-cell), `caves.rs` (chamber count, entrance rolls), `climate.rs` (Voronoi jitter), etc.

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-1-foundations/1.2-determinism.mdx` with the following. Adapt to whatever you found in Step 1 — if the function signatures differ from what's shown here, update.

```mdx
---
id: 1.2-determinism
title: 1.2 Deterministic Randomness
---

In the previous chapter we said the worldgen pipeline's job is to turn a `(seed, ChunkCoord)` into a `DenseChunk`. The word *"deterministic"* doing all the heavy lifting in that sentence is what makes the engine work. Every roll — *which plate is at this column, where do trees go in this cell, does this cave chamber roll a sinkhole entrance* — has to give the same answer every time, on every machine, with no per-process state.

The trick is to never use a stateful random number generator. Instead, every random roll is a pure function of three things: the world seed, some integer coordinate, and a salt that distinguishes one kind of roll from another. We call that function the **hash mixer**.

## The picture

A hash mixer takes a small bundle of integers and returns a single integer that *looks* random but is fully determined by its inputs.

```
mix(seed: u64, words: &[i32]) -> u64
```

`mix(42, &[10, 20])` always returns the same `u64`. But `mix(42, &[10, 20])` and `mix(42, &[20, 10])` return very different `u64`s — the function is sensitive to argument order, which lets us use the position of an integer to mean something (e.g., "this is an X coordinate; this is the salt that says I'm rolling for a sinkhole").

From a `u64`, you can derive any kind of randomness you need:

- A `[0, 1)` float: take the top 24 bits, divide by $2^{24}$.
- A roll in `[lo, hi)`: scale the float.
- A "yes with probability p" decision: compare the float to `p`.

Every random thing the engine does — every one — routes through this same function. No `rand::thread_rng()`, no `RandomState`, no `Instant::now()` mixed into a seed somewhere.

## Build it

A solid 64-bit mixer is about 15 lines of code:

```rust
// simplified for exposition; see src/worldgen/hash.rs for the real one.
pub fn mix(seed: u64, words: &[i32]) -> u64 {
    let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for (i, w) in words.iter().enumerate() {
        h ^= (*w as i64 as u64).wrapping_mul(LARGE_PRIMES[i % LARGE_PRIMES.len()]);
        h = h.rotate_left((i as u32 % 5) * 4 + 13);
    }
    // Finaliser: xor-shift + multiply, twice. Spreads any bias in the
    // input across all 64 output bits.
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    h
}
```

The two-stage finaliser at the bottom is the **avalanche** step. Without it, flipping one bit of input changes only ~one bit of output on average — which means correlated inputs (adjacent coords, similar seeds) produce correlated outputs, and the world would have visible patterns at small scales. With the finaliser, flipping any input bit flips roughly half the output bits — the gold standard for a non-cryptographic hash.

You can derive a `[0, 1)` float by taking the top 24 bits:

```rust
pub fn mix_unit(seed: u64, words: &[i32]) -> f32 {
    ((mix(seed, words) >> 40) as f32) / ((1u64 << 24) as f32)
}
```

Top bits, not low bits — the avalanche step distributes bias more evenly there. And $2^{24}$ is what `f32` mantissa precision supports cleanly.

## In the engine

The real mixer at `src/worldgen/hash.rs:mix` uses 5 large primes (rotating with `i % 5`) and gives every word position a different multiplier and rotation. That structure is what makes the mixer order-sensitive: rearranging the words array would route each word through a different multiplier slot.

Look at a typical call site — biome-edge Voronoi jitter:

```rust
// src/worldgen/climate.rs (paraphrased)
let jitter_x = (hash::mix_unit(seed, &[wx, wz, JITTER_X_SALT]) * 2.0 - 1.0) * BIOME_JITTER_RADIUS;
let jitter_z = (hash::mix_unit(seed, &[wx, wz, JITTER_Z_SALT]) * 2.0 - 1.0) * BIOME_JITTER_RADIUS;
```

Two rolls — `jitter_x` and `jitter_z` — at the same column. Each gets a different *salt* (`JITTER_X_SALT` vs `JITTER_Z_SALT`), which means the two rolls are uncorrelated even though they share `(seed, wx, wz)`. If both rolls used the same salt they'd produce identical values, and biome edges would jitter diagonally (only along $x = z$). The salt is what makes them independent.

This is the recurring trick — anywhere you see `hash::mix*(seed, &[...some coords..., SOME_SALT])`, the salt is buying independence between rolls that share the same coordinates.

## What you can now do in the code

You can read:

- `src/worldgen/hash.rs` — every mixer in the engine.
- Every `hash::mix*(...)` call site in the worldgen. The salts named in `tuning.rs` tell you what each roll is for.

When a future chapter says "the engine rolls a value here," you'll know it means `mix_unit(seed, &[coord, salt])`, and you can find the salt in `tuning.rs` to see what it controls.

## Next

→ [1.3 Coherent Noise](./1.3-coherent-noise.mdx) — the *other* fundamental primitive: turning hash-randomness into smooth spatial fields.
```

- [ ] **Step 3: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

Expected: `[SUCCESS]` line. If there's a math or import error, fix and rebuild.

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-1-foundations/1.2-determinism.mdx
git commit -m "content(book): write 1.2 Deterministic Randomness

Foundation chapter on the hash mixer in src/worldgen/hash.rs.
Teaches (seed, coord, salt) as the universal random-input bundle,
the avalanche step, and the salt-for-independence trick.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Chapter 1.3 — Coherent Noise + `NoiseField` widget

This chapter introduces Perlin and Simplex noise as the "smooth random scalar field" primitive. It sits between 1.2 (random integers) and 1.4 (which already exists and uses FBM = stacked noise).

**Widget:** `NoiseField` — a canvas + sliders for `seed`, plus a toggle for "show gradient grid overlay". Default values match the book seed.

**Files:**
- Create: `docs/book/src/widgets/NoiseField.tsx`
- Modify: `docs/book/content/part-1-foundations/1.3-coherent-noise.mdx`

### Steps

- [ ] **Step 1: Read where the engine uses coherent noise**

```bash
grep -rn "Simplex::new\|Perlin::new\|noise::Simplex\|noise::Perlin" src/worldgen/ | head -20
```

The engine uses `noise::Simplex` everywhere (no Perlin). The `noise` crate is in `Cargo.toml`. Confirm with `grep "^noise" Cargo.toml`.

- [ ] **Step 2: Build the widget**

Model after `docs/book/src/widgets/FbmExplorer.tsx`. The widget should:
- Take no props.
- Use `useBookSeed()` for the default seed; allow override via a slider.
- Render a 256×256 grayscale canvas showing one octave of simplex noise.
- Have a checkbox "show gradient grid overlay" that, when on, draws thin grid lines at `frequency`-spaced intervals.
- Have a frequency slider (default `1/64`).

Create `docs/book/src/widgets/NoiseField.tsx`:

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { createNoise2D } from 'simplex-noise';
import Alea from 'alea';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 256;

export default function NoiseField() {
  const seedBigInt = useBookSeed();
  const defaultSeed = Number(seedBigInt & 0xffff_ffffn);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [seed, setSeed] = useState(defaultSeed);
  const [frequency, setFrequency] = useState(1 / 64);
  const [showGrid, setShowGrid] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prng = Alea(String(seed));
    const noise = createNoise2D(prng);
    const img = ctx.createImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = noise(x * frequency, y * frequency); // [-1, 1]
        const g = Math.round(((v + 1) * 0.5) * 255);
        const i = (y * SIZE + x) * 4;
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    if (showGrid) {
      const period = 1 / frequency; // grid spacing in pixels
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
      ctx.lineWidth = 1;
      for (let g = 0; g <= SIZE; g += period) {
        ctx.beginPath();
        ctx.moveTo(g, 0); ctx.lineTo(g, SIZE);
        ctx.moveTo(0, g); ctx.lineTo(SIZE, g);
        ctx.stroke();
      }
    }
  }, [seed, frequency, showGrid]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
      <canvas ref={canvasRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444' }} />
      <div>
        <label>
          Seed: {seed}
          <input type="range" min={1} max={1000} step={1} value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10))} />
        </label>
        <label>
          Frequency: {frequency.toFixed(4)}
          <input type="range" min={0.005} max={0.1} step={0.005} value={frequency}
            onChange={(e) => setFrequency(parseFloat(e.target.value))} />
        </label>
        <label>
          <input type="checkbox" checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)} />
          Show gradient grid overlay
        </label>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write the chapter**

Replace `docs/book/content/part-1-foundations/1.3-coherent-noise.mdx` with a chapter following the standard skeleton:

```mdx
---
id: 1.3-coherent-noise
title: 1.3 Coherent Noise
---

import LazyWidget from '@site/src/widgets/LazyWidget';
import NoiseField from '@site/src/widgets/NoiseField';

In [1.2](./1.2-determinism.mdx) we built `mix`, the function that turns `(seed, integer coords, salt)` into a `u64`. That gives us *uncorrelated* randomness — every adjacent column gets a different value with no relationship to its neighbours. That's perfect for things like "should a tree spawn in this cell" but useless for "how tall is this column" — terrain needs to be *smooth*. Two columns ten blocks apart should be at similar heights.

**Coherent noise** is the bridge: a function from real-valued coordinates $(x, y)$ to a scalar in roughly $[-1, 1]$ that varies smoothly, looks random at scale, but has zero discontinuities anywhere. It's the fundamental primitive of every procedural texture, terrain, cloud, and water shader you've ever seen.

## The picture

Imagine an invisible grid of integer-coordinate points. At each grid point, the noise function pins a random unit-vector "gradient" (computed from the grid coordinate and the seed). For a query point $(x, y)$ between four grid points, the noise function:

1. Looks up the four corner gradients.
2. Computes the dot product of each gradient with the offset from its corner to $(x, y)$ — that's a per-corner "contribution".
3. Smoothly interpolates the four contributions across the cell.

The result is a smooth surface that wiggles up and down on the order of the grid spacing. The interpolation function (a fifth-degree polynomial in Perlin, a simplex-shape weighting in Simplex) is the magic: the noise has continuous *derivatives* across grid lines, not just continuous values.

Try it. Drag the seed to see how the random gradient grid changes the whole pattern; drag frequency to change the grid spacing; toggle the overlay to see where the lattice sits:

<LazyWidget>
  <NoiseField />
</LazyWidget>

Two flavors are common:

- **Perlin** (1985) — the classic. Hypercubic grid; gradients picked from 12 fixed unit vectors.
- **Simplex** (2001) — Perlin's own follow-up. Uses a *triangular* grid (in 2D) — fewer corners per cell (3 instead of 4 in 2D, 4 instead of 8 in 3D), faster, fewer artifacts at axis-aligned features.

The engine uses Simplex exclusively (via the `noise` crate). The "looks like terrain" properties are essentially the same; Simplex just avoids subtle grid-axis bias.

## Build it

Both Perlin and Simplex follow the same shape:

```rust
// simplified for exposition.
fn noise2d(x: f64, y: f64, hash: impl Fn(i32, i32) -> [f64; 2]) -> f64 {
    // 1. Find the cell.
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f64;       // [0, 1) — local position in cell
    let yf = y - yi as f64;

    // 2. Per-corner gradient and dot product with offset.
    let n00 = dot(hash(xi,     yi    ), [xf,        yf       ]);
    let n10 = dot(hash(xi + 1, yi    ), [xf - 1.0,  yf       ]);
    let n01 = dot(hash(xi,     yi + 1), [xf,        yf - 1.0 ]);
    let n11 = dot(hash(xi + 1, yi + 1), [xf - 1.0,  yf - 1.0 ]);

    // 3. Smooth interpolation. Perlin uses fifth-degree fade.
    let u = fade(xf);
    let v = fade(yf);
    lerp(v, lerp(u, n00, n10), lerp(u, n01, n11))
}

fn fade(t: f64) -> f64 { t * t * t * (t * (t * 6.0 - 15.0) + 10.0) }
```

The `fade` function is what gives noise its smoothness across grid lines. It's a fifth-degree polynomial chosen specifically so that `fade(0) = 0`, `fade(1) = 1`, `fade'(0) = fade'(1) = 0`, *and* `fade''(0) = fade''(1) = 0`. That last property — second derivative continuous — is what prevents visible "ringing" at grid lines.

The `hash(xi, yi) → unit_vector` lookup is where our friend from 1.2 reappears. Production implementations bake the gradients into a 256-entry permutation table, but conceptually you can just hash each integer corner with the world seed and salt into one of N fixed directions.

## In the engine

You won't find a hand-written noise implementation in the engine — we use the `noise` crate. Every place that wants smooth random structure writes:

```rust
// src/worldgen/mod.rs (Generator::new_internal)
let temperature_map = Fbm::<Simplex>::new(seed.wrapping_add(8) as u32)
    .set_octaves(2)
    .set_frequency(1.0 / 512.0)
    .set_persistence(0.5);
```

`Fbm` is "fractional Brownian motion" — stacked octaves of noise — and we explore it in detail in [1.4](./1.4-fbm.mdx). The `<Simplex>` type parameter says "use simplex noise as the base". `seed.wrapping_add(8)` is the salt; each different noise field in the `Generator` (temperature, humidity, weirdness, FBM base, …) uses a different small constant offset so the fields are uncorrelated, even though they share the world seed.

The `noise` crate handles the gradient lattice for you. You feed it a `u32` seed; it deterministically generates its permutation table; calls to `.get([x, y])` return a value in `[-1, 1]`.

## What you can now do in the code

You can read:

- Any `Simplex::new(seed_offset)` or `Fbm::<Simplex>::new(seed_offset)` construction. The `seed.wrapping_add(N)` pattern is the salt; the result is a deterministic 2D smooth random scalar field.
- The `noise` crate docs and source — the implementation matches the algorithm above with the addition of a 256-entry gradient permutation table.

## Next

→ [1.4 Fractional Brownian Motion](./1.4-fbm.mdx) — the moment you stack octaves of this and unlock fractal terrain.
```

- [ ] **Step 4: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/src/widgets/NoiseField.tsx docs/book/content/part-1-foundations/1.3-coherent-noise.mdx
git commit -m "content(book): write 1.3 Coherent Noise + NoiseField widget

Chapter explaining Perlin/Simplex as the smooth scalar field primitive,
the gradient-lattice algorithm, the fade polynomial. Widget renders
one octave of 2D simplex with sliders for seed + frequency and a
gradient-grid overlay.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Chapter 1.5 — Voronoi Diagrams + `VoronoiCells` widget

**Widget:** `VoronoiCells` — a canvas showing a jittered grid of seed points and the cells they generate, with click-to-move-seed and toggleable overlays for "show 2nd-nearest" and "show boundary t".

**Files:**
- Create: `docs/book/src/widgets/VoronoiCells.tsx`
- Modify: `docs/book/content/part-1-foundations/1.5-voronoi.mdx`

### Steps

- [ ] **Step 1: Read `src/worldgen/plates.rs`**

```bash
cat src/worldgen/plates.rs
```

Note: cell size is `PLATE_CELL_SIZE` (1024 blocks); each cell rolls one seed at a hashed jitter offset; nearest-seed search scans the 3×3 neighbourhood. The "boundary intensity" $t = (d_b - d_a) / (d_b + d_a)$ is the centerpiece.

- [ ] **Step 2: Build the widget**

Create `docs/book/src/widgets/VoronoiCells.tsx`. The widget should:
- Use a 5×5 visible grid of cells, each ~50px wide in a 256×256 canvas.
- Each cell has a jittered seed point inside it (jitter offset hashed from cell coords + a fixed seed).
- Render the cells: each pixel is colored by its nearest seed.
- Toggle: "show 2nd-nearest" overlays a striped pattern in cells where the 2nd-nearest is from a different "kind" (alternate-coloured if `(cellX + cellZ) % 2 === 0`).
- Toggle: "show boundary t" — render the boundary intensity field `t = (d_b - d_a) / (d_b + d_a)` as grayscale; 0 (boundary) is black, 1 (deep inside a cell) is white.
- Draw the seed points as small dots.
- Click on a seed to "regenerate" all jitters (reseed the widget).

Model after `FbmExplorer.tsx` for structure. Use 5×5 grid; sample only the 9 cells around each pixel (3×3 neighborhood — same as the engine).

The math doesn't need to be ported to a separate file since it's so small; inline it in the widget. (For the parity test, the engine's `plate_at` function uses different inputs/outputs so a 1:1 parity isn't meaningful — skip parity entries for this task.)

- [ ] **Step 3: Write the chapter**

Replace `docs/book/content/part-1-foundations/1.5-voronoi.mdx`:

```mdx
---
id: 1.5-voronoi
title: 1.5 Voronoi Diagrams
---

import LazyWidget from '@site/src/widgets/LazyWidget';
import VoronoiCells from '@site/src/widgets/VoronoiCells';

In [1.3](./1.3-coherent-noise.mdx) and [1.4](./1.4-fbm.mdx) we built *continuous* fields — every nearby point has a similar value to its neighbours. But the world also needs *cellular* structure: continents that are clearly separated by ocean, biomes with definite edges, cave systems that don't overlap. For that we use **Voronoi diagrams**.

## The picture

Drop a bunch of points (call them *seeds*) onto a 2D plane. For every other point on the plane, ask: which seed is closest? Color that point with the seed's colour. The result is a tiling — each seed "owns" a polygonal region, and the boundary between two regions is exactly the perpendicular bisector of the line between their seeds.

That tiling is a **Voronoi diagram**. The regions are cells; the boundaries form a connected network.

Click on the canvas below to drop new seeds; toggle the overlays to see how the cells form and how the "boundary intensity" field $t$ rises smoothly from 0 at edges to 1 deep inside cells:

<LazyWidget>
  <VoronoiCells />
</LazyWidget>

For the engine's purposes, dropping seeds at truly random locations is overkill. We use a **jittered grid**: divide the world into a regular grid of cells (1024 blocks per side in the engine — see `PLATE_CELL_SIZE`), and roll one Voronoi seed at a hashed offset inside each cell. Every world coordinate's nearest seed is then guaranteed to live in either the same grid cell or one of its 8 neighbours — so a 3×3 search is all you need, regardless of how big the world is.

## Build it

Finding the nearest seed at a query point `(qx, qz)`:

```rust
// simplified for exposition; see src/worldgen/plates.rs for the real one.
fn nearest_seed(seed: u64, qx: f32, qz: f32) -> (PlateId, f32) {
    let cx = (qx / PLATE_CELL_SIZE as f32).floor() as i32;
    let cz = (qz / PLATE_CELL_SIZE as f32).floor() as i32;
    let mut best = (PlateId { cell_x: 0, cell_z: 0 }, f32::INFINITY);
    for dz in -1..=1 {
        for dx in -1..=1 {
            let id = PlateId { cell_x: cx + dx, cell_z: cz + dz };
            // Seed location = cell center + jittered offset.
            let center = Vec2::new(
                (id.cell_x as f32 + 0.5) * PLATE_CELL_SIZE as f32,
                (id.cell_z as f32 + 0.5) * PLATE_CELL_SIZE as f32,
            );
            let jx = (hash::mix_unit(seed, &[id.cell_x, id.cell_z, JITTER_X_SALT]) - 0.5) * PLATE_CELL_SIZE as f32;
            let jz = (hash::mix_unit(seed, &[id.cell_x, id.cell_z, JITTER_Z_SALT]) - 0.5) * PLATE_CELL_SIZE as f32;
            let seed_pos = center + Vec2::new(jx, jz);
            let d = (Vec2::new(qx, qz) - seed_pos).length();
            if d < best.1 { best = (id, d); }
        }
    }
    best
}
```

Note our friend from 1.2 — the jitter is `hash::mix_unit(seed, &[cell_x, cell_z, salt])`. The seeds aren't random; they're hashed deterministically from cell coordinates. Recompute them on demand at query time, no need to store anything.

## The boundary intensity field

The Voronoi diagram by itself only tells you which seed is closest. For terrain we want something smoother: a continuous field that's 0 on a cell boundary and 1 deep inside a cell. That's what enables tricks like *mountain ranges form along plate boundaries* (we crank ridge lift when $t$ is small) and *continental shelves taper into oceans* (we lerp base elevations using $t$).

Get it like this:

```rust
fn nearest_two(seed: u64, qx: f32, qz: f32) -> (f32, f32) {
    // Same 3x3 scan as above, but track the 2 smallest distances.
    // Returns (d_a, d_b) where d_a <= d_b.
    todo!()
}

let (d_a, d_b) = nearest_two(seed, qx, qz);
let t = (d_b - d_a) / (d_b + d_a);  // 0 on boundary, → 1 deep inside cell
```

The expression $(d_b - d_a) / (d_b + d_a)$ has the nice property that it's scale-invariant — the field looks the same regardless of how big the cells are.

## In the engine

`src/worldgen/plates.rs` does exactly this. Every plate carries `kind: PlateKind { Continental, Oceanic }`, a `base_elevation`, and a `roughness` multiplier, all derived deterministically from the plate ID via `hash::mix_range(seed, &[cell_x, cell_z, salt], lo, hi)`. The boundary intensity $t$ shapes the heightmap two ways:

- **Continental shelf taper.** The base elevation at a column is `lerp(plate_a.base, plate_b.base, smooth(t))` — so the elevation transitions smoothly across a plate boundary instead of stepping.
- **Ridge lift.** When $t < \text{BOUNDARY\_RIDGE\_WIDTH}$ (default 0.12), we add a mountain range — `(1 - t/width)^2 * peak`. The closer to the boundary, the taller the ridge. This is what produces the engine's distinctive mountain *chains* along plate edges.

The same Voronoi pattern shows up in two other places:

- **Biome edge jitter** (`src/worldgen/climate.rs`) — per-block hash jitter perturbs the `(temperature, humidity)` query before the biome R-tree lookup, so biome edges aren't perfectly aligned with noise contours.
- **Aquifer cells** (`src/worldgen/aquifer.rs`) — 16×12×16 jittered cell grid, each cell holding a `y_top` and a fluid kind. Same nearest-seed logic, finer cell size.

## What you can now do in the code

You can read:

- `src/worldgen/plates.rs` — the Voronoi plate decomposition, top to bottom.
- `src/worldgen/climate.rs::voronoi_jitter_offset` — biome edge jitter using the same pattern.
- `src/worldgen/aquifer.rs` — aquifer cell sampling.

Whenever a chapter says "the engine partitions space into cells," you'll recognize this pattern.

## Next

→ [1.6 Vector Fields & Domain Warping](./1.6-domain-warping.mdx) — the trick that turns *blobby* noise into things that look like terrain.
```

- [ ] **Step 4: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/src/widgets/VoronoiCells.tsx docs/book/content/part-1-foundations/1.5-voronoi.mdx
git commit -m "content(book): write 1.5 Voronoi + VoronoiCells widget

Chapter on jittered-grid Voronoi (the engine's plate decomposition).
Teaches the 3x3 nearest-seed scan, the boundary intensity t = (d_b -
d_a) / (d_b + d_a), and how t drives continental shelf taper + ridge
lift. Widget visualizes the jittered grid, the cells, and the t field.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Chapter 1.6 — Vector Fields & Domain Warping + `WarpToggle` widget

**Widget:** `WarpToggle` — two canvases side-by-side: same FBM, with vs without domain warp. Slider for warp amplitude. Demonstrates that domain warping is what turns blobby FBM into terrain-shaped FBM.

**Important note on engine reality:** the engine's CURRENT heightmap (`heightmap.rs::h_pre`) does NOT use classical domain warping — it uses nested cubic Hermite splines on 6 climate axes (continentalness, ridges, terrain shape, weirdness, depth, T+H). The plan spec mentioned domain warping as a previously-used technique. **The chapter teaches the general technique and grounds it in `src/worldgen/heightmap.rs::DensityNoise::evaluate_base_3d` (which still applies warp-style perturbation to the 3D base density)**, and notes the historical use in the heightmap. Verify the actual current usage with `grep -n "warp\|noise_x\|noise_z" src/worldgen/heightmap.rs`.

**Files:**
- Create: `docs/book/src/widgets/WarpToggle.tsx`
- Modify: `docs/book/content/part-1-foundations/1.6-domain-warping.mdx`

### Steps

- [ ] **Step 1: Find the engine's domain warping (if any)**

```bash
grep -rn "warp\|noise_x\|noise_z\|distort\|x_offset\|z_offset" src/worldgen/heightmap.rs src/worldgen/mod.rs
cat src/worldgen/heightmap.rs | grep -A5 "fn evaluate_base_3d"
```

Document the *real* current state in your chapter — don't pretend the engine does classical domain warping if it doesn't. If the engine has NO domain warping currently, frame the chapter as "the technique we used to use, why we use a different scheme now, and how the principle still applies in `evaluate_base_3d`".

- [ ] **Step 2: Build the widget**

Create `docs/book/src/widgets/WarpToggle.tsx` with two side-by-side canvases. Both render the same FBM (octaves=4, persistence=0.5). The right canvas warps each `(x, y)` query through a 2D vector field before sampling: `query(x', y')` where `x' = x + warpAmp * fbmX(x*0.02, y*0.02)` and `y' = y + warpAmp * fbmZ(x*0.02, y*0.02)` (using two independently-seeded FBMs for X and Z displacement). Slider for `warpAmp` (0 to 80). At `warpAmp=0` the two canvases should be identical; cranking it produces the characteristic "fingered" terrain look.

Use `makeFbm2D` from `docs/book/src/math/fbm.ts`. No need to add new math.

- [ ] **Step 3: Write the chapter**

Write the chapter following the standard skeleton. Hook: "FBM looks blobby and round, but real terrain has fingers, ridges, and valleys that bend around things. The trick is to perturb the INPUT to the noise before sampling — that's domain warping." Build-it: show the formula `noise(x + ε·warpX(x,y), y + ε·warpY(x,y))`. In-the-engine: ground in `evaluate_base_3d` or note that the engine moved away from classical warp toward spline-driven heightmaps (whichever is true based on Step 1). Next: `→ 1.7 Splines`.

Provide a complete MDX file — don't leave placeholders. Use the existing 1.3 and 1.5 as templates.

- [ ] **Step 4: Verify the build**

```bash
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/src/widgets/WarpToggle.tsx docs/book/content/part-1-foundations/1.6-domain-warping.mdx
git commit -m "content(book): write 1.6 Domain Warping + WarpToggle widget

Chapter on the input-perturbation trick that turns FBM into
terrain-shaped FBM. Widget shows the same FBM with and without warp
side-by-side. Engine grounding describes the technique's role
historically and where similar perturbation lives now.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Chapter 1.7 — Splines + `SplineEditor` widget

The engine uses **cubic Hermite** splines (not Bezier, not Catmull-Rom). The implementation is in `src/worldgen/spline.rs`. The chapter teaches Hermite specifically. Widget lets the reader drag knots and slopes and see the curve update.

**Widget:** `SplineEditor` — canvas with N draggable control points; each control point has both a position and an editable slope (tangent). Show the resulting cubic Hermite curve. Toggle "show linear interpolation" overlays a straight-line version for comparison.

**Files:**
- Create: `docs/book/src/math/spline.ts`
- Create: `docs/book/src/widgets/SplineEditor.tsx`
- Modify: `src/bin/doc_render/parity_dump.rs` (add spline parity cases)
- Modify: `docs/book/static/parity.json` (regenerate via `cargo run --release --bin doc_render -- parity --output docs/book/static/parity.json`)
- Modify: `docs/book/src/pages/parity.tsx` (add a `runSpline` case to the parity-runner switch)
- Modify: `docs/book/content/part-1-foundations/1.7-splines.mdx`

### Steps

- [ ] **Step 1: Read the spline module**

```bash
cat src/worldgen/spline.rs
```

Understand the `Knot { loc, val, slope }` struct, the `Constant | Multipoint(Vec<Knot>)` enum, and the evaluation formula in the file's doc comment. Note the linear-extrapolation behavior outside the knot range.

- [ ] **Step 2: TS port of cubic Hermite evaluation**

Create `docs/book/src/math/spline.ts`:

```typescript
// parity: src/worldgen/spline.rs (CubicSpline::evaluate)

export type Knot = { loc: number; val: number; slope: number };

/**
 * Evaluate a cubic Hermite spline at `input`. Mirrors
 * `src/worldgen/spline.rs::CubicSpline::Multipoint::evaluate`.
 *
 * Knots must be sorted ascending by `loc`. Outside the knot range,
 * the result extrapolates linearly using the endpoint slope.
 */
export function evaluateSpline(knots: Knot[], input: number): number {
  if (knots.length === 0) throw new Error('empty spline');
  if (knots.length === 1) return knots[0].val;
  // Below the first knot: linear extrap using the first knot's slope.
  if (input < knots[0].loc) {
    const k = knots[0];
    return k.val + k.slope * (input - k.loc);
  }
  // Above the last knot: linear extrap using the last knot's slope.
  const last = knots[knots.length - 1];
  if (input > last.loc) {
    return last.val + last.slope * (input - last.loc);
  }
  // Find the bracketing knots.
  for (let i = 0; i + 1 < knots.length; i++) {
    const k1 = knots[i], k2 = knots[i + 1];
    if (input >= k1.loc && input <= k2.loc) {
      const dx = k2.loc - k1.loc;
      const t = (input - k1.loc) / dx;
      const a = k1.slope * dx - (k2.val - k1.val);
      const b = -k2.slope * dx + (k2.val - k1.val);
      // Hermite formula: lerp(t, y1, y2) + t*(1-t)*lerp(t, a, b)
      return (1 - t) * k1.val + t * k2.val + t * (1 - t) * ((1 - t) * a + t * b);
    }
  }
  // Shouldn't reach here; defensive.
  throw new Error('unreachable in evaluateSpline');
}
```

- [ ] **Step 3: Add parity entries in `parity_dump.rs`**

After the existing FBM loop in `src/bin/doc_render/parity_dump.rs`, add:

```rust
// -- spline --
use crate::worldgen::spline::{CubicSpline, Knot};

let spline_cases: &[(&str, &[(f32, f32, f32)], f32)] = &[
    // (name, knots [(loc, val, slope), ...], input)
    ("ramp", &[(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)], 0.5),
    ("plateau", &[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 0.5),
    ("plateau", &[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)], 0.25),
    ("dip", &[(0.0, 1.0, -2.0), (0.5, 0.0, 0.0), (1.0, 1.0, 2.0)], 0.5),
    ("dip", &[(0.0, 1.0, -2.0), (0.5, 0.0, 0.0), (1.0, 1.0, 2.0)], 0.25),
];
for (name, knots_raw, input) in spline_cases {
    let knots = CubicSpline::Multipoint(
        knots_raw.iter().map(|&(loc, val, slope)| Knot { loc, val, slope }).collect(),
    );
    let v = knots.evaluate(*input);
    let knots_json: Vec<String> = knots_raw.iter()
        .map(|(loc, val, slope)| format!("[{loc},{val},{slope}]")).collect();
    entries.push(format!(
        "    {{\"fn\":\"spline\",\"args\":{{\"name\":\"{name}\",\"knots\":[{}],\"input\":{input}}},\"out\":{v}}}",
        knots_json.join(",")
    ));
}
```

Note: this `use` line goes inside `pub fn run(...)`, not at module top, to avoid an unused import warning if the spline path is the only one that uses it.

Rebuild and regenerate the parity reference:

```bash
cargo build --release --bin doc_render
cargo run --release --bin doc_render -- parity --output docs/book/static/parity.json
```

Verify the JSON now has both `fn:"fbm"` and `fn:"spline"` entries with `cat docs/book/static/parity.json`.

- [ ] **Step 4: Extend the parity page**

Modify `docs/book/src/pages/parity.tsx`:

1. Add an `Entry` variant for spline: `{ fn: 'spline'; args: { name: string; knots: [number, number, number][]; input: number }; out: number }`.
2. Add `import { evaluateSpline } from '../math/spline';`.
3. Add a `runSpline(e)` function that builds knots from `e.args.knots` (each `[loc, val, slope]`) and calls `evaluateSpline`.
4. In the row-builder switch, dispatch on `e.fn`: `fbm → runFbm(e); spline → runSpline(e); else NaN`.
5. Spline is pure math (no PRNG involved); set the tolerance for spline rows to `1e-5` so a real bug would surface. (Keep the loose FBM tolerance.)

The tolerance handling: change `TOLERANCE` constant into a per-fn map: `const TOLERANCES: Record<string, number> = { fbm: 1.5, spline: 1e-5 };`. In the row builder, use `delta < (TOLERANCES[e.fn] ?? 1e-5)`.

- [ ] **Step 5: Build the widget**

Create `docs/book/src/widgets/SplineEditor.tsx`. The widget:
- Has 4 default knots, e.g. `[(0, 0, 1), (0.33, 0.5, 0), (0.66, 0.5, 0), (1, 1, 1)]`.
- Each knot's `(loc, val)` is draggable on the canvas.
- Each knot also has a draggable "slope handle" — a short line segment whose angle from the knot represents the slope.
- Render the resulting Hermite curve at high resolution.
- Toggle: "show linear" overlays a straight-line connect-the-dots in a different color.

Use `evaluateSpline` from `src/math/spline.ts`.

- [ ] **Step 6: Write the chapter**

Write the chapter teaching:
1. What a piecewise polynomial spline is — bracket the input between two knots; evaluate a polynomial within that bracket.
2. Why **Hermite** specifically: each knot's slope (tangent) is a *first-class input*. You don't compute the slope from neighbouring knots (Catmull-Rom) or set it via off-curve handles (Bezier) — you specify it directly. The engine's biome configs (RON files) literally list `slope: <float>` per knot, which is the most direct way to author a "smooth at this knot" vs "sharp at this knot" curve.
3. The formula: `lerp(t, y1, y2) + t·(1-t)·lerp(t, a, b)` with the `a` and `b` derived from the knot slopes scaled by `(x2 - x1)`.
4. In-the-engine: cite `src/worldgen/heightmap.rs` and any `offset_spline`, `factor_spline`, `jaggedness_spline` it uses. Note that splines are nested in the engine — a spline's *value* at a knot can itself be a spline keyed off a different climate axis. (Don't dive into nesting too far; just note it exists.)

Use the standard chapter skeleton. End with `→ 1.8 Signed Distance Functions`.

- [ ] **Step 7: Verify the build**

```bash
cargo build --release --bin doc_render
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 8: Commit**

```bash
git add docs/book/src/math/spline.ts docs/book/src/widgets/SplineEditor.tsx \
        docs/book/src/pages/parity.tsx docs/book/static/parity.json \
        src/bin/doc_render/parity_dump.rs \
        docs/book/content/part-1-foundations/1.7-splines.mdx
git commit -m "content(book): write 1.7 Splines + SplineEditor widget + parity

Chapter teaches cubic Hermite splines, the same scheme the engine uses
in src/worldgen/spline.rs (climate-to-height mapping, tunnel paths).
TS port mirrors the Rust evaluate(); 5 new parity-test cases asserting
the TS matches Rust to 1e-5 (pure math, no PRNG gap). Widget allows
dragging knot positions and slope tangents.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Chapter 1.8 — Signed Distance Functions + `SdfComposer` widget

The engine carves caves using SDFs — ellipsoid SDF for chambers, capsule-along-spline SDF for tunnels — composed with `min` and `max`. This chapter teaches the principle and the composition rules.

**Widget:** `SdfComposer` — canvas where you can place ellipses and capsules and see the SDF visualized as iso-contours; toggle between `min` (union) and `max` (intersection) composition modes.

**Files:**
- Create: `docs/book/src/math/sdf.ts`
- Create: `docs/book/src/widgets/SdfComposer.tsx`
- Modify: `src/bin/doc_render/parity_dump.rs` (add SDF parity cases)
- Modify: `docs/book/src/pages/parity.tsx` (add `runSdf`)
- Modify: `docs/book/static/parity.json`
- Modify: `docs/book/content/part-1-foundations/1.8-sdfs.mdx`

### Steps

- [ ] **Step 1: Read the cave SDF code**

```bash
sed -n '400,470p' src/worldgen/caves.rs
```

Note the `cave_sdf` and `entrance_sdf` function shapes. Understand how they compose (each ellipsoid/capsule returns a positive intensity when the point is *inside* the shape; the overall cave SDF takes `max` over all shapes; the final density composition takes `min(-sdf, raw_density)`).

- [ ] **Step 2: TS port of ellipsoid + capsule SDFs**

Create `docs/book/src/math/sdf.ts`:

```typescript
// parity: src/worldgen/caves.rs (ellipsoid SDF inside cave_sdf, capsule SDF inside tunnel evaluation)

/** 2D ellipsoid SDF. Returns positive intensity inside, 0 at boundary, negative outside (clamped to 0 in code). */
export function ellipsoid2D(
  px: number, py: number,
  cx: number, cy: number,
  rx: number, ry: number,
): number {
  const dx = (px - cx) / rx;
  const dy = (py - cy) / ry;
  // Inside: 1 - distance² (in normalized ellipsoid space); clamp to ≥0.
  return Math.max(0, 1 - (dx * dx + dy * dy));
}

/** 2D capsule SDF: shortest distance to a line segment, projected to a falloff. */
export function capsule2D(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number,
  radius: number,
): number {
  // Vector from A to B.
  const abx = bx - ax, aby = by - ay;
  const apx = px - ax, apy = py - ay;
  const denom = abx * abx + aby * aby || 1e-9;
  let t = (apx * abx + apy * aby) / denom;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * abx, cy = ay + t * aby;
  const d = Math.hypot(px - cx, py - cy);
  return Math.max(0, 1 - d / radius);
}

/** Union of multiple SDF intensities: take the max (most-inside wins). */
export function unionSdf(values: number[]): number {
  return values.reduce((a, b) => Math.max(a, b), 0);
}

/** Intersection: take the min. */
export function intersectionSdf(values: number[]): number {
  return values.reduce((a, b) => Math.min(a, b), Infinity);
}
```

- [ ] **Step 3: Add parity entries**

In `src/bin/doc_render/parity_dump.rs`, add a small block testing the ellipsoid SDF math. Since the engine's `cave_sdf` is more complex than just one ellipsoid, write the test cases as direct evaluations of the ellipsoid formula in Rust (mirroring exactly what the TS function does — both are pure analytical math; tolerance can be `1e-5`).

```rust
// -- ellipsoid_sdf --
// Pure math — no engine call, just verify TS port matches the formula
// directly. (The engine's cave_sdf wraps this in additional logic.)
let sdf_cases: &[(f32, f32, f32, f32, f32, f32)] = &[
    // (px, py, cx, cy, rx, ry)
    (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  // dead center → 1.0
    (1.0, 0.0, 0.0, 0.0, 1.0, 1.0),  // on boundary → 0.0
    (0.5, 0.0, 0.0, 0.0, 1.0, 1.0),  // halfway → 0.75
    (2.0, 0.0, 0.0, 0.0, 1.0, 1.0),  // outside → 0.0
    (0.0, 0.5, 0.0, 0.0, 1.0, 2.0),  // off-y in elongated → 1 - 0.0625
];
for &(px, py, cx, cy, rx, ry) in sdf_cases {
    let dx = (px - cx) / rx;
    let dy = (py - cy) / ry;
    let v = (1.0 - (dx * dx + dy * dy)).max(0.0);
    entries.push(format!(
        "    {{\"fn\":\"ellipsoid2D\",\"args\":{{\"px\":{px},\"py\":{py},\"cx\":{cx},\"cy\":{cy},\"rx\":{rx},\"ry\":{ry}}},\"out\":{v}}}",
    ));
}
```

Rebuild, regen `parity.json`.

- [ ] **Step 4: Extend the parity page**

Add an `ellipsoid2D` variant to the `Entry` type, a `runEllipsoid` function in `docs/book/src/pages/parity.tsx`, and an entry in the `TOLERANCES` map: `'ellipsoid2D': 1e-5`.

- [ ] **Step 5: Build the widget**

Create `docs/book/src/widgets/SdfComposer.tsx`. The widget:
- Has a list of "shapes" — initially 2 ellipsoids and 1 capsule.
- Drag handles for each shape's center/endpoints.
- Render the SDF as an iso-contour heatmap on the canvas.
- Toggle: `union` (min) vs `intersection` (max) composition.
- Toggle: "subtract from terrain" — render the result as if it were carved out of a hypothetical solid (just visualize differently).

Use `unionSdf`, `intersectionSdf`, `ellipsoid2D`, `capsule2D` from `docs/book/src/math/sdf.ts`.

- [ ] **Step 6: Write the chapter**

Standard chapter skeleton:
- Hook: "Caves aren't carved by writing 'air' at specific voxels. The engine writes a *field* — a value at every voxel that says 'how far inside a cave am I?'. That field is an SDF."
- Intuition: define SDF as a function returning positive inside / 0 at boundary / negative outside; show the widget.
- Build it: derive the ellipsoid and capsule formulas; show how min/max compose. Explain why min = union (most-inside-of-anything wins) and max = intersection (least-inside-of-anything wins).
- In the engine: cite `src/worldgen/caves.rs::cave_sdf` and `entrance_sdf`. Note that the engine actually uses *signed density* (negative is solid, positive is air-ish) and composes with `min(density, -sdf)` to subtract the cave from solid terrain.
- Next: `→ 1.9 Trilinear Interpolation`.

- [ ] **Step 7: Verify the build**

```bash
cargo build --release --bin doc_render
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 8: Commit**

```bash
git add docs/book/src/math/sdf.ts docs/book/src/widgets/SdfComposer.tsx \
        docs/book/src/pages/parity.tsx docs/book/static/parity.json \
        src/bin/doc_render/parity_dump.rs \
        docs/book/content/part-1-foundations/1.8-sdfs.mdx
git commit -m "content(book): write 1.8 SDFs + SdfComposer widget + parity

Chapter teaches signed distance functions and Boolean composition via
min/max — the language of caves in src/worldgen/caves.rs. Widget lets
readers place ellipses + capsules and switch between union and
intersection. TS port + parity entries for ellipsoid2D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Chapter 1.9 — Trilinear Interpolation + `TrilerpDemo` widget

The chapter that explains why the engine evaluates the density graph at a 9×9×9 corner lattice and trilerps the interior — the 45× speedup over per-voxel evaluation.

**Widget:** `TrilerpDemo` — 3D-but-rendered-as-2D-slice. Shows a 2×2 grid of corner values that the reader can edit; mouse hover inside the grid shows the bilerped value; toggle "show error vs exact" overlays a heatmap of |exact - trilerp| at every interior point.

(2D is fine for the widget; trilerp is just bilerp + lerp; the math principle is the same.)

**Files:**
- Create: `docs/book/src/math/trilerp.ts`
- Create: `docs/book/src/widgets/TrilerpDemo.tsx`
- Modify: `src/bin/doc_render/parity_dump.rs` (add trilerp parity cases)
- Modify: `docs/book/src/pages/parity.tsx`
- Modify: `docs/book/static/parity.json`
- Modify: `docs/book/content/part-1-foundations/1.9-trilerp.mdx`

### Steps

- [ ] **Step 1: Read the cell evaluator**

```bash
sed -n '220,340p' src/worldgen/density_graph.rs
```

Note: `CORNER_COUNT = CELL_COUNT + 1`. The grid is stored as `[x][y][z]` indexing. The trilerp is at line ~285.

- [ ] **Step 2: TS port**

Create `docs/book/src/math/trilerp.ts`:

```typescript
// parity: src/worldgen/density_graph.rs (CellEvaluator::evaluate trilerp inside the cell)

/**
 * Trilinear interpolation. Given the 8 corner values of a unit cube,
 * compute the value at a point (tx, ty, tz) where each t ∈ [0, 1].
 *
 * Corner order: c[xi][yi][zi] for xi, yi, zi ∈ {0, 1}, supplied as a
 * length-8 array indexed `xi + 2*yi + 4*zi`.
 */
export function trilerp(corners: [number, number, number, number, number, number, number, number], tx: number, ty: number, tz: number): number {
  // Lerp along x at each of 4 (yi, zi) pairs.
  const c00 = corners[0] * (1 - tx) + corners[1] * tx;
  const c10 = corners[2] * (1 - tx) + corners[3] * tx;
  const c01 = corners[4] * (1 - tx) + corners[5] * tx;
  const c11 = corners[6] * (1 - tx) + corners[7] * tx;
  // Lerp along y at each of 2 zi.
  const c0 = c00 * (1 - ty) + c10 * ty;
  const c1 = c01 * (1 - ty) + c11 * ty;
  // Lerp along z.
  return c0 * (1 - tz) + c1 * tz;
}

/** 2D bilinear interpolation — used in the chapter's 2D widget. */
export function bilerp(c00: number, c10: number, c01: number, c11: number, tx: number, ty: number): number {
  const a = c00 * (1 - tx) + c10 * tx;
  const b = c01 * (1 - tx) + c11 * tx;
  return a * (1 - ty) + b * ty;
}
```

- [ ] **Step 3: Add parity entries**

In `parity_dump.rs`:

```rust
// -- trilerp --
// Pure math — straightforward to assert exact match.
let trilerp_cases: &[(([f32; 8]), f32, f32, f32)] = &[
    // (corners c[xi + 2*yi + 4*zi], tx, ty, tz)
    ([0.0; 8], 0.5, 0.5, 0.5),                              // all zero → 0
    ([1.0; 8], 0.0, 0.0, 0.0),                              // all one  → 1
    ([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 0.5, 0.0, 0.0),  // gradient in x → 0.5
    ([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 0.0, 0.0, 0.5),  // gradient in z → 0.5
    ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 0.5, 0.5),  // single corner → 0.125
];
for (corners, tx, ty, tz) in trilerp_cases {
    let c = corners;
    let c00 = c[0] * (1.0 - tx) + c[1] * tx;
    let c10 = c[2] * (1.0 - tx) + c[3] * tx;
    let c01 = c[4] * (1.0 - tx) + c[5] * tx;
    let c11 = c[6] * (1.0 - tx) + c[7] * tx;
    let c0  = c00 * (1.0 - ty) + c10 * ty;
    let c1  = c01 * (1.0 - ty) + c11 * ty;
    let v   = c0 * (1.0 - tz) + c1 * tz;
    let corners_json: Vec<String> = c.iter().map(|x| format!("{x}")).collect();
    entries.push(format!(
        "    {{\"fn\":\"trilerp\",\"args\":{{\"corners\":[{}],\"tx\":{tx},\"ty\":{ty},\"tz\":{tz}}},\"out\":{v}}}",
        corners_json.join(",")
    ));
}
```

Rebuild, regen `parity.json`.

- [ ] **Step 4: Extend the parity page**

Add `trilerp` variant + `runTrilerp` to `parity.tsx`; tolerance `1e-5`.

- [ ] **Step 5: Build the widget**

`docs/book/src/widgets/TrilerpDemo.tsx`. Use 2D bilinear for the visualization (mention in the chapter that trilerp is the natural 3D extension). The widget has 4 editable corner values (sliders below the canvas), and renders the bilerped field as a heatmap. Toggle: "show error vs exact" — define "exact" as some test function (e.g., `0.5 * (Math.sin(x*8) + Math.cos(y*8))`), let the user pick this function from a small dropdown, and render `|exact(x,y) - bilerp(x,y)|` as a separate heatmap.

- [ ] **Step 6: Write the chapter**

Standard skeleton:
- Hook: "Per-voxel density evaluation in a 32³ chunk = 32,768 noise lookups per chunk. Most of those values change very little from their neighbours. The engine evaluates the density graph at a 9×9×9 lattice of corner points and interpolates the rest — same image, 45× cheaper."
- Build it: derive trilerp as a chain of 7 lerps (or 4 bilerps + 1 lerp), explain why it's the "natural" multidimensional linear interpolation.
- In the engine: cite `density_graph.rs::CellEvaluator` with the `CORNER_COUNT = CELL_COUNT + 1` constant. Show the speedup numbers (730 evaluations vs 32,768).
- The honest caveat: trilerp is an approximation, not exact. Near voxels right at the density=0 threshold, the interpolated value can sign-flip differently from the exact one. The engine accepts this — the visual difference is invisible, the speedup is real. (Cite `probe.rs::DensityBreakdown` which uses exact evaluation for inspection.)
- Next: `→ Part II — Pipeline Overview` (or wherever 1.9 leads in the existing sidebar — verify).

- [ ] **Step 7: Verify the build**

```bash
cargo build --release --bin doc_render
cd docs/book && npm run build 2>&1 | tail -5 && cd ../..
```

- [ ] **Step 8: Commit**

```bash
git add docs/book/src/math/trilerp.ts docs/book/src/widgets/TrilerpDemo.tsx \
        docs/book/src/pages/parity.tsx docs/book/static/parity.json \
        src/bin/doc_render/parity_dump.rs \
        docs/book/content/part-1-foundations/1.9-trilerp.mdx
git commit -m "content(book): write 1.9 Trilerp + TrilerpDemo widget + parity

Chapter teaches trilinear interpolation as the 45x speedup behind
src/worldgen/density_graph.rs::CellEvaluator. Widget uses 2D bilerp
for visualization; toggle shows the error against an exact function.
Parity entries verify the TS port matches Rust on pure math (1e-5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

### Spec coverage

Cross-checking against the Phase 1 design spec's Part I outline:

| Spec chapter | Plan task |
|---|---|
| 1.1 The Voxel World | Already done in Plan 1 |
| 1.2 Deterministic Randomness | Task 1 |
| 1.3 Coherent Noise | Task 2 |
| 1.4 FBM | Already done in Plan 1 |
| 1.5 Voronoi Diagrams | Task 3 |
| 1.6 Vector Fields & Domain Warping | Task 4 |
| 1.7 Splines | Task 5 |
| 1.8 SDFs | Task 6 |
| 1.9 Trilinear Interpolation | Task 7 |

All seven remaining Part I chapters have a task. ✅

Widget coverage:

| Spec widget | Plan task |
|---|---|
| NoiseField (1.3) | Task 2 |
| FbmExplorer (1.4) | Already done |
| VoronoiCells (1.5) | Task 3 |
| WarpToggle (1.6) | Task 4 |
| SplineEditor (1.7) | Task 5 |
| SdfComposer (1.8) | Task 6 |
| TrilerpDemo (1.9) | Task 7 |

All six remaining widgets have a task. ✅

Parity test growth:

- Task 5 adds `spline` cases
- Task 6 adds `ellipsoid2D` cases
- Task 7 adds `trilerp` cases
- These three are pure math (no PRNG gap), so tolerance is `1e-5`. The `TOLERANCES` map approach lets per-fn tolerances coexist cleanly.

### Placeholder scan

No "TBD", "TODO" placeholders in task code. Step 6 of Task 4 says "Provide a complete MDX file — don't leave placeholders" but that's an instruction, not a placeholder. ✅

### Type consistency

- `Knot { loc, val, slope }` — matches `src/worldgen/spline.rs::Knot` field names exactly.
- `evaluateSpline(knots: Knot[], input: number)` — only place this function is named.
- `trilerp(corners: [number, number, number, number, number, number, number, number], tx, ty, tz)` — same signature in Rust parity case (corner indexing `c[xi + 2*yi + 4*zi]`).
- `ellipsoid2D` / `capsule2D` / `unionSdf` / `intersectionSdf` — names used consistently across math file, parity entries, widget.
- `TOLERANCES` map shape — `Record<string, number>` with default fallback. Set once in Task 5; extended in Tasks 6 and 7.

Plan is internally consistent.

### Risks worth flagging

1. **The widget code is sketched, not complete.** Tasks 3, 4, 5, 6, 7 each describe a widget but don't paste the full `.tsx` content — they direct the implementer to model after `FbmExplorer.tsx`. The implementer subagents will write the actual widget code. This is intentional given plan length but means each task has a non-trivial creative-implementation moment.

2. **Engine reality may differ from the spec.** Task 4 already flags this for domain warping. Other potential traps: the spline file's API may have changed since I read it (verify in Step 1 of Task 5); the cave SDF function signatures may differ. **Every chapter task starts with reading the relevant source.** That's the safeguard.

3. **Tolerance for FBM stays at 1.5; pure-math entries use 1e-5.** Verify the parity page handles per-fn tolerances correctly — Task 5 Step 4 changes the single `TOLERANCE` constant into a map; subsequent tasks just add entries.
