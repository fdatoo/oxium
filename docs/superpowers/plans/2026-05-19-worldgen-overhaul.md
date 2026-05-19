# Worldgen overhaul — Implementation Plan

> **Spec:** `docs/superpowers/specs/2026-05-19-worldgen-overhaul-design.md`
> **Mode:** Autonomous; commit each PR to `main`, push periodically.

**Goal:** Replace single-file `worldgen/mod.rs` with a region-cached, plate-driven layered pipeline producing continental geography, hierarchical river networks, graph-based caves, and blended biomes.

**Architecture:** Six-phase rollout (each phase = one or more atomic commits on `main`). Per spec.

**Tech Stack:** Rust 2024 / `noise = 0.9` / `bitvec = 1` / `glam = 0.29` / `toml = 0.8` / new dep: `lru = 0.12`.

---

## PR 1 — Module split + region cache scaffolding + save manifest

**Files (create):**
- `src/worldgen/region.rs`
- `src/worldgen/plates.rs`
- `src/worldgen/climate.rs`
- `src/worldgen/heightmap.rs`
- `src/worldgen/hydrology.rs`
- `src/worldgen/caves.rs`
- `src/worldgen/surface.rs`
- `src/worldgen/trees.rs`
- `src/worldgen/tuning.rs`
- `src/persistence/manifest.rs`

**Files (modify):**
- `src/worldgen/mod.rs` — re-export module surface, public API only
- `src/persistence/mod.rs` — add `manifest` module
- `src/app.rs` — read/write manifest, get seed from manifest
- `Cargo.toml` — add `lru = "0.12"`, `rand = "0.8"`

**Step 1.1 — Add deps:** `lru`, `rand` (`rand` for one-shot seed gen at world creation).
**Step 1.2 — Tuning module:** Verbatim from spec's constants table.
**Step 1.3 — Region module:** `RegionCoord`, `MacroRegionCoord`, `FineRegion`, `MacroRegion` skeleton structs, `LruCache` wiring on `Generator`. Build functions return default-filled regions for now (overwritten in PR 2/3/4).
**Step 1.4 — Plates module:** Full `plate_at` implementation per spec — this is a stable foundation that downstream PRs depend on, worth landing it correct now.
**Step 1.5 — Other modules:** Each is a stub with one function: `legacy_*` containing the v1 logic moved verbatim. `mod.rs::fill_chunk` calls `legacy_fill_chunk` so behaviour is unchanged.
**Step 1.6 — Save manifest:** `WorldManifest { seed: u64, worldgen_version: u32, created_at: i64 }`, TOML serde. Read on world load; write on creation; auto-generate on first open of legacy save (with `seed = 42`, version = 1).
**Step 1.7 — Wire app.rs to manifest:** Replace `let seed = 42;` with manifest read/write.
**Step 1.8 — Tests:** Plate determinism, plate ratio, save-manifest roundtrip, generator output unchanged from baseline.
**Step 1.9 — Run tests, commit, push.**

## PR 2 — New heightmap (plates + warped FBM + cliffs)

**Files (modify):**
- `src/worldgen/heightmap.rs` — replace `legacy_height` with `h_pre`
- `src/worldgen/plates.rs` — add `ridge_lift` helper
- `src/worldgen/surface.rs` — add cliff detection rule
- `src/worldgen/mod.rs::fill_chunk` — call new heightmap pipeline (rivers still legacy, caves still legacy)

**Step 2.1 — `h_pre` implementation.** Plate base lerp + ridge_lift + warped FBM.
**Step 2.2 — `ridge_lift`.** Per spec: triangular falloff inside `BOUNDARY_RIDGE_WIDTH`, modulated by 1D noise along the boundary, peak scaled by plate-pair type.
**Step 2.3 — Warped FBM.** Two 2-octave warp noises offset coords; height FBM evaluated at warped coord.
**Step 2.4 — Cliff detection.** Sample `h_pre` at ±2 in x/z; if `|∇h| > CLIFF_SLOPE_THRESH`, surface = `Stone`, subsurface = `Stone` (no dirt).
**Step 2.5 — Remove v1 mountain rules.** Drop `mountain_noise`, `mountainness_map`, `MOUNTAIN_ROCK_LINE` from the new heightmap path. v1 trees still use legacy `mountain_weight` until PR 5 — keep a shim that derives mountain weight from cliff detection.
**Step 2.6 — Tests.** Axis-symmetry breakage, cliff exposure, height cap.
**Step 2.7 — Visual.** Take a few screenshots from elevated spawn to confirm continents/archipelagos appear.
**Step 2.8 — Re-baseline `golden_seed42_chunk_0_2_0`.**
**Step 2.9 — Commit, push.**

## PR 3 — Hydrology

**Files (modify):**
- `src/worldgen/hydrology.rs` — full fine + macro D8 + flow accumulation + valley carve
- `src/worldgen/region.rs` — populate `FineRegion::rivers`, `MacroRegion::trunk`
- `src/worldgen/heightmap.rs` — `h_final` subtracts valley carve
- `src/worldgen/mod.rs::fill_chunk` — sea-level flood uses h_final and lake_rim; old river/lake noise removed

**Step 3.1 — D8 + sink fill + accumulation.** Pure functions over a grid input; testable in isolation with synthetic heightmaps.
**Step 3.2 — Macro pass.** Same algorithm at macro scale.
**Step 3.3 — Trunk injection.** Fine flow accumulation starts each macro-trunk's fine cell with `macro_acc * 64`.
**Step 3.4 — `RiverSegment` + kd-tree spatial index.** Built lazily on first `valley_carve` query per region.
**Step 3.5 — Valley carve.** U-profile against perturbed centerline.
**Step 3.6 — Lake handling.** `lake_rim` data populated; chunk fill turns cells below rim in lake regions into water.
**Step 3.7 — Tests.** Flow path termination, width monotonicity, trunk presence, lake continuity across regions.
**Step 3.8 — Visual.** Screenshot a continental scene to confirm rivers + valleys.
**Step 3.9 — Re-baseline, commit, push.**

## PR 4 — Caves

**Files (modify):**
- `src/worldgen/caves.rs` — full `CaveSystem`, chamber MST, spline tunnels, entrances
- `src/worldgen/region.rs` — `FineRegion::cave_systems` populated; cross-region neighbour fetch helper
- `src/worldgen/mod.rs::fill_chunk` — cave carve precedence per spec

**Step 4.1 — Cave system rolls.** Per-region deterministic.
**Step 4.2 — Chamber Poisson-disk.** Bridson 3D.
**Step 4.3 — MST + extra loops.** Kruskal on chamber pairs.
**Step 4.4 — Spline tunnels.** Catmull–Rom through control points with domain-warped offsets.
**Step 4.5 — SDF carve.** Ellipsoid chamber, capsule-along-spline tunnel.
**Step 4.6 — Surface entrances.** Three types per spec.
**Step 4.7 — Deep wormhole noise.** Only `y < WORMHOLE_BAND_Y`.
**Step 4.8 — Cross-region neighbour fetch.** `chunk_fill` asks 9 regions for systems that intersect.
**Step 4.9 — Tests.** Connectivity, surface buffer, sinkhole punch-through.
**Step 4.10 — Visual.** Screenshot a sinkhole and a cliff mouth.
**Step 4.11 — Re-baseline, commit, push.**

## PR 5 — Biome refresh + palm trees

**Files (modify):**
- `src/worldgen/climate.rs` — `Biome::Tropical` + threshold perturbation
- `src/worldgen/surface.rs` — sand/grass transition band, beach width 4, snow line cleanup
- `src/worldgen/trees.rs` — `TreeKind::{Oak, Palm}`, palm shape stamping

**Step 5.1 — Add `Tropical` variant.**
**Step 5.2 — Threshold perturbation noise field.**
**Step 5.3 — Sand transition band.**
**Step 5.4 — Tree-rate interpolation.**
**Step 5.5 — Palm tree stamp function.**
**Step 5.6 — Tests:** All biomes including Tropical; transition band sanity.
**Step 5.7 — Visual:** Tropical island screenshot.
**Step 5.8 — Re-baseline, commit, push.**

## PR 6 — Cleanup + map fingerprint

**Files (modify):**
- `src/worldgen/mod.rs` — remove legacy shims, thin public surface only
- `src/worldgen/tuning.rs` — final pass on constants
- `tests/worldgen_fingerprint.rs` (new) — PNG hash map fingerprint test

**Step 6.1 — Remove all legacy_* shims.**
**Step 6.2 — Final tuning pass with screenshots.**
**Step 6.3 — Map fingerprint test:** Render 256×256 heightmap PNG at seed 42, hash, pin.
**Step 6.4 — Re-baseline golden hash for the last time.**
**Step 6.5 — Commit, push.**

---

## Validation gates

After each PR:
1. `cargo check` clean.
2. `cargo test --lib worldgen` passes.
3. `cargo run -- --screenshot-and-exit /tmp/wg-prN.png --spawn 0,140,0 --look 0,-45` produces a valid PNG.
4. Visual review of screenshot for regressions.

## Open questions resolved by judgement during execution

- **Macro cache shared vs per-instance:** per-instance (simplest invariant; pool later if needed).
- **kd-tree eager vs lazy:** lazy (`OnceCell<KdTree>` per region).
- **Tunnel spline domain-warp:** 2-octave Simplex, ~24-block period, amplitude = tunnel_radius × 1.5.
- **LRU crate:** `lru = "0.12"` — well-established, no extra dependencies.
- **Hash function for plate ID & cave rolls:** xor-shift / golden-ratio multiply pattern already used in `tree_hash`. Re-export from a shared module.
