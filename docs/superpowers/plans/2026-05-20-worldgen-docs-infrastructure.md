# Documentation Site Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Docusaurus-based *How Oxium Builds a World* documentation site at `docs/book/`, with the `doc_render` Rust binary for generating engine-authoritative images, the widget framework (lazy-loaded React+Canvas), CI deploy to GitHub Pages, and two demonstration chapters (1.1 Voxel World — no widget; 1.4 FBM — with the headline `FbmExplorer` widget) that prove the chapter-skeleton and widget systems work end-to-end.

**Architecture:** Docusaurus 3 (TypeScript) at `docs/book/`, separate from `cargo build`. Engine-authoritative images come from a new `src/bin/doc_render/` binary that reuses overlay rendering code from `worldgen_viz` (which we promote into the library to enable sharing). Widgets are React+Canvas components in `docs/book/src/widgets/` with math ported to TS in `docs/book/src/math/`, kept honest via a parity-test page that compares TS outputs to JSON dumped by `doc_render --dump-parity`. CI: a GitHub Action on push to `main` runs `cargo build --release --bin doc_render`, regenerates images, runs the parity check, builds the site, deploys to `gh-pages`.

**Tech Stack:** Docusaurus 3 + TypeScript + React 18; remark-math + rehype-katex for math; `@docusaurus/theme-mermaid` for system diagrams; existing Rust 2024 edition + wgpu/glam stack for `doc_render`; `image = "0.25"` (already a dep) for PNG output; GitHub Actions; GitHub Pages.

**Out of plan scope:** Phase 1 chapter content beyond the two demos (1.1, 1.4) — those are separate writing plans (Plans 2–7). This plan delivers the *infrastructure* needed for Plans 2–7 to drop chapters into.

---

## Pre-flight: enter a worktree

Per the user's instruction, all implementation work for the docs project lives in a dedicated git worktree, not on `main`.

Before starting Task 1:

```bash
# From the main repo root, create a worktree for this project:
git worktree add .worktrees/docs-infra -b docs/book-infra
cd .worktrees/docs-infra
```

(If using the harness's `EnterWorktree` tool: `EnterWorktree(name: "docs-infra")`.)

The plan file itself lives on `main` at `docs/superpowers/plans/`. All file paths in tasks below are relative to the repo root **inside the worktree**.

---

## File structure overview

New files this plan introduces:

```
docs/book/                         # Docusaurus site (new)
├── docusaurus.config.ts           # site config (title, KaTeX, Mermaid, GH Pages URL)
├── sidebars.ts                    # full Phase 1 sidebar with stub chapter entries
├── package.json                   # Docusaurus deps + scripts
├── tsconfig.json
├── content/                       # all .mdx chapters
│   ├── intro.mdx                  # landing page
│   ├── part-1-foundations/
│   │   ├── 1.1-voxel-world.mdx    # written end-to-end (demo)
│   │   ├── 1.2-determinism.mdx    # stub
│   │   ├── …                      # stubs
│   │   └── 1.4-fbm.mdx            # written end-to-end (widget demo)
│   ├── part-2-overview/2.1-big-picture.mdx           # stub
│   ├── part-3-region-build/*.mdx                     # stubs
│   ├── part-4-chunk-fill/*.mdx                       # stubs
│   ├── part-5-engineering/*.mdx                      # stubs
│   └── appendices/{a,b,c,d}.mdx                      # stubs
├── src/
│   ├── book.constants.ts          # BOOK_SEED + other widget constants
│   ├── components/Stub.tsx        # the placeholder component used by stub chapters
│   ├── math/fbm.ts                # TS port of FBM (used by FbmExplorer + parity test)
│   ├── widgets/FbmExplorer.tsx    # the headline widget
│   ├── widgets/LazyWidget.tsx     # shared lazy-loading wrapper with static fallback
│   ├── hooks/useBookSeed.ts       # widget hook returning BOOK_SEED
│   └── pages/parity.tsx           # hidden parity-test page (TS-vs-Rust)
├── static/
│   ├── img/generated/.gitkeep
│   ├── img/diagrams/.gitkeep
│   └── img/figures/.gitkeep
├── tools/
│   └── gen-images.sh              # manifest of every generated image
└── .gitignore                     # node_modules/, build/, .docusaurus/

src/bin/doc_render/                # new Rust binary
├── main.rs                        # CLI parsing + dispatch
├── snapshot.rs                    # writes a PNG by sampling generator.sample_stage per pixel
└── parity_dump.rs                 # writes the JSON reference set for the parity test

src/                               # library changes
└── viz_render/                    # NEW — promoted from worldgen_viz so doc_render can reuse
    ├── mod.rs                     # pub mod colormap; pub mod stages; pub use overlay primitives;
    ├── colormap.rs                # MOVED from src/bin/worldgen_viz/overlays/colormap.rs
    └── stages.rs                  # MOVED from src/bin/worldgen_viz/overlays/stages.rs

.github/workflows/
├── docs-build-deploy.yml          # build + deploy site on push to main
└── docs-image-drift.yml           # PR check: regenerate images, comment if any moved
```

Files modified:

```
Cargo.toml                                   # add [[bin]] doc_render entry; expose viz_render lib
src/lib.rs                                   # pub mod viz_render;
src/bin/worldgen_viz/overlays/mod.rs         # re-export from oxium::viz_render
src/bin/worldgen_viz/overlays/colormap.rs    # DELETED (moved to lib)
src/bin/worldgen_viz/overlays/stages.rs      # DELETED (moved to lib)
.gitignore                                   # add docs/book/node_modules/, docs/book/build/, docs/book/.docusaurus/
README.md                                    # add a small "Docs" section linking to the published book URL
```

---

## Task 1: Promote worldgen_viz overlay rendering into the library

The `overlays::colormap` and `overlays::stages` modules in `worldgen_viz` already do exactly what `doc_render` needs: given a `Generator`, a `Stage`, and a world column `(wx, wz)`, return `[u8; 4]` RGBA. To reuse from a sibling binary, they need to live in the library.

**Files:**
- Create: `src/viz_render/mod.rs`
- Create: `src/viz_render/colormap.rs` (content moved from worldgen_viz)
- Create: `src/viz_render/stages.rs` (content moved from worldgen_viz)
- Modify: `src/lib.rs`
- Modify: `src/bin/worldgen_viz/overlays/mod.rs`
- Delete: `src/bin/worldgen_viz/overlays/colormap.rs`
- Delete: `src/bin/worldgen_viz/overlays/stages.rs`

- [ ] **Step 1: Read the current overlay files**

```bash
cat src/bin/worldgen_viz/overlays/colormap.rs
cat src/bin/worldgen_viz/overlays/stages.rs
cat src/bin/worldgen_viz/overlays/mod.rs
```

Expected: see ~200–400 lines of pure rendering helpers (no egui/wgpu dependencies). If any of these files imports from elsewhere in the `worldgen_viz` binary, note those imports — they need to come too or be left behind.

- [ ] **Step 2: Verify the move is safe — check for binary-only deps**

Run:

```bash
grep -E "^use" src/bin/worldgen_viz/overlays/colormap.rs src/bin/worldgen_viz/overlays/stages.rs
```

Expected: imports should be from `oxium::worldgen`, `glam`, `std`, or sibling modules within `overlays/`. If any line shows `use crate::{paint, session, app, camera}::…`, the file is binary-coupled and we either move that dep too or skip moving the file. Based on the design and prior `grep`, both files are pure — proceed.

- [ ] **Step 3: Create the new library module**

Create `src/viz_render/mod.rs`:

```rust
//! Engine-authoritative rendering helpers shared between the `worldgen_viz`
//! debug app and the `doc_render` snapshot binary. Pure layer on top of
//! `crate::worldgen::Generator` + `crate::worldgen::probe::Stage`.

pub mod colormap;
pub mod stages;

pub use stages::{pixel, render_pixel};
```

Move the contents of `src/bin/worldgen_viz/overlays/colormap.rs` → `src/viz_render/colormap.rs` (verbatim — same `pub fn` signatures).

Move the contents of `src/bin/worldgen_viz/overlays/stages.rs` → `src/viz_render/stages.rs` and rewrite the `use crate::overlays::colormap;` line to `use crate::viz_render::colormap;`.

- [ ] **Step 4: Expose from the library root**

Add to `src/lib.rs` (alphabetical with the existing `pub mod` lines):

```rust
pub mod viz_render;
```

- [ ] **Step 5: Re-export from worldgen_viz's `overlays` so the binary keeps compiling**

Replace the contents of `src/bin/worldgen_viz/overlays/mod.rs` with:

```rust
//! Overlay map + per-stage sampler.
//!
//! The pure rendering primitives (colormap, per-stage pixel) live in
//! the library at `oxium::viz_render` so the `doc_render` binary can
//! reuse them. This module re-exports them and keeps the egui/UI-side
//! `MapView` plumbing here (binary-only).

pub use oxium::viz_render::{colormap, stages};

// ... keep the existing MapView struct and pixel_to_world / world_to_pixel ...
```

Keep everything else in `mod.rs` that isn't a `mod` declaration for the moved files. Remove the lines `pub mod colormap;` and `pub mod stages;` (they're now re-exports).

- [ ] **Step 6: Delete the moved-from files**

```bash
git rm src/bin/worldgen_viz/overlays/colormap.rs
git rm src/bin/worldgen_viz/overlays/stages.rs
```

- [ ] **Step 7: Verify both binaries still compile**

Run:

```bash
cargo build --bin worldgen_viz
cargo build
```

Expected: both succeed with no warnings about missing modules. If there are unresolved imports elsewhere, search-and-replace `crate::overlays::colormap` → `crate::overlays::colormap` (no change; the re-export keeps the path the same) and same for `stages`.

- [ ] **Step 8: Run tests**

Run:

```bash
cargo test --bin worldgen_viz
```

Expected: existing tests still pass. The pixel determinism test (`render_pixel_is_deterministic`) verifies the move didn't break anything.

- [ ] **Step 9: Commit**

```bash
git add src/viz_render/ src/lib.rs src/bin/worldgen_viz/overlays/mod.rs
git rm src/bin/worldgen_viz/overlays/colormap.rs src/bin/worldgen_viz/overlays/stages.rs
git commit -m "refactor(viz): promote overlay rendering primitives into the library

Lets the new doc_render binary reuse the same colormap + per-stage
sampler that worldgen_viz uses, so book images are engine-authoritative
with no duplication.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add the `doc_render` binary — scaffold + `--help`

**Files:**
- Create: `src/bin/doc_render/main.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Register the binary in Cargo.toml**

Add to `Cargo.toml` under existing `[[bin]]` entries (search for `[[bin]]` to find the right block):

```toml
[[bin]]
name = "doc_render"
path = "src/bin/doc_render/main.rs"
```

- [ ] **Step 2: Create the binary scaffold with CLI parsing via `std::env`**

We deliberately avoid pulling in `clap` for one small binary. Create `src/bin/doc_render/main.rs`:

```rust
//! Snapshot tool that turns the engine's worldgen pipeline into PNGs
//! for the documentation book. Pure on `(seed, center, zoom, stage)`.
//!
//! Usage:
//!   doc_render --seed <u64> --center <wx>,<wz> --zoom <N> \
//!              --stage <stage> --width <px> --output <path>
//!   doc_render --dump-parity --output <path>
//!   doc_render --help

use std::process::ExitCode;

fn print_help() {
    eprintln!(
        "doc_render — generate engine-authoritative images for the docs book

Usage:
  doc_render snapshot --seed <u64> --center <wx,wz> --zoom <N> \\
                      --stage <stage> --width <px> --output <path>
  doc_render parity --output <path>
  doc_render --help

Stages (snapshot):
  continentalness, plate-id, temperature, humidity, desertness,
  weirdness, h-pre, valley-carve, h-target, flow-accum, biome-id,
  aquifer-y, aquifer-substance
"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    let sub = args[0].as_str();
    let rest = &args[1..];
    let result: Result<(), String> = match sub {
        "snapshot" => snapshot::run(rest),
        "parity"   => parity_dump::run(rest),
        other      => Err(format!("unknown subcommand: {other}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

mod snapshot;
mod parity_dump;
```

- [ ] **Step 3: Create empty submodules so the file compiles**

Create `src/bin/doc_render/snapshot.rs`:

```rust
pub fn run(_args: &[String]) -> Result<(), String> {
    Err("snapshot not yet implemented".into())
}
```

Create `src/bin/doc_render/parity_dump.rs`:

```rust
pub fn run(_args: &[String]) -> Result<(), String> {
    Err("parity dump not yet implemented".into())
}
```

- [ ] **Step 4: Verify it builds**

Run:

```bash
cargo build --bin doc_render
cargo run --bin doc_render -- --help
```

Expected: build succeeds; running with `--help` prints the usage text and exits 0.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/bin/doc_render/
git commit -m "feat(doc_render): scaffold the snapshot binary

Stub binary with --help and subcommand dispatch. Snapshot and parity
implementations land in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement `doc_render snapshot`

**Files:**
- Modify: `src/bin/doc_render/snapshot.rs`

- [ ] **Step 1: Write the snapshot implementation**

Replace `src/bin/doc_render/snapshot.rs` with:

```rust
//! Snapshot mode: render a (zoom × width) top-down image of one
//! pipeline stage and write it as a PNG.

use image::{ImageBuffer, Rgba};
use oxium::viz_render;
use oxium::worldgen::{probe::Stage, Generator};
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), String> {
    let opts = parse(args)?;
    let stage = parse_stage(&opts.stage)?;
    let gen = Generator::new(opts.seed);

    let w = opts.width;
    let h = opts.width; // square images for now; can add --height later
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);

    let half = (w as i32) / 2;
    for py in 0..h {
        for px in 0..w {
            // Pixel (px, py) → world coordinate.
            // px=0 → wx = center.0 - half*zoom; py=0 → wz = center.1 - half*zoom.
            let wx = opts.center.0 + (px as i32 - half) * (opts.zoom as i32);
            let wz = opts.center.1 + (py as i32 - half) * (opts.zoom as i32);
            let rgba = viz_render::render_pixel(&gen, stage, wx, wz);
            img.put_pixel(px, py, Rgba(rgba));
        }
    }

    img.save(&opts.output).map_err(|e| format!("write {:?}: {e}", opts.output))?;
    Ok(())
}

#[derive(Debug)]
struct Opts {
    seed: u64,
    center: (i32, i32),
    zoom: u32,
    stage: String,
    width: u32,
    output: PathBuf,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut seed: Option<u64> = None;
    let mut center: Option<(i32, i32)> = None;
    let mut zoom: Option<u32> = None;
    let mut stage: Option<String> = None;
    let mut width: Option<u32> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let val = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--seed" => seed = Some(val.parse().map_err(|_| format!("bad seed: {val}"))?),
            "--center" => {
                let parts: Vec<&str> = val.split(',').collect();
                if parts.len() != 2 {
                    return Err(format!("--center expects wx,wz, got {val}"));
                }
                let wx: i32 = parts[0].parse().map_err(|_| format!("bad wx: {}", parts[0]))?;
                let wz: i32 = parts[1].parse().map_err(|_| format!("bad wz: {}", parts[1]))?;
                center = Some((wx, wz));
            }
            "--zoom"   => zoom   = Some(val.parse().map_err(|_| format!("bad zoom: {val}"))?),
            "--stage"  => stage  = Some(val.clone()),
            "--width"  => width  = Some(val.parse().map_err(|_| format!("bad width: {val}"))?),
            "--output" => output = Some(PathBuf::from(val)),
            other      => return Err(format!("unknown flag: {other}")),
        }
        i += 2;
    }

    Ok(Opts {
        seed:   seed.ok_or("--seed required")?,
        center: center.ok_or("--center required")?,
        zoom:   zoom.ok_or("--zoom required")?,
        stage:  stage.ok_or("--stage required")?,
        width:  width.ok_or("--width required")?,
        output: output.ok_or("--output required")?,
    })
}

fn parse_stage(s: &str) -> Result<Stage, String> {
    match s {
        "continentalness"     => Ok(Stage::Continentalness),
        "plate-id"            => Ok(Stage::PlateId),
        "temperature"         => Ok(Stage::Temperature),
        "humidity"            => Ok(Stage::Humidity),
        "desertness"          => Ok(Stage::Desertness),
        "weirdness"           => Ok(Stage::Weirdness),
        "h-pre"               => Ok(Stage::HPre),
        "valley-carve"        => Ok(Stage::ValleyCarve),
        "h-target"            => Ok(Stage::HTarget),
        "flow-accum"          => Ok(Stage::FlowAccum),
        "biome-id"            => Ok(Stage::BiomeId),
        "aquifer-y"           => Ok(Stage::AquiferY),
        "aquifer-substance"   => Ok(Stage::AquiferSubstance),
        other                 => Err(format!("unknown stage: {other}")),
    }
}
```

- [ ] **Step 2: Verify it builds**

Run:

```bash
cargo build --release --bin doc_render
```

Expected: builds in release mode (we'll be running it from CI; release matters for speed).

- [ ] **Step 3: Smoke-test with one image**

Run:

```bash
mkdir -p /tmp/doc_render_test
cargo run --release --bin doc_render -- snapshot \
  --seed 42 --center 0,0 --zoom 4 --stage h-pre \
  --width 256 --output /tmp/doc_render_test/h_pre_seed42.png
file /tmp/doc_render_test/h_pre_seed42.png
```

Expected: file output reports `PNG image data, 256 x 256`. Open the file with `open /tmp/doc_render_test/h_pre_seed42.png` (macOS Preview) — you should see a terrain-colored top-down heightmap, not a solid color.

- [ ] **Step 4: Smoke-test determinism**

Run:

```bash
cargo run --release --bin doc_render -- snapshot \
  --seed 42 --center 0,0 --zoom 4 --stage h-pre \
  --width 256 --output /tmp/doc_render_test/h_pre_seed42_b.png
diff /tmp/doc_render_test/h_pre_seed42.png /tmp/doc_render_test/h_pre_seed42_b.png && echo "DETERMINISTIC"
```

Expected: prints `DETERMINISTIC`. (The renderer is pure on `(seed, coord)`, so two runs at the same flags must produce byte-identical output.)

- [ ] **Step 5: Commit**

```bash
git add src/bin/doc_render/snapshot.rs
git commit -m "feat(doc_render): implement snapshot subcommand

Top-down per-pixel sampling of any Stage, reusing viz_render::render_pixel
for color. Deterministic in (seed, center, zoom, stage).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement `doc_render parity` (dumps Rust reference outputs for TS parity tests)

**Files:**
- Modify: `src/bin/doc_render/parity_dump.rs`

- [ ] **Step 1: Implement the parity dump**

Replace `src/bin/doc_render/parity_dump.rs` with:

```rust
//! Parity dump: writes a JSON file containing reference outputs of
//! the Rust math functions that have TS ports under
//! `docs/book/src/math/`. The TS parity-test page imports this JSON
//! and asserts the TS implementations match.
//!
//! New functions are added here as their TS ports land. Each entry
//! is a `(input, output)` pair using a deterministic input set.

use noise::{Fbm, MultiFractal, NoiseFn, Simplex};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), String> {
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let val = args.get(i + 1).ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--output" => output = Some(PathBuf::from(val)),
            other      => return Err(format!("unknown flag: {other}")),
        }
        i += 2;
    }
    let output = output.ok_or("--output required")?;

    let mut entries: Vec<String> = Vec::new();

    // -- fbm --
    // Inputs span the unit square at several scales / seeds /
    // (octaves, persistence) pairs so a TS port that gets octave
    // composition wrong will show.
    let cases: &[(u32, usize, f64, f64, f64)] = &[
        // (seed, octaves, persistence, x, y)
        (42, 4, 0.5, 0.0, 0.0),
        (42, 4, 0.5, 0.25, 0.0),
        (42, 4, 0.5, 0.5, 0.5),
        (42, 1, 0.5, 0.5, 0.5),
        (42, 6, 0.6, 0.123, 0.789),
        (1337, 4, 0.5, 0.0, 0.0),
    ];
    for &(seed, oct, pers, x, y) in cases {
        let fbm = Fbm::<Simplex>::new(seed)
            .set_octaves(oct)
            .set_frequency(1.0)
            .set_persistence(pers);
        let v = fbm.get([x, y]);
        entries.push(format!(
            "    {{\"fn\":\"fbm\",\"args\":{{\"seed\":{seed},\"octaves\":{oct},\"persistence\":{pers},\"x\":{x},\"y\":{y}}},\"out\":{v}}}",
        ));
    }

    // Write as JSON array. We hand-build the JSON so we don't pull
    // in serde_json just for this (the schema is simple and stable).
    let json = format!("[\n{}\n]\n", entries.join(",\n"));
    let mut f = File::create(&output).map_err(|e| format!("create {:?}: {e}", output))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("write {:?}: {e}", output))?;
    Ok(())
}
```

- [ ] **Step 2: Verify it builds**

Run:

```bash
cargo build --release --bin doc_render
```

- [ ] **Step 3: Smoke-test the dump**

Run:

```bash
cargo run --release --bin doc_render -- parity --output /tmp/parity.json
cat /tmp/parity.json
```

Expected: JSON array of 6 entries with `fn: "fbm"`. Numeric `out` values are floats.

- [ ] **Step 4: Commit**

```bash
git add src/bin/doc_render/parity_dump.rs
git commit -m "feat(doc_render): implement parity subcommand

Dumps reference outputs of math functions that have TS ports in the
book's widget code. The TS parity test loads this JSON and asserts
its implementations match Rust on the same inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Scaffold the Docusaurus site

**Files:**
- Create: `docs/book/` (entire directory tree via Docusaurus CLI)
- Modify: `.gitignore`

- [ ] **Step 1: Confirm Node is available**

Run:

```bash
node --version
npm --version
```

Expected: both print versions. Docusaurus 3 requires Node ≥ 18.

- [ ] **Step 2: Scaffold via `create-docusaurus`**

Run from the repo root (inside the worktree):

```bash
npx create-docusaurus@latest docs/book classic --typescript
```

Expected: creates `docs/book/` with `package.json`, `docusaurus.config.ts`, `sidebars.ts`, `docs/`, `blog/`, `src/`, `static/`. Takes ~30 seconds.

- [ ] **Step 3: Strip the template content we don't want**

Run:

```bash
rm -rf docs/book/blog
rm -rf docs/book/docs
rm -rf docs/book/src/pages/markdown-page.md
```

- [ ] **Step 4: Create the content directory**

Run:

```bash
mkdir -p docs/book/content/{part-1-foundations,part-2-overview,part-3-region-build,part-4-chunk-fill,part-5-engineering,appendices}
mkdir -p docs/book/static/img/{generated,diagrams,figures}
touch docs/book/static/img/generated/.gitkeep
touch docs/book/static/img/diagrams/.gitkeep
touch docs/book/static/img/figures/.gitkeep
```

- [ ] **Step 5: Update .gitignore at repo root**

Add to `.gitignore` (at repo root, not inside docs/book):

```
# Docusaurus
docs/book/node_modules/
docs/book/build/
docs/book/.docusaurus/
```

- [ ] **Step 6: Verify the scaffold builds**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: `docs/book/build/` produced. Since we removed `docs/` and `blog/`, the build will print warnings about empty content — that's fine, we'll wire content in the next task.

- [ ] **Step 7: Commit**

```bash
git add docs/book/ .gitignore
git commit -m "scaffold(book): create Docusaurus 3 site at docs/book/

Standard create-docusaurus output, TypeScript template. Default
template content (blog/, docs/, markdown-page) removed. Empty content/
tree created for Phase 1 part structure. Static image directories
seeded with .gitkeep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Configure docusaurus.config.ts (KaTeX + Mermaid + GH Pages URL)

**Files:**
- Modify: `docs/book/docusaurus.config.ts`
- Modify: `docs/book/package.json`

- [ ] **Step 1: Install math and Mermaid deps**

Run:

```bash
cd docs/book
npm install remark-math@^6 rehype-katex@^7 katex@^0.16
npm install @docusaurus/theme-mermaid@latest
cd ../..
```

Expected: deps land in `docs/book/package.json` under `dependencies`.

- [ ] **Step 2: Replace docusaurus.config.ts**

Replace `docs/book/docusaurus.config.ts` with:

```typescript
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'How Oxium Builds a World',
  tagline: 'A deep dive into the Oxium voxel engine',
  favicon: 'img/favicon.ico',

  url: 'https://fdatoo.github.io', // update if a custom domain is set later
  baseUrl: '/oxium/',

  organizationName: 'fdatoo',
  projectName: 'oxium',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/fdatoo/oxium/tree/main/docs/book/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    navbar: {
      title: 'How Oxium Builds a World',
      logo: { alt: 'Oxium', src: 'img/logo.svg' },
      items: [
        { to: '/', label: 'Read', position: 'left' },
        { href: 'https://github.com/fdatoo/oxium', label: 'GitHub', position: 'right' },
      ],
    },
    colorMode: { defaultMode: 'dark', respectPrefersColorScheme: true },
    docs: { sidebar: { hideable: true, autoCollapseCategories: false } },
  } satisfies Preset.ThemeConfig,
};

export default config;
```

- [ ] **Step 3: Verify the config parses and the site still builds**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: build succeeds (will still warn about empty content — wired up next).

- [ ] **Step 4: Commit**

```bash
git add docs/book/docusaurus.config.ts docs/book/package.json docs/book/package-lock.json
git commit -m "config(book): wire KaTeX, Mermaid, content path, GH Pages URL

Math via remark-math + rehype-katex; Mermaid via theme-mermaid;
content lives under ./content (not the default ./docs). Site builds
empty for now — content lands in Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Define the sidebar (full Phase 1 structure, stubs allowed)

**Files:**
- Modify: `docs/book/sidebars.ts`

- [ ] **Step 1: Replace sidebars.ts**

Replace `docs/book/sidebars.ts` with:

```typescript
import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  book: [
    'intro',
    {
      type: 'category',
      label: 'Part I — Foundations',
      collapsed: false,
      items: [
        'part-1-foundations/1.1-voxel-world',
        'part-1-foundations/1.2-determinism',
        'part-1-foundations/1.3-coherent-noise',
        'part-1-foundations/1.4-fbm',
        'part-1-foundations/1.5-voronoi',
        'part-1-foundations/1.6-domain-warping',
        'part-1-foundations/1.7-splines',
        'part-1-foundations/1.8-sdfs',
        'part-1-foundations/1.9-trilerp',
      ],
    },
    {
      type: 'category',
      label: 'Part II — Pipeline Overview',
      items: ['part-2-overview/2.1-big-picture'],
    },
    {
      type: 'category',
      label: 'Part III — Per-Region Build',
      items: [
        'part-3-region-build/3.1-plates',
        'part-3-region-build/3.2-climate',
        'part-3-region-build/3.3-heightmap',
        'part-3-region-build/3.4-hydrology',
        'part-3-region-build/3.5-rivers-lakes',
        'part-3-region-build/3.6-cave-systems',
      ],
    },
    {
      type: 'category',
      label: 'Part IV — Per-Chunk Fill',
      items: [
        'part-4-chunk-fill/4.1-density-graph',
        'part-4-chunk-fill/4.2-cell-evaluator',
        'part-4-chunk-fill/4.3-composing-caves',
        'part-4-chunk-fill/4.4-noise-carvers',
        'part-4-chunk-fill/4.5-procedural-carvers',
        'part-4-chunk-fill/4.6-surface-rules',
        'part-4-chunk-fill/4.7-aquifers',
        'part-4-chunk-fill/4.8-fluid-settle',
        'part-4-chunk-fill/4.9-trees',
      ],
    },
    {
      type: 'category',
      label: 'Part V — Engineering Scaffolding',
      items: [
        'part-5-engineering/5.1-determinism',
        'part-5-engineering/5.2-region-cache',
        'part-5-engineering/5.3-config-hot-reload',
        'part-5-engineering/5.4-visualizer',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      collapsed: true,
      items: [
        'appendices/a-module-index',
        'appendices/b-technique-index',
        'appendices/c-constants-catalog',
        'appendices/d-glossary',
      ],
    },
  ],
};

export default sidebars;
```

- [ ] **Step 2: Verify sidebar references resolve when content lands**

Don't try to build yet — Docusaurus will throw because the referenced doc IDs don't exist as files. We create those next.

- [ ] **Step 3: Commit**

```bash
git add docs/book/sidebars.ts
git commit -m "config(book): define Phase 1 sidebar (parts + chapters + appendices)

Full structure for all ~33 entries. Content files land in Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Create all stub chapter files + intro page

For every chapter the sidebar references, create a stub `.mdx` file using a shared `<Stub />` component. This lets the site build with everything wired up; Plans 2–7 then replace stubs with real content.

**Files:**
- Create: `docs/book/src/components/Stub.tsx`
- Create: `docs/book/content/intro.mdx`
- Create: 33 stub `.mdx` files (paths match sidebar IDs)

- [ ] **Step 1: Create the Stub component**

Create `docs/book/src/components/Stub.tsx`:

```tsx
import React from 'react';
import Admonition from '@theme/Admonition';

type Props = { children?: React.ReactNode };

export default function Stub({ children }: Props) {
  return (
    <Admonition type="note" title="Chapter under construction">
      <p>
        This chapter is part of <strong>How Oxium Builds a World — Phase 1</strong> and
        has not been written yet.
      </p>
      {children}
      <p>
        Track progress at{' '}
        <a href="https://github.com/fdatoo/oxium/tree/main/docs/superpowers/specs/2026-05-20-worldgen-docs-design.md">
          docs/superpowers/specs/2026-05-20-worldgen-docs-design.md
        </a>
        .
      </p>
    </Admonition>
  );
}
```

- [ ] **Step 2: Write the intro page**

Create `docs/book/content/intro.mdx`:

```mdx
---
id: intro
title: How Oxium Builds a World
sidebar_position: 1
slug: /
---

import Stub from '@site/src/components/Stub';

This is the documentation for **Oxium**, a voxel engine. The book teaches how the engine works from first principles up — starting with the math primitives, building the worldgen pipeline layer by layer, then (in later phases) the renderer, mesher, lighting, and engineering substrate.

## Who this is for

You have a CS degree but no specialized background in graphics or games math. You can read Rust well enough to follow types and ownership. By the end of Phase 1 you will be able to read the entire `src/worldgen/` module and know what every line is doing.

## What is in Phase 1

| Part | What it covers |
|---|---|
| **I.** Foundations | Noise, FBM, Voronoi, vector fields, splines, SDFs, trilerp, determinism — the math primitives the engine reuses. |
| **II.** Pipeline Overview | One chapter that walks the whole worldgen pipeline end-to-end, no math. |
| **III.** Per-Region Build | Plates, climate, heightmap, hydrology, cave systems — what the engine caches per region. |
| **IV.** Per-Chunk Fill | Density graph, cell evaluator, cave composition, surface rules, aquifers, trees — the runtime hot path. |
| **V.** Engineering Scaffolding | Determinism, region cache, hot-reload, the visualizer. |

Phase 2 (renderer, mesher, lighting) and Phase 3 (ECS, jobs, persistence, UI) land in subsequent versions of this book.

<Stub />
```

- [ ] **Step 3: Create stub files for every sidebar entry**

Run the following from the repo root (inside the worktree) — this is one paste-block that produces all 33 stubs:

```bash
cd docs/book/content

# Helper: write a stub MDX with the given path, id, title.
stub() {
  local path="$1" id="$2" title="$3"
  mkdir -p "$(dirname "$path")"
  cat > "$path" <<EOF
---
id: $id
title: $title
---

import Stub from '@site/src/components/Stub';

<Stub />
EOF
}

# Part I — Foundations
stub part-1-foundations/1.1-voxel-world.mdx     1.1-voxel-world      "1.1 The Voxel World"
stub part-1-foundations/1.2-determinism.mdx     1.2-determinism      "1.2 Deterministic Randomness"
stub part-1-foundations/1.3-coherent-noise.mdx  1.3-coherent-noise   "1.3 Coherent Noise"
stub part-1-foundations/1.4-fbm.mdx             1.4-fbm              "1.4 Fractional Brownian Motion"
stub part-1-foundations/1.5-voronoi.mdx         1.5-voronoi          "1.5 Voronoi Diagrams"
stub part-1-foundations/1.6-domain-warping.mdx  1.6-domain-warping   "1.6 Vector Fields & Domain Warping"
stub part-1-foundations/1.7-splines.mdx         1.7-splines          "1.7 Splines"
stub part-1-foundations/1.8-sdfs.mdx            1.8-sdfs             "1.8 Signed Distance Functions"
stub part-1-foundations/1.9-trilerp.mdx         1.9-trilerp          "1.9 Trilinear Interpolation"

# Part II — Pipeline Overview
stub part-2-overview/2.1-big-picture.mdx        2.1-big-picture      "2.1 The Big Picture"

# Part III — Per-Region Build
stub part-3-region-build/3.1-plates.mdx         3.1-plates           "3.1 Plates"
stub part-3-region-build/3.2-climate.mdx        3.2-climate          "3.2 Climate"
stub part-3-region-build/3.3-heightmap.mdx      3.3-heightmap        "3.3 Heightmap (h_pre)"
stub part-3-region-build/3.4-hydrology.mdx      3.4-hydrology        "3.4 Hydrology"
stub part-3-region-build/3.5-rivers-lakes.mdx   3.5-rivers-lakes     "3.5 Rivers, Valleys, Lakes"
stub part-3-region-build/3.6-cave-systems.mdx   3.6-cave-systems     "3.6 Cave Systems"

# Part IV — Per-Chunk Fill
stub part-4-chunk-fill/4.1-density-graph.mdx       4.1-density-graph    "4.1 The Density Graph"
stub part-4-chunk-fill/4.2-cell-evaluator.mdx      4.2-cell-evaluator   "4.2 The Cell Evaluator"
stub part-4-chunk-fill/4.3-composing-caves.mdx     4.3-composing-caves  "4.3 Composing Caves"
stub part-4-chunk-fill/4.4-noise-carvers.mdx       4.4-noise-carvers    "4.4 Noise Carvers"
stub part-4-chunk-fill/4.5-procedural-carvers.mdx  4.5-procedural-carvers "4.5 Procedural Carvers"
stub part-4-chunk-fill/4.6-surface-rules.mdx       4.6-surface-rules    "4.6 Surface Rules"
stub part-4-chunk-fill/4.7-aquifers.mdx            4.7-aquifers         "4.7 Aquifers"
stub part-4-chunk-fill/4.8-fluid-settle.mdx        4.8-fluid-settle     "4.8 Fluid Settle"
stub part-4-chunk-fill/4.9-trees.mdx               4.9-trees            "4.9 Trees"

# Part V — Engineering Scaffolding
stub part-5-engineering/5.1-determinism.mdx        5.1-determinism      "5.1 Determinism Everywhere"
stub part-5-engineering/5.2-region-cache.mdx       5.2-region-cache     "5.2 The Region Cache"
stub part-5-engineering/5.3-config-hot-reload.mdx  5.3-config-hot-reload "5.3 Config & Hot-Reload"
stub part-5-engineering/5.4-visualizer.mdx         5.4-visualizer       "5.4 The Visualizer"

# Appendices
stub appendices/a-module-index.mdx              a-module-index       "Appendix A: Module Index"
stub appendices/b-technique-index.mdx           b-technique-index    "Appendix B: Technique Index"
stub appendices/c-constants-catalog.mdx         c-constants-catalog  "Appendix C: Constants Catalog"
stub appendices/d-glossary.mdx                  d-glossary           "Appendix D: Glossary"

cd ../../..
```

- [ ] **Step 4: Build the site to verify the sidebar resolves**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: builds cleanly, no broken-link warnings, all 34 pages (intro + 33 stubs) appear in `docs/book/build/`.

- [ ] **Step 5: Spot-check locally**

Run:

```bash
cd docs/book && npm run serve & sleep 3 && open http://localhost:3000/oxium/ && cd ../..
```

Expected: browser opens the landing page; sidebar shows all parts; clicking through any chapter shows the "Chapter under construction" admonition.

Press Ctrl+C in the terminal when done viewing.

- [ ] **Step 6: Commit**

```bash
git add docs/book/src/components/Stub.tsx docs/book/content/
git commit -m "scaffold(book): create stub MDX for every Phase 1 chapter + intro

All 33 chapter stubs + the intro page wired into the sidebar. Each
stub renders a 'Chapter under construction' admonition. Plans 2–7
replace stubs with real content.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Set up widget infrastructure (book.constants.ts, useBookSeed, LazyWidget)

**Files:**
- Create: `docs/book/src/book.constants.ts`
- Create: `docs/book/src/hooks/useBookSeed.ts`
- Create: `docs/book/src/widgets/LazyWidget.tsx`

- [ ] **Step 1: Create the constants file**

Create `docs/book/src/book.constants.ts`:

```typescript
/**
 * Constants the book is parameterized by. The Rust binary
 * `doc_render` consumes the same values via CLI flags — when you
 * update BOOK_SEED here, also update docs/book/tools/gen-images.sh.
 */

/** The seed every chapter's images use unless explicitly overridden. */
export const BOOK_SEED = 42n;

/** Default world-space center for top-down maps (the "home" of the book world). */
export const BOOK_CENTER: { wx: number; wz: number } = { wx: 0, wz: 0 };
```

- [ ] **Step 2: Create the seed hook**

Create `docs/book/src/hooks/useBookSeed.ts`:

```typescript
import { BOOK_SEED } from '../book.constants';

/**
 * Returns the book seed. A hook (not a constant import) so future
 * widgets can override per-instance via a context provider without
 * touching widget call sites.
 */
export function useBookSeed(): bigint {
  return BOOK_SEED;
}
```

- [ ] **Step 3: Create the lazy-loading wrapper**

Create `docs/book/src/widgets/LazyWidget.tsx`:

```tsx
import React, { Suspense } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

type Props = {
  /** Static-image fallback shown during SSR + before client hydration. */
  fallbackSrc?: string;
  fallbackAlt?: string;
  children: React.ReactNode;
};

/**
 * Wraps a widget in BrowserOnly + Suspense so it doesn't run during
 * static-site build (where there's no canvas) and the page can paint
 * a fallback image before the JS bundle is parsed.
 */
export default function LazyWidget({ fallbackSrc, fallbackAlt, children }: Props) {
  const fallback = fallbackSrc ? (
    <img src={fallbackSrc} alt={fallbackAlt ?? 'Widget loading'} />
  ) : (
    <div style={{ minHeight: 200 }}>Loading interactive widget…</div>
  );
  return (
    <BrowserOnly fallback={fallback}>
      {() => <Suspense fallback={fallback}>{children}</Suspense>}
    </BrowserOnly>
  );
}
```

- [ ] **Step 4: Verify the site still builds (these files are imported in Task 10)**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: still builds (the new files aren't referenced by any chapter yet).

- [ ] **Step 5: Commit**

```bash
git add docs/book/src/book.constants.ts docs/book/src/hooks/ docs/book/src/widgets/LazyWidget.tsx
git commit -m "feat(book): widget infrastructure (BOOK_SEED, useBookSeed, LazyWidget)

LazyWidget wraps each widget in BrowserOnly + Suspense so SSR and the
pre-hydration paint use a static-image fallback. BOOK_SEED is the
single source of truth for the book's world seed — gen-images.sh
imports the same value via its own copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Implement the TS port of FBM + the FbmExplorer widget

**Files:**
- Create: `docs/book/src/math/fbm.ts`
- Create: `docs/book/src/widgets/FbmExplorer.tsx`

- [ ] **Step 1: Add a simplex-noise dep for the TS port**

Run:

```bash
cd docs/book && npm install simplex-noise@^4 && cd ../..
```

Expected: `simplex-noise` lands in `package.json`. We use this so the TS port matches the same algorithm family as the Rust `noise` crate (both are 2D simplex).

- [ ] **Step 2: Write the TS FBM port**

Create `docs/book/src/math/fbm.ts`:

```typescript
// parity: src/worldgen/* (anywhere Fbm<Simplex> is constructed)
//
// The Rust code uses `noise::Fbm<Simplex>`. This TS port uses the
// `simplex-noise` npm package and reproduces the same octave loop.
// The Rust crate's default frequency is 1.0; persistence default 0.5;
// lacunarity default 2.0 — match them here so the widget defaults
// align with what the engine uses.

import { createNoise2D } from 'simplex-noise';
import { alea } from 'seedrandom-alea';

export type FbmParams = {
  seed: number;
  octaves: number;
  persistence: number;  // amplitude multiplier per octave (default 0.5)
  lacunarity: number;   // frequency multiplier per octave (default 2.0)
  frequency: number;    // base frequency (default 1.0)
};

/**
 * Build a 2D FBM sampler matching the Rust `Fbm<Simplex>` semantics.
 * The returned function maps (x, y) → a scalar in approximately [-1, 1].
 */
export function makeFbm2D(params: FbmParams): (x: number, y: number) => number {
  const prng = alea(String(params.seed));
  const noise = createNoise2D(prng);
  const { octaves, persistence, lacunarity, frequency } = params;
  return (x, y) => {
    let amp = 1.0;
    let freq = frequency;
    let total = 0.0;
    let maxAmp = 0.0;
    for (let o = 0; o < octaves; o++) {
      total += amp * noise(x * freq, y * freq);
      maxAmp += amp;
      amp *= persistence;
      freq *= lacunarity;
    }
    return total / maxAmp;
  };
}
```

Note on the `seedrandom-alea` import: if `simplex-noise@4` no longer requires an external PRNG (its API may have shifted), use whatever PRNG injection point that version provides. Verify with `npm view simplex-noise@4 main` and adjust the import.

- [ ] **Step 3: Install the PRNG dep**

Run:

```bash
cd docs/book && npm install alea && cd ../..
```

(If `simplex-noise@4` ships its own PRNG, skip this and adjust `fbm.ts` to use the built-in one.)

- [ ] **Step 4: Write the FbmExplorer widget**

Create `docs/book/src/widgets/FbmExplorer.tsx`:

```tsx
import React, { useEffect, useRef, useState } from 'react';
import { makeFbm2D } from '../math/fbm';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 256;

export default function FbmExplorer() {
  const seedBigInt = useBookSeed();
  const seed = Number(seedBigInt & 0xffff_ffffn);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [octaves, setOctaves] = useState(4);
  const [persistence, setPersistence] = useState(0.5);
  const [lacunarity, setLacunarity] = useState(2.0);
  const [frequency, setFrequency] = useState(1 / 64);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const fbm = makeFbm2D({ seed, octaves, persistence, lacunarity, frequency });
    const img = ctx.createImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = fbm(x, y);            // ~ [-1, 1]
        const g = Math.round(((v + 1) * 0.5) * 255);
        const i = (y * SIZE + x) * 4;
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [seed, octaves, persistence, lacunarity, frequency]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
      <canvas ref={canvasRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444' }} />
      <div>
        <label>
          Octaves: {octaves}
          <input type="range" min={1} max={8} step={1} value={octaves}
            onChange={(e) => setOctaves(parseInt(e.target.value, 10))} />
        </label>
        <label>
          Persistence: {persistence.toFixed(2)}
          <input type="range" min={0.1} max={0.9} step={0.05} value={persistence}
            onChange={(e) => setPersistence(parseFloat(e.target.value))} />
        </label>
        <label>
          Lacunarity: {lacunarity.toFixed(2)}
          <input type="range" min={1.5} max={3.0} step={0.1} value={lacunarity}
            onChange={(e) => setLacunarity(parseFloat(e.target.value))} />
        </label>
        <label>
          Frequency: {frequency.toFixed(4)}
          <input type="range" min={0.005} max={0.1} step={0.005} value={frequency}
            onChange={(e) => setFrequency(parseFloat(e.target.value))} />
        </label>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify the site still builds (widget isn't imported yet)**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: builds. (The widget is referenced from chapter 1.4 in Task 12.)

- [ ] **Step 6: Commit**

```bash
git add docs/book/src/math/ docs/book/src/widgets/FbmExplorer.tsx docs/book/package.json docs/book/package-lock.json
git commit -m "feat(book): FBM TS port + FbmExplorer widget

makeFbm2D mirrors noise::Fbm<Simplex> octave composition. The widget
renders 256x256 grayscale FBM with sliders for octaves, persistence,
lacunarity, frequency. Lazy-loaded; no SSR — wrapped in LazyWidget at
use sites.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Write the parity-test page

A hidden page at `/parity` runs the TS math against `parity.json` (produced by `doc_render parity`) and renders a pass/fail table. CI fails the build if any row fails.

**Files:**
- Create: `docs/book/src/pages/parity.tsx`
- Create: `docs/book/static/parity.json` (generated by `doc_render parity`)

- [ ] **Step 1: Generate the parity reference JSON**

Run from the repo root:

```bash
cargo run --release --bin doc_render -- parity --output docs/book/static/parity.json
```

Expected: `docs/book/static/parity.json` exists, ~1 KB, contains the FBM entries from Task 4.

- [ ] **Step 2: Write the parity page**

Create `docs/book/src/pages/parity.tsx`:

```tsx
import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import { makeFbm2D } from '../math/fbm';

type Entry =
  | {
      fn: 'fbm';
      args: { seed: number; octaves: number; persistence: number; x: number; y: number };
      out: number;
    };

type Row = {
  fn: string;
  args: string;
  rust: number;
  ts: number;
  delta: number;
  ok: boolean;
};

const TOLERANCE = 1e-6;

function runFbm(e: Entry): number {
  const fbm = makeFbm2D({
    seed: e.args.seed,
    octaves: e.args.octaves,
    persistence: e.args.persistence,
    lacunarity: 2.0,
    frequency: 1.0,
  });
  return fbm(e.args.x, e.args.y);
}

export default function ParityPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch('/oxium/parity.json')
      .then((r) => r.json())
      .then((entries: Entry[]) => {
        const rs = entries.map((e) => {
          const ts = e.fn === 'fbm' ? runFbm(e) : NaN;
          const delta = Math.abs(ts - e.out);
          return {
            fn: e.fn,
            args: JSON.stringify(e.args),
            rust: e.out,
            ts,
            delta,
            ok: delta < TOLERANCE,
          };
        });
        setRows(rs);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const allOk = rows.every((r) => r.ok);

  return (
    <Layout title="Parity">
      <main style={{ padding: '2rem' }}>
        <h1>TS-vs-Rust math parity</h1>
        {err && <p style={{ color: 'red' }}>Error: {err}</p>}
        <p>
          Status:{' '}
          <span data-testid="parity-status">
            {rows.length === 0 ? 'loading' : allOk ? 'PASS' : 'FAIL'}
          </span>
        </p>
        <table>
          <thead>
            <tr>
              <th>fn</th>
              <th>args</th>
              <th>rust</th>
              <th>ts</th>
              <th>|Δ|</th>
              <th>ok</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ background: r.ok ? undefined : '#fdd' }}>
                <td>{r.fn}</td>
                <td><code>{r.args}</code></td>
                <td>{r.rust.toFixed(6)}</td>
                <td>{Number.isNaN(r.ts) ? 'n/a' : r.ts.toFixed(6)}</td>
                <td>{r.delta.toExponential(2)}</td>
                <td>{r.ok ? 'ok' : 'FAIL'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </main>
    </Layout>
  );
}
```

- [ ] **Step 3: Build and view the page locally**

Run:

```bash
cd docs/book && npm run build && npm run serve & sleep 3 && open http://localhost:3000/oxium/parity && cd ../..
```

Expected: the page renders a table. The `simplex-noise` TS port will *not* match Rust's `noise` crate exactly out of the box because the two crates use different PRNG seeding for the gradient lattice. **This is expected and discussed in the next step.**

Press Ctrl+C in the terminal when done viewing.

- [ ] **Step 4: Reconcile the seed-handling gap**

This is the hardest part of the plan and worth doing in a separate commit. Two paths:

- **Option A (preferred):** in `docs/book/src/math/fbm.ts`, seed the `simplex-noise` PRNG identically to how the `noise` crate seeds its lattice. Cross-language fidelity for a single function (FBM) is reasonable — read `noise`'s source for `Simplex::new(seed)` and reproduce. May require writing a small permutation-table generator in TS.

- **Option B (fallback):** loosen the parity tolerance for FBM to "≤ 0.05 absolute difference" and call it close-enough. Acknowledge in a comment that the *behaviour* matches (octave composition is correct) but the underlying gradient field is from a different PRNG. Future widgets that need byte-exact match (e.g., Voronoi seeded by `hash::mix_*`) cannot use this fallback; they'll need Option A.

Try Option A first. If it ends up taking more than ~2 hours, fall back to B for the FBM case and revisit in a follow-up. The widget itself is educational either way — it shows what octaves do, regardless of whether the noise lattice matches the engine byte-for-byte.

- [ ] **Step 5: Commit whichever resolution lands**

```bash
git add docs/book/src/math/fbm.ts docs/book/src/pages/parity.tsx docs/book/static/parity.json
git commit -m "feat(book): TS-vs-Rust parity test page

Hidden /parity page loads docs/book/static/parity.json (produced by
doc_render parity) and asserts each TS math function matches Rust on
the same inputs. Pass/fail rendered in a table; allOk reflected in a
data-testid for CI scraping.

See task 11 step 4 for the Rust↔TS seed-handling notes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Write chapter 1.1 — The Voxel World (no-widget demo)

This chapter proves the chapter skeleton works end-to-end without any widget dependency. It teaches `Block`, `Chunk`, `ChunkCoord`, the palette — just enough that worldgen output makes sense.

**Files:**
- Modify: `docs/book/content/part-1-foundations/1.1-voxel-world.mdx`
- Modify: `docs/book/tools/gen-images.sh` (created in Task 14)

- [ ] **Step 1: Read the voxel module so the chapter is accurate**

Run:

```bash
ls src/voxel/
cat src/voxel/block.rs | head -80
cat src/voxel/chunk.rs | head -80
cat src/voxel/coords.rs | head -80
```

Note the actual type definitions, especially `Block`, `DenseChunk`, `ChunkCoord`, and `CHUNK_DIM`.

- [ ] **Step 2: Write the chapter**

Replace `docs/book/content/part-1-foundations/1.1-voxel-world.mdx`:

```mdx
---
id: 1.1-voxel-world
title: 1.1 The Voxel World
---

We're going to spend a lot of time talking about what the worldgen produces. So let's start with the thing it produces: a **chunk**.

## The picture

The world is a 3D grid of unit cubes called **blocks**. Each block has a kind — `Grass`, `Stone`, `Sand`, `Water`, `Air`, and so on. The grid is enormous and conceptually infinite, so the engine doesn't store it as one big array. It stores it in **chunks**: cubes of $32 \times 32 \times 32$ blocks.

The world is identified by its global `(x, y, z)` integer coordinates. Each chunk's bottom-southwest-west corner sits at a position that is a multiple of 32 on each axis — so the chunk at `ChunkCoord(0, 0, 0)` covers world coordinates `[0..32] × [0..32] × [0..32]`, the chunk at `ChunkCoord(1, 0, 0)` covers `[32..64] × [0..32] × [0..32]`, and so on.

## Build it

A `Block` is just an enum. The actual definition lives in `src/voxel/block.rs` and looks (simplified) like this:

```rust
// simplified for exposition; the real enum has more variants.
// src/voxel/block.rs
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Block {
    Air,
    Stone,
    Dirt,
    Grass,
    Sand,
    Snow,
    Water,
    Lava,
    Wood,
    Leaves,
    // ...
}
```

A `ChunkCoord` is a thin wrapper around an `IVec3` of integer chunk indices:

```rust
// src/voxel/coords.rs
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct ChunkCoord(pub glam::IVec3);

pub const CHUNK_DIM: i32 = 32;
pub const CHUNK_DIM_U: u32 = 32;
```

And a `DenseChunk` is — at its heart — a $32^3$ array of `Block`:

```rust
// simplified for exposition; the real type uses a palette + bitpacking to save memory.
// src/voxel/chunk.rs
pub struct DenseChunk {
    blocks: Box<[Block; (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) as usize]>,
}

impl DenseChunk {
    pub fn set(&mut self, local: LocalPos, b: Block) { /* index, store */ }
    pub fn get(&self, local: LocalPos) -> Block      { /* index, load  */ }
}
```

The worldgen pipeline's whole job is: given a `ChunkCoord`, produce a `DenseChunk`. Every subsequent chapter is "how do we decide which `Block` to put in each of the $32^3 = 32768$ cells?"

## In the engine

The real `DenseChunk` doesn't actually store one `Block` per cell. Storing 32 KB per chunk for `Block` (assuming a byte per cell) would be wasteful — most chunks contain only a few block kinds. Instead it uses a **palette**:

```
src/voxel/chunk.rs : DenseChunk (real)
  palette: Vec<Block>      // small (typically 1–6 entries)
  cells:   BitVec          // each cell is log2(palette.len()) bits, packed
```

This is an optimization, not a conceptual change. Code reading the chunk still asks "what block is at local position `(x, y, z)`?" and gets a `Block` back; the palette is transparent. We'll mention this again in **Phase 3, Chapter XI (Persistence)** when we look at how chunks are serialized to disk.

## What you can now do in the code

You can read:

- `src/voxel/block.rs` — every block kind in the engine.
- `src/voxel/chunk.rs` — chunk storage layout.
- `src/voxel/coords.rs` — coordinate conversions between world, chunk, and local space.

These three files are the language the rest of the book speaks. From here on out, when a chapter says "the worldgen writes `Water` at this voxel," you know what that means.

## Next

→ [1.2 Deterministic Randomness](./1.2-determinism.md) — the hash mixer that makes the entire engine pure on `(seed, coord)`.
```

- [ ] **Step 3: Verify the build**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: builds cleanly. The chapter renders with the chapter skeleton sections visible.

- [ ] **Step 4: Commit**

```bash
git add docs/book/content/part-1-foundations/1.1-voxel-world.mdx
git commit -m "content(book): write 1.1 The Voxel World

End-to-end chapter demonstrating the chapter skeleton: opening hook,
intuition, build-it, in-the-engine, what-you-can-now-do, next. No
widget, no generated images — the simplest viable demo chapter.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Write chapter 1.4 — FBM (widget demo)

This chapter proves the widget system works end-to-end. It introduces FBM as the canonical "smooth random terrain" technique and embeds `<FbmExplorer />` inline.

**Files:**
- Modify: `docs/book/content/part-1-foundations/1.4-fbm.mdx`

- [ ] **Step 1: Write the chapter**

Replace `docs/book/content/part-1-foundations/1.4-fbm.mdx`:

```mdx
---
id: 1.4-fbm
title: 1.4 Fractional Brownian Motion
---

import LazyWidget from '@site/src/widgets/LazyWidget';
import FbmExplorer from '@site/src/widgets/FbmExplorer';

In the previous chapter we built **coherent noise** — a smooth scalar field over $\mathbb{R}^2$ that goes up and down in a random-looking but continuous way. One octave of coherent noise gives you blobby gradients. Real terrain has structure at *every scale*: continental shapes, ridges, gravel-sized roughness. Stacking copies of noise at different frequencies gives you all of that, cheaply. That stack is called **Fractional Brownian Motion**, or FBM.

## The picture

FBM is a sum of noise *octaves*. The first octave is your base noise, sampled at frequency $f_0$. Each subsequent octave samples at a higher frequency and lower amplitude. The result is a fractal-looking field with detail across orders of magnitude.

$$
\mathrm{fbm}(x, y) = \sum_{i=0}^{N-1} a_i \cdot \mathrm{noise}(f_i x, f_i y)
$$

with $a_{i+1} = a_i \cdot p$ (persistence, default $0.5$) and $f_{i+1} = f_i \cdot l$ (lacunarity, default $2.0$).

Three parameters control it:

- **Octaves ($N$)** — how many copies you stack. More = more detail.
- **Persistence ($p$)** — how quickly amplitude decays. Lower = smoother (high octaves contribute less). Higher = noisier.
- **Lacunarity ($l$)** — how quickly frequency rises. $2.0$ doubles per octave; rarely tuned away from that.

Drag the sliders to see what each parameter does:

<LazyWidget fallbackSrc="/img/figures/fbm-default-fallback.png" fallbackAlt="FBM at default parameters">
  <FbmExplorer />
</LazyWidget>

A few things worth playing with:

- Set octaves to **1** and persistence to anything. You get a single octave of pure simplex noise — blobby, no fine detail.
- Set octaves to **8** with persistence **0.5**. You should see noticeable fine detail; the 8th octave is at frequency $f_0 \cdot 2^7$, so it's contributing rapid wiggles.
- Set octaves to **8** with persistence **0.85**. The high octaves no longer decay — the image looks noisy, almost speckled. This is why default persistence is $0.5$: high octaves should be visible but subordinate.

## Build it

A clean FBM implementation is short:

```rust
// simplified for exposition.
// (The engine uses noise::Fbm<Simplex> from the `noise` crate, which
//  does essentially this in a stateful builder.)
fn fbm_2d(
    noise: impl Fn(f64, f64) -> f64,
    x: f64, y: f64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,
    frequency: f64,
) -> f64 {
    let mut amp = 1.0;
    let mut freq = frequency;
    let mut total = 0.0;
    let mut max_amp = 0.0;
    for _ in 0..octaves {
        total += amp * noise(x * freq, y * freq);
        max_amp += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    total / max_amp
}
```

The `/ max_amp` at the end normalizes back to roughly $[-1, 1]$ regardless of how many octaves you stacked. Without that, the magnitude grows with $N$, and downstream code that compares to thresholds breaks every time you add an octave.

## In the engine

FBM shows up in many places — every time the engine wants smooth random structure at multiple scales. A representative use is the temperature map in `src/worldgen/mod.rs`:

```rust
// src/worldgen/mod.rs (Generator::new_internal)
let temperature_map = Fbm::<Simplex>::new(seed.wrapping_add(8) as u32)
    .set_octaves(2)
    .set_frequency(1.0 / 512.0)
    .set_persistence(0.5);
```

Notes:

- The seed is offset by a small constant (`+8`) per noise field. Each FBM in the engine gets a different axis-salted seed so two fields with the same conceptual role don't accidentally produce the same pattern. This is a recurring trick — we'll come back to it in **5.1 Determinism Everywhere**.
- Only **2** octaves. Temperature varies slowly across the world; we don't want gravel-scale wiggles in temperature.
- Frequency $1/512$ means the period of the first octave is $\sim 512$ blocks — about 16 chunks. A player walks for a while between biome bands instead of crossing one every few steps.

Compare to the heightmap's 3D base density, which uses **8** octaves at higher frequencies — terrain has detail at every scale, climate doesn't.

## What you can now do in the code

You can read:

- Every `Fbm::<Simplex>::new(...).set_octaves(N).set_frequency(f).set_persistence(p)` call site in `src/worldgen/`. The parameters tell you the spatial scale and roughness of what's being generated.
- The `noise` crate's FBM implementation — it's a few hundred lines and matches the pseudocode above.

## Next

→ [1.5 Voronoi Diagrams](./1.5-voronoi.md) — the *other* spatial pattern we use everywhere, this time for *cellular* structure (plates, biomes-with-edges, cave system layout).
```

- [ ] **Step 2: Add the static fallback image**

To make the widget fallback work, we need a static image. The simplest path is a screenshot of the widget at its default parameters — but since we can't take screenshots in this plan, generate one with the same parameters using `doc_render` (it doesn't render FBM directly, but we can use the `humidity` stage which is FBM at known params):

```bash
cargo run --release --bin doc_render -- snapshot \
  --seed 42 --center 0,0 --zoom 4 --stage humidity \
  --width 256 --output docs/book/static/img/figures/fbm-default-fallback.png
```

This isn't a *perfect* match for the widget output (the widget shows the math directly; this shows the engine's humidity FBM with its own seed handling) — but it's representative and good enough for a pre-hydration paint.

- [ ] **Step 3: Verify the build**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

Expected: builds. The FBM chapter renders; the widget appears below the heading "Drag the sliders...".

- [ ] **Step 4: Smoke-test the widget locally**

Run:

```bash
cd docs/book && npm run serve & sleep 3 && open http://localhost:3000/oxium/part-1-foundations/1.4-fbm && cd ../..
```

Expected: the page renders, the canvas appears, dragging any slider updates the image in <100 ms.

Press Ctrl+C in the terminal when done.

- [ ] **Step 5: Commit**

```bash
git add docs/book/content/part-1-foundations/1.4-fbm.mdx docs/book/static/img/figures/fbm-default-fallback.png
git commit -m "content(book): write 1.4 FBM with FbmExplorer widget

End-to-end demo of the widget system: chapter introduces FBM, embeds
<FbmExplorer /> inline via <LazyWidget>, with a static-image fallback
produced by doc_render. Math intuition, code, in-the-engine grounding,
links to 1.5 next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Create the gen-images.sh manifest

The book's images live at `docs/book/static/img/generated/` and are produced by running `doc_render snapshot` once per image. We keep the list of all images in a single shell script so re-generating everything is one command, and so CI can diff outputs against checked-in PNGs.

**Files:**
- Create: `docs/book/tools/gen-images.sh`

- [ ] **Step 1: Write the script**

Create `docs/book/tools/gen-images.sh`:

```bash
#!/usr/bin/env bash
# Regenerates every PNG the book uses from the engine.
#
# Edit this file (don't hand-edit individual PNGs). To add a new
# image, add a line below and re-run.
#
# Convention: filenames are <stage>-<seedhex>-<center>-z<zoom>-w<width>.png

set -euo pipefail

OUT_DIR="$(dirname "$0")/../static/img/generated"
SEED=42      # keep in sync with docs/book/src/book.constants.ts BOOK_SEED
mkdir -p "$OUT_DIR"

# Cargo target dir is at the repo root, two levels up.
CARGO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DOC_RENDER="$CARGO_ROOT/target/release/doc_render"

if [[ ! -x "$DOC_RENDER" ]]; then
  echo "Building doc_render..."
  (cd "$CARGO_ROOT" && cargo build --release --bin doc_render)
fi

render() {
  local stage="$1" center="$2" zoom="$3" width="$4"
  local name="${stage}-s${SEED}-${center//,/_}-z${zoom}-w${width}.png"
  echo "rendering $name"
  "$DOC_RENDER" snapshot \
    --seed   "$SEED" \
    --center "$center" \
    --zoom   "$zoom" \
    --stage  "$stage" \
    --width  "$width" \
    --output "$OUT_DIR/$name"
}

# Phase 1 — initial image set. Plans 2–7 add entries as their
# chapters need images.

# Chapter 1.4 FBM — static fallback for the widget (also produced
# by Task 13 step 2; this entry exists so the script is idempotent).
render humidity 0,0 4 256

# Re-generate parity reference JSON (lives in static/, not img/).
"$DOC_RENDER" parity --output "$OUT_DIR/../parity.json"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x docs/book/tools/gen-images.sh
```

- [ ] **Step 3: Run it end-to-end**

Run:

```bash
bash docs/book/tools/gen-images.sh
```

Expected: rebuilds `doc_render` if needed; generates each PNG; regenerates `parity.json`. Output files appear in `docs/book/static/img/generated/` and `docs/book/static/parity.json`.

- [ ] **Step 4: Verify the site still builds**

Run:

```bash
cd docs/book && npm run build && cd ../..
```

- [ ] **Step 5: Commit**

```bash
git add docs/book/tools/gen-images.sh docs/book/static/
git commit -m "tools(book): gen-images.sh manifest

One shell script regenerates every PNG and the parity.json reference.
Plans 2-7 add entries as chapters need images. Re-runnable; output
is byte-deterministic given a fixed engine + seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: GitHub Action — build & deploy to GitHub Pages

**Files:**
- Create: `.github/workflows/docs-build-deploy.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/docs-build-deploy.yml`:

```yaml
name: Build and deploy docs book

on:
  push:
    branches: [main]
    paths:
      - 'docs/book/**'
      - 'src/bin/doc_render/**'
      - 'src/viz_render/**'
      - 'src/worldgen/**'
      - 'src/voxel/**'
      - '.github/workflows/docs-build-deploy.yml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Build doc_render
        run: cargo build --release --bin doc_render

      - name: Regenerate images
        run: bash docs/book/tools/gen-images.sh

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: docs/book/package-lock.json

      - name: Install npm deps
        run: npm ci
        working-directory: docs/book

      - name: Build site
        run: npm run build
        working-directory: docs/book

      - name: Verify parity
        # The /parity page renders pass/fail at runtime. Headless check:
        # serve the built site and grep for "PASS" in the parity HTML.
        run: |
          npx --yes http-server docs/book/build -p 4180 >/dev/null 2>&1 &
          sleep 2
          curl -s http://localhost:4180/oxium/parity > /tmp/parity.html
          if ! grep -q 'data-testid="parity-status">PASS<' /tmp/parity.html; then
            echo "PARITY FAILED"
            cat /tmp/parity.html | sed -n '/data-testid="parity-status"/,/<\/p>/p'
            exit 1
          fi
          echo "PARITY OK"

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/book/build

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docs-build-deploy.yml
git commit -m "ci(book): GitHub Action to build and deploy on push to main

Builds doc_render in release, regenerates images, builds the Docusaurus
site, runs the headless parity check against the built /parity page,
and deploys to GitHub Pages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Configure GitHub Pages in the repo settings**

This step is one-time and happens in the GitHub web UI (not in code):

1. Go to `https://github.com/fdatoo/oxium/settings/pages`.
2. Under "Build and deployment" → "Source", select **GitHub Actions**.
3. Save.

There is no commit for this step — it's a UI change. Note in the next commit message (Task 16) that Pages must be enabled before the action will deploy successfully.

---

## Task 16: GitHub Action — image-drift PR check

**Files:**
- Create: `.github/workflows/docs-image-drift.yml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/docs-image-drift.yml`:

```yaml
name: Docs image drift check

on:
  pull_request:
    paths:
      - 'src/worldgen/**'
      - 'src/voxel/**'
      - 'src/viz_render/**'
      - 'src/bin/doc_render/**'
      - 'docs/book/tools/gen-images.sh'
      - 'docs/book/static/img/generated/**'

permissions:
  contents: read
  pull-requests: write

jobs:
  drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Regenerate images
        run: bash docs/book/tools/gen-images.sh

      - name: Diff against committed
        id: diff
        run: |
          set +e
          CHANGED=$(git status --porcelain docs/book/static/img/generated docs/book/static/parity.json)
          if [[ -n "$CHANGED" ]]; then
            echo "changed=true" >> "$GITHUB_OUTPUT"
            {
              echo "drift<<EOF"
              echo "The following docs images changed in this PR:"
              echo ""
              echo "$CHANGED" | awk '{print "- " $2}'
              echo ""
              echo "Run \`bash docs/book/tools/gen-images.sh\` locally and commit the result if these are expected."
              echo "EOF"
            } >> "$GITHUB_OUTPUT"
          else
            echo "changed=false" >> "$GITHUB_OUTPUT"
          fi

      - name: Comment on PR
        if: steps.diff.outputs.changed == 'true'
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: docs-image-drift
          message: ${{ steps.diff.outputs.drift }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docs-image-drift.yml
git commit -m "ci(book): PR check that comments when docs images would change

Doesn't fail the build — just surfaces the drift so the author notices.
Reruns gen-images.sh; diffs against committed; posts/updates a sticky
comment listing which PNGs moved.

Reminder: GitHub Pages must be enabled (Settings → Pages → Source:
GitHub Actions) before the deploy workflow's first run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: README pointer + final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Check whether README.md exists with content**

Run:

```bash
test -s README.md && head -20 README.md || echo "no README content"
```

If empty / nonexistent, create with a minimal stub. If it has content, add a "Documentation" section near the top.

- [ ] **Step 2: Add the docs section**

Either replace the empty README.md with this, or splice the "Documentation" section into an existing one:

```markdown
# Oxium

A voxel engine in Rust.

## Documentation

The engine's design and internals are documented in **[How Oxium Builds a World](https://fdatoo.github.io/oxium/)** — a deep-dive book published as a Docusaurus site. Phase 1 covers the foundations and worldgen pipeline.

- **Site source:** `docs/book/`
- **Design spec:** `docs/superpowers/specs/2026-05-20-worldgen-docs-design.md`
- **Image generator:** `cargo run --release --bin doc_render -- --help`

## Building the engine

```bash
cargo build --release
cargo run --release --bin oxium
```

## Building the docs site

```bash
cd docs/book
npm install
bash tools/gen-images.sh   # regenerate engine-authored PNGs
npm run start              # local dev server at http://localhost:3000/oxium/
```
```

- [ ] **Step 3: Final whole-system verification**

Run all the smoke tests in sequence:

```bash
# Rust side
cargo build --release --bin doc_render
cargo build --bin worldgen_viz
cargo test --bin worldgen_viz

# Docs side
cd docs/book
npm ci
bash tools/gen-images.sh
npm run build
cd ../..

# Final: does the built site contain everything we expect?
test -f docs/book/build/index.html && echo "intro OK"
test -f docs/book/build/part-1-foundations/1.1-voxel-world/index.html && echo "1.1 OK"
test -f docs/book/build/part-1-foundations/1.4-fbm/index.html && echo "1.4 OK"
test -f docs/book/build/parity/index.html && echo "parity OK"
test -f docs/book/build/img/generated/humidity-s42-0_0-z4-w256.png && echo "image OK"
```

Expected: all five "OK" lines print. Any failure indicates a path or build issue from earlier tasks — go back and fix before committing this final task.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: link the docs book from the README"
```

- [ ] **Step 5: Push and watch the action**

```bash
git push -u origin docs/book-infra
```

Then watch the action at `https://github.com/fdatoo/oxium/actions`. Once `docs-build-deploy` succeeds on `main` (after a merge), the site goes live at `https://fdatoo.github.io/oxium/`.

---

## Self-review — spec coverage

Run through the spec section by section, point at the task that implements it.

- **Site identity (title, audience, voice).** → Task 6 (docusaurus.config.ts), Task 8 (intro page).
- **Three navigation modes (linear, by module, by technique).** → Linear via Task 7 sidebar. By-module + by-technique are appendix pages — created as stubs in Task 8; populated in a later plan once content exists.
- **Phase 1 chapter outline (33 chapters + 4 appendices).** → Task 7 sidebar (structural), Task 8 stubs (files), Tasks 12 + 13 (2 demonstration chapters fully written).
- **Image strategy: 3 types (generated / Mermaid / Excalidraw).** → Generated: Tasks 2–4, 14. Mermaid: enabled in Task 6 (`theme-mermaid`). Excalidraw: no automation needed; authoring convention only — committed source + SVG export per the spec, lives in `docs/book/static/img/diagrams/`.
- **Pinned book seed.** → Task 9 (`book.constants.ts`); Task 14 (`gen-images.sh` SEED variable with comment to keep in sync).
- **CI parity check.** → Task 15 build job has the parity check; Task 16 image drift action.
- **Widget strategy + parity test.** → Tasks 9, 10, 11.
- **Repo layout.** → Task 5 scaffold + matches the file-structure-overview section above.
- **Build + hosting.** → Tasks 5, 15.
- **Phase 1 completion criteria.** → This plan satisfies items 4, 5 (appendices stubbed), 6, 7. Items 1, 2, 3 (full chapters, full widgets, all images) are the *content* deliverable from Plans 2–7.

**Gaps identified during self-review:**

- The spec mentions a `--height` flag for non-square images; the snapshot tool in Task 3 only handles square output. Note left in the code comment; if a chapter needs portrait/landscape, the implementer extends snapshot in that chapter's plan.
- The Excalidraw workflow is convention-only. Decision deferred: there's no enforcement of the "commit both `.excalidraw.json` source and `.svg` export" rule until a chapter actually authors a diagram. Plans 2–7 should include a one-step pre-flight that documents the convention before they begin.

## Self-review — placeholder scan

Searched for the disallowed patterns:

- No "TBD", "TODO", "fill in details", "similar to Task N" — all task steps contain actual content.
- "Add appropriate error handling": none. The Rust binaries either propagate `Result` errors or print them and exit; explicit and minimal.
- Every code block in a step is complete and runnable as shown.

## Self-review — type consistency

- `Generator`, `Stage`, `viz_render::render_pixel` referenced consistently across Tasks 1, 3, 4.
- `BOOK_SEED` typed as `bigint` in TS (Task 9), `u64` in Rust (Task 14 `SEED=42`); the script comment notes they must stay in sync. No drift risk if a single source-of-truth is documented (it is).
- `Stub` component (Task 8) → imported in Task 12 (intro page imports it). Re-import path is `@site/src/components/Stub`.
- `LazyWidget`, `FbmExplorer` (Tasks 9, 10) → imported in Task 13's chapter 1.4 with the same `@site/src/widgets/*` paths.

Plan is internally consistent.
