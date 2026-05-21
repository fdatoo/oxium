# Render regression baselines

These PNGs are the visual baseline used to detect unintended changes during the
lighting overhaul (PR 1 onward). Each was captured against world seed 42, which
is hardcoded in `src/app.rs` as `TEST_SEED_OVERRIDE`.

## Important: warmup frames

The `--screenshot-and-exit` flag captures after a warmup period during which
chunks stream in. The default 60-frame warmup is **not enough at radius 16** —
chunks require wall-clock time to generate on worker threads. Set
`OXIUM_SCREENSHOT_WARMUP_FRAMES=1800` (≈30 seconds at 60 FPS) to give the
streaming system time to populate the world. The harness also waits for chunk
streaming to quiesce (no growth for 60 consecutive frames) before capturing.

The window is automatically hidden when `--screenshot-and-exit` is set.

## Determinism + diff threshold

Even with quiesce + a zeroed shader time uniform, ~6% of pixels still differ by
±1-2 between two identical-code runs (floating-point physics drift carries into
LOD selection and per-chunk vertex jitter). To diff after a refactor, use:

```
python3 tests/screenshots/diff.py tests/screenshots/baseline_<scene>.png /tmp/new.png
```

The script flags a regression only when differences exceed calibrated noise-floor
thresholds (>2% of pixels differ by >5 channels, or >0.5% by >25, etc.). Exit
code 0 = within noise floor, 1 = regression.

## Regenerate

To recapture a baseline after an intentional visual change, run the command
listed below. To diff after a non-visual change (e.g., a pure refactor),
regenerate to a temp path and run `cmp -l old.png new.png | wc -l` — should
be 0 for a true no-visual-change refactor.

| Baseline | Command |
|----------|---------|
| `baseline_noon_outdoor.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_noon_outdoor.png --look 45,-15 --time 0.5` |
| `baseline_underwater.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_underwater.png --spawn 64,60,-12 --look 0,-10 --time 0.5` |
| `baseline_cave.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_cave.png --spawn 0,30,0 --look 0,-30 --time 0.5` |
| `baseline_sunset.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_sunset.png --look 90,-10 --time 0.78` |
| `baseline_fog_horizon.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_fog_horizon.png --look 0,0 --time 0.5` |
| `baseline_sun_in_frame.png` | `OXIUM_SCREENSHOT_WARMUP_FRAMES=1800 cargo run --release --bin oxium -- --screenshot-and-exit tests/screenshots/baseline_sun_in_frame.png --spawn 0,200,0 --look 0,89 --time 0.25` |

## Known limitations

- **`baseline_underwater.png` does not currently exercise `underwater_factor > 0`** — the camera spawned above water rather than inside it. Stable as a baseline (same input → same output) but won't catch regressions in the underwater tint code path. Future work: a `--spawn-underwater` helper that locates a deep water column and spawns the camera 4 blocks below the surface.
- The "noon" baseline at `--time 0.5` and the "sunset" baseline at `--time 0.78` are misnamed — both correspond to dim twilight sun positions per the `sun_state` formula (`sin(t * TAU) + 0.1` ≈ 0.1 at those times). They were kept for continuity with PR1-PR3's regression suite but do not produce HDR-bright pixels and therefore do not exercise the bloom pass.
- **`baseline_sun_in_frame.png`** (added in PR4) captures `--spawn 0,200,0 --look 0,89 --time 0.25` — overhead sun at true noon, from 200 blocks up looking near-vertical so only sky is in frame (no terrain → no chunk-streaming variance). The wide soft halo around the sun disc is the PR4 visual diff.

## PR 4 notes

All five legacy baselines were refreshed in PR 4 — the previous PNGs had silently drifted from main due to PR 2 (colored block light) and PR 3 (per-pixel light volume sampling, wrap diffuse) altering the lit appearance of surfaces beyond `diff.py`'s noise floor without anyone re-baselining at the time. PR 4 leaves them at the same `--time` / `--spawn` / `--look` parameters but with current content.

Bloom tuning constants:
- `BLOOM_THRESHOLD` and `BLOOM_KNEE` in `assets/shaders/bloom.wgsl` (`1.0` / `0.5`).
- `BLOOM_STRENGTH` in `assets/shaders/composite.wgsl` (`0.06`, matching the spec).
