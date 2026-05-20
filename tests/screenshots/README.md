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

## Known limitations

- **`baseline_underwater.png` does not currently exercise `underwater_factor > 0`** — the camera spawned above water rather than inside it. Stable as a baseline (same input → same output) but won't catch regressions in the underwater tint code path. Future work: a `--spawn-underwater` helper that locates a deep water column and spawns the camera 4 blocks below the surface.
- The "sunset" baseline at `--time 0.78` reads as dusk/twilight, not golden hour. Adjust `--time` to taste if a brighter sunset is wanted.
