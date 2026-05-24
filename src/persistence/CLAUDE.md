# Persistence Module - Agent & Contributor Guide

## Orientation

`persistence/` owns durable world state: the manifest, edited chunk region files, `SaveIndex`, and the background I/O thread. Procedural chunks are still seed-derived; persistence stores chunks that differ from generation.

Never silence I/O or decode errors. Use `RegionError` or explicit channel results, and make lossy fallbacks visible at the call site.

## Layout

| Path | Role |
| --- | --- |
| `manifest.rs` | World seed and manifest metadata |
| `save_index.rs` | Region occupancy cache for edited chunks |
| `thread.rs` | Background load/save worker |
| `region/` | Region file format, coordinates, presence bitmap, compatibility reads |

`region/` is split by responsibility:

| Path | Role |
| --- | --- |
| `coords.rs` | `RegionCoord`, `RegionSlot`, path and slot math |
| `format.rs` | Magic, sector sizing, header layout, `RegionError` |
| `io.rs` | zstd/bincode encode, write, read |
| `presence.rs` | Header-only occupancy reads for `SaveIndex` |
| `compat.rs` | Legacy payload decode |

## Region Rules

- Region files cover 16 x 16 x 16 chunks.
- The header has 4096 little-endian slot entries.
- New writes append compressed blobs at EOF; do not add compaction without format tests.
- Use `RegionCoord` and `RegionSlot` instead of raw tuples or `usize` values.
- `slot_index` exists for compatibility with old call sites. New code should use `region_slot`.
- v2 payloads are the current write format. v1 decode paths are read-only compatibility.

## Thread Rules

- Keep persistence async from the frame loop.
- Do not block render or streaming systems on disk I/O.
- Clone or `Arc` chunk payloads intentionally; avoid hidden mutation crossing the I/O channel.

## Tests

Run focused tests after local changes:

```bash
cargo test persistence
cargo test persistence::region
```

Run `cargo test --test smoke` when region format, `SaveIndex`, or world round-trip behavior changes.
