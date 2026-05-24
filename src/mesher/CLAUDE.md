# Mesher Module - Agent & Contributor Guide

## Orientation

`mesher/` turns dense chunk snapshots into renderer-ready mesh data. Keep the hot path allocation-aware and deterministic; generation and lighting should have already produced the block and light arrays before this package runs.

## Layout

| Path | Role |
| --- | --- |
| `types.rs` | Public `Face`, `Vertex`, and `ChunkMesh` types |
| `greedy/` | Production greedy mesher |
| `naive.rs` | Reference/debug mesher |
| `lod.rs` | Coarse chunk downsampling and LOD mesh generation |
| `ao.rs` | Ambient-occlusion sampling helpers |

`greedy/` is split by invariant:

| Path | Role |
| --- | --- |
| `sampling.rs` | Chunk and neighbour sampling |
| `mask.rs` | Per-face visibility masks and merge spans |
| `emit.rs` | Quad emission, winding, texture, AO, light packing |
| `water.rs` | Per-block water-top quads |

## Hot-Path Rules

- Avoid heap allocations inside per-voxel and per-slice loops beyond fixed masks.
- Do not introduce `dyn` dispatch in the meshing inner loops.
- Preserve the public imports `crate::mesher::{Face, Vertex, ChunkMesh}` and `crate::mesher::greedy::mesh_greedy`.
- Keep vertex layout changes coordinated with the renderer; `Vertex` is a GPU-facing type.
- Water tops are intentionally per-block quads. Do not greedy-merge them unless the renderer contract changes.

## Visual Rules

- `Face` order is part of neighbour indexing. Update all callers together if it changes.
- Winding must stay outward for every emitted face.
- AO split flipping lives in quad emission; keep tests around diagonal choices.
- Texture and light packing should stay in typed helpers where possible, not scattered bit math.

## Tests

Run focused tests after local changes:

```bash
cargo test mesher
cargo test every_face_winds_outward
```

Run `cargo test --test smoke` after greedy output, vertex layout, AO, or lighting interactions change. Use screenshot regression checks for intentional visual changes.
