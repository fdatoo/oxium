# Voxel Module - Agent & Contributor Guide

## Orientation

`voxel/` owns the engine's block domain model: blocks, coordinate spaces, chunk storage, world storage, and raycasting. Keep this package independent of windowing and rendering backends.

Prefer domain types over raw integers:

- `BlockPos` for global voxel positions.
- `ChunkCoord` for 32 x 32 x 32 chunk positions. Negative coordinates are valid.
- `LocalPos` for in-chunk positions.
- `LightLevel` and `PackedRgbLight` for packed lighting values.

## Layout

| Path | Role |
| --- | --- |
| `block.rs` | Block enum and registry metadata |
| `coords.rs` | Coordinate conversions and chunk-local indexing |
| `chunk/` | Dense/paletted chunk storage, lighting metadata, dirty flags, neighbour views |
| `world.rs` | Sparse loaded-chunk map and edit propagation |
| `raycast.rs` | Grid DDA raycasting |
| `packed.rs` | Packed 4-bit storage helpers |

`chunk/` re-exports its public surface from `crate::voxel::chunk::*`. Preserve that import path unless you are doing an explicit API migration.

## Chunk Rules

- `DenseChunk` is the hot, unpacked working form for generation, lighting, meshing, and edits.
- `PalettedChunk` is the compressed in-memory and on-disk form stored in `World` and region files.
- `PalettedChunkV1` is compatibility-only. Do not write new v1 payloads.
- Keep RGB light bit layout inside `chunk::lighting`; callers should use `LightLevel`, `PackedRgbLight`, `pack_rgb`, and `unpack_rgb`.
- Use `FaceMask` for neighbour/light input validity. Avoid ad hoc bit masks.

## World Rules

- `Pending` chunks cannot be read. Wait for a stored chunk.
- User edits should go through `World::set_block` so mesh dirty, light dirty, version, and light-input metadata stay consistent.
- When changing coordinates or local indexing, test negative chunk coordinates explicitly.
- Keep decompression out of tight loops unless the caller already chose a dense working set.

## Tests

Run focused tests after local changes:

```bash
cargo test voxel
cargo test voxel::chunk
```

Run `cargo test --test smoke` when chunk storage, edits, light metadata, or persistence-facing shapes change.
