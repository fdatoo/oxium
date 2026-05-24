//! Legacy region payload decoding.

use crate::voxel::chunk::PalettedChunk;

use super::format::{CHUNK_V2_MAGIC, RegionError};

/// Decode a zstd-inflated chunk payload.
///
/// v2 payloads start with [`CHUNK_V2_MAGIC`] and contain the current
/// `PalettedChunk` shape. v1 payloads had no prefix and carried a single block
/// light channel; converting them mirrors that brightness into RGB.
pub(super) fn decode_chunk_payload(blob_bin: &[u8]) -> Result<PalettedChunk, RegionError> {
    if blob_bin.starts_with(CHUNK_V2_MAGIC) {
        return bincode::deserialize(&blob_bin[CHUNK_V2_MAGIC.len()..])
            .map_err(|e| RegionError::Decode(e.to_string()));
    }

    let v1: crate::voxel::chunk::PalettedChunkV1 =
        bincode::deserialize(blob_bin).map_err(|e| RegionError::Decode(e.to_string()))?;
    Ok(v1.into())
}
