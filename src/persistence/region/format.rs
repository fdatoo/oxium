//! Shared constants and error types for region files.

/// Magic prefix written before bincode payload to mark the v2 (RGB block
/// light) chunk format. Old saves have no prefix; the read path falls back
/// to deserialising a v1 payload when the prefix is absent.
pub(super) const CHUNK_V2_MAGIC: &[u8; 4] = b"OX2\0";

/// On-disk sector size. Slot offsets and lengths are measured in sectors.
pub(super) const SECTOR: u64 = 4096;

/// Edge length of the region grid, in chunks.
pub(super) const REGION_DIM: i32 = 16;

/// Header size in bytes: 4096 slots x 4-byte entries.
pub(super) const HEADER_BYTES: u64 = (REGION_DIM as u64).pow(3) * 4;

/// Header size in sectors. Always >= 1, so the first chunk blob lives at
/// `HEADER_SECTORS * SECTOR`.
pub(super) const HEADER_SECTORS: u64 = HEADER_BYTES.div_ceil(SECTOR);

/// Errors that can arise while reading/writing a region file.
#[derive(Debug, thiserror::Error)]
pub enum RegionError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("zstd: {0}")]
    Zstd(String),
    #[error("chunk not present in region")]
    NotPresent,
}
