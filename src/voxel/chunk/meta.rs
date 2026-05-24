//! Chunk lifecycle and scheduler metadata.

use super::ChunkLightInputs;

/// Lifecycle marker for a chunk slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkState {
    #[default]
    Empty,
    Generating,
    Generated,
    Meshing,
    Ready,
}

/// Tracks which expensive recomputes the chunk owes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkDirty {
    pub mesh: bool,
    pub light: bool,
}

/// Six-bit mask keyed by [`crate::mesher::Face`] discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FaceMask(u8);

impl FaceMask {
    pub const NONE: Self = Self(0);

    #[inline]
    pub fn set(&mut self, face: crate::mesher::Face) {
        self.0 |= 1 << face as u8;
    }

    #[inline]
    pub fn contains(self, face: crate::mesher::Face) -> bool {
        self.0 & (1 << face as u8) != 0
    }
}

impl From<[bool; 6]> for FaceMask {
    fn from(value: [bool; 6]) -> Self {
        let mut mask = Self::NONE;
        for (i, changed) in value.iter().enumerate() {
            if *changed {
                mask.0 |= 1 << i;
            }
        }
        mask
    }
}

/// Chunk-owned lighting lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightState {
    #[default]
    Unlit,
    Queued,
    Lighting {
        version: u64,
    },
    Lit {
        version: u64,
    },
    NeedsBorderReconcile,
}

/// Per-chunk bookkeeping. Lives alongside the [`super::PalettedChunk`] in
/// `ChunkSlot::Stored`.
#[derive(Debug, Default)]
pub struct ChunkMeta {
    pub state: ChunkState,
    pub dirty: ChunkDirty,
    /// True if this chunk has been edited by the player since load.
    pub modified: bool,
    /// Monotonic version of chunk data for stale mesh-result rejection.
    pub mesh_version: u64,
    /// Monotonic version of block data relevant to lighting.
    pub light_version: u64,
    pub light_state: LightState,
    pub unresolved_borders: FaceMask,
    /// Per-column world-Y of the lowest sky-source cell.
    pub sky_sources: crate::lighting::ChunkSkyLightSources,
    /// Generator/block manifest consumed by the relight worker.
    pub light_inputs: ChunkLightInputs,
    /// True when a relight commit wrote new voxel light.
    pub light_gpu_dirty: bool,
}
