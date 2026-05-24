//! Face-neighbour view used by meshing and lighting jobs.

use super::DenseChunk;

/// Read-only references to up to six neighbouring chunks in
/// [`crate::mesher::Face`] order.
pub struct Neighbors<'a> {
    pub chunks: [Option<&'a DenseChunk>; 6],
}
