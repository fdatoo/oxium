//! Translate voxel volumes into GPU-friendly triangle meshes.
//!
//! M1 ships a *naive culled* mesher (one quad per visible block face); M4
//! replaces the hot path with a *greedy* mesher that fuses coplanar
//! same-appearance faces into rectangles, dramatically cutting vertex count.
//! Ambient occlusion is *baked* per-vertex: the mesher samples three diagonal
//! neighbours and stores a 0..3 darkening factor on each corner.
