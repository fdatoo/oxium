//! Bit-packed boolean arrays: `bitset_get` and `bitset_set`.
//!
//! Used by `FineRegion` and `MacroRegion` to store per-cell boolean
//! flags (`is_river`, `is_lake`, `is_trunk`) without the 8× memory
//! overhead of a `bool` array. Each byte holds 8 flags; index `i`
//! maps to bit `i & 7` of byte `i >> 3`.
//!
//! Both functions are `#[inline]` — they appear on the hot path inside
//! the hydrology fill loops.

/// Read bit `i` from a packed bytestring.
#[inline]
pub fn bitset_get(bytes: &[u8], i: usize) -> bool {
    (bytes[i >> 3] >> (i & 7)) & 1 != 0
}

/// Set bit `i` in a packed bytestring.
#[inline]
pub fn bitset_set(bytes: &mut [u8], i: usize, v: bool) {
    let mask = 1u8 << (i & 7);
    if v {
        bytes[i >> 3] |= mask;
    } else {
        bytes[i >> 3] &= !mask;
    }
}
