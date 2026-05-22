//! Bit-packed dense arrays. The voxel light arrays (sky and block) each
//! hold 32 768 values in the range 0..=15, so packing them at 4 bits per
//! entry halves storage vs. one byte per entry.
//!
//! Two 4-bit values share each byte: the **even** index occupies the *low*
//! nibble, the **odd** index the *high* nibble. This convention keeps
//! `idx == 0` at the bottom — easier to spot in a hex dump.

use serde::{Deserialize, Serialize};

/// Fixed-length array of 4-bit unsigned values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Packed4Bit {
    bytes: Vec<u8>,
    len: usize,
}

impl Packed4Bit {
    /// Allocate a `Packed4Bit` of length `len`, every entry initialised to 0.
    pub fn zeros(len: usize) -> Self {
        // div_ceil(2) so odd lengths still fit; the high nibble of the last
        // byte is unused but zeroed.
        Self {
            bytes: vec![0; len.div_ceil(2)],
            len,
        }
    }

    /// Number of logical 4-bit entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the array has no entries (len 0).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read the 4-bit value at index `idx`. Returns a `u8` in `0..=15`.
    /// Panics in debug if `idx >= len`; in release the access is
    /// out-of-bounds-checked by `Vec::index`.
    #[inline]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.len);
        let byte = self.bytes[idx >> 1];
        if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 }
    }

    /// Write a 4-bit value. Only the low 4 bits of `value` are honoured; in
    /// debug, supplying a larger value panics so bugs surface early.
    #[inline]
    pub fn set(&mut self, idx: usize, value: u8) {
        debug_assert!(idx < self.len);
        debug_assert!(value <= 0x0F, "Packed4Bit values must fit in 4 bits");
        let i = idx >> 1;
        if idx & 1 == 0 {
            // Even index: clear the low nibble, then OR in the new low nibble.
            self.bytes[i] = (self.bytes[i] & 0xF0) | (value & 0x0F);
        } else {
            // Odd index: clear the high nibble, then OR in the new high nibble.
            self.bytes[i] = (self.bytes[i] & 0x0F) | ((value & 0x0F) << 4);
        }
    }

    /// Fill every entry with the same value (low 4 bits only).
    pub fn fill(&mut self, value: u8) {
        let b = (value & 0x0F) | ((value & 0x0F) << 4);
        self.bytes.iter_mut().for_each(|x| *x = b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_initialized() {
        let p = Packed4Bit::zeros(100);
        assert_eq!(p.len(), 100);
        for i in 0..100 {
            assert_eq!(p.get(i), 0);
        }
    }

    #[test]
    fn round_trip_each_index() {
        let mut p = Packed4Bit::zeros(33);
        for i in 0..33 {
            p.set(i, (i as u8) & 0x0F);
        }
        for i in 0..33 {
            assert_eq!(p.get(i), (i as u8) & 0x0F, "idx {i}");
        }
    }

    #[test]
    fn fill_sets_all() {
        let mut p = Packed4Bit::zeros(50);
        p.fill(0xA);
        for i in 0..50 {
            assert_eq!(p.get(i), 0xA);
        }
    }

    #[test]
    fn neighbor_writes_dont_corrupt() {
        // Writing the odd-index neighbor must not stomp the even-index one
        // sharing the same byte.
        let mut p = Packed4Bit::zeros(10);
        p.set(2, 0x3);
        p.set(3, 0xC);
        assert_eq!(p.get(2), 0x3);
        assert_eq!(p.get(3), 0xC);
    }
}
