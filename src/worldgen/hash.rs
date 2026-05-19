//! Deterministic 64-bit mixer used throughout worldgen.
//!
//! The same `(seed, coord, salt)` always returns the same value — no
//! `RandomState`, no thread-local RNG. Every roll in the generator
//! (plate seed jitter, tree placement, cave system count, chamber
//! placement, entrance type) routes through this function so that
//! every world is byte-reproducible from its seed.
//!
//! The pattern is the standard xor-shift / golden-ratio multiply you
//! see in shader hash functions. Not cryptographic; comfortably fast
//! and well-distributed.

/// Hash 64-bit `seed` plus an arbitrary number of 32-bit ints.
///
/// Order matters: `mix(s, &[a, b])` differs from `mix(s, &[b, a])`,
/// so the call sites can use the salt position to keep independent
/// rolls uncorrelated.
#[inline]
pub fn mix(seed: u64, words: &[i32]) -> u64 {
    let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for (i, w) in words.iter().enumerate() {
        // Per-position multipliers — large primes with good
        // bit-distribution — keep `mix(s, &[a, b])` distinct from
        // `mix(s, &[b, a])`.
        let mult = [
            0xC2B2_AE3D_27D4_EB4F_u64,
            0x1656_67B1_9E37_79F9,
            0xCC9E_2D51_1B87_3593,
            0x85EB_CA6B_27D4_EB4F,
            0xC2B2_AE3D_85EB_CA6B,
        ][i % 5];
        h ^= (*w as i64 as u64).wrapping_mul(mult);
        h = h.rotate_left(((i as u32) % 5) * 4 + 13);
    }
    // Finaliser — full avalanche.
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    h
}

/// Convenience: returns `mix(...) as u32`.
#[inline]
pub fn mix_u32(seed: u64, words: &[i32]) -> u32 {
    mix(seed, words) as u32
}

/// `mix(...)` reduced to the unit interval `[0, 1)`.
#[inline]
pub fn mix_unit(seed: u64, words: &[i32]) -> f32 {
    // Use the top 24 bits so the result is a multiple of 1/2²⁴ —
    // plenty of precision for any spawn-rate roll.
    ((mix(seed, words) >> 40) as f32) / ((1u64 << 24) as f32)
}

/// `mix(...)` sampled uniformly from `[lo, hi)`.
#[inline]
pub fn mix_range(seed: u64, words: &[i32], lo: f32, hi: f32) -> f32 {
    lo + mix_unit(seed, words) * (hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_is_deterministic() {
        assert_eq!(mix(42, &[1, 2, 3]), mix(42, &[1, 2, 3]));
    }

    #[test]
    fn mix_is_order_sensitive() {
        assert_ne!(mix(42, &[1, 2]), mix(42, &[2, 1]));
    }

    #[test]
    fn mix_unit_in_range() {
        for i in 0..1024 {
            let v = mix_unit(7, &[i, i * 31]);
            assert!((0.0..1.0).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn mix_range_spans_target() {
        let mut min_seen: f32 = 100.0;
        let mut max_seen: f32 = -100.0;
        for i in 0..1024 {
            let v = mix_range(7, &[i], -5.0, 5.0);
            min_seen = min_seen.min(v);
            max_seen = max_seen.max(v);
        }
        assert!(min_seen < -4.0, "min {min_seen} too high");
        assert!(max_seen > 4.0, "max {max_seen} too low");
    }
}
