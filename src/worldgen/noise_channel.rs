//! Build `Fbm<Simplex>` instances from a `(first_octave,
//! amplitudes[])` channel descriptor.
//!
//! `first_octave` sets the base wavelength (`2^-first_octave` blocks
//! per cycle at the largest scale). `amplitudes` is a vector of
//! per-octave weights; the count of non-zero entries determines
//! the FBM's octave count, and the geometric mean of successive
//! ratios approximates a persistence value for the noise crate's
//! `Fbm` builder.
//!
//! The bridge is deliberately approximate — exact octave-amplitude
//! semantics would require a custom `NoiseFn` impl. The descriptor
//! is enough to give us the *shape* of noise we want; the visible
//! tuning knobs live in `assets/worldgen/default.ron`.

use crate::worldgen::config::ChannelParams;
use noise::{Fbm, MultiFractal, NoiseFn, Simplex};

/// Build an `Fbm<Simplex>` from a [`ChannelParams`] and a seed salt.
///
/// - `first_octave` → base frequency = `2^first_octave` cycles/block.
/// - `amplitudes` → octave count (nonzero entries).
/// - Persistence is derived from the geometric average of successive
///   nonzero amplitude ratios; defaults to 0.5 if amplitudes are
///   uniform or only one octave is present.
pub fn build_channel(params: &ChannelParams, seed: u64, salt: u32) -> Fbm<Simplex> {
    let octaves = params.octave_count().max(1);
    let freq = params.first_frequency();
    let persistence = derive_persistence(&params.amplitudes);
    Fbm::<Simplex>::new(seed.wrapping_add(salt as u64) as u32)
        .set_octaves(octaves)
        .set_frequency(freq)
        .set_persistence(persistence)
}

/// Approximate persistence from an amplitudes vector: geometric mean
/// of ratios between successive nonzero amplitudes. Falls back to
/// 0.5 (the `Fbm` default) when amplitudes are flat or sparse.
fn derive_persistence(amplitudes: &[f32]) -> f64 {
    let nonzero: Vec<f64> = amplitudes
        .iter()
        .filter(|&&a| a > 0.0)
        .map(|&a| a as f64)
        .collect();
    if nonzero.len() < 2 {
        return 0.5;
    }
    let mut log_sum = 0.0;
    let mut count = 0;
    for w in nonzero.windows(2) {
        log_sum += (w[1] / w[0]).ln();
        count += 1;
    }
    if count == 0 {
        return 0.5;
    }
    let mean_ratio = (log_sum / count as f64).exp();
    mean_ratio.clamp(0.1, 0.9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise::NoiseFn;

    #[test]
    fn build_channel_is_deterministic_in_seed_salt() {
        let p = ChannelParams {
            first_octave: -7,
            amplitudes: vec![1.0],
        };
        let a = build_channel(&p, 42, 1);
        let b = build_channel(&p, 42, 1);
        // Use a non-origin sample point — simplex returns 0 at the
        // exact origin which would let any two channels falsely
        // compare equal.
        let p_pt = [123.4, 56.7, -89.1];
        assert_eq!(a.get(p_pt), b.get(p_pt));
        let c = build_channel(&p, 42, 2);
        assert_ne!(a.get(p_pt), c.get(p_pt));
    }

    #[test]
    fn build_channel_different_first_octave_changes_field() {
        // Different first_octave → different base frequency → noise
        // samples at the same point differ for non-trivial inputs.
        let a = build_channel(
            &ChannelParams { first_octave: -7, amplitudes: vec![1.0] },
            42,
            0,
        );
        let b = build_channel(
            &ChannelParams { first_octave: -3, amplitudes: vec![1.0] },
            42,
            0,
        );
        // Same seed, different frequency: sampling at a non-zero
        // point should produce different values.
        assert_ne!(a.get([100.0, 100.0, 100.0]), b.get([100.0, 100.0, 100.0]));
    }

    #[test]
    fn empty_amplitudes_does_not_panic() {
        let p = ChannelParams {
            first_octave: -7,
            amplitudes: vec![],
        };
        let _ = build_channel(&p, 42, 0);
    }
}
