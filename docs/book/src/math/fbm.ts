// parity: src/worldgen/* (anywhere Fbm<Simplex> is constructed)
//
// The Rust code uses `noise::Fbm<Simplex>`. This TS port uses the
// `simplex-noise` npm package and reproduces the same octave loop.
// The Rust crate's default frequency is 1.0; persistence default 0.5;
// lacunarity default 2.0 — match them here so the widget defaults
// align with what the engine uses.
//
// PRNG: simplex-noise@4 takes an optional `() => number` argument.
// We use the `alea` npm package (https://npmjs.org/package/alea) as
// the seeded PRNG — the same companion recommended in the simplex-noise
// README.

import { createNoise2D } from 'simplex-noise';
import Alea from 'alea';

export type FbmParams = {
  seed: number;
  octaves: number;
  persistence: number; // amplitude multiplier per octave (default 0.5)
  lacunarity: number;  // frequency multiplier per octave (default 2.0)
  frequency: number;   // base frequency (default 1.0)
};

/**
 * Build a 2D FBM sampler matching the Rust `Fbm<Simplex>` semantics.
 * The returned function maps (x, y) → a scalar in approximately [-1, 1].
 */
export function makeFbm2D(params: FbmParams): (x: number, y: number) => number {
  const prng = Alea(String(params.seed));
  const noise = createNoise2D(prng);
  const { octaves, persistence, lacunarity, frequency } = params;
  return (x, y) => {
    let amp = 1.0;
    let freq = frequency;
    let total = 0.0;
    let maxAmp = 0.0;
    for (let o = 0; o < octaves; o++) {
      total += amp * noise(x * freq, y * freq);
      maxAmp += amp;
      amp *= persistence;
      freq *= lacunarity;
    }
    return total / maxAmp;
  };
}
