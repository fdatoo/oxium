import React, { useEffect, useRef, useState } from 'react';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 256;
// 5 cells across → each cell is ~51px wide (256/5 ≈ 51.2)
const CELLS = 5;
const CELL_PX = SIZE / CELLS;

// ---------------------------------------------------------------------------
// Hash helpers (self-contained 32-bit xorshift / murmur finaliser)
// Mirrors the engine's hash::mix pattern: order-sensitive, salt-based.
// ---------------------------------------------------------------------------

function hash32(seed: number, x: number, y: number, salt: number): number {
  let h = (seed | 0) ^ Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca77) ^ Math.imul(salt, 0xc2b2ae3d);
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Maps (seed, x, y, salt) → [0, 1) */
function hashUnit(seed: number, x: number, y: number, salt: number): number {
  return hash32(seed, x, y, salt) / 0x1_0000_0000;
}

// ---------------------------------------------------------------------------
// Voronoi helpers — matching the engine's plates.rs convention exactly.
// Engine: seed_x = (cell_x + jx) * PLATE_CELL_SIZE,  jx ∈ [0, 1)
// At widget scale: seed_x = (cell_x + jx) * CELL_PX
// (No centering at 0.5; jitter is over the full cell width, just as in Rust.)
// ---------------------------------------------------------------------------

function seedPosition(
  seed: number,
  cellX: number,
  cellY: number,
): [number, number] {
  const jx = hashUnit(seed, cellX, cellY, 0); // salt 0 for X
  const jy = hashUnit(seed, cellX, cellY, 1); // salt 1 for Y (Z in engine)
  return [(cellX + jx) * CELL_PX, (cellY + jy) * CELL_PX];
}

interface NearestResult {
  d1: number;
  d2: number;
  id1x: number;
  id1y: number;
}

function nearestTwo(seed: number, px: number, py: number): NearestResult {
  const cx = Math.floor(px / CELL_PX);
  const cy = Math.floor(py / CELL_PX);
  let d1 = Infinity, d2 = Infinity;
  let id1x = 0, id1y = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const [sx, sy] = seedPosition(seed, cx + dx, cy + dy);
      const d = Math.hypot(px - sx, py - sy);
      if (d < d1) {
        d2 = d1;
        d1 = d;
        id1x = cx + dx;
        id1y = cy + dy;
      } else if (d < d2) {
        d2 = d;
      }
    }
  }
  return { d1, d2, id1x, id1y };
}

// ---------------------------------------------------------------------------
// Cell color — deterministic HSL derived from (cellX, cellY)
// ---------------------------------------------------------------------------

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const hf = h / 360;
  const sf = s / 100;
  const lf = l / 100;
  const q = lf < 0.5 ? lf * (1 + sf) : lf + sf - lf * sf;
  const p = 2 * lf - q;
  const hue2rgb = (t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [
    Math.round(hue2rgb(hf + 1 / 3) * 255),
    Math.round(hue2rgb(hf) * 255),
    Math.round(hue2rgb(hf - 1 / 3) * 255),
  ];
}

function cellColor(cellX: number, cellY: number): [number, number, number] {
  const hue = hash32(0xcafe, cellX, cellY, 0) % 360;
  return hslToRgb(hue, 60, 50);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function VoronoiCells() {
  const seedBigInt = useBookSeed();
  const defaultSeed = Number(seedBigInt & 0xffff_ffffn);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [seed, setSeed] = useState(defaultSeed);
  const [showBoundaryT, setShowBoundaryT] = useState(false);
  const [showStripes, setShowStripes] = useState(false);
  const [showSeeds, setShowSeeds] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = ctx.createImageData(SIZE, SIZE);

    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const { d1, d2, id1x, id1y } = nearestTwo(seed, px + 0.5, py + 0.5);

        let r: number, g: number, b: number;

        if (showBoundaryT) {
          // t = (d2 - d1) / (d2 + d1) — 0 on boundary, →1 deep inside
          const denom = d1 + d2;
          const t = denom < 1e-9 ? 1.0 : (d2 - d1) / denom;
          const v = Math.round(t * 255);
          r = v; g = v; b = v;
        } else {
          [r, g, b] = cellColor(id1x, id1y);

          if (showStripes && ((id1x + id1y) & 1) === 0) {
            // Alternating stripe modulation on even-parity cells
            const stripe = ((px + py) & 7) < 4 ? 1.0 : 0.7;
            r = Math.round(r * stripe);
            g = Math.round(g * stripe);
            b = Math.round(b * stripe);
          }
        }

        const i = (py * SIZE + px) * 4;
        img.data[i + 0] = r;
        img.data[i + 1] = g;
        img.data[i + 2] = b;
        img.data[i + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);

    // Draw seed dots
    if (showSeeds) {
      ctx.fillStyle = 'white';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      // Draw seeds for a slightly extended range so edge cells look right
      for (let cy = -1; cy <= CELLS; cy++) {
        for (let cx = -1; cx <= CELLS; cx++) {
          const [sx, sy] = seedPosition(seed, cx, cy);
          if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE) continue;
          ctx.beginPath();
          ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    }
  }, [seed, showBoundaryT, showStripes, showSeeds]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        style={{ border: '1px solid #444', imageRendering: 'pixelated' }}
      />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        <label style={{ display: 'block' }}>
          Seed: {seed}
          <input
            type="range" min={1} max={1000} step={1} value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10))}
            style={{ display: 'block', width: '100%' }}
          />
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
          <input
            type="checkbox" checked={showBoundaryT}
            onChange={(e) => setShowBoundaryT(e.target.checked)}
          />
          Show boundary t (grayscale)
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
          <input
            type="checkbox" checked={showStripes}
            onChange={(e) => setShowStripes(e.target.checked)}
          />
          Show 2nd-nearest stripes
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer' }}>
          <input
            type="checkbox" checked={showSeeds}
            onChange={(e) => setShowSeeds(e.target.checked)}
          />
          Show seed points
        </label>

        <button
          onClick={() => setSeed((s) => (s % 1000) + 1)}
          style={{ marginTop: '0.4rem', padding: '0.3rem 0.8rem', cursor: 'pointer' }}
        >
          Reseed
        </button>

        <p style={{ fontSize: '0.8rem', color: '#888', margin: 0 }}>
          Each cell rolls one jittered seed point. The 3&times;3 neighbourhood
          scan finds the nearest seed for every pixel &mdash; the same scan the
          engine runs in <code>plates.rs</code>.
        </p>
      </div>
    </div>
  );
}
