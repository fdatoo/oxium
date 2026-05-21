import React, { useEffect, useRef, useState, useMemo } from 'react';
import { bilerp } from '../math/trilerp';

const N = 9;       // 9 corners per side (CORNER_COUNT)
const CELLS = 8;   // 8 cells between corners (CELL_COUNT)
const SIZE = 256;
const PX_PER_CELL = SIZE / CELLS;

type Preset = 'gradient' | 'singleBump' | 'twoBumps' | 'noisy' | 'cliff';

const PRESET_LABELS: Record<Preset, string> = {
  gradient:   'gradient',
  singleBump: 'single bump',
  twoBumps:   'two bumps',
  noisy:      'noisy',
  cliff:      'cliff',
};

function buildCorners(preset: Preset): number[] {
  const c: number[] = new Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      let v = 0;
      switch (preset) {
        case 'gradient':
          v = (i + j) / (N + N - 2) - 0.5;
          break;
        case 'singleBump':
          v = Math.exp(-((i - 4) ** 2 + (j - 4) ** 2) / 4);
          break;
        case 'twoBumps':
          v = Math.exp(-((i - 2) ** 2 + (j - 2) ** 2) / 3)
            - Math.exp(-((i - 6) ** 2 + (j - 6) ** 2) / 3);
          break;
        case 'noisy':
          v = Math.sin(i * 0.7) * Math.cos(j * 0.5);
          break;
        case 'cliff':
          v = i < N / 2 ? -0.5 : 0.5;
          break;
      }
      c[i + j * N] = v;
    }
  }
  return c;
}

// The smooth exact function underlying each preset, at continuous (fx, fy) in [0, CELLS].
function sampleExact(preset: Preset, fx: number, fy: number): number {
  // Map from cell-space [0, CELLS] to corner-space [0, N-1].
  const ci = (fx / CELLS) * (N - 1);
  const cj = (fy / CELLS) * (N - 1);
  switch (preset) {
    case 'gradient':
      return (ci + cj) / (N + N - 2) - 0.5;
    case 'singleBump':
      return Math.exp(-((ci - 4) ** 2 + (cj - 4) ** 2) / 4);
    case 'twoBumps':
      return Math.exp(-((ci - 2) ** 2 + (cj - 2) ** 2) / 3)
        - Math.exp(-((ci - 6) ** 2 + (cj - 6) ** 2) / 3);
    case 'noisy':
      return Math.sin(ci * 0.7) * Math.cos(cj * 0.5);
    case 'cliff':
      return ci < (N - 1) / 2 ? -0.5 : 0.5;
  }
}

// Evaluate bilerp at a continuous point (fx, fy) in [0, CELLS] using the corner grid.
function sampleBilerp(corners: number[], fx: number, fy: number): number {
  const cx = Math.min(Math.floor(fx), CELLS - 1);
  const cy = Math.min(Math.floor(fy), CELLS - 1);
  const tx = fx - cx;
  const ty = fy - cy;
  const c00 = corners[cx       + cy       * N];
  const c10 = corners[(cx + 1) + cy       * N];
  const c01 = corners[cx       + (cy + 1) * N];
  const c11 = corners[(cx + 1) + (cy + 1) * N];
  return bilerp(c00, c10, c01, c11, tx, ty);
}

// Divergent ramp: blue (negative) → white (zero) → red (positive).
// v is clamped to [-1, 1].
function divergentColor(v: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    // white → red: r=255, g/b decreasing
    const g = Math.round(255 * (1 - t));
    return [255, g, g];
  } else {
    // white → blue: b=255, r/g decreasing
    const g = Math.round(255 * (1 + t));
    return [g, g, 255];
  }
}

// Error heatmap: 0 (black/dark) → 1 (bright yellow).
function errorColor(err: number, maxErr: number): [number, number, number] {
  const t = Math.min(1, err / Math.max(maxErr, 1e-9));
  const r = Math.round(220 * t);
  const g = Math.round(180 * t);
  const b = Math.round(20  * t);
  return [r, g, b];
}

type Mode = 'bilerp' | 'exact' | 'error';

export default function CellGridSlice() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [preset, setPreset] = useState<Preset>('gradient');
  const [mode, setMode] = useState<Mode>('bilerp');
  const [showCorners, setShowCorners] = useState(false);

  const corners = useMemo(() => buildCorners(preset), [preset]);

  // Precompute max error for normalising the heatmap.
  const maxError = useMemo(() => {
    let max = 0;
    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const fx = (px + 0.5) / PX_PER_CELL;
        const fy = (py + 0.5) / PX_PER_CELL;
        const bi = sampleBilerp(corners, fx, fy);
        const ex = sampleExact(preset, fx, fy);
        const err = Math.abs(ex - bi);
        if (err > max) max = err;
      }
    }
    return max;
  }, [corners, preset]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = ctx.createImageData(SIZE, SIZE);

    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const fx = (px + 0.5) / PX_PER_CELL;
        const fy = (py + 0.5) / PX_PER_CELL;

        let rgb: [number, number, number];
        if (mode === 'error') {
          const bi = sampleBilerp(corners, fx, fy);
          const ex = sampleExact(preset, fx, fy);
          rgb = errorColor(Math.abs(ex - bi), maxError);
        } else {
          const v = mode === 'exact'
            ? sampleExact(preset, fx, fy)
            : sampleBilerp(corners, fx, fy);
          rgb = divergentColor(v);
        }

        const idx = (py * SIZE + px) * 4;
        img.data[idx]     = rgb[0];
        img.data[idx + 1] = rgb[1];
        img.data[idx + 2] = rgb[2];
        img.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);

    // Draw cell grid lines.
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 0.5;
    for (let k = 0; k <= CELLS; k++) {
      const p = k * PX_PER_CELL;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(SIZE, p); ctx.stroke();
    }

    // Overlay corner values as small text.
    if (showCorners) {
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          const v = corners[i + j * N];
          const px = i * PX_PER_CELL;
          const py = j * PX_PER_CELL;
          // Filled pill background so text is readable on any field colour.
          const label = v.toFixed(2);
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,0.55)';
          ctx.beginPath();
          ctx.roundRect(px - tw / 2 - 2, py - 7, tw + 4, 14, 3);
          ctx.fill();
          ctx.fillStyle = v >= 0 ? '#88ffcc' : '#ff9090';
          ctx.fillText(label, px, py);
        }
      }
    }
  }, [corners, mode, showCorners, preset, maxError]);

  const labelStyle: React.CSSProperties = {
    display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.9rem',
  };
  const selectStyle: React.CSSProperties = {
    fontSize: '0.9rem', padding: '0.1rem 0.4rem', cursor: 'pointer',
  };

  return (
    <div style={{ fontFamily: 'var(--ifm-font-family-base, sans-serif)', display: 'flex', flexDirection: 'column', gap: '0.6rem', alignItems: 'flex-start' }}>
      {/* Controls row */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <label style={labelStyle}>
          Preset:
          <select
            style={selectStyle}
            value={preset}
            onChange={(e) => setPreset(e.target.value as Preset)}
          >
            {(Object.keys(PRESET_LABELS) as Preset[]).map((p) => (
              <option key={p} value={p}>{PRESET_LABELS[p]}</option>
            ))}
          </select>
        </label>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {(['bilerp', 'exact', 'error'] as Mode[]).map((m) => (
            <label key={m} style={labelStyle}>
              <input
                type="radio"
                name="mode"
                value={m}
                checked={mode === m}
                onChange={() => setMode(m)}
              />
              {m === 'error' ? 'show error' : m === 'exact' ? 'show exact' : 'bilerp'}
            </label>
          ))}
        </div>

        <label style={labelStyle}>
          <input
            type="checkbox"
            checked={showCorners}
            onChange={(e) => setShowCorners(e.target.checked)}
          />
          show corner values
        </label>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        style={{ border: '1px solid #444', imageRendering: 'pixelated', display: 'block' }}
      />

      {/* Caption */}
      <div style={{ fontSize: '0.8rem', color: '#888', maxWidth: SIZE }}>
        {mode === 'error' ? (
          <>Error heatmap: black = 0, bright&nbsp;=&nbsp;max error ({maxError.toFixed(3)}). The {preset === 'cliff' ? <strong>cliff</strong> : 'cliff'} preset shows bilerp smearing a discontinuity into a diagonal gradient.</>
        ) : mode === 'exact' ? (
          <>Exact underlying function — continuous, not limited to cell corners.</>
        ) : (
          <>Bilerp field: 9&times;9 corner samples (dots at grid intersections), interpolated across 8&times;8 cells. Color: blue&nbsp;=&nbsp;negative, white&nbsp;=&nbsp;zero, red&nbsp;=&nbsp;positive.</>
        )}
      </div>
    </div>
  );
}
