import React, { useEffect, useRef, useState, useMemo } from 'react';

const N = 16;
const PX = 16; // pixels per cell; canvas is N * PX = 256

type Cell = { x: number; y: number; elev: number; acc: number; downhill: number | null };

// D8 offset table: [dx, dy] for each of the 8 neighbours.
const DIRS: [number, number][] = [
  [-1, -1], [0, -1], [1, -1],
  [-1,  0],          [1,  0],
  [-1,  1], [0,  1], [1,  1],
];

function buildGrid(): Cell[] {
  const cells: Cell[] = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      // North-to-south gradient with sinusoidal east-west variation.
      // A couple of bumps create ridges that split flow into multiple paths.
      let elev = (N - 1 - y) * 1.0 + Math.sin(x * 0.7) * 0.8;
      if (x === 4 && y === 5) elev += 2.5;   // ridge on west side
      if (x === 10 && y === 9) elev += 1.8;  // bump on east side
      if (x === 7 && y === 2) elev += 1.2;   // small peak near top
      cells.push({ x, y, elev, acc: 1, downhill: null });
    }
  }
  return cells;
}

function idxOf(x: number, y: number): number | null {
  if (x < 0 || x >= N || y < 0 || y >= N) return null;
  return x + y * N;
}

function computeD8(cells: Cell[]): void {
  for (const c of cells) {
    let bestSlope = 0;
    let best: number | null = null;
    for (const [dx, dy] of DIRS) {
      const nIdx = idxOf(c.x + dx, c.y + dy);
      if (nIdx === null) continue;
      const drop = c.elev - cells[nIdx].elev;
      if (drop <= 0) continue;
      // Diagonal moves are √2 longer → steeper slope wins only when
      // the drop-per-unit-distance is actually greater.
      const slope = drop / Math.hypot(dx, dy);
      if (slope > bestSlope) {
        bestSlope = slope;
        best = nIdx;
      }
    }
    c.downhill = best;
  }
}

// Return cell indices sorted by elevation descending (highest first).
// Cells are processed in this order so a cell always donates after all
// uphill neighbours have already donated to it.
function topoOrder(cells: Cell[]): number[] {
  return cells.map((_, i) => i).sort((a, b) => cells[b].elev - cells[a].elev);
}

// Re-run flow accumulation through `step` cells in topo order.
// Mutates `cells` in place (resets acc to 1 first).
function accumulateThrough(cells: Cell[], order: number[], step: number): void {
  for (const c of cells) c.acc = 1;
  const limit = Math.min(step, order.length);
  for (let s = 0; s < limit; s++) {
    const i = order[s];
    const c = cells[i];
    if (c.downhill !== null) {
      cells[c.downhill].acc += c.acc;
    }
  }
}

// Build a fast lookup: cell index → its rank in the topo order.
function buildRankMap(order: number[]): number[] {
  const rank = new Array<number>(N * N).fill(Infinity as unknown as number);
  for (let r = 0; r < order.length; r++) {
    rank[order[r]] = r;
  }
  return rank;
}

export default function D8Stepper() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [step, setStep] = useState(0);
  const [showDirs, setShowDirs] = useState(false);
  const [autoStep, setAutoStep] = useState(false);

  // Build the grid, compute D8 directions, and derive topo order — all fixed.
  const baseCells = useMemo<Cell[]>(() => {
    const cells = buildGrid();
    computeD8(cells);
    return cells;
  }, []);

  const order = useMemo<number[]>(() => topoOrder(baseCells), [baseCells]);
  const rankMap = useMemo<number[]>(() => buildRankMap(order), [order]);

  // Recompute accumulation for current step, then redraw.
  useEffect(() => {
    // Clone so we don't mutate the base cells.
    const cells = baseCells.map((c) => ({ ...c, acc: 1 }));
    // Transfer downhill references (map doesn't carry object references).
    for (let i = 0; i < baseCells.length; i++) {
      cells[i].downhill = baseCells[i].downhill;
    }
    accumulateThrough(cells, order, step);
    draw(canvasRef.current, cells, order, rankMap, step, showDirs);
  }, [step, showDirs, baseCells, order, rankMap]);

  // Auto-step interval: one cell every 150 ms.
  useEffect(() => {
    if (!autoStep) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= order.length) {
          setAutoStep(false);
          return s;
        }
        return s + 1;
      });
    }, 150);
    return () => clearInterval(id);
  }, [autoStep, order.length]);

  const total = order.length;

  return (
    <div style={{ fontFamily: 'var(--ifm-font-family-base, sans-serif)' }}>
      <canvas
        ref={canvasRef}
        width={N * PX}
        height={N * PX}
        style={{ border: '1px solid #444', display: 'block', imageRendering: 'pixelated' }}
      />
      <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={() => setStep((s) => Math.min(s + 1, total))}
          disabled={step >= total}
          style={{ cursor: step >= total ? 'not-allowed' : 'pointer' }}
        >
          Step
        </button>
        <button
          onClick={() => setAutoStep((a) => !a)}
          disabled={step >= total}
          style={{ cursor: step >= total ? 'not-allowed' : 'pointer' }}
        >
          {autoStep ? 'Pause' : 'Auto'}
        </button>
        <button onClick={() => { setStep(0); setAutoStep(false); }}>
          Reset
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <input
            type="checkbox"
            checked={showDirs}
            onChange={(e) => setShowDirs(e.target.checked)}
          />
          Show D8 directions
        </label>
        <span style={{ marginLeft: 'auto', fontSize: '0.85em', opacity: 0.75 }}>
          {step} / {total} cells
        </span>
      </div>
    </div>
  );
}

function draw(
  canvas: HTMLCanvasElement | null,
  cells: Cell[],
  order: number[],
  rankMap: number[],
  step: number,
  showDirs: boolean,
): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Find max acc among processed cells for log-scale colour mapping.
  let maxAcc = 1;
  for (let i = 0; i < step && i < order.length; i++) {
    const acc = cells[order[i]].acc;
    if (acc > maxAcc) maxAcc = acc;
  }

  const elevs = cells.map((c) => c.elev);
  const minElev = Math.min(...elevs);
  const maxElev = Math.max(...elevs);
  const elevRange = maxElev - minElev || 1;

  // Draw cells.
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    const processed = rankMap[i] < step;

    if (processed) {
      // Log-scaled red ramp: low accumulation → dark red; high → bright yellow-white.
      const t = maxAcc > 1 ? Math.log(c.acc) / Math.log(maxAcc) : 0;
      const tt = Math.max(0, Math.min(1, t));
      // Map: 0 → (80, 10, 10), 1 → (255, 220, 100)
      const r = Math.round(80 + 175 * tt);
      const g = Math.round(10 + 210 * tt);
      const b = Math.round(10 +  90 * tt);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
    } else {
      // Grayscale elevation: darker = lower.
      const t = (c.elev - minElev) / elevRange;
      const v = Math.round(40 + 175 * t);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
    }

    ctx.fillRect(c.x * PX, c.y * PX, PX, PX);
  }

  // Draw downhill arrows.
  // Show arrow for a cell if: showDirs is on (all cells), or the cell
  // has been processed (rank < step).
  ctx.lineWidth = 1;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    if (c.downhill === null) continue;
    const isProcessed = rankMap[i] < step;
    if (!showDirs && !isProcessed) continue;

    const d = cells[c.downhill];
    const x0 = c.x * PX + PX / 2;
    const y0 = c.y * PX + PX / 2;
    const x1 = d.x * PX + PX / 2;
    const y1 = d.y * PX + PX / 2;

    const arrowColor = isProcessed
      ? 'rgba(200, 235, 255, 0.9)'
      : 'rgba(150, 180, 210, 0.5)';
    ctx.strokeStyle = arrowColor;
    ctx.fillStyle = arrowColor;

    // Line from center to center.
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    // Arrowhead at destination end.
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const ahx = x1 - Math.cos(ang) * 5;
    const ahy = y1 - Math.sin(ang) * 5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(ahx + Math.cos(ang + 2.5) * 4, ahy + Math.sin(ang + 2.5) * 4);
    ctx.lineTo(ahx + Math.cos(ang - 2.5) * 4, ahy + Math.sin(ang - 2.5) * 4);
    ctx.closePath();
    ctx.fill();
  }

  // Highlight the cell just processed (the "current" cell).
  if (step > 0 && step <= order.length) {
    const lastIdx = order[step - 1];
    const lc = cells[lastIdx];
    ctx.strokeStyle = 'rgba(255, 255, 100, 0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(lc.x * PX + 1, lc.y * PX + 1, PX - 2, PX - 2);
  }
}
