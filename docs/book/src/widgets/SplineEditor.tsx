import React, { useEffect, useRef, useState, useCallback } from 'react';
import { evaluateSpline, Knot } from '../math/spline';

const W = 400;
const H = 300;
const KNOT_R = 7;          // px radius for hit detection on knots
const HANDLE_LEN = 32;     // px length of slope-tangent handle arm
const HANDLE_R = 6;        // px radius for hit detection on handle endpoints

type DragTarget = { idx: number; kind: 'knot' | 'slope' } | null;

const DEFAULT_KNOTS: Knot[] = [
  { loc: 0.0,  val: 0.1, slope: 0.5  },
  { loc: 0.33, val: 0.5, slope: 0.0  },
  { loc: 0.66, val: 0.5, slope: 0.0  },
  { loc: 1.0,  val: 0.9, slope: 0.5  },
];

function toCanvas(loc: number, val: number): [number, number] {
  return [loc * W, (1 - val) * H];
}

function fromCanvas(cx: number, cy: number): [number, number] {
  return [cx / W, 1 - cy / H];
}

// Given slope in (loc, val) space, compute the canvas-space handle endpoint.
function handleEndpoint(loc: number, val: number, slope: number): [number, number] {
  const [kx, ky] = toCanvas(loc, val);
  // In canvas space: moving right by HANDLE_LEN pixels.
  // val-per-loc slope → canvas: dy/dx = -slope * H/W (y axis is flipped, scaled).
  const dx = HANDLE_LEN;
  const dy = -slope * (H / W) * HANDLE_LEN;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = dx / len * HANDLE_LEN;
  const ny = dy / len * HANDLE_LEN;
  return [kx + nx, ky + ny];
}

function dist2(ax: number, ay: number, bx: number, by: number): number {
  return (ax - bx) ** 2 + (ay - by) ** 2;
}

export default function SplineEditor() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [knots, setKnots] = useState<Knot[]>(DEFAULT_KNOTS);
  const [showLinear, setShowLinear] = useState(false);
  const dragRef = useRef<DragTarget>(null);

  // Draw the curve whenever knots or toggle change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = (i / 4) * W;
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, H);
      ctx.stroke();
      const y = (i / 4) * H;
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Linear interpolation overlay
    if (showLinear && knots.length >= 2) {
      ctx.strokeStyle = '#e07040';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      knots.forEach((k, i) => {
        const [cx, cy] = toCanvas(k.loc, k.val);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Spline curve
    if (knots.length >= 2) {
      ctx.strokeStyle = '#7ec8e3';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const STEPS = 400;
      const minLoc = knots[0].loc;
      const maxLoc = knots[knots.length - 1].loc;
      // Also draw a bit of extrapolation outside the range
      const extLo = minLoc - 0.05;
      const extHi = maxLoc + 0.05;
      for (let i = 0; i <= STEPS; i++) {
        const loc = extLo + (extHi - extLo) * (i / STEPS);
        const val = evaluateSpline(knots, loc);
        const [cx, cy] = toCanvas(loc, val);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }

    // Slope handles
    knots.forEach((k) => {
      const [kx, ky] = toCanvas(k.loc, k.val);
      const [hx, hy] = handleEndpoint(k.loc, k.val, k.slope);
      // Handle line
      ctx.strokeStyle = '#a0c878';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(kx, ky);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      // Handle endpoint dot
      ctx.fillStyle = '#a0c878';
      ctx.beginPath();
      ctx.arc(hx, hy, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Knot dots
    knots.forEach((k) => {
      const [cx, cy] = toCanvas(k.loc, k.val);
      ctx.fillStyle = '#f0c040';
      ctx.beginPath();
      ctx.arc(cx, cy, KNOT_R, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }, [knots, showLinear]);

  const getCanvasXY = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY] as [number, number];
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const [mx, my] = getCanvasXY(e);

    // Check slope handles first (smaller target, prefer them when overlapping).
    for (let i = 0; i < knots.length; i++) {
      const k = knots[i];
      const [hx, hy] = handleEndpoint(k.loc, k.val, k.slope);
      if (dist2(mx, my, hx, hy) < HANDLE_R * HANDLE_R * 2.5) {
        dragRef.current = { idx: i, kind: 'slope' };
        return;
      }
    }

    // Check knot centers.
    for (let i = 0; i < knots.length; i++) {
      const k = knots[i];
      const [kx, ky] = toCanvas(k.loc, k.val);
      if (dist2(mx, my, kx, ky) < KNOT_R * KNOT_R * 2.5) {
        dragRef.current = { idx: i, kind: 'knot' };
        return;
      }
    }
  }, [knots, getCanvasXY]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const [mx, my] = getCanvasXY(e);
    const [newLoc, newVal] = fromCanvas(mx, my);

    setKnots((prev) => {
      const next = prev.map((k) => ({ ...k }));
      const k = next[drag.idx];

      if (drag.kind === 'knot') {
        // Clamp loc between neighbours to keep sorted order.
        const minLoc = drag.idx === 0 ? 0 : next[drag.idx - 1].loc + 0.01;
        const maxLoc = drag.idx === next.length - 1 ? 1 : next[drag.idx + 1].loc - 0.01;
        k.loc = Math.min(maxLoc, Math.max(minLoc, newLoc));
        k.val = Math.min(1, Math.max(0, newVal));
      } else {
        // Compute slope from handle vector in canvas space.
        const [kx, ky] = toCanvas(k.loc, k.val);
        const dx = mx - kx;
        const dy = my - ky;
        if (Math.abs(dx) > 1) {
          // slope = dval/dloc; in canvas: dval = -dy/H, dloc = dx/W
          k.slope = (-dy / H) / (dx / W);
        }
      }
      return next;
    });
  }, [getCanvasXY]);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{ border: '1px solid #444', cursor: 'crosshair', maxWidth: '100%' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem' }}>
        <input
          type="checkbox"
          checked={showLinear}
          onChange={(e) => setShowLinear(e.target.checked)}
        />
        Show linear interpolation
      </label>
      <div style={{ fontSize: '0.8rem', color: '#888' }}>
        Drag yellow knot dots to move; drag green handles to change slope.
      </div>
    </div>
  );
}
