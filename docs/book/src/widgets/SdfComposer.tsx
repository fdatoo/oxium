import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ellipsoid2D, capsule2D, unionSdf, intersectionSdf } from '../math/sdf';

const W = 400;
const H = 300;
const HANDLE_R = 7; // px radius for hit detection

type EllipsoidShape = {
  kind: 'ellipsoid';
  cx: number; cy: number; // center in canvas pixels
  rx: number; ry: number; // radii in canvas pixels
};

type CapsuleShape = {
  kind: 'capsule';
  ax: number; ay: number; // endpoint A in canvas pixels
  bx: number; by: number; // endpoint B in canvas pixels
  radius: number;          // radius in canvas pixels
};

type Shape = EllipsoidShape | CapsuleShape;

// Each shape exposes 3 draggable handles.
type Handle = { shapeIdx: number; role: string };

const DEFAULT_SHAPES: Shape[] = [
  { kind: 'ellipsoid', cx: 140, cy: 130, rx: 80, ry: 55 },
  { kind: 'ellipsoid', cx: 260, cy: 150, rx: 60, ry: 80 },
  { kind: 'capsule',   ax: 120, ay: 220, bx: 300, by: 200, radius: 45 },
];

// Evaluate a shape's intensity at a canvas pixel (px, py).
// Shapes are in pixel space; we normalise for display only in rendering.
function evalShape(s: Shape, px: number, py: number): number {
  if (s.kind === 'ellipsoid') {
    return ellipsoid2D(px, py, s.cx, s.cy, s.rx, s.ry);
  } else {
    return capsule2D(px, py, s.ax, s.ay, s.bx, s.by, s.radius);
  }
}

// List handles for a shape.
function handlesFor(s: Shape, idx: number): Array<{ x: number; y: number; role: string; shapeIdx: number }> {
  if (s.kind === 'ellipsoid') {
    return [
      { x: s.cx,        y: s.cy,        role: 'center', shapeIdx: idx },
      { x: s.cx + s.rx, y: s.cy,        role: 'rx',     shapeIdx: idx },
      { x: s.cx,        y: s.cy - s.ry, role: 'ry',     shapeIdx: idx },
    ];
  } else {
    const mx = (s.ax + s.bx) / 2;
    const my = (s.ay + s.by) / 2;
    // radius handle: perpendicular to AB, offset by radius
    const abx = s.bx - s.ax, aby = s.by - s.ay;
    const len = Math.hypot(abx, aby) || 1;
    const nx = -aby / len, ny = abx / len; // perpendicular unit
    return [
      { x: s.ax,              y: s.ay,              role: 'a',      shapeIdx: idx },
      { x: s.bx,              y: s.by,              role: 'b',      shapeIdx: idx },
      { x: mx + nx * s.radius, y: my + ny * s.radius, role: 'radius', shapeIdx: idx },
    ];
  }
}

function dist2(ax: number, ay: number, bx: number, by: number): number {
  return (ax - bx) ** 2 + (ay - by) ** 2;
}

// ISO contour lines at these intensity levels.
const ISO_LEVELS = [0.25, 0.5, 0.75];
// Colours per shape for handles.
const SHAPE_COLOURS = ['#f0c040', '#60d8c0', '#e07090'];

export default function SdfComposer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [shapes, setShapes] = useState<Shape[]>(DEFAULT_SHAPES);
  const [mode, setMode] = useState<'union' | 'intersection'>('union');
  const [showIso, setShowIso] = useState(false);
  const dragRef = useRef<Handle | null>(null);

  // ── Render ──────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    // Heatmap: compute per-pixel SDF, render as red gradient.
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    // We'll also store the intensity grid for iso-line extraction.
    const grid = new Float32Array(W * H);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const vals = shapes.map((s) => evalShape(s, x, y));
        const v = mode === 'union' ? unionSdf(vals) : intersectionSdf(vals);
        const vClamped = Math.min(1, Math.max(0, v));
        grid[y * W + x] = vClamped;
        const i4 = (y * W + x) * 4;
        data[i4]     = Math.round(200 * vClamped);  // R
        data[i4 + 1] = Math.round(40  * vClamped);  // G
        data[i4 + 2] = Math.round(60  * vClamped);  // B
        data[i4 + 3] = vClamped > 0 ? Math.round(80 + 175 * vClamped) : 0;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Iso-lines via marching squares (simple edge-crossing).
    if (showIso) {
      ISO_LEVELS.forEach((level, li) => {
        ctx.strokeStyle = ['#ffe080', '#80ffcc', '#ff80c0'][li];
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (let y = 0; y < H - 1; y++) {
          for (let x = 0; x < W - 1; x++) {
            const v00 = grid[ y      * W + x    ];
            const v10 = grid[ y      * W + x + 1];
            const v01 = grid[(y + 1) * W + x    ];
            const v11 = grid[(y + 1) * W + x + 1];
            // Collect crossing edges and draw them.
            // For each cell edge, if it crosses `level`, compute the crossing point.
            type Pt = [number, number];
            const crossings: Pt[] = [];
            const lerp = (a: number, b: number, va: number, vb: number): number =>
              a + (b - a) * (level - va) / (vb - va);
            // top edge
            if ((v00 < level) !== (v10 < level))
              crossings.push([lerp(x, x + 1, v00, v10), y]);
            // bottom edge
            if ((v01 < level) !== (v11 < level))
              crossings.push([lerp(x, x + 1, v01, v11), y + 1]);
            // left edge
            if ((v00 < level) !== (v01 < level))
              crossings.push([x, lerp(y, y + 1, v00, v01)]);
            // right edge
            if ((v10 < level) !== (v11 < level))
              crossings.push([x + 1, lerp(y, y + 1, v10, v11)]);

            if (crossings.length >= 2) {
              ctx.moveTo(crossings[0][0], crossings[0][1]);
              ctx.lineTo(crossings[1][0], crossings[1][1]);
            }
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // Grid guides
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = (i / 4) * W;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      const y = (i / 4) * H;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Handles
    shapes.forEach((s, si) => {
      const handles = handlesFor(s, si);
      const col = SHAPE_COLOURS[si % SHAPE_COLOURS.length];

      // For capsule: draw the spine so the shape is visible.
      if (s.kind === 'capsule') {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(s.ax, s.ay);
        ctx.lineTo(s.bx, s.by);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Handle connector lines for ellipsoid radius handles.
      if (s.kind === 'ellipsoid') {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        ctx.moveTo(handles[0].x, handles[0].y);
        ctx.lineTo(handles[1].x, handles[1].y);
        ctx.moveTo(handles[0].x, handles[0].y);
        ctx.lineTo(handles[2].x, handles[2].y);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      handles.forEach((h, hi) => {
        ctx.fillStyle = hi === 0 ? col : '#ffffff';
        ctx.beginPath();
        ctx.arc(h.x, h.y, HANDLE_R, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    });
  }, [shapes, mode, showIso]);

  // ── Mouse helpers ───────────────────────────────────────────────────
  const getXY = useCallback((e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (W / rect.width),
      (e.clientY - rect.top)  * (H / rect.height),
    ];
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const [mx, my] = getXY(e);
    for (let si = shapes.length - 1; si >= 0; si--) {
      const handles = handlesFor(shapes[si], si);
      for (const h of handles) {
        if (dist2(mx, my, h.x, h.y) <= HANDLE_R * HANDLE_R * 2.5) {
          dragRef.current = { shapeIdx: h.shapeIdx, role: h.role };
          return;
        }
      }
    }
  }, [shapes, getXY]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const [mx, my] = getXY(e);

    setShapes((prev) => {
      const next = prev.map((s) => ({ ...s })) as Shape[];
      const s = next[drag.shapeIdx];

      if (s.kind === 'ellipsoid') {
        if (drag.role === 'center') {
          (s as EllipsoidShape).cx = mx;
          (s as EllipsoidShape).cy = my;
        } else if (drag.role === 'rx') {
          (s as EllipsoidShape).rx = Math.max(5, Math.abs(mx - s.cx));
        } else if (drag.role === 'ry') {
          (s as EllipsoidShape).ry = Math.max(5, Math.abs(my - s.cy));
        }
      } else {
        const cs = s as CapsuleShape;
        if (drag.role === 'a') {
          cs.ax = mx; cs.ay = my;
        } else if (drag.role === 'b') {
          cs.bx = mx; cs.by = my;
        } else if (drag.role === 'radius') {
          const midX = (cs.ax + cs.bx) / 2;
          const midY = (cs.ay + cs.by) / 2;
          cs.radius = Math.max(5, Math.hypot(mx - midX, my - midY));
        }
      }
      return next;
    });
  }, [getXY]);

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
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem' }}>
          Composition:
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as 'union' | 'intersection')}
            style={{ fontSize: '0.9rem' }}
          >
            <option value="union">Union (max)</option>
            <option value="intersection">Intersection (min)</option>
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem' }}>
          <input
            type="checkbox"
            checked={showIso}
            onChange={(e) => setShowIso(e.target.checked)}
          />
          Show iso-lines (0.25 / 0.5 / 0.75)
        </label>
      </div>
      <div style={{ fontSize: '0.8rem', color: '#888' }}>
        Drag coloured handles to reshape. Yellow = shape 1, teal = shape 2, pink = capsule.
        Center handle moves the shape; outer handles adjust radii.
      </div>
    </div>
  );
}
