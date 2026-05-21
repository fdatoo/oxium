import React, { useEffect, useRef, useState, useMemo } from 'react';
import { bilerp } from '../math/trilerp';

const SIZE = 256;

function exact(x: number, y: number): number {
  return Math.cos(x * Math.PI * 2) * Math.cos(y * Math.PI * 2);
}

export default function TrilerpDemo() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [c00, setC00] = useState(0.8);
  const [c10, setC10] = useState(-0.4);
  const [c01, setC01] = useState(-0.4);
  const [c11, setC11] = useState(0.8);
  const [showExact, setShowExact] = useState(false);
  const [showError, setShowError] = useState(false);

  const maxError = useMemo(() => {
    let max = 0;
    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const x = px / SIZE;
        const y = py / SIZE;
        const bi = bilerp(c00, c10, c01, c11, x, y);
        const err = Math.abs(exact(x, y) - bi);
        if (err > max) max = err;
      }
    }
    return max;
  }, [c00, c10, c01, c11]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(SIZE, SIZE);

    for (let py = 0; py < SIZE; py++) {
      for (let px = 0; px < SIZE; px++) {
        const x = px / SIZE;
        const y = py / SIZE;
        const bi = bilerp(c00, c10, c01, c11, x, y);
        const ex = exact(x, y);
        const i = (py * SIZE + px) * 4;

        if (showError) {
          // |exact - bilerp| as red gradient
          const err = Math.abs(ex - bi);
          const t = Math.min(1, err / 2);
          img.data[i]     = Math.round(255 * t);
          img.data[i + 1] = Math.round(30  * (1 - t));
          img.data[i + 2] = Math.round(30  * (1 - t));
          img.data[i + 3] = 255;
        } else {
          const v = showExact ? ex : bi;
          // map [-1, 1] → [0, 255] grayscale
          const g = Math.round(((v + 1) * 0.5) * 255);
          img.data[i]     = g;
          img.data[i + 1] = g;
          img.data[i + 2] = g;
          img.data[i + 3] = 255;
        }
      }
    }

    ctx.putImageData(img, 0, 0);

    // Draw corner labels
    ctx.font = '11px monospace';
    ctx.textBaseline = 'top';
    const labels: Array<[number, number, number]> = [
      [4,         4,         c00],
      [SIZE - 36, 4,         c10],
      [4,         SIZE - 16, c01],
      [SIZE - 36, SIZE - 16, c11],
    ];
    labels.forEach(([lx, ly, val]) => {
      ctx.fillStyle = val >= 0 ? '#00ff88' : '#ff6060';
      ctx.fillText(val.toFixed(1), lx, ly);
    });
  }, [c00, c10, c01, c11, showExact, showError]);

  const sliderStyle: React.CSSProperties = {
    display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
    fontSize: '0.85rem', gap: '0.2rem',
  };
  const rowStyle: React.CSSProperties = {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem 1.5rem',
    marginTop: '0.5rem',
  };

  function cornerSlider(label: string, val: number, set: (v: number) => void) {
    return (
      <div style={sliderStyle}>
        <span>{label}: {val.toFixed(2)}</span>
        <input type="range" min={-1} max={1} step={0.05} value={val}
          onChange={(e) => set(parseFloat(e.target.value))} style={{ width: 120 }} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.5rem' }}>
      <canvas ref={canvasRef} width={SIZE} height={SIZE}
        style={{ border: '1px solid #444', imageRendering: 'pixelated' }} />

      <div style={rowStyle}>
        {cornerSlider('c(0,0)', c00, setC00)}
        {cornerSlider('c(1,0)', c10, setC10)}
        {cornerSlider('c(0,1)', c01, setC01)}
        {cornerSlider('c(1,1)', c11, setC11)}
      </div>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center', fontSize: '0.9rem' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showExact}
            onChange={(e) => { setShowExact(e.target.checked); if (e.target.checked) setShowError(false); }} />
          show exact
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <input type="checkbox" checked={showError}
            onChange={(e) => { setShowError(e.target.checked); if (e.target.checked) setShowExact(false); }} />
          show error
        </label>
        {showError && (
          <span style={{ fontSize: '0.85rem', color: '#aaa' }}>
            max error: {maxError.toFixed(3)}
          </span>
        )}
      </div>
    </div>
  );
}
