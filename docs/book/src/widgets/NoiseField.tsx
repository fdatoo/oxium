import React, { useEffect, useRef, useState } from 'react';
import { createNoise2D } from 'simplex-noise';
import Alea from 'alea';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 256;

export default function NoiseField() {
  const seedBigInt = useBookSeed();
  const defaultSeed = Number(seedBigInt & 0xffff_ffffn);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [seed, setSeed] = useState(defaultSeed);
  const [frequency, setFrequency] = useState(1 / 64);
  const [showGrid, setShowGrid] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prng = Alea(String(seed));
    const noise = createNoise2D(prng);
    const img = ctx.createImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = noise(x * frequency, y * frequency); // ~ [-1, 1]
        const g = Math.round(((v + 1) * 0.5) * 255);
        const i = (y * SIZE + x) * 4;
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    if (showGrid) {
      const period = 1 / frequency; // grid spacing in pixels
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
      ctx.lineWidth = 1;
      for (let g = 0; g <= SIZE; g += period) {
        ctx.beginPath();
        ctx.moveTo(g, 0); ctx.lineTo(g, SIZE);
        ctx.moveTo(0, g); ctx.lineTo(SIZE, g);
        ctx.stroke();
      }
    }
  }, [seed, frequency, showGrid]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
      <canvas ref={canvasRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444' }} />
      <div>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>
          Seed: {seed}
          <input type="range" min={1} max={1000} step={1} value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10))}
            style={{ display: 'block', width: '100%' }} />
        </label>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>
          Frequency: {frequency.toFixed(4)}
          <input type="range" min={0.005} max={0.1} step={0.005} value={frequency}
            onChange={(e) => setFrequency(parseFloat(e.target.value))}
            style={{ display: 'block', width: '100%' }} />
        </label>
        <label style={{ display: 'block' }}>
          <input type="checkbox" checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)} />
          {' '}Show gradient grid overlay
        </label>
      </div>
    </div>
  );
}
