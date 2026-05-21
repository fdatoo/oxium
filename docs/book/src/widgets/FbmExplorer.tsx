import React, { useEffect, useRef, useState } from 'react';
import { makeFbm2D } from '../math/fbm';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 256;

export default function FbmExplorer() {
  const seedBigInt = useBookSeed();
  const seed = Number(seedBigInt & 0xffff_ffffn);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [octaves, setOctaves] = useState(4);
  const [persistence, setPersistence] = useState(0.5);
  const [lacunarity, setLacunarity] = useState(2.0);
  const [frequency, setFrequency] = useState(1 / 64);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const fbm = makeFbm2D({ seed, octaves, persistence, lacunarity, frequency });
    const img = ctx.createImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = fbm(x, y);            // ~ [-1, 1]
        const g = Math.round(((v + 1) * 0.5) * 255);
        const i = (y * SIZE + x) * 4;
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [seed, octaves, persistence, lacunarity, frequency]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start' }}>
      <canvas ref={canvasRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444' }} />
      <div>
        <label>
          Octaves: {octaves}
          <input type="range" min={1} max={8} step={1} value={octaves}
            onChange={(e) => setOctaves(parseInt(e.target.value, 10))} />
        </label>
        <label>
          Persistence: {persistence.toFixed(2)}
          <input type="range" min={0.1} max={0.9} step={0.05} value={persistence}
            onChange={(e) => setPersistence(parseFloat(e.target.value))} />
        </label>
        <label>
          Lacunarity: {lacunarity.toFixed(2)}
          <input type="range" min={1.5} max={3.0} step={0.1} value={lacunarity}
            onChange={(e) => setLacunarity(parseFloat(e.target.value))} />
        </label>
        <label>
          Frequency: {frequency.toFixed(4)}
          <input type="range" min={0.005} max={0.1} step={0.005} value={frequency}
            onChange={(e) => setFrequency(parseFloat(e.target.value))} />
        </label>
      </div>
    </div>
  );
}
