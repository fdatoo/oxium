import React, { useEffect, useRef, useState } from 'react';
import { makeFbm2D } from '../math/fbm';
import { useBookSeed } from '../hooks/useBookSeed';

const SIZE = 200;

export default function WarpToggle() {
  const seedBigInt = useBookSeed();
  const seed = Number(seedBigInt & 0xffff_ffffn);
  const plainRef = useRef<HTMLCanvasElement | null>(null);
  const warpedRef = useRef<HTMLCanvasElement | null>(null);
  const [warpAmp, setWarpAmp] = useState(40);
  const [warpFreq, setWarpFreq] = useState(0.02);

  useEffect(() => {
    const fbm = makeFbm2D({ seed, octaves: 4, persistence: 0.5, lacunarity: 2.0, frequency: 1 / 64 });
    const warpX = makeFbm2D({ seed: seed + 1, octaves: 2, persistence: 0.5, lacunarity: 2.0, frequency: warpFreq });
    const warpY = makeFbm2D({ seed: seed + 2, octaves: 2, persistence: 0.5, lacunarity: 2.0, frequency: warpFreq });

    drawFbm(plainRef.current, (x, y) => fbm(x, y));
    drawFbm(warpedRef.current, (x, y) => {
      const dx = warpAmp * warpX(x, y);
      const dy = warpAmp * warpY(x, y);
      return fbm(x + dx, y + dy);
    });
  }, [seed, warpAmp, warpFreq]);

  return (
    <div>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <div>
          <div style={{ textAlign: 'center', fontSize: '0.85em', marginBottom: '0.25rem' }}>FBM (no warp)</div>
          <canvas ref={plainRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444', display: 'block' }} />
        </div>
        <div>
          <div style={{ textAlign: 'center', fontSize: '0.85em', marginBottom: '0.25rem' }}>FBM + domain warp</div>
          <canvas ref={warpedRef} width={SIZE} height={SIZE} style={{ border: '1px solid #444', display: 'block' }} />
        </div>
      </div>
      <div style={{ maxWidth: 420 }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>
          Warp amplitude: {warpAmp}px
          <input
            type="range" min={0} max={80} step={2} value={warpAmp}
            onChange={(e) => setWarpAmp(parseInt(e.target.value, 10))}
            style={{ display: 'block', width: '100%' }}
          />
        </label>
        <label style={{ display: 'block' }}>
          Warp frequency: {warpFreq.toFixed(3)}
          <input
            type="range" min={0.005} max={0.05} step={0.005} value={warpFreq}
            onChange={(e) => setWarpFreq(parseFloat(e.target.value))}
            style={{ display: 'block', width: '100%' }}
          />
        </label>
      </div>
    </div>
  );
}

function drawFbm(canvas: HTMLCanvasElement | null, sample: (x: number, y: number) => number) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const img = ctx.createImageData(SIZE, SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = sample(x, y);
      const g = Math.round(((v + 1) * 0.5) * 255);
      const i = (y * SIZE + x) * 4;
      img.data[i + 0] = g;
      img.data[i + 1] = g;
      img.data[i + 2] = g;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
