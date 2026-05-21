import React, { Suspense } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';

type Props = {
  /** Static-image fallback shown during SSR + before client hydration. */
  fallbackSrc?: string;
  fallbackAlt?: string;
  children: React.ReactNode;
};

/**
 * Wraps a widget in BrowserOnly + Suspense so it doesn't run during
 * static-site build (where there's no canvas) and the page can paint
 * a fallback image before the JS bundle is parsed.
 */
export default function LazyWidget({ fallbackSrc, fallbackAlt, children }: Props) {
  const fallback = fallbackSrc ? (
    <img src={fallbackSrc} alt={fallbackAlt ?? 'Widget loading'} />
  ) : (
    <div style={{ minHeight: 200 }}>Loading interactive widget…</div>
  );
  return (
    <BrowserOnly fallback={fallback}>
      {() => <Suspense fallback={fallback}>{children}</Suspense>}
    </BrowserOnly>
  );
}
