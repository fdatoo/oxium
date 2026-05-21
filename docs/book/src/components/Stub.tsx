import React from 'react';
import Admonition from '@theme/Admonition';

type Props = { children?: React.ReactNode };

export default function Stub({ children }: Props) {
  return (
    <Admonition type="note" title="Chapter under construction">
      <p>
        This chapter is part of <strong>How Oxium Builds a World — Phase 1</strong> and
        has not been written yet.
      </p>
      {children}
      <p>
        Track progress at{' '}
        <a href="https://github.com/fdatoo/oxium/tree/main/docs/superpowers/specs/2026-05-20-worldgen-docs-design.md">
          docs/superpowers/specs/2026-05-20-worldgen-docs-design.md
        </a>
        .
      </p>
    </Admonition>
  );
}
