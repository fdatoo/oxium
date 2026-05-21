import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  book: [
    'intro',
    {
      type: 'category',
      label: 'Part I — Foundations',
      collapsed: false,
      items: [
        'part-1-foundations/1.1-voxel-world',
        'part-1-foundations/1.2-determinism',
        'part-1-foundations/1.3-coherent-noise',
        'part-1-foundations/1.4-fbm',
        'part-1-foundations/1.5-voronoi',
        'part-1-foundations/1.6-domain-warping',
        'part-1-foundations/1.7-splines',
        'part-1-foundations/1.8-sdfs',
        'part-1-foundations/1.9-trilerp',
      ],
    },
    {
      type: 'category',
      label: 'Part II — Pipeline Overview',
      items: ['part-2-overview/2.1-big-picture'],
    },
    {
      type: 'category',
      label: 'Part III — Per-Region Build',
      items: [
        'part-3-region-build/3.1-plates',
        'part-3-region-build/3.2-climate',
        'part-3-region-build/3.3-heightmap',
        'part-3-region-build/3.4-hydrology',
        'part-3-region-build/3.5-rivers-lakes',
        'part-3-region-build/3.6-cave-systems',
      ],
    },
    {
      type: 'category',
      label: 'Part IV — Per-Chunk Fill',
      items: [
        'part-4-chunk-fill/4.1-density-graph',
        'part-4-chunk-fill/4.2-cell-evaluator',
        'part-4-chunk-fill/4.3-composing-caves',
        'part-4-chunk-fill/4.4-noise-carvers',
        'part-4-chunk-fill/4.5-procedural-carvers',
        'part-4-chunk-fill/4.6-surface-rules',
        'part-4-chunk-fill/4.7-aquifers',
        'part-4-chunk-fill/4.8-fluid-settle',
        'part-4-chunk-fill/4.9-trees',
      ],
    },
    {
      type: 'category',
      label: 'Part V — Engineering Scaffolding',
      items: [
        'part-5-engineering/5.1-determinism',
        'part-5-engineering/5.2-region-cache',
        'part-5-engineering/5.3-config-hot-reload',
        'part-5-engineering/5.4-visualizer',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      collapsed: true,
      items: [
        'appendices/a-module-index',
        'appendices/b-technique-index',
        'appendices/c-constants-catalog',
        'appendices/d-glossary',
      ],
    },
  ],
};

export default sidebars;
