import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'How Oxium Builds a World',
  tagline: 'A deep dive into the Oxium voxel engine',
  favicon: 'img/favicon.ico',

  url: 'https://fdatoo.github.io', // update if a custom domain is set later
  baseUrl: '/oxium/',

  organizationName: 'fdatoo',
  projectName: 'oxium',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/fdatoo/oxium/tree/main/docs/book/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    navbar: {
      title: 'How Oxium Builds a World',
      logo: { alt: 'Oxium', src: 'img/logo.svg' },
      items: [
        { to: '/', label: 'Read', position: 'left' },
        { href: 'https://github.com/fdatoo/oxium', label: 'GitHub', position: 'right' },
      ],
    },
    colorMode: { defaultMode: 'dark', respectPrefersColorScheme: true },
    docs: { sidebar: { hideable: true, autoCollapseCategories: false } },
  } satisfies Preset.ThemeConfig,
};

export default config;
