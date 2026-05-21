# Oxium

A voxel engine in Rust.

## Documentation

The engine's design and internals are documented in **[How Oxium Builds a World](https://fdatoo.github.io/oxium/)** — a deep-dive book published as a Docusaurus site. Phase 1 covers the foundations and worldgen pipeline.

- **Site source:** `docs/book/`
- **Design spec:** `docs/superpowers/specs/2026-05-20-worldgen-docs-design.md`
- **Image generator:** `cargo run --release --bin doc_render -- --help`

## Building the engine

```bash
cargo build --release
cargo run --release --bin oxium
```

## Building the docs site

```bash
cd docs/book
npm install
bash tools/gen-images.sh   # regenerate engine-authored PNGs
npm run start              # local dev server at http://localhost:3000/oxium/
```
