#!/usr/bin/env bash
# Regenerates every PNG the book uses from the engine.
#
# Edit this file (don't hand-edit individual PNGs). To add a new
# image, add a line below and re-run.
#
# Convention: filenames are <stage>-<seedhex>-<center>-z<zoom>-w<width>.png

set -euo pipefail

OUT_DIR="$(dirname "$0")/../static/img/generated"
SEED=42      # keep in sync with docs/book/src/book.constants.ts BOOK_SEED
mkdir -p "$OUT_DIR"

# Cargo target dir is at the repo root, two levels up.
CARGO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DOC_RENDER="$CARGO_ROOT/target/release/doc_render"

if [[ ! -x "$DOC_RENDER" ]]; then
  echo "Building doc_render..."
  (cd "$CARGO_ROOT" && cargo build --release --bin doc_render)
fi

render() {
  local stage="$1" center="$2" zoom="$3" width="$4"
  local name="${stage}-s${SEED}-${center//,/_}-z${zoom}-w${width}.png"
  echo "rendering $name"
  "$DOC_RENDER" snapshot \
    --seed   "$SEED" \
    --center "$center" \
    --zoom   "$zoom" \
    --stage  "$stage" \
    --width  "$width" \
    --output "$OUT_DIR/$name"
}

# Phase 1 — initial image set. Plans 2–7 add entries as their
# chapters need images.

# Chapter 1.4 FBM — static fallback for the widget (also produced
# by Task 13 step 2; this entry exists so the script is idempotent).
render humidity 0,0 4 256

# Re-generate parity reference JSON (lives in static/, not img/).
STATIC_DIR="$(dirname "$0")/../static"
"$DOC_RENDER" parity --output "$STATIC_DIR/parity.json"
