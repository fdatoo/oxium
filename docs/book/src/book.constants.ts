/**
 * Constants the book is parameterized by. The Rust binary
 * `doc_render` consumes the same values via CLI flags — when you
 * update BOOK_SEED here, also update docs/book/tools/gen-images.sh.
 */

/** The seed every chapter's images use unless explicitly overridden. */
export const BOOK_SEED = 42n;

/** Default world-space center for top-down maps (the "home" of the book world). */
export const BOOK_CENTER: { wx: number; wz: number } = { wx: 0, wz: 0 };
