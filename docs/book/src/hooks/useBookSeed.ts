import { BOOK_SEED } from '../book.constants';

/**
 * Returns the book seed. A hook (not a constant import) so future
 * widgets can override per-instance via a context provider without
 * touching widget call sites.
 */
export function useBookSeed(): bigint {
  return BOOK_SEED;
}
