#!/usr/bin/env python3
"""Pixel-tolerance diff between two screenshots.

The screenshot harness has ~6% of pixels differing by ±1-2 between two
otherwise identical runs (chunk streaming order is non-deterministic
even with the in-Harness quiesce + zeroed shader time). This script
flags a regression only if the diff exceeds noise-floor thresholds.

Usage:
    python3 tests/screenshots/diff.py <baseline.png> <new.png>

Exit 0 = within noise floor (no regression). Exit 1 = regression.
"""

import sys
from PIL import Image, ImageChops

# Noise-floor thresholds, calibrated against 4 captures of identical code.
# Empirical noise from rayon's non-deterministic scheduling + minor physics
# drift: two captures of the same build can differ by 1-8% at the >5 bucket.
# Thresholds are set well above that observed noise floor, so the regression
# signal is real visual change (a missing pass, wrong operator, etc.) rather
# than environmental jitter.
#
# Each row: (max-diff threshold, max allowed pixel fraction).
THRESH = [
    (5,   0.150),   # max-diff > 5    must be < 15% of pixels   (noise floor ~8%)
    (25,  0.015),   # max-diff > 25   must be < 1.5%            (noise floor ~0.3%)
    (100, 0.005),   # max-diff > 100  must be < 0.5%            (noise floor ~0.1%)
    (200, 0.001),   # max-diff > 200  must be < 0.1%            (catastrophic)
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff.py <baseline.png> <new.png>", file=sys.stderr)
        return 2

    a = Image.open(sys.argv[1]).convert("RGB")
    b = Image.open(sys.argv[2]).convert("RGB")
    if a.size != b.size:
        print(f"SIZE MISMATCH: {a.size} vs {b.size}", file=sys.stderr)
        return 1

    diff = ImageChops.difference(a, b)
    data = list(diff.getdata())
    total = len(data)
    hist = [0] * 256
    for p in data:
        hist[max(p)] += 1

    regression = False
    for thresh, frac_limit in THRESH:
        above = sum(hist[thresh + 1:])
        frac = above / total
        marker = "OK"
        if frac > frac_limit:
            marker = "REGRESSION"
            regression = True
        print(
            f"  pixels with max-diff > {thresh:>3}: {above:>7} "
            f"({100 * frac:.3f}%, limit {100 * frac_limit:.3f}%)  [{marker}]"
        )

    if regression:
        print("DIFF: regression detected", file=sys.stderr)
        return 1
    print("DIFF: within noise floor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
