"""A pass rate is a proportion, and a proportion measured on a small sample
has a wide confidence interval. This topic computes a real Wilson score
interval two ways -- two algebraically different orderings of the same
formula, cross-checked against each other -- and applies the same
regression threshold to two runs with the identical 70% pass rate but very
different sample sizes: n=20 cannot confidently call it a regression;
n=200 can.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Is the drop real? Statistical significance"

Z = 1.96  # 95% confidence
THRESHOLD = 0.80  # the previously-shipped baseline pass rate


def wilson_ci(x: int, n: int, z: float = Z) -> tuple[float, float]:
    """The standard closed form: (phat + z^2/2n +- margin) / (1 + z^2/n)."""
    phat = x / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


def wilson_ci_alt(x: int, n: int, z: float = Z) -> tuple[float, float]:
    """An algebraically equivalent rearrangement (multiply num/denom by n
    before dividing), computed via a different sequence of floating-point
    operations -- an independent recomputation of the same formula."""
    denom = n + z**2
    center = (x + z**2 / 2) / denom
    margin = (z / denom) * math.sqrt(x * (n - x) / n + z**2 / 4)
    return center - margin, center + margin


def main():
    clear(FIGURES)

    draw_card(
        "A pass rate is a proportion. A proportion measured on a small\n"
        "sample has a wide confidence interval -- the same 70% pass rate\n"
        "can mean 'clearly fine' or 'clearly broken' depending on n.\n\n"
        f"Wilson score interval, z={Z} (95% CI), threshold={THRESHOLD:.0%}.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="sample size, variance, and a regression threshold",
        note="Regression rule: flagged only if the ENTIRE interval sits below the threshold.",
    )

    cases = [("small sample", 20, 14), ("large sample", 200, 140)]
    rows = []
    verdicts = {}
    for label, n, x in cases:
        lo, hi = wilson_ci(x, n)
        lo_alt, hi_alt = wilson_ci_alt(x, n)
        assert math.isclose(lo, lo_alt, abs_tol=1e-9), (lo, lo_alt)
        assert math.isclose(hi, hi_alt, abs_tol=1e-9), (hi, hi_alt)
        regression = hi < THRESHOLD
        verdicts[label] = (n, x, lo, hi, regression)
        rows.append(
            {
                "label": f"{label} (n={n})",
                "cells": [f"{x}/{n} = {x / n:.0%}", f"[{lo:.3f}, {hi:.3f}]"],
                "verdict": "fail" if regression else "pass",
            }
        )

    draw_scorecard(
        rows,
        FIGURES / "step-02.png",
        TITLE,
        columns=["pass rate", "95% Wilson CI"],
        note="Same 70% pass rate, two sample sizes. PASS here means 'not a confirmed regression', not 'good'.",
    )

    small = verdicts["small sample"]
    large = verdicts["large sample"]
    draw_card(
        f"n=20:  CI=[{small[2]:.3f}, {small[3]:.3f}] -- includes {THRESHOLD:.0%} -> "
        f"can't confidently call it a regression\n"
        f"n=200: CI=[{large[2]:.3f}, {large[3]:.3f}] -- entirely below {THRESHOLD:.0%} -> "
        f"regression confirmed\n\n"
        "More samples narrow the interval. The same nominal drop is noise\n"
        "at n=20 and signal at n=200 -- report the interval, not just the rate.",
        FIGURES / "step-03.png",
        TITLE,
        tone="bad",
        subtitle="the same 70%, two verdicts",
        note="A regression threshold without a confidence interval is a coin flip at small n.",
    )

    # ---- oracle ----
    n_s, x_s, lo_s, hi_s, reg_s = verdicts["small sample"]
    n_l, x_l, lo_l, hi_l, reg_l = verdicts["large sample"]
    assert (n_s, x_s) == (20, 14) and (n_l, x_l) == (200, 140)
    assert 0.0 <= lo_s <= x_s / n_s <= hi_s <= 1.0
    assert 0.0 <= lo_l <= x_l / n_l <= hi_l <= 1.0
    assert hi_l < hi_s, "more samples must narrow the interval"
    assert reg_s is False, (
        "small-sample CI includes the threshold -- not a confirmed regression"
    )
    assert reg_l is True, (
        "large-sample CI sits entirely below the threshold -- confirmed regression"
    )
    assert 0.45 < lo_s < 0.51 and 0.83 < hi_s < 0.88
    assert 0.60 < lo_l < 0.66 and 0.73 < hi_l < 0.78

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, n=20 CI=[{lo_s:.3f},{hi_s:.3f}] regression={reg_s}, "
        f"n=200 CI=[{lo_l:.3f},{hi_l:.3f}] regression={reg_l}. All checks passed."
    )


if __name__ == "__main__":
    main()
