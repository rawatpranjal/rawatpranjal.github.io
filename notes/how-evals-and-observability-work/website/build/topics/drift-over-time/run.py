"""Drift: the same eval, the same test set, run against two model
versions. v1's scripted predictions are all correct; v2's script drops two
of them. The pass rate is real arithmetic on real exact-match comparisons,
not a claimed number, and a fixed regression threshold turns the delta
into a decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_card, draw_test_set  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Drift over time: same eval, two model versions"

CASES = [
    ("c1", "one large oat milk latte", "latte"),
    ("c2", "a cappuccino and a croissant to go", "cappuccino"),
    ("c3", "medium cold brew with vanilla", "cold brew"),
    ("c4", "two small espressos, one with an extra shot", "espresso"),
    ("c5", "a small mocha and a cookie", "mocha"),
]
V1_PREDICTIONS = ["latte", "cappuccino", "cold brew", "espresso", "mocha"]  # 5/5
V2_PREDICTIONS = ["latte", "drip coffee", "cold brew", "americano", "mocha"]  # 3/5
DRIFT_THRESHOLD = -0.10  # flag if pass rate drops by more than 10 points


def score(scripted_replies: list[str]) -> tuple[list[str], list[bool], float]:
    model = scripted(*scripted_replies)
    predicted = [
        model.invoke([HumanMessage(content=text)]).content for _, text, _ in CASES
    ]
    hits = [p == e for p, (_, _, e) in zip(predicted, CASES)]
    return predicted, hits, sum(hits) / len(CASES)


def main():
    clear(FIGURES)

    draw_card(
        "Same 5-case test set. Same exact-match metric.\n"
        "Only the model version under test changes.\n\n"
        f"regression threshold: pass-rate delta <= {DRIFT_THRESHOLD:.0%}",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the setup",
        note="Drift detection is an offline eval run twice, diffed.",
    )

    predicted1, hits1, rate1 = score(V1_PREDICTIONS)
    draw_test_set(
        [
            {
                "id": cid,
                "input": text,
                "expected": e,
                "verdict": "pass" if h else "fail",
            }
            for (cid, text, e), h in zip(CASES, hits1)
        ],
        FIGURES / "step-02.png",
        TITLE,
        note=f"model v1: {sum(hits1)}/{len(CASES)} = {rate1:.0%} pass rate.",
    )

    predicted2, hits2, rate2 = score(V2_PREDICTIONS)
    draw_test_set(
        [
            {
                "id": cid,
                "input": text,
                "expected": e,
                "verdict": "pass" if h else "fail",
            }
            for (cid, text, e), h in zip(CASES, hits2)
        ],
        FIGURES / "step-03.png",
        TITLE,
        note=f"model v2: {sum(hits2)}/{len(CASES)} = {rate2:.0%} pass rate. Same cases, same metric.",
    )

    delta = round(rate2 - rate1, 4)
    regression = delta <= DRIFT_THRESHOLD
    draw_card(
        f"v1 pass rate: {rate1:.0%}\n"
        f"v2 pass rate: {rate2:.0%}\n"
        f"delta: {delta:+.0%}\n"
        f"threshold: {DRIFT_THRESHOLD:+.0%}\n\n"
        f"delta <= threshold  ->  {'REGRESSION CAUGHT' if regression else 'no regression'}",
        FIGURES / "step-04.png",
        TITLE,
        tone="bad" if regression else "good",
        subtitle="the drift check",
        note="Fixed test set, real arithmetic, one number to watch across versions.",
    )

    # ---- oracle ----
    assert predicted1 == V1_PREDICTIONS and predicted2 == V2_PREDICTIONS
    assert rate1 == 1.0
    assert rate2 == 0.6
    assert delta == -0.4
    assert regression is True
    assert hits2 == [True, False, True, False, True]

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, v1={rate1:.0%}, v2={rate2:.0%}, delta={delta:+.0%}, "
        f"threshold={DRIFT_THRESHOLD:+.0%}, regression={regression}. All checks passed."
    )


if __name__ == "__main__":
    main()
