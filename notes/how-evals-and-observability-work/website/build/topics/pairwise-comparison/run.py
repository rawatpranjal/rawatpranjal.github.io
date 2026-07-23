"""Pairwise comparison: instead of scoring one answer against a rubric,
show a judge two candidate answers -- A and B -- for the same request, and
ask which is better. A scripted judge makes this exactly as deterministic
as any other metric: four comparisons, four scripted preferences, one
aggregate.
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
from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Pairwise comparison: A vs B"

COMPARISONS = [
    (
        "one large oat milk latte",
        "That's a large latte with oat milk, $6.25.",
        "Latte, large, oat milk added. Total comes to $6.25.",
    ),
    (
        "a cappuccino and a croissant to go",
        "Cappuccino plus a croissant, to go -- $8.00.",
        "Sure! Cappuccino and croissant, on its way!",
    ),
    (
        "medium cold brew with vanilla",
        "Cold brew, medium, vanilla -- $5.25.",
        "One vanilla cold brew, medium size. $5.25 total.",
    ),
    (
        "a small mocha and a cookie",
        "Great choice! Enjoy your mocha and cookie!",
        "Small mocha plus a cookie: $7.75.",
    ),
]
# scripted preference per comparison -- "A" 3 times, "B" once.
PREFERENCES = ["A", "A", "B", "A"]


def main():
    clear(FIGURES)

    draw_card(
        "Show the judge both candidates for the same request, ask which is\n"
        "better, and let it pick a side -- no absolute score required.\n\n"
        "Pairwise comparison is what most human-preference evals actually run:\n"
        "it is easier to say 'A is better' than to assign a 1-10 score.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the pattern",
        note="4 comparisons, one Beanline order each, scripted judge picks A or B.",
    )

    judge = scripted(*PREFERENCES)
    picks = []
    rows = []
    for (req, a, b), expected_pick in zip(COMPARISONS, PREFERENCES):
        shown = f"request: {req}\nA: {a}\nB: {b}\nWhich answer is better, A or B?"
        reply = judge.invoke([HumanMessage(content=shown)])
        picks.append(reply.content)
        rows.append(
            {
                "label": req,
                "cells": [a[:28], b[:28], reply.content],
            }
        )

    draw_scorecard(
        rows,
        FIGURES / "step-02.png",
        TITLE,
        columns=["answer A", "answer B", "chosen"],
        note="Every request gets both candidates, the specific price, the exact total -- neither is wrong, but one is clearer.",
    )

    a_count = picks.count("A")
    b_count = picks.count("B")
    draw_card(
        f"A preferred: {a_count}/{len(picks)}\n"
        f"B preferred: {b_count}/{len(picks)}\n\n"
        f"A preferred in {a_count}/{len(picks)} = {a_count / len(picks):.0%} of comparisons.",
        FIGURES / "step-03.png",
        TITLE,
        tone="good",
        subtitle="the aggregate",
        note="Pairwise preference aggregates to a win rate, the same way exact match aggregates to a pass rate.",
    )

    # ---- oracle ----
    assert len(judge.calls) == 4
    assert picks == PREFERENCES
    assert a_count == 3 and b_count == 1
    assert a_count / len(picks) == 0.75

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, picks={picks}, A preferred {a_count}/{len(picks)} "
        f"({a_count / len(picks):.0%}). All checks passed."
    )


if __name__ == "__main__":
    main()
