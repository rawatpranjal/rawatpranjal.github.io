"""The pivot. Part 1 stayed offline: a frozen test set, a known-correct
answer for every input, a metric that is a pure function of (prediction,
expected). Production traffic has none of that -- no fixed input list, no
known-correct answer, and the eval question changes from "did it get the
known cases right" to "what actually happened on this one live run".
That's observability: not scoring, watching.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The pivot: scored offline, traced online"


def main():
    clear(FIGURES)

    draw_card(
        "fixed test set          known inputs, known-correct answers\n"
        "deterministic metric    a pure function of (prediction, expected)\n"
        "one number              exact match, recall@k, pass rate, CI\n\n"
        "Offline scoring answers one question: on the cases we already know,\n"
        "does the bot still get them right.",
        FIGURES / "step-01.png",
        TITLE,
        tone="good",
        subtitle="offline, scored",
        note="Every topic so far ran against a frozen test set.",
    )

    draw_card(
        "live input               nobody wrote down the expected answer\n"
        "a real run               nested model calls, tool calls, retries\n"
        "spans, tokens, cost      read after the fact, not graded against a key\n\n"
        "Observability answers a different question: what actually happened\n"
        "on this one run, and what did it cost.",
        FIGURES / "step-02.png",
        TITLE,
        tone="neutral",
        subtitle="online, traced",
        note="Once traffic is live, there is no answer key left to score against.",
    )

    draw_card(
        "Same bot, same code. The pivot is the QUESTION, not the tool:\n"
        "'did it get this right' (needs an answer key) becomes\n"
        "'what happened, and what did it cost' (needs an instrument).\n\n"
        "Part 2 builds that instrument: nested spans with token, cost, and\n"
        "latency accounting, and the same eval logic watching for drift.",
        FIGURES / "step-03.png",
        TITLE,
        subtitle="the pivot",
        note="A trace does not replace a test set -- it answers what a test set structurally cannot.",
    )

    # ---- oracle ----
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures: offline-scored, online-traced, the pivot. All checks passed."
    )


if __name__ == "__main__":
    main()
