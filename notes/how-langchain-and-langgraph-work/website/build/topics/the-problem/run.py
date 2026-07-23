"""Gluing model calls by hand: the problem LangChain exists to solve.

Beanline Coffee wants a barista bot with three requirements: greet with the
menu, quote a price, remember the previous turn. Built with no framework,
each requirement demands bespoke glue: an f-string prompt, a regex price
extractor, a hand-spliced history list. The regex really grabs the wrong
price on a second phrasing, and the hand-spliced history really drops the
system message on turn two -- model.calls photographs the actual context
the model was shown, so both bugs are real, not staged claims.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from beanline import menu_board, scripted  # noqa: E402
from langviz import (  # noqa: E402
    clear,
    draw_card,
    draw_messages,
    draw_scorecard,
)

FIGURES = HERE / "figures"
TITLE = "Gluing model calls by hand"

PHRASING_A = "A small latte is $4.50."
PHRASING_B = "For a large it's $5.50, but a small latte is just $4.50."


def naive_price(reply: str) -> float | None:
    """The hand-rolled extractor: grab the first dollar figure in the prose."""
    m = re.search(r"\$?(\d+\.\d{2})", reply)
    return float(m.group(1)) if m else None


def main():
    clear(FIGURES)

    system = SystemMessage(
        content=f"You are the barista at Beanline Coffee. Menu: {menu_board()}"
    )
    model = scripted(
        "Welcome to Beanline! Today we have lattes, cappuccinos, espresso, "
        "mocha and cold brew. What can I get you?",
        PHRASING_B,
        "Sure -- one large coming up!",
    )

    # requirement 1: greet with the menu. glue: an f-string prompt.
    human1 = HumanMessage(content="hi! what's good today?")
    reply1 = model.invoke([system, human1])

    draw_card(
        "Beanline Coffee wants a barista bot.\n\n"
        "  1. greet customers with today's menu\n"
        "  2. quote a price when asked\n"
        "  3. remember the previous turn\n\n"
        "No framework. Just a model endpoint and python.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="three requirements, zero plumbing (so far)",
        note="The shop: a real menu, a model that answers, and us writing glue.",
    )

    draw_messages(
        model.calls[0],
        FIGURES / "step-02.png",
        TITLE,
        new=(0, 1),
        right_text=(
            "system = (\n"
            '  "You are the barista at\n'
            "   Beanline Coffee.\n"
            '   Menu: {menu_board()}"\n'
            ")\n\n"
            "model.invoke([system, human])"
        ),
        right_title="glue #1: prompt by f-string",
        note="Requirement 1 works: the menu is pasted into a system string by hand.",
    )

    # requirement 2: quote a price. glue: a regex over prose.
    human2 = HumanMessage(content="how much is a small latte?")
    reply2 = model.invoke([system, human2])
    got_a = naive_price(PHRASING_A)
    got_b = naive_price(reply2.content)

    draw_scorecard(
        [
            {
                "label": f'"{PHRASING_A}"',
                "cells": [f"extracted {got_a:.2f}", "want 4.50"],
                "verdict": "pass",
            }
        ],
        FIGURES / "step-03.png",
        TITLE,
        columns=["model prose", 'regex r"\\$?(\\d+\\.\\d{2})"', "expected"],
        note="Requirement 2, glue #2: a regex pulls the price out of prose. First phrasing: fine.",
    )

    draw_scorecard(
        [
            {
                "label": f'"{PHRASING_A}"',
                "cells": [f"extracted {got_a:.2f}", "want 4.50"],
                "verdict": "pass",
            },
            {
                "label": f'"{reply2.content}"',
                "cells": [f"extracted {got_b:.2f}", "want 4.50"],
                "verdict": "fail",
            },
        ],
        FIGURES / "step-04.png",
        TITLE,
        columns=["model prose", 'regex r"\\$?(\\d+\\.\\d{2})"', "expected"],
        note="Same question, new phrasing: the regex grabs the large price for a small latte. Really wrong.",
    )

    # requirement 3: remember the previous turn. glue: splice the history by hand.
    human3 = HumanMessage(content="actually, make it a large.")
    hand_spliced = [human2, reply2, human3]  # oops: the system message got lost
    model.invoke(hand_spliced)

    draw_messages(
        model.calls[2],
        FIGURES / "step-05.png",
        TITLE,
        new=(2,),
        right_text=(
            "history = []\n"
            "history.append(human2)\n"
            "history.append(reply2)\n"
            "history.append(human3)\n"
            "model.invoke(history)\n"
            "\n"
            "# the system message\n"
            "# never made the list"
        ),
        right_title="glue #3: memory by hand",
        note="Turn two, as the model ACTUALLY received it: the barista persona and menu are gone.",
    )

    draw_card(
        "glue #1  prompting     an f-string\n"
        "glue #2  parsing       a regex (wrong on phrasing 2)\n"
        "glue #3  memory        a list (lost the system msg)\n"
        "glue #4  tools         not even attempted\n\n"
        "Every app rebuilds this same plumbing.\n"
        "The plumbing, standardized, is LangChain.",
        FIGURES / "step-06.png",
        TITLE,
        tone="bad",
        subtitle="the tally after three requirements",
        note="None of these bugs were staged: both live in the recorded calls above.",
    )

    # ---- oracle ----
    assert naive_price(PHRASING_A) == 4.50
    assert naive_price(reply2.content) == 5.50  # the extractor is really wrong
    assert naive_price(reply2.content) != 4.50
    assert len(model.calls) == 3
    assert any(isinstance(m, SystemMessage) for m in model.calls[0])
    assert not any(isinstance(m, SystemMessage) for m in model.calls[2]), (
        "the hand-spliced turn-2 context must really be missing the system message"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, regex extracted {naive_price(reply2.content)} "
        f"for a 4.50 question, turn-2 context lost the system message. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
