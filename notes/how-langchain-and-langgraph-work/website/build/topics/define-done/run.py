"""Define done, verifiably: the booking-agent parable, at the till.

Maya orders a large oat milk latte. Run 1 is the demo: a scripted bot
chirps "Your order is placed!" but the harness never touches the till or
the stock room -- the model produced dialogue, nothing else moved. A
transcript eval ("placed" in reply.content) passes anyway, because it can
only read the words. A state eval, one plain function that checks the
till, the stock, and the ledger, fails on every count: the world never
moved. Run 2 replays the identical dialogue script, but this time the
harness really executes the actions -- till.charge and stock.take -- and
the same state eval now passes, with a trajectory check confirming the
actions fired in the right order.
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

from beanline import OrderItem, Stock, Till, price, scripted  # noqa: E402
from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Define done, verifiably"

MAYA_ORDER = OrderItem(drink="latte", size="large", extras=["oat milk"])
ITEM = "oat milk"
EXPECTED_TRAJECTORY = ["charge", "take"]


def state_eval(
    till: Till,
    stock: Stock,
    before_balance: float,
    before_count: int,
    item: str,
    total: float,
) -> tuple[bool, bool, bool, bool]:
    """Does the world actually reflect the order. One function, called on
    both runs -- it does not know or care which run it is checking."""
    till_moved = round(till.balance - before_balance, 2) == total
    stock_dropped = stock.check(item) == before_count - 1
    ledger_has_charge = any(
        kind == "charge" and amount == total for kind, amount, _balance in till.ledger
    )
    return (
        till_moved,
        stock_dropped,
        ledger_has_charge,
        (till_moved and stock_dropped and ledger_has_charge),
    )


def main():
    clear(FIGURES)

    order_request = HumanMessage(content="One large oat milk latte, please.")
    total = price(MAYA_ORDER)

    # ---- run 1: the demo -- dialogue only, nothing in the world moves ----
    till1 = Till()
    stock1 = Stock()
    before_balance = till1.balance
    before_count = stock1.check(ITEM)

    model1 = scripted("Your order is placed!")
    reply1 = model1.invoke([order_request])

    draw_card(
        f'bot says: "{reply1.content}"',
        FIGURES / "step-01.png",
        TITLE,
        subtitle="Maya orders a large oat milk latte",
        note="A booking agent succeeds when the reservation exists, not when it says booked.",
    )

    transcript_passed = "placed" in reply1.content
    draw_scorecard(
        [
            {
                "label": "transcript eval",
                "cells": [reply1.content, str(transcript_passed)],
                "verdict": "pass" if transcript_passed else "fail",
            }
        ],
        FIGURES / "step-02.png",
        TITLE,
        columns=["reply.content", "'placed' in reply?"],
        note="The demo eval grades the transcript, so it cannot see the till.",
    )

    till1_moved, stock1_dropped, ledger1_ok, run1_passed = state_eval(
        till1, stock1, before_balance, before_count, ITEM, total
    )
    draw_scorecard(
        [
            {
                "label": "till moved?",
                "cells": [f"{before_balance:.2f}", f"{till1.balance:.2f}"],
                "verdict": "pass" if till1_moved else "fail",
            },
            {
                "label": "stock decremented?",
                "cells": [str(before_count), str(stock1.check(ITEM))],
                "verdict": "pass" if stock1_dropped else "fail",
            },
            {
                "label": "ledger entry?",
                "cells": [str(len(till1.ledger)), "1 expected"],
                "verdict": "pass" if ledger1_ok else "fail",
            },
        ],
        FIGURES / "step-03.png",
        TITLE,
        columns=["before", "after"],
        note="The model said done. The world says no.",
    )

    # ---- run 2: the fix -- same dialogue script, the actions really fire ----
    till2 = Till()
    stock2 = Stock()
    actions: list[str] = []

    model2 = scripted("Your order is placed!")
    reply2 = model2.invoke([order_request])

    till2.charge(total)
    actions.append("charge")
    stock2.take(ITEM)
    actions.append("take")

    draw_card(
        f"till.charge({total:.2f})     {before_balance:.2f} -> {till2.balance:.2f}\n"
        f"stock.take('{ITEM}')   {before_count} -> {stock2.check(ITEM)}",
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="the harness executes the actions the model asked for",
        note="Same dialogue script as run 1, but this time the actions really execute.",
    )

    till2_moved, stock2_dropped, ledger2_ok, run2_passed = state_eval(
        till2, stock2, before_balance, before_count, ITEM, total
    )
    draw_scorecard(
        [
            {
                "label": "till moved?",
                "cells": [f"{before_balance:.2f}", f"{till2.balance:.2f}"],
                "verdict": "pass" if till2_moved else "fail",
            },
            {
                "label": "stock decremented?",
                "cells": [str(before_count), str(stock2.check(ITEM))],
                "verdict": "pass" if stock2_dropped else "fail",
            },
            {
                "label": "ledger entry?",
                "cells": [str(len(till2.ledger)), "1 expected"],
                "verdict": "pass" if ledger2_ok else "fail",
            },
        ],
        FIGURES / "step-05.png",
        TITLE,
        columns=["before", "after"],
        note="Same transcript as run 1. The world moved this time, and the eval can tell.",
    )

    draw_card(
        "outcome checks         did the end state change correctly\n"
        "trajectory checks      did the actions fire in the right order\n"
        "repeated trials        does it work more than once\n"
        "regression thresholds  pass rate held above a bar, run after run\n\n"
        "Success is a property of the world, not the transcript.",
        FIGURES / "step-06.png",
        TITLE,
        subtitle="the eval discipline",
        note="Define done before you build it, or you cannot tell a fix from a chirp.",
    )

    # ---- oracle ----
    assert "placed" in reply1.content
    assert till1.balance == 200.00
    assert stock1.check(ITEM) == 2
    assert till1.ledger == []
    assert run1_passed is False

    assert "placed" in reply2.content
    assert till2.balance == 206.25
    assert len(till2.ledger) == 1
    assert till2.ledger[0] == ("charge", 6.25, 206.25)
    assert stock2.check(ITEM) == 1
    assert total == price(MAYA_ORDER) == 6.25
    assert run2_passed is True
    assert actions == EXPECTED_TRAJECTORY

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, run1 state_eval={run1_passed}, run2 state_eval={run2_passed}, "
        f"till {before_balance:.2f} -> {till2.balance:.2f}, stock {before_count} -> "
        f"{stock2.check(ITEM)}, actions={actions}. All checks passed."
    )


if __name__ == "__main__":
    main()
