"""Approve before you act: authority tiers and a human gate on the till.

Maya's latte came out wrong. Checking the fridge or the menu costs nothing
-- check_stock and get_menu just run, no one asks. But a refund moves real
money OUT of the till, so the refund node calls interrupt() and the graph
STOPS, durably, in the checkpointer: get_state(cfg).next still names the
refund node, and the till still reads $200.00. Sam approves and the till
really drops to $193.75. On a second thread Ben's refund is declined and
the till never moves -- same graph, same checkpointer, two outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from beanline import Till  # noqa: E402
from langviz import (  # noqa: E402
    clear,
    draw_card,
    draw_graph,
    draw_scorecard,
    draw_thread_lanes,
)

FIGURES = HERE / "figures"
TITLE = "Approve before you act"

POSITIONS = {"__start__": (8, 30), "refund": (32, 30), "__end__": (56, 30)}


class RefundState(TypedDict):
    amount: float
    reason: str
    approved: bool


def make_refund_node(till: Till):
    """The refund node. Everything that touches the till sits AFTER
    interrupt() -- the node re-runs from its top on every resume, so a
    till.refund() call placed before interrupt() would fire on resume too
    and refund Maya twice."""

    def refund_node(state: RefundState) -> dict:
        answer = interrupt(
            {"amount": state["amount"], "reason": state["reason"], "needs": "manager"}
        )
        if answer["approved"]:
            till.refund(state["amount"])
        return {"approved": answer["approved"]}

    return refund_node


def main():
    clear(FIGURES)

    till = Till()
    g = StateGraph(RefundState)
    g.add_node("refund", make_refund_node(till))
    g.add_edge(START, "refund")
    g.add_edge("refund", END)
    graph = g.compile(checkpointer=InMemorySaver())  # the ONE compile, both threads
    drawable = graph.get_graph()

    # ---- frame 1: the authority tiers, before any run starts ----
    draw_card(
        "read-only        check_stock, get_menu -- just run, no approval\n"
        "write, logged     till.charge() for a sale -- runs, gets logged\n"
        "irreversible / money   till.refund() -- STOP, call interrupt()\n\n"
        "The higher the blast radius, the fewer things do it alone.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="who is allowed to do what",
        note="Guardrails are not a substitute for authorization.",
    )

    # ---- Run A: Maya's latte, $6.25, thread refund-maya ----
    cfg_maya = {"configurable": {"thread_id": "refund-maya"}}
    result = graph.invoke({"amount": 6.25, "reason": "wrong drink"}, cfg_maya)
    interrupted = result["__interrupt__"][0]

    # frame 2: the run hits refund and hands a real payload to a human
    draw_graph(
        drawable,
        POSITIONS,
        FIGURES / "step-02.png",
        TITLE,
        active="refund",
        state_rows=[
            {
                "channel": "amount",
                "value": f"{interrupted.value['amount']:.2f}",
                "changed": True,
            },
            {
                "channel": "reason",
                "value": interrupted.value["reason"],
                "changed": True,
            },
            {"channel": "needs", "value": interrupted.value["needs"], "changed": True},
        ],
        note="The refund node hits interrupt() and hands the till decision to a human.",
    )

    # frame 3: THE FROZEN FRAME -- a fresh get_state read, not the invoke() return
    fresh = graph.get_state(cfg_maya)
    fresh_again = graph.get_state(cfg_maya)  # a second, independent read
    assert fresh_again.next == ("refund",), (
        "the interrupt must survive a fresh get_state read"
    )
    draw_graph(
        drawable,
        POSITIONS,
        FIGURES / "step-03.png",
        TITLE,
        active="refund",
        state_rows=[
            {"channel": "next", "value": str(fresh.next)},
            {"channel": "till.balance", "value": f"{till.balance:.2f}"},
        ],
        note="Paused is a durable place in the checkpointer, not a spinning thread.",
    )

    # frame 4: Sam's answer, as a card, before it is applied
    draw_card(
        "Command(resume={'approved': True})",
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="Sam, the manager, answers",
        note="The resume value is arbitrary data -- exactly what the node's interrupt() call returns.",
    )

    balance_before_a = till.balance
    graph.invoke(Command(resume={"approved": True}), cfg_maya)
    after_a = graph.get_state(cfg_maya)

    # frame 5: the node resumes and the till REALLY moves
    draw_graph(
        drawable,
        POSITIONS,
        FIGURES / "step-05.png",
        TITLE,
        visited=("refund",),
        taken_edges=(("refund", "__end__"),),
        state_rows=[
            {
                "channel": "till.balance",
                "value": f"{till.balance:.2f}",
                "reducer": "overwrite",
                "delta": f"{balance_before_a:.2f}",
                "changed": True,
            },
            {"channel": "ledger", "value": str(till.ledger[-1]), "changed": True},
        ],
        note="The side effect sat after interrupt() -- only now, approved, does the till move.",
    )

    # ---- Run B: Ben, a fresh thread, $7.00, Sam declines ----
    cfg_ben = {"configurable": {"thread_id": "refund-ben"}}
    graph.invoke({"amount": 7.00, "reason": "wrong drink"}, cfg_ben)
    graph.invoke(Command(resume={"approved": False}), cfg_ben)
    after_b = graph.get_state(cfg_ben)

    # frame 6: two threads, one checkpointer, two outcomes
    draw_thread_lanes(
        [
            {
                "label": "refund-maya",
                "state": "done",
                "checkpoints": [
                    {"step": "refund", "hint": "interrupt"},
                    {"step": "resume", "hint": "approved"},
                ],
                "resume_from": None,
            },
            {
                "label": "refund-ben",
                "state": "done",
                "checkpoints": [
                    {"step": "refund", "hint": "interrupt"},
                    {"step": "resume", "hint": "declined"},
                ],
                "resume_from": None,
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        note="A second thread, the same checkpointer: Sam declines and the world does not move.",
    )

    # frame 7: the two ledger lines, side by side
    draw_scorecard(
        [
            {
                "label": "refund-maya  approved",
                "cells": ["6.25", "200.00", "193.75"],
                "verdict": "pass",
            },
            {
                "label": "refund-ben  declined",
                "cells": ["7.00", "193.75", "193.75"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-07.png",
        TITLE,
        columns=["amount", "till before", "till after"],
        note="A confused agent can only do bounded damage when the till waits for a key.",
    )

    # ---- oracle ----
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["amount"] == 6.25
    assert fresh.next == ("refund",), "a fresh get_state read must still see the pause"
    assert balance_before_a == 200.00, "the till must be untouched while paused"
    assert till.balance == 193.75
    assert after_a.next == ()
    assert till.ledger == [("refund", 6.25, 193.75)]
    assert till.balance == 193.75, "run B's decline must leave the till untouched"
    assert len(till.ledger) == 1, "a declined refund must not append a second entry"
    assert after_b.next == ()
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, run A paused at next={fresh.next} with till.balance="
        f"{balance_before_a:.2f}, approved -> {till.balance:.2f}, run B declined -> "
        f"till.balance stayed {till.balance:.2f}, ledger={till.ledger}. "
        "All checks passed."
    )


if __name__ == "__main__":
    main()
