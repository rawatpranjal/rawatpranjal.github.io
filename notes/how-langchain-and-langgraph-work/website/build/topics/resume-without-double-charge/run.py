"""Resume without double-charging: the deck's sharpest teeth.

Maya's refund gets approved, the process pauses at interrupt() to wait for
that approval, and something has to answer for a resumed run: did the world
move once, or twice? A naive node puts the side effect ABOVE interrupt(),
and LangGraph really does replay that code on resume -- the till really
gets refunded twice for one approval. Two fixes are shown, each proven
against the same real Till: move the effect below the pause, and add an
idempotency key that survives a killed process resuming on a brand-new
compiled graph object.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
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
TITLE = "Resume without double-charging"


class RefundState(TypedDict, total=False):
    amount: float
    approved: bool
    op: str


class IdempotencyGuard:
    """Lives outside the till: refuses a repeated op id even if the node re-runs."""

    def __init__(self):
        self.applied: dict[str, bool] = {}
        self.refused: list[str] = []

    def try_apply(self, op: str) -> bool:
        if op in self.applied:
            self.refused.append(op)
            return False
        self.applied[op] = True
        return True


def build_naive_graph(till: Till):
    """Act 1: the side effect sits ABOVE interrupt() -- a real, documented bug."""

    def naive_refund(state):
        till.refund(state["amount"])  # ABOVE the interrupt
        answer = interrupt({"amount": state["amount"]})
        return {"approved": answer["approved"]}

    g = StateGraph(RefundState)
    g.add_node("refund", naive_refund)
    g.add_edge(START, "refund")
    g.add_edge("refund", END)
    return g.compile(checkpointer=InMemorySaver()), naive_refund


def build_fix_a_graph(till: Till):
    """Act 2: the side effect moves BELOW interrupt() -- it only runs once."""

    def refund_after(state):
        answer = interrupt({"amount": state["amount"]})
        if answer["approved"]:
            till.refund(state["amount"])
        return {"approved": answer["approved"]}

    g = StateGraph(RefundState)
    g.add_node("refund_after", refund_after)
    g.add_edge(START, "refund_after")
    g.add_edge("refund_after", END)
    return g.compile(checkpointer=InMemorySaver())


def build_fix_b_graph(till: Till, guard: IdempotencyGuard, saver: InMemorySaver):
    """Act 3: below interrupt(), plus an idempotency key the guard checks."""

    def refund_guarded(state):
        answer = interrupt({"amount": state["amount"], "op": state["op"]})
        if answer["approved"] and guard.try_apply(state["op"]):
            till.refund(state["amount"])
        return {"approved": answer["approved"]}

    g = StateGraph(RefundState)
    g.add_node("refund_guarded", refund_guarded)
    g.add_edge(START, "refund_guarded")
    g.add_edge("refund_guarded", END)
    return g.compile(checkpointer=saver)


def main():
    clear(FIGURES)

    # ======================================================= ACT 1: THE BUG
    till1 = Till()
    graph1, naive_refund_fn = build_naive_graph(till1)
    cfg1 = {"configurable": {"thread_id": "t-naive"}}

    code_text = textwrap.dedent(inspect.getsource(naive_refund_fn)).rstrip("\n")
    draw_card(
        code_text
        + "\n\n"
        + "LangGraph replays a node from its top on resume,\n"
        + "so the line above interrupt() runs again.",
        FIGURES / "step-01.png",
        TITLE,
        tone="bad",
        subtitle="Act 1: the bug, for real -- a naive refund node",
        note="LangGraph's docs warn that code before an interrupt runs again on resume.",
    )

    result1 = graph1.invoke({"amount": 6.25}, cfg1)
    assert "__interrupt__" in result1
    state1 = graph1.get_state(cfg1)

    positions_1 = {"__start__": (10, 42), "refund": (50, 42), "__end__": (90, 42)}
    draw_graph(
        graph1.get_graph(),
        positions_1,
        FIGURES / "step-02.png",
        TITLE,
        active="refund",
        visited={"refund"},
        taken_edges={("__start__", "refund")},
        state_rows=[
            {"channel": "balance", "value": f"{till1.balance:.2f}", "changed": True},
            {"channel": "ledger", "value": f"{len(till1.ledger)} entry"},
            {"channel": "next", "value": str(state1.next)},
        ],
        note="The node paused at interrupt() -- but the line above it already fired.",
    )

    graph1.invoke(Command(resume={"approved": True}), cfg1)

    draw_scorecard(
        [
            {
                "label": "ledger[0]",
                "cells": [
                    till1.ledger[0][0],
                    f"{till1.ledger[0][1]:.2f}",
                    f"{till1.ledger[0][2]:.2f}",
                ],
                "verdict": "fail",
            },
            {
                "label": "ledger[1]",
                "cells": [
                    till1.ledger[1][0],
                    f"{till1.ledger[1][1]:.2f}",
                    f"{till1.ledger[1][2]:.2f}",
                ],
                "verdict": "fail",
            },
        ],
        FIGURES / "step-03.png",
        TITLE,
        columns=["kind", "amount", "balance after"],
        note=f"One approval, two refunds -- the node re-ran from the top, balance {till1.balance:.2f}.",
    )

    # =============================================================== THE RULE
    draw_card(
        "A replayed node must not repeat a side effect.\n\n"
        "  1. move the effect BELOW interrupt()\n"
        "  2. or guard it behind an idempotency key\n"
        "  3. or wrap it in a transaction\n\n"
        "Act 2 uses (1). Act 3 adds (2), and survives a kill.",
        FIGURES / "step-04.png",
        TITLE,
        subtitle="the rule",
        note="interrupt() does not remember what already ran -- the code around it has to.",
    )

    # ================================================== ACT 2: EFFECT BELOW
    till2 = Till()
    graph2 = build_fix_a_graph(till2)
    cfg2 = {"configurable": {"thread_id": "t-fixa"}}

    graph2.invoke({"amount": 6.25}, cfg2)
    balance_during_pause = till2.balance
    state2 = graph2.get_state(cfg2)

    graph2.invoke(Command(resume={"approved": True}), cfg2)

    positions_2 = {"__start__": (10, 42), "refund_after": (50, 42), "__end__": (90, 42)}
    draw_graph(
        graph2.get_graph(),
        positions_2,
        FIGURES / "step-05.png",
        TITLE,
        visited={"refund_after"},
        taken_edges={("__start__", "refund_after"), ("refund_after", "__end__")},
        state_rows=[
            {
                "channel": "balance",
                "value": f"{till2.balance:.2f}",
                "delta": f"{balance_during_pause:.2f}",
                "reducer": "overwrite",
                "changed": True,
            },
            {"channel": "ledger", "value": f"{len(till2.ledger)} entry"},
            {"channel": "next", "value": str(graph2.get_state(cfg2).next)},
        ],
        note="Effect moved below interrupt(): untouched at the pause, moved exactly once on resume.",
    )

    # ============================================ ACT 3: GUARD + KILL/RESUME
    till3 = Till()
    guard = IdempotencyGuard()
    saver = InMemorySaver()
    old_graph = build_fix_b_graph(till3, guard, saver)
    old_graph_ref = old_graph
    cfg3 = {"configurable": {"thread_id": "t-fixb"}}

    old_graph.invoke({"amount": 6.25, "op": "refund-maya-001"}, cfg3)
    balance_before_kill = till3.balance
    assert old_graph.get_state(cfg3).next == ("refund_guarded",)

    del old_graph  # THROW AWAY the compiled object -- simulate a killed process
    new_graph = build_fix_b_graph(till3, guard, saver)  # a fresh object, same saver

    new_graph.invoke(Command(resume={"approved": True}), cfg3)

    # a deliberate second submission of the same op id, a genuinely fresh thread
    cfg3b = {"configurable": {"thread_id": "t-fixb-retry"}}
    new_graph.invoke({"amount": 6.25, "op": "refund-maya-001"}, cfg3b)
    new_graph.invoke(Command(resume={"approved": True}), cfg3b)

    draw_thread_lanes(
        [
            {
                "label": "t-fixb",
                "state": "done",
                "checkpoints": [
                    {"step": "paused", "hint": "op refund-maya-001"},
                    {"step": "old_graph killed", "hint": "object discarded"},
                    {"step": "resumed", "hint": "new_graph, same saver"},
                ],
                "resume_from": 2,
            },
            {
                "label": "t-fixb-retry",
                "state": "done",
                "checkpoints": [
                    {"step": "same op id", "hint": "refund-maya-001 again"},
                    {"step": "guard refuses", "hint": "already applied"},
                ],
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        note="A fresh graph from the same checkpointer resumed the thread; a duplicate op id was refused.",
    )

    draw_scorecard(
        [
            {"label": "naive (Act 1)", "cells": ["2", "1"], "verdict": "fail"},
            {
                "label": "fix A: effect after interrupt",
                "cells": ["1", "1"],
                "verdict": "pass",
            },
            {
                "label": "fix B: idempotency key + kill/resume",
                "cells": ["1", "1"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-07.png",
        TITLE,
        columns=["refunds fired", "expected"],
        note="Kill it anywhere -- the world moves once. That is what production-ready means.",
    )

    # ---- oracle ----
    assert till1.balance == 187.50, till1.balance
    assert len(till1.ledger) == 2, till1.ledger
    assert till1.ledger[0][0] == "refund" and till1.ledger[1][0] == "refund"
    assert state1.next == ("refund",)

    assert balance_during_pause == 200.00, balance_during_pause
    assert till2.balance == 193.75, till2.balance
    assert len(till2.ledger) == 1, till2.ledger
    assert state2.next == ("refund_after",)

    assert balance_before_kill == 200.00, balance_before_kill
    assert new_graph is not old_graph_ref
    assert till3.balance == 193.75, till3.balance
    assert len(till3.ledger) == 1, till3.ledger
    assert guard.refused == ["refund-maya-001"], guard.refused

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, naive fired {len(till1.ledger)} refunds for one "
        f"approval (balance {till1.balance:.2f}), fix A fired {len(till2.ledger)} "
        f"(untouched at {balance_during_pause:.2f} during the pause, {till2.balance:.2f} "
        f"after), fix B survived a kill onto a fresh compiled graph "
        f"(new_graph is not old_graph: {new_graph is not old_graph_ref}) and refused "
        f"a duplicate op id {guard.refused}, ledger stayed at {len(till3.ledger)} entry. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
