"""Branching on real state: one compiled graph, two mornings.

Beanline Coffee wires ONE graph: take_order -> check_stock, then a
conditional edge whose router reads state (the real Stock.check result
check_stock just wrote) and returns "in_stock" or "out". "out" detours to
suggest_alt, which proposes an alternative and loops back to take_order --
a real cycle in the compiled graph, not a diagram fiction. Run A's fridge
has oat milk; the router sends it straight through. Run B's fridge does
not; the router detours it, and the second lap actually clears check_stock.
Same compiled graph object, same router function, different weather.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from beanline import Stock  # noqa: E402
from langviz import clear, draw_graph, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Branching on real state"


class OrderState(TypedDict, total=False):
    stock: Stock
    drink: str
    proposed: str
    stock_count: int
    lap: int
    confirmed: bool


def state_rows_for(snapshot: dict, changed: set[str]) -> list[dict]:
    """Real channels off the real snapshot -- "changed" is literally the
    key set of the delta the node just returned, not a guess."""
    rows = []
    for ch in ("drink", "stock_count", "proposed", "lap", "confirmed"):
        if ch not in snapshot:
            continue
        rows.append(
            {
                "channel": ch,
                "value": str(snapshot[ch]),
                "reducer": None,
                "delta": None,
                "changed": ch in changed,
            }
        )
    return rows


def main():
    clear(FIGURES)

    router_returns: list[str] = []

    def take_order_node(state: OrderState) -> dict:
        proposed = state.get("proposed")
        return {
            "drink": proposed if proposed else "oat milk",
            "lap": state.get("lap", 0) + 1,
            "proposed": None,
        }

    def check_stock_node(state: OrderState) -> dict:
        stock = state["stock"]
        return {"stock_count": stock.check(state["drink"])}

    def router(state: OrderState) -> str:
        ret = "in_stock" if state.get("stock_count", 0) > 0 else "out"
        router_returns.append(ret)
        return ret

    def suggest_alt_node(state: OrderState) -> dict:
        alt = "whole milk" if state["drink"] == "oat milk" else "oat milk"
        return {"proposed": alt}

    def confirm_node(state: OrderState) -> dict:
        return {"confirmed": True}

    g = StateGraph(OrderState)
    g.add_node("take_order", take_order_node)
    g.add_node("check_stock", check_stock_node)
    g.add_node("confirm", confirm_node)
    g.add_node("suggest_alt", suggest_alt_node)
    g.add_edge(START, "take_order")
    g.add_edge("take_order", "check_stock")
    g.add_conditional_edges(
        "check_stock", router, {"in_stock": "confirm", "out": "suggest_alt"}
    )
    g.add_edge("suggest_alt", "take_order")
    g.add_edge("confirm", END)
    compiled = g.compile()  # the ONE compile call, reused for both runs below
    drawable = compiled.get_graph()

    positions = {
        "__start__": (5, 40),
        "take_order": (18, 40),
        "check_stock": (33, 40),
        "confirm": (52, 50),
        "suggest_alt": (45, 14),
        "__end__": (60, 50),
    }

    draw_graph(
        drawable,
        positions,
        FIGURES / "step-01.png",
        TITLE,
        note="One compiled graph, both branches wired before either run starts -- the weather decides which one gets taken.",
    )

    # ------------------------------------------------------------ run A
    stock_a = Stock({"oat milk": 2, "almond milk": 0, "whole milk": 10, "croissant": 3})
    used_graph_ids = [id(compiled)]
    updates_a = list(compiled.stream({"stock": stock_a}, stream_mode="updates"))
    seq_a = [next(iter(u)) for u in updates_a]
    deltas_a = [next(iter(u.values())) for u in updates_a]
    router_returns_a = list(router_returns)
    router_returns.clear()

    running: dict = {}
    snapshots_a = []
    for delta in deltas_a:
        running = {**running, **delta}
        snapshots_a.append(dict(running))

    idx_a = seq_a.index("check_stock")
    draw_graph(
        drawable,
        positions,
        FIGURES / "step-02.png",
        TITLE,
        active="check_stock",
        visited=tuple(seq_a[:idx_a]),
        taken_edges=[("take_order", "check_stock")],
        edge_label=("check_stock", "confirm", router_returns_a[0]),
        state_rows=state_rows_for(snapshots_a[idx_a], set(deltas_a[idx_a])),
        note='Run A: check_stock reads the real fridge, oat milk is 2 -- router(state) actually returns "in_stock".',
    )

    draw_graph(
        drawable,
        positions,
        FIGURES / "step-03.png",
        TITLE,
        visited=tuple(seq_a),
        taken_edges=[
            ("__start__", "take_order"),
            ("take_order", "check_stock"),
            ("check_stock", "confirm"),
            ("confirm", "__end__"),
        ],
        note="Run A complete: a straight green path to confirm, suggest_alt never fires.",
    )

    # ------------------------------------------------------------ run B
    stock_b = Stock({"oat milk": 0, "almond milk": 0, "whole milk": 10, "croissant": 3})
    used_graph_ids.append(id(compiled))
    updates_b = list(compiled.stream({"stock": stock_b}, stream_mode="updates"))
    seq_b = [next(iter(u)) for u in updates_b]
    deltas_b = [next(iter(u.values())) for u in updates_b]
    router_returns_b = list(router_returns)

    running = {}
    snapshots_b = []
    for delta in deltas_b:
        running = {**running, **delta}
        snapshots_b.append(dict(running))

    idx_b1 = seq_b.index("check_stock")
    draw_graph(
        drawable,
        positions,
        FIGURES / "step-04.png",
        TITLE,
        active="check_stock",
        visited=tuple(seq_b[:idx_b1]),
        taken_edges=[("__start__", "take_order"), ("take_order", "check_stock")],
        state_rows=state_rows_for(snapshots_b[idx_b1], set(deltas_b[idx_b1])),
        note="Run B begins: check_stock reads the real fridge, and this morning oat milk is 0.",
    )

    idx_b2 = seq_b.index("suggest_alt")
    draw_graph(
        drawable,
        positions,
        FIGURES / "step-05.png",
        TITLE,
        active="suggest_alt",
        visited=tuple(seq_b[:idx_b2]),
        taken_edges=[
            ("__start__", "take_order"),
            ("take_order", "check_stock"),
            ("check_stock", "suggest_alt"),
            ("suggest_alt", "take_order"),
        ],
        edge_label=("check_stock", "suggest_alt", router_returns_b[0]),
        state_rows=state_rows_for(snapshots_b[idx_b2], set(deltas_b[idx_b2])),
        note='router(state) returns "out" -- suggest_alt fires and the back-edge to take_order is real, drawn from the compiled edges.',
    )

    idx_b3 = [i for i, n in enumerate(seq_b) if n == "check_stock"][1]
    draw_graph(
        drawable,
        positions,
        FIGURES / "step-06.png",
        TITLE,
        active="confirm",
        visited=tuple(seq_b[: idx_b3 + 1]),
        taken_edges=[
            ("__start__", "take_order"),
            ("take_order", "check_stock"),
            ("check_stock", "suggest_alt"),
            ("suggest_alt", "take_order"),
            ("take_order", "check_stock"),
            ("check_stock", "confirm"),
        ],
        edge_label=("check_stock", "confirm", router_returns_b[1]),
        state_rows=state_rows_for(snapshots_b[idx_b3], set(deltas_b[idx_b3])),
        note="Second lap: take_order proposes whole milk, check_stock finds 10, confirm is reached.",
    )

    draw_scorecard(
        [
            {
                "label": "Run A (oat milk = 2)",
                "cells": [str(len(seq_a)), str(seq_a.count("take_order")), "no"],
                "verdict": "pass",
            },
            {
                "label": "Run B (oat milk = 0)",
                "cells": [str(len(seq_b)), str(seq_b.count("take_order")), "yes"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-07.png",
        TITLE,
        columns=["steps", "take_order visits", "hit suggest_alt"],
        note="Same graph, different weather -- run A never visits suggest_alt, run B visits take_order twice.",
    )

    # ---- oracle ----
    assert "suggest_alt" not in seq_a
    assert seq_b.count("take_order") == 2 and "suggest_alt" in seq_b
    assert router_returns_a == ["in_stock"]
    assert router_returns_b == ["out", "in_stock"]
    assert stock_b.check("oat milk") == 0, (
        "run B's detour must be caused by a real empty fridge"
    )
    assert len(used_graph_ids) == 2 and len(set(used_graph_ids)) == 1, (
        "both runs must use the SAME compiled graph object"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, router returned {router_returns_a} for run A and "
        f"{router_returns_b} for run B, take_order visited {seq_b.count('take_order')}x "
        f"on the real detour. All checks passed."
    )


if __name__ == "__main__":
    main()
