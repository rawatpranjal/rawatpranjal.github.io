"""Graphs are data: the compiled StateGraph is an object, not a diagram.

Beanline's happy path is four stations -- greet, take_order, price_order,
confirm -- each a plain function state -> partial dict. The graph figure is
drawn straight FROM compiled.get_graph(), so it cannot drift from the code.
Stepping it with graph.stream(..., stream_mode="updates") turns each real
super-step into one frame. The point: price_order touches NO model at all,
it is plain python arithmetic over beanline.price -- only greet and
take_order consult the scripted model. Models interpret; ordinary code
decides.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from beanline import OrderItem, price, scripted  # noqa: E402
from langviz import clear, draw_graph  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Graphs are data"
SCHEMA_KEYS = {"messages", "item", "total"}

# node boxes stay left of the state panel (which owns x 66..98)
POSITIONS = {
    "__start__": (4, 30),
    "greet": (14, 30),
    "take_order": (27, 30),
    "price_order": (41, 30),
    "confirm": (55, 30),
    "__end__": (63, 30),
}
EDGE_ORDER = [
    ("__start__", "greet"),
    ("greet", "take_order"),
    ("take_order", "price_order"),
    ("price_order", "confirm"),
    ("confirm", "__end__"),
]


class OrderState(TypedDict):
    messages: Annotated[list[str], operator.add]
    item: dict | None
    total: float


def state_rows(state: dict, delta: dict, prior: dict) -> list[dict]:
    """Photograph the three channels: 'add' shows what got appended, plain
    python 'overwrite' shows what got replaced (delta = the discarded old
    value, per langviz's convention)."""
    changed_msg = "messages" in delta
    changed_item = "item" in delta
    changed_total = "total" in delta
    return [
        {
            "channel": "messages",
            "reducer": "add" if changed_msg else None,
            "delta": " / ".join(delta["messages"]) if changed_msg else None,
            "value": " / ".join(state["messages"]),
            "changed": changed_msg,
        },
        {
            "channel": "item",
            "reducer": "overwrite" if changed_item else None,
            "delta": str(prior["item"]) if changed_item else None,
            "value": str(state["item"]),
            "changed": changed_item,
        },
        {
            "channel": "total",
            "reducer": "overwrite" if changed_total else None,
            "delta": f"{prior['total']:.2f}" if changed_total else None,
            "value": f"{state['total']:.2f}",
            "changed": changed_total,
        },
    ]


def main():
    clear(FIGURES)

    model = scripted(
        "Welcome to Beanline! What can I get you today?",
        "Got it -- one large oat milk latte coming right up.",
    )

    # ---- the four stations: plain functions, state -> partial dict ----

    def greet(state: OrderState) -> dict:
        system = SystemMessage(content="You are the barista at Beanline Coffee.")
        human = HumanMessage(content="hi! what's good today?")
        reply = model.invoke([system, human])
        return {"messages": [reply.content]}

    def take_order(state: OrderState) -> dict:
        human = HumanMessage(content="one large oat milk latte, please")
        reply = model.invoke([human])
        # the fuzzy-prose -> typed-object step is its own topic
        # (from-prose-to-objects); here the item is the known order itself.
        item = {"drink": "latte", "size": "large", "extras": ["oat milk"]}
        return {"messages": [reply.content], "item": item}

    def price_order(state: OrderState) -> dict:
        # no model call: plain python over the state's order.
        total = price(OrderItem(**state["item"]))
        return {"total": total}

    def confirm(state: OrderState) -> dict:
        ticket = f"Ticket: large latte, oat milk -- ${state['total']:.2f}. Thanks!"
        return {"messages": [ticket]}

    g = StateGraph(OrderState)
    g.add_node("greet", greet)
    g.add_node("take_order", take_order)
    g.add_node("price_order", price_order)
    g.add_node("confirm", confirm)
    g.add_edge(START, "greet")
    g.add_edge("greet", "take_order")
    g.add_edge("take_order", "price_order")
    g.add_edge("price_order", "confirm")
    g.add_edge("confirm", END)
    compiled = g.compile()
    drawable = compiled.get_graph()

    # frame 1: the graph, drawn straight from the compiled object
    draw_graph(
        drawable,
        POSITIONS,
        FIGURES / "step-01.png",
        TITLE,
        note="Four stations between START and END, drawn FROM compiled.get_graph() -- it cannot drift from the code.",
    )

    # frame 2-5: one frame per real super-step
    state: dict = {"messages": [], "item": None, "total": 0.0}
    step_order: list[str] = []
    calls_log: list[tuple[str, int, int]] = []
    visited: list[str] = []
    taken_edges: list[tuple[str, str]] = []
    edge_iter = iter(EDGE_ORDER)
    prev_calls = len(model.calls)

    step_notes = {
        "greet": "Step 1: greet -- the scripted model answers, its reply lands in messages.",
        "take_order": "Step 2: take_order -- the model acknowledges, a real OrderItem lands in state.",
        "price_order": "Step 3: price_order -- this node has no LLM: the till does arithmetic, the model does not.",
        "confirm": "Step 4: confirm -- plain python quotes the ticket back, still no model call.",
    }

    for frame_i, update in enumerate(
        compiled.stream(state, stream_mode="updates"), start=2
    ):
        node_name, delta = next(iter(update.items()))
        assert set(delta.keys()) <= SCHEMA_KEYS, (node_name, delta.keys())
        step_order.append(node_name)
        taken_edges.append(next(edge_iter))

        prior = dict(state)
        state["messages"] = state["messages"] + delta.get("messages", [])
        if "item" in delta:
            state["item"] = delta["item"]
        if "total" in delta:
            state["total"] = delta["total"]

        new_calls = len(model.calls)
        calls_log.append((node_name, prev_calls, new_calls))
        prev_calls = new_calls

        draw_graph(
            drawable,
            POSITIONS,
            FIGURES / f"step-{frame_i:02d}.png",
            TITLE,
            active=node_name,
            visited=tuple(visited),
            taken_edges=tuple(taken_edges),
            state_rows=state_rows(state, delta, prior),
            note=step_notes[node_name],
        )
        visited.append(node_name)

    # frame 6: END reached, the final trace
    taken_edges.append(next(edge_iter))
    draw_graph(
        drawable,
        POSITIONS,
        FIGURES / "step-06.png",
        TITLE,
        visited=tuple(visited),
        taken_edges=tuple(taken_edges),
        state_rows=state_rows(state, {}, state),
        note="END reached -- compile once, invoke many: the graph is a program, the run is a trace.",
    )

    # ---- oracle ----
    assert set(compiled.get_graph().nodes) >= {
        "greet",
        "take_order",
        "price_order",
        "confirm",
    }
    assert step_order == ["greet", "take_order", "price_order", "confirm"], step_order

    expected_total = price(OrderItem(drink="latte", size="large", extras=["oat milk"]))
    assert expected_total == 6.25
    assert state["total"] == expected_total

    calls_by_node = {n: (before, after) for n, before, after in calls_log}
    before, after = calls_by_node["price_order"]
    assert before == after, "price_order must make zero model calls"
    for n in ("greet", "take_order"):
        before, after = calls_by_node[n]
        assert after > before, f"{n} must really consult the model"
    before, after = calls_by_node["confirm"]
    assert before == after, "confirm must make zero model calls too"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, super-step order {step_order}, "
        f"total ${state['total']:.2f} recomputed independently, "
        f"model calls unchanged across price_order ({calls_by_node['price_order']}). "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
