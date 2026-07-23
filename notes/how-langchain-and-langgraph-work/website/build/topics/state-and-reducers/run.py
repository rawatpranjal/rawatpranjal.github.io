"""State and reducers: nodes return deltas, the schema decides how they land.

Beanline Coffee grows one ticket across two nodes. TicketState has three
channels with three different merge behaviors: items accumulates
(operator.add), messages appends but replaces a same-id entry
(add_messages), and total is last-write-wins (a plain field, no Annotated).
add_latte and add_cappuccino each return a PARTIAL dict -- never the whole
state -- and the compiled graph's reducers do the real merging. A second,
one-node graph proves add_messages' id-replace: the same message id gets a
corrected body instead of piling up a duplicate.
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

from langchain_core.messages import AIMessage, AnyMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from beanline import EXTRAS, MENU, SIZES, OrderItem, price  # noqa: E402
from langviz import (  # noqa: E402
    clear,
    draw_card,
    draw_graph,
    draw_messages,
    draw_state,
)

FIGURES = HERE / "figures"
TITLE = "State and reducers"


class TicketState(TypedDict):
    items: Annotated[list[dict], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]
    total: float


def add_latte(state: TicketState) -> dict:
    """A partial dict: only items and total, never messages."""
    latte = OrderItem(drink="latte", size="large", extras=["oat milk"])
    return {
        "items": [{"drink": "latte", "size": "large", "extras": ["oat milk"]}],
        "total": price(latte),
    }


def add_cappuccino(state: TicketState) -> dict:
    """Reads the running total so far, returns the new one -- overwrite, not add."""
    cappuccino = OrderItem(drink="cappuccino", size="small")
    return {
        "items": [{"drink": "cappuccino", "size": "small", "extras": []}],
        "total": round(state["total"] + price(cappuccino), 2),
    }


def revise_order(state: TicketState) -> dict:
    """Same message id as the seeded state -- add_messages must replace, not append."""
    return {"messages": [AIMessage(id="order-1", content="one LARGE latte")]}


def main():
    clear(FIGURES)

    builder = StateGraph(TicketState)
    builder.add_node("add_latte", add_latte)
    builder.add_node("add_cappuccino", add_cappuccino)
    builder.add_edge(START, "add_latte")
    builder.add_edge("add_latte", "add_cappuccino")
    builder.add_edge("add_cappuccino", END)
    graph = builder.compile()

    init_state: TicketState = {"items": [], "messages": [], "total": 0.0}

    # both views of one real run: the deltas nodes actually returned, and
    # the accumulated state after each superstep.
    updates = list(graph.stream(init_state, stream_mode="updates"))
    values = list(graph.stream(init_state, stream_mode="values"))
    delta_latte = updates[0]["add_latte"]
    delta_cappuccino = updates[1]["add_cappuccino"]
    state_after_latte = values[1]
    final = values[2]

    drawable = graph.get_graph()
    positions = {
        "__start__": (7, 30),
        "add_latte": (24, 30),
        "add_cappuccino": (46, 30),
        "__end__": (60, 30),
    }

    draw_card(
        "TicketState: three channels, three reducers\n\n"
        "  items      Annotated[list[dict], operator.add]        add\n"
        "  messages   Annotated[list[AnyMessage], add_messages]   add_messages\n"
        "  total      float, no Annotated                         overwrite\n\n"
        "Nodes will return DELTAS; reducers decide how deltas land.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="three channels, three merge behaviors",
        note="Each channel decides for itself how a node's partial return gets folded in.",
    )

    draw_state(
        [
            {
                "channel": "items",
                "value": str(delta_latte["items"]),
                "reducer": "add",
                "delta": None,
                "changed": True,
            },
            {
                "channel": "messages",
                "value": "[]",
                "reducer": "add_messages",
                "delta": None,
                "changed": False,
            },
            {
                "channel": "total",
                "value": str(delta_latte["total"]),
                "reducer": "overwrite",
                "delta": None,
                "changed": True,
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        note="Only the two keys add_latte actually returned show as changed here -- messages was never touched.",
    )

    draw_graph(
        drawable,
        positions,
        FIGURES / "step-03.png",
        TITLE,
        active="add_latte",
        taken_edges=(("__start__", "add_latte"),),
        state_rows=[
            {
                "channel": "items",
                "value": str(state_after_latte["items"]),
                "reducer": "add",
                "delta": str(delta_latte["items"]),
                "changed": True,
            },
            {
                "channel": "total",
                "value": str(state_after_latte["total"]),
                "reducer": "overwrite",
                "delta": str(init_state["total"]),
                "changed": True,
            },
            {
                "channel": "messages",
                "value": "[]",
                "reducer": "add_messages",
                "delta": None,
                "changed": False,
            },
        ],
        note="Two different merges land in the same step -- items appends the delta, total gets overwritten.",
    )

    draw_graph(
        drawable,
        positions,
        FIGURES / "step-04.png",
        TITLE,
        active="add_cappuccino",
        visited=("add_latte",),
        taken_edges=(("__start__", "add_latte"), ("add_latte", "add_cappuccino")),
        state_rows=[
            {
                "channel": "items",
                "value": str(final["items"]),
                "reducer": "add",
                "delta": str(delta_cappuccino["items"]),
                "changed": True,
            },
            {
                "channel": "total",
                "value": str(final["total"]),
                "reducer": "overwrite",
                "delta": str(state_after_latte["total"]),
                "changed": True,
            },
            {
                "channel": "messages",
                "value": "[]",
                "reducer": "add_messages",
                "delta": None,
                "changed": False,
            },
        ],
        note="Same pattern on the second node -- items now holds both drinks, total is struck and overwritten again.",
    )

    # a separate one-node graph, just to isolate add_messages' id-replace.
    msg_builder = StateGraph(TicketState)
    msg_builder.add_node("revise_order", revise_order)
    msg_builder.add_edge(START, "revise_order")
    msg_builder.add_edge("revise_order", END)
    msg_graph = msg_builder.compile()

    seeded_state: TicketState = {
        "items": [],
        "messages": [AIMessage(id="order-1", content="one small latte")],
        "total": 0.0,
    }
    before_messages = list(seeded_state["messages"])
    msg_result = msg_graph.invoke(seeded_state)
    after_messages = msg_result["messages"]

    draw_messages(
        after_messages,
        FIGURES / "step-05.png",
        TITLE,
        new=(0,),
        right_text=(
            f"before: id={before_messages[0].id!r}\n"
            f"  content={before_messages[0].content!r}\n\n"
            f"after:  id={after_messages[0].id!r}\n"
            f"  content={after_messages[0].content!r}\n\n"
            f"same id -> add_messages REPLACES\n"
            f"len before {len(before_messages)}, len after {len(after_messages)}"
        ),
        right_title="add_messages: same id replaces",
        note="The id matched, so add_messages replaced the message in place -- length unchanged, content changed.",
    )

    draw_card(
        "Deltas + reducers is what makes parallel branches mergeable.\n\n"
        "Two nodes can each return {'items': [x]} in the same step\n"
        "and nobody clobbers anybody -- the add reducer folds both in.\n\n"
        "Same story for messages: a same-id return replaces instead\n"
        "of piling up a duplicate. total (overwrite) is the one\n"
        "channel where the last write really does win.",
        FIGURES / "step-06.png",
        TITLE,
        tone="good",
        subtitle="why this matters",
        note=(
            f"This real run ended with {len(final['items'])} items and a total of "
            f"{final['total']} -- deltas plus reducers made the merge automatic."
        ),
    )

    # ---- oracle ----
    assert len(final["items"]) == 2
    assert final["items"][0]["drink"] == "latte"
    assert final["items"][1]["drink"] == "cappuccino"

    latte_price = round(MENU["latte"] + SIZES["large"] + EXTRAS["oat milk"], 2)
    cappuccino_price = round(MENU["cappuccino"] + SIZES["small"], 2)
    assert (
        latte_price
        == price(OrderItem(drink="latte", size="large", extras=["oat milk"]))
        == 6.25
    )
    assert (
        cappuccino_price == price(OrderItem(drink="cappuccino", size="small")) == 4.00
    )
    assert final["total"] == round(latte_price + cappuccino_price, 2) == 10.25

    assert set(delta_latte.keys()) == {"items", "total"}, (
        "add_latte's captured update must be a real partial, no messages key"
    )
    assert delta_latte["items"] == [
        {"drink": "latte", "size": "large", "extras": ["oat milk"]}
    ]
    assert delta_latte["total"] == 6.25

    assert len(before_messages) == len(after_messages) == 1
    assert before_messages[0].id == after_messages[0].id == "order-1"
    assert before_messages[0].content == "one small latte"
    assert after_messages[0].content == "one LARGE latte"
    assert before_messages[0].content != after_messages[0].content

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, final total {final['total']} from {len(final['items'])} items, "
        f"add_messages replaced order-1 in place (len {len(before_messages)} -> "
        f"{len(after_messages)}). All checks passed."
    )


if __name__ == "__main__":
    main()
