"""Chains can't loop: a pipe has no arrow back into itself.

Maya wants an oat-milk latte. Beanline is out (Stock({"oat milk": 0,
"almond milk": 0, "whole milk": 10})). The real conversation is propose ->
check the real Stock -> out -> ask again -> BACK to propose: three laps,
the model naming a different milk each lap, until whole milk (in stock)
lands. A chain runs once, so the only way to get that loop is a hand-rolled
python while-loop wrapped around it, re-splicing the message list by hand
each lap -- invisible control flow, no state object, unrestartable. The
same beat as a real langgraph StateGraph makes the back-edge a real,
inspectable edge in the compiled graph: ask_again -> propose.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import (  # noqa: E402
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from beanline import Stock, scripted, tool_call_msg  # noqa: E402
from langviz import (  # noqa: E402
    clear,
    draw_card,
    draw_graph,
    draw_messages,
    draw_pipeline,
    draw_scorecard,
)

FIGURES = HERE / "figures"
TITLE = "Chains can't loop"

MILKS_IN_ORDER = ["oat milk", "almond milk", "whole milk"]


class MilkState(TypedDict):
    attempt: int
    milk: str | None
    stock_count: int


def main():
    clear(FIGURES)
    stock = Stock({"oat milk": 0, "almond milk": 0, "whole milk": 10})

    # ---- frame 1: recap the pipe from earlier topics -----------------------
    draw_pipeline(
        [
            {
                "label": "prompt",
                "type_label": "ChatPromptTemplate",
                "payload": "",
                "state": "done",
            },
            {
                "label": "model",
                "type_label": "ScriptedChatModel",
                "payload": "",
                "state": "done",
            },
            {
                "label": "parser",
                "type_label": "StrOutputParser",
                "payload": "",
                "state": "done",
            },
        ],
        FIGURES / "step-01.png",
        TITLE,
        note="A pipe flows one way: input in, one answer out. Nothing routes backward.",
    )

    # ---- frame 2: the demand a pipe cannot satisfy -------------------------
    draw_card(
        'Maya: "oat milk\'s out -- can I pick again?"\n\n'
        "propose  ->  check the real Stock  ->  out  ->  ask again  ->  propose\n\n"
        "That last arrow points BACKWARD, into the pipe's own start.\n"
        "A chain has one direction. Once it finishes, it is done.",
        FIGURES / "step-02.png",
        TITLE,
        tone="bad",
        subtitle="the demand: re-enter the pipe",
        note="Chains have no such arrow: there is nothing built in to re-enter.",
    )

    # ---- the while-loop version ---------------------------------------------
    model = scripted(
        tool_call_msg("propose_milk", {"milk": MILKS_IN_ORDER[0]}, "call_1"),
        tool_call_msg("propose_milk", {"milk": MILKS_IN_ORDER[1]}, "call_2"),
        tool_call_msg("propose_milk", {"milk": MILKS_IN_ORDER[2]}, "call_3"),
    )
    system = SystemMessage(
        content="You are the barista at Beanline Coffee. Maya wants a latte; "
        "find a milk that's actually in stock."
    )
    context = [system, HumanMessage(content="Maya wants an oat-milk latte.")]

    chosen_milk = None
    while chosen_milk is None:
        ai = model.invoke(context)  # the chain, re-invoked by hand
        context.append(ai)
        call = ai.tool_calls[0]
        milk = call["args"]["milk"]
        if stock.check(milk) > 0:
            context.append(
                ToolMessage(content=f"{milk}: in stock", tool_call_id=call["id"])
            )
            chosen_milk = milk
        else:
            context.append(
                ToolMessage(content=f"{milk}: out of stock", tool_call_id=call["id"])
            )
            context.append(
                HumanMessage(content=f"{milk} is out -- please propose another milk.")
            )

    # ---- frame 3: lap 1, the loop's invisible control flow ------------------
    draw_messages(
        model.calls[0],
        FIGURES / "step-03.png",
        TITLE,
        new=(0, 1),
        right_text=(
            "chosen_milk = None\n"
            "while chosen_milk is None:\n"
            "    ai = model.invoke(context)\n"
            "    milk = ai.tool_calls[0][\n"
            '        "args"]["milk"]\n'
            "    if stock.check(milk) > 0:\n"
            "        chosen_milk = milk\n"
            "    else:\n"
            "        context.append(...)\n"
            "        # lap again"
        ),
        right_title="lap 1: the hand-rolled loop",
        note="Lap 1's context, exactly what the model saw. The loop lives only in python -- invisible, unrestartable, undrawable.",
    )

    # ---- frame 4: lap 3, success by hand-splicing ----------------------------
    draw_messages(
        model.calls[2],
        FIGURES / "step-04.png",
        TITLE,
        new=(2, 3, 4, 5, 6, 7),
        right_text=(
            "# context re-spliced by hand\n"
            "# after every missed lap:\n"
            "context = [\n"
            "  system, human0,\n"
            "  ai1, tool1, human1,\n"
            "  ai2, tool2, human2,\n"
            "]\n"
            "ai3 = model.invoke(context)\n"
            "# -> whole milk: in stock"
        ),
        right_title="lap 3: re-spliced by hand",
        note="Lap 3's context has grown to 8 messages, hand-appended lap by lap. That splice IS the loop.",
    )

    # ---- the explicit graph version -------------------------------------------
    def propose(state: MilkState) -> dict:
        idx = state["attempt"]
        return {"milk": MILKS_IN_ORDER[idx], "attempt": idx + 1}

    def check_stock_node(state: MilkState) -> dict:
        return {"stock_count": stock.check(state["milk"])}

    def ask_again(state: MilkState) -> dict:
        return {}

    def route_after_check(state: MilkState) -> str:
        return END if state["stock_count"] > 0 else "ask_again"

    g = StateGraph(MilkState)
    g.add_node("propose", propose)
    g.add_node("check_stock", check_stock_node)
    g.add_node("ask_again", ask_again)
    g.add_edge(START, "propose")
    g.add_edge("propose", "check_stock")
    g.add_conditional_edges(
        "check_stock", route_after_check, {END: END, "ask_again": "ask_again"}
    )
    g.add_edge("ask_again", "propose")
    compiled = g.compile()
    drawable = compiled.get_graph()

    visited = []
    taken_edges = []
    prev = "__start__"
    for update in compiled.stream(
        {"attempt": 0, "milk": None, "stock_count": 0}, stream_mode="updates"
    ):
        node_name = next(iter(update))
        visited.append(node_name)
        taken_edges.append((prev, node_name))
        prev = node_name
    taken_edges.append((prev, "__end__"))

    graph_result = compiled.invoke({"attempt": 0, "milk": None, "stock_count": 0})

    # ---- frame 5: the explicit graph, a real drawable back-edge -------------
    positions = {
        "__start__": (6, 42),
        "propose": (26, 42),
        "check_stock": (50, 42),
        "ask_again": (50, 18),
        "__end__": (76, 42),
    }
    draw_graph(
        drawable,
        positions,
        FIGURES / "step-05.png",
        TITLE,
        visited=set(visited),
        taken_edges=set(taken_edges),
        edge_label=("ask_again", "propose", "retry"),
        note="Same three laps, now a real StateGraph edge: ask_again -> propose is solid and drawable, not hidden python control flow.",
    )

    # ---- frame 6: the comparison ---------------------------------------------
    draw_scorecard(
        [
            {
                "label": "python while-loop",
                "cells": ["invisible", "no state object", "no"],
                "verdict": "fail",
            },
            {
                "label": "langgraph StateGraph",
                "cells": ["drawable", "State object", "yes"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        columns=["control flow", "state", "checkpointable"],
        note="LangGraph's move: control flow becomes an object.",
    )

    # ---- oracle ----
    assert len(model.calls) == 3, "the while-loop must invoke the model exactly 3 times"
    assert stock.check("oat milk") == 0
    assert stock.check("almond milk") == 0
    assert stock.check("whole milk") > 0
    assert chosen_milk == "whole milk"
    real_edges = [(e.source, e.target) for e in compiled.get_graph().edges]
    assert ("ask_again", "propose") in real_edges, (
        "the compiled graph must keep the back-edge"
    )
    assert graph_result["milk"] == "whole milk"
    assert graph_result["milk"] == chosen_milk == "whole milk"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, while-loop invoked the model {len(model.calls)} times "
        f"(oat {stock.check('oat milk')}, almond {stock.check('almond milk')}, "
        f"whole {stock.check('whole milk')} in stock), graph back-edge "
        f"ask_again->propose present in {len(real_edges)} real edges, both versions "
        f"landed on {chosen_milk!r}. All checks passed."
    )


if __name__ == "__main__":
    main()
