"""Walk away, come back: langgraph's checkpointer keeps a ticket per thread.

Beanline's ticket graph (add_item -> price) is compiled ONCE with
checkpointer=InMemorySaver(). Maya orders a latte on thread_id "maya", then
steps aside for a phone call. Ben orders an espresso on thread_id "ben" --
same graph, same saver, his own rail. Maya returns and adds a croissant: a
second invoke on thread "maya" that EXTENDS her existing ticket and never
sees Ben's. graph.get_state and graph.get_state_history are the physical
evidence -- checkpoints that outlive the walk-away.
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

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from langviz import clear, draw_scorecard, draw_thread_lanes  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Walk away, come back"

MENU = {"latte": 4.50, "espresso": 3.00, "croissant": 3.50}


class TicketState(TypedDict):
    items: Annotated[list[dict], operator.add]
    total: float


def add_item(state: TicketState) -> dict:
    """Validate every item ordered so far is really on the menu."""
    for item in state["items"]:
        if item["drink"] not in MENU:
            raise ValueError(f"not on the menu: {item['drink']}")
    return {}


def price(state: TicketState) -> dict:
    return {"total": round(sum(MENU[i["drink"]] for i in state["items"]), 2)}


def build_graph(saver: InMemorySaver):
    g = StateGraph(TicketState)
    g.add_node("add_item", add_item)
    g.add_node("price", price)
    g.add_edge(START, "add_item")
    g.add_edge("add_item", "price")
    g.add_edge("price", END)
    return g.compile(checkpointer=saver)


def step_label(snap) -> str:
    """What runs next from this checkpoint, or "done" once the graph finished."""
    return snap.next[0] if snap.next else "done"


def item_hint(snap) -> str:
    names = [i["drink"] for i in snap.values.get("items", [])]
    return ", ".join(names) if names else "(empty)"


def as_checkpoints(history: list) -> list[dict]:
    """history is newest-first from get_state_history; a lane reads left to
    right in time, so reverse it to chronological order."""
    return [{"step": step_label(h), "hint": item_hint(h)} for h in reversed(history)]


def main():
    clear(FIGURES)

    saver = InMemorySaver()
    graph = build_graph(saver)
    maya_cfg = {"configurable": {"thread_id": "maya"}}
    ben_cfg = {"configurable": {"thread_id": "ben"}}

    # ---- frame 1: two empty rails over one checkpointer ----
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "dimmed",
                "checkpoints": [],
                "resume_from": None,
            },
            {
                "label": "thread-ben",
                "state": "dimmed",
                "checkpoints": [],
                "resume_from": None,
            },
        ],
        FIGURES / "step-01.png",
        TITLE,
        note="One graph, one saver, compiled once with checkpointer=InMemorySaver(). A lane per thread_id.",
    )

    # ---- frame 2: Maya orders a latte -- her checkpoints land ----
    graph.invoke({"items": [{"drink": "latte"}]}, maya_cfg)
    maya_hist_1 = list(graph.get_state_history(maya_cfg))
    maya_cps_1 = as_checkpoints(maya_hist_1)
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "active",
                "checkpoints": maya_cps_1,
                "resume_from": None,
            },
            {
                "label": "thread-ben",
                "state": "dimmed",
                "checkpoints": [],
                "resume_from": None,
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        note=f"{len(maya_hist_1)} checkpoints, one per super-step: start, add_item, price, done.",
    )

    # ---- frame 3: Maya walks away -- checkpoints stay, nothing is running ----
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "dimmed",
                "checkpoints": maya_cps_1,
                "resume_from": None,
            },
            {
                "label": "thread-ben",
                "state": "dimmed",
                "checkpoints": [],
                "resume_from": None,
            },
        ],
        FIGURES / "step-03.png",
        TITLE,
        note="The pause is durable state sitting in the saver, not a running process.",
    )

    # ---- frame 4: Ben orders an espresso on his own thread ----
    graph.invoke({"items": [{"drink": "espresso"}]}, ben_cfg)
    ben_hist = list(graph.get_state_history(ben_cfg))
    ben_cps = as_checkpoints(ben_hist)
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "dimmed",
                "checkpoints": maya_cps_1,
                "resume_from": None,
            },
            {
                "label": "thread-ben",
                "state": "active",
                "checkpoints": ben_cps,
                "resume_from": None,
            },
        ],
        FIGURES / "step-04.png",
        TITLE,
        note="Ben's checkpoints land on his rail only. Maya's four are untouched.",
    )

    # ---- frame 5: Maya returns -- resume from her last checkpoint ----
    maya_resumed = graph.get_state(maya_cfg).values
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "active",
                "checkpoints": maya_cps_1,
                "resume_from": len(maya_cps_1) - 1,
            },
            {
                "label": "thread-ben",
                "state": "done",
                "checkpoints": ben_cps,
                "resume_from": None,
            },
        ],
        FIGURES / "step-05.png",
        TITLE,
        note=f"get_state(maya_cfg).values['items'] == {maya_resumed['items']} -- Ben's espresso never leaked in.",
    )

    # ---- frame 6: "and a croissant" -- the second invoke extends the SAME lane ----
    graph.invoke({"items": [{"drink": "croissant"}]}, maya_cfg)
    maya_hist_2 = list(graph.get_state_history(maya_cfg))
    maya_cps_2 = as_checkpoints(maya_hist_2)
    maya_final = graph.get_state(maya_cfg).values
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "active",
                "checkpoints": maya_cps_2,
                "resume_from": None,
            },
            {
                "label": "thread-ben",
                "state": "done",
                "checkpoints": ben_cps,
                "resume_from": None,
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        note=f"items = {maya_final['items']} -- four more checkpoints appended to the same rail, not a new one.",
    )

    # ---- frame 7: the history unrolled, newest first ----
    rows = [
        {
            "label": f"step {h.metadata['step']}",
            "cells": [step_label(h), item_hint(h)],
        }
        for h in maya_hist_2
    ]
    draw_scorecard(
        rows,
        FIGURES / "step-07.png",
        TITLE,
        columns=["next node", "items"],
        note="get_state_history(maya_cfg): the ticket rail, replayable -- 8 snapshots, newest first.",
    )

    # ---- oracle ----
    ben_final = graph.get_state(ben_cfg).values
    assert graph.checkpointer is saver, (
        "maya and ben must share ONE checkpointer instance"
    )
    assert maya_resumed["items"] == [{"drink": "latte"}], (
        "isolation must survive ben's interleaved run"
    )
    assert maya_final["items"] == [{"drink": "latte"}, {"drink": "croissant"}]
    assert ben_final["items"] == [{"drink": "espresso"}]
    assert not any(i["drink"] == "latte" for i in ben_final["items"])
    assert len(maya_hist_2) > len(maya_hist_1)
    ids = [h.config["configurable"]["checkpoint_id"] for h in maya_hist_2]
    assert len(set(ids)) == len(ids), "checkpoint ids must all be distinct"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, maya's ticket held {len(maya_hist_1)} then "
        f"{len(maya_hist_2)} checkpoints, ben's espresso never touched her rail, "
        f"her final items={maya_final['items']}. All checks passed."
    )


if __name__ == "__main__":
    main()
