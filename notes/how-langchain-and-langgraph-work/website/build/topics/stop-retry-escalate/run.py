"""Stop, retry, escalate: bounded responses to the four ways an agent step fails.

Three real langgraph runs at Beanline Coffee, each a different failure class.
(a) The espresso machine jams twice, then works: a RetryPolicy on the node
absorbs a transient fault -- the library retries, we do not. (b) A customer
keeps asking for decaf unicorn milk, an impossible order: the router always
loops, so nothing but a recursion_limit and a real GraphRecursionError stops
it. (c) check_stock is really out of oat milk, twice: a strikes counter in
state routes the third try to ask_human instead of hammering the stock room
again. All three run through real langgraph graphs; the machine and the
router are deterministic counters, never random.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from typing_extensions import TypedDict  # noqa: E402

from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import RetryPolicy  # noqa: E402

from beanline import Stock, make_tools  # noqa: E402
from langviz import clear, draw_card, draw_graph, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Stop, retry, escalate"


# ---------------------------------------------------------- (a) retry


class MachineJam(Exception):
    """The espresso machine jammed. A real, named, transient fault."""


class EspressoMachine:
    """A deterministic flaky callable: jams on attempts 1 and 2, works on 3."""

    def __init__(self):
        self.attempts = 0

    def pull_shot(self) -> str:
        self.attempts += 1
        if self.attempts < 3:
            raise MachineJam(f"jam on attempt {self.attempts}")
        return "espresso ready"


class ShotState(TypedDict):
    result: str


def build_shot_graph(machine: EspressoMachine, attempt_log: list):
    def make_shot(state: ShotState) -> dict:
        try:
            result = machine.pull_shot()
        except MachineJam as exc:
            attempt_log.append((machine.attempts, type(exc).__name__, str(exc)))
            raise
        return {"result": result}

    g = StateGraph(ShotState)
    g.add_node(
        "make_shot",
        make_shot,
        retry_policy=RetryPolicy(max_attempts=3, retry_on=MachineJam),
    )
    g.add_edge(START, "make_shot")
    g.add_edge("make_shot", END)
    return g.compile()


# ---------------------------------------------------------- (b) bound


class LoopState(TypedDict):
    laps: int


def ask(state: LoopState) -> dict:
    return {"laps": state.get("laps", 0) + 1}


def try_decaf_unicorn_milk(state: LoopState) -> dict:
    return {}


def route_ask(state: LoopState) -> str:
    return "try_decaf_unicorn_milk"  # the router always loops -- no way out


def build_loop_graph():
    g = StateGraph(LoopState)
    g.add_node("ask", ask)
    g.add_node("try_decaf_unicorn_milk", try_decaf_unicorn_milk)
    g.add_edge(START, "ask")
    g.add_conditional_edges(
        "ask", route_ask, {"try_decaf_unicorn_milk": "try_decaf_unicorn_milk"}
    )
    g.add_edge("try_decaf_unicorn_milk", "ask")
    return g.compile()


def edges_from_seq(seq: list[str]) -> list[tuple[str, str]]:
    return list(zip(seq, seq[1:]))


# ---------------------------------------------------------- (c) escalate


class StockState(TypedDict):
    strikes: int


def build_stock_graph(stock: Stock, checked_log: list):
    check_stock, _get_menu, _compute_price = make_tools(stock)

    def check_stock_node(state: StockState) -> dict:
        raw = check_stock.invoke({"item": "oat milk"})  # "0 left"
        n = int(raw.split()[0])
        checked_log.append(n)
        strikes = state.get("strikes", 0)
        if n == 0:
            strikes += 1
        return {"strikes": strikes}

    def ask_human_node(state: StockState) -> dict:
        return {}

    def route_stock(state: StockState) -> str:
        return "ask_human" if state.get("strikes", 0) >= 2 else "check_stock"

    g = StateGraph(StockState)
    g.add_node("check_stock", check_stock_node)
    g.add_node("ask_human", ask_human_node)
    g.add_edge(START, "check_stock")
    g.add_conditional_edges(
        "check_stock",
        route_stock,
        {"check_stock": "check_stock", "ask_human": "ask_human"},
    )
    g.add_edge("ask_human", END)
    return g.compile()


def main():
    clear(FIGURES)

    # ---- frame 1: the four failure classes ----
    draw_card(
        "transient fault    ->  retry     (RetryPolicy: bounded attempts)\n"
        "impossible task     ->  bounded loop  (a hard recursion_limit)\n"
        "missing info         ->  clarify    (ask instead of guessing)\n"
        "repeated failure      ->  escalate   (a strikes counter hands off)\n\n"
        "Three real langgraph runs below, one per policy.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="four failure classes, four deterministic responses",
        note="Without policies: a transient fault crashes the run, an impossible "
        "task loops forever, and a repeated failure just hammers the same broken thing.",
    )

    # ---- (a) retry: the espresso machine ----
    machine = EspressoMachine()
    attempt_log: list = []
    shot_graph = build_shot_graph(machine, attempt_log)

    shot_raised = False
    try:
        shot_out = shot_graph.invoke({"result": ""})
    except Exception:
        shot_raised = True
        shot_out = {"result": ""}

    draw_card(
        "add_node(\n"
        "  'make_shot', make_shot,\n"
        "  retry_policy=RetryPolicy(max_attempts=3, retry_on=MachineJam),\n"
        ")\n\n"
        f"attempt {attempt_log[0][0]}: {attempt_log[0][1]}('{attempt_log[0][2]}')\n"
        "langgraph catches it, matches the retry_policy, retries automatically.",
        FIGURES / "step-02.png",
        TITLE,
        tone="bad",
        subtitle="make_shot node, attempt 1",
        note="The machine is a deterministic counter, not randomness: attempts 1 and 2 always jam.",
    )

    draw_card(
        f"attempt {machine.attempts}: '{shot_out['result']}'\n\n"
        f"machine.attempts == {machine.attempts}\n"
        "shot_graph.invoke(...) did NOT raise.",
        FIGURES / "step-03.png",
        TITLE,
        tone="good",
        subtitle="make_shot node, attempt 3",
        note="RetryPolicy(max_attempts=3) absorbed a transient fault inside ONE graph.invoke() call.",
    )

    # ---- (b) bound: the doom loop ----
    loop_graph = build_loop_graph()
    loop_drawable = loop_graph.get_graph()
    loop_positions = {
        "__start__": (8, 40),
        "ask": (35, 40),
        "try_decaf_unicorn_milk": (65, 40),
        "__end__": (90, 40),
    }

    loop_seq = ["__start__"]
    loop_caught = None
    spin_snapshot = None
    gen = loop_graph.stream({"laps": 0}, {"recursion_limit": 6}, stream_mode="updates")
    try:
        for step in gen:
            node = list(step.keys())[0]
            loop_seq.append(node)
            if len(loop_seq) - 1 == 5:
                spin_snapshot = list(loop_seq)
    except GraphRecursionError as exc:
        loop_caught = exc

    spin_laps = (len(spin_snapshot) - 1) // 2
    draw_graph(
        loop_drawable,
        loop_positions,
        FIGURES / "step-04.png",
        TITLE,
        active=spin_snapshot[-1],
        visited=set(spin_snapshot[1:-1]),
        taken_edges=edges_from_seq(spin_snapshot),
        note=f"Streamed live: {len(spin_snapshot) - 1} node executions in, lap {spin_laps} "
        "of ask -> try_decaf_unicorn_milk -> ask. Still spinning.",
    )

    total_executions = len(loop_seq) - 1
    draw_card(
        f"{type(loop_caught).__name__}\n{loop_caught}\n\n"
        f"{total_executions} node executions ran, recursion_limit=6.",
        FIGURES / "step-05.png",
        TITLE,
        tone="bad",
        subtitle="loop_graph.stream(..., {'recursion_limit': 6})",
        note="The loop was BOUNDED -- nothing about the task improved, so the budget is what saved us.",
    )

    # ---- (c) escalate: check_stock, out of oat milk ----
    stock = Stock({"oat milk": 0})
    checked_log: list = []
    stock_graph = build_stock_graph(stock, checked_log)
    stock_drawable = stock_graph.get_graph()
    stock_positions = {
        "__start__": (8, 30),
        "check_stock": (30, 30),
        "ask_human": (58, 30),
        "__end__": (90, 30),
    }

    stock_seq: list = []
    final_stock_state = None
    for mode, chunk in stock_graph.stream(
        {"strikes": 0}, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            stock_seq.append(list(chunk.keys())[0])
        elif mode == "values":
            final_stock_state = chunk

    draw_graph(
        stock_drawable,
        stock_positions,
        FIGURES / "step-06.png",
        TITLE,
        active="ask_human",
        visited={"check_stock"},
        taken_edges=edges_from_seq(["__start__"] + stock_seq),
        edge_label=("check_stock", "ask_human", "ask_human"),
        note=f"check_stock really returned {checked_log} for oat milk -- two strikes, "
        "then the router's real return sends the third try to ask_human instead.",
    )

    # ---- frame 7: the policy scorecard ----
    loop_laps = total_executions // 2
    draw_scorecard(
        [
            {
                "label": "transient fault (jam)",
                "cells": ["attempts<=3", f"{machine.attempts}, succeeded"],
                "verdict": "pass",
            },
            {
                "label": "impossible task (loop)",
                "cells": ["limit=6", f"{loop_laps} laps, error"],
                "verdict": "pass",
            },
            {
                "label": "repeated failure (stock)",
                "cells": ["strikes>=2", f"{final_stock_state['strikes']} -> human"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-07.png",
        TITLE,
        columns=["policy", "real result"],
        note="Three failure classes, three bounded policies, three real graphs run above.",
    )

    # ---- oracle ----
    assert not shot_raised, "the retried invoke must not raise"
    assert machine.attempts == 3
    assert shot_out["result"] == "espresso ready"
    assert [a[1] for a in attempt_log] == ["MachineJam", "MachineJam"]

    assert loop_caught is not None
    assert type(loop_caught) is GraphRecursionError
    assert total_executions <= 6, "the loop must be bounded by the recursion_limit"

    assert stock_seq[-1] == "ask_human"
    assert final_stock_state["strikes"] == 2
    assert checked_log == [0, 0], "check_stock must really have returned 0 both times"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, retry: attempts={machine.attempts} succeeded, "
        f"loop: {total_executions} executions then {type(loop_caught).__name__}, "
        f"escalate: strikes={final_stock_state['strikes']} -> ask_human. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
