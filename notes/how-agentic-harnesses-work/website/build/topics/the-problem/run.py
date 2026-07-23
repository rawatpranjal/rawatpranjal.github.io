"""The problem: one prompt, one model, a task bigger than a context window.

A real naive run: no memory files, no compaction, no plan on disk. The context
really fills past its ceiling mid-feature and the session dies with todo.py
half-written (it genuinely does not compile). The next sessions wake with
empty columns and do exactly what the keystone paper names: one guesses the
project state wrong, one declares victory at 5/12.

Source: Anthropic, "Effective harnesses for long-running agents" (2025).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import PYTHON, _mark, build_workspace, todo_source  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    builtin_tools,
    total_tokens,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The problem"
CEILING = 1000  # the naive context ceiling, in tokens


class ContextFull(Exception):
    pass


def main():
    clear(FIGURES)
    # A bare world: the user's feature list and the tests. No plan, no
    # progress file, no init.sh, no git -- nothing a future session could read.
    ws = build_workspace(stage=0, git=False)
    ws.path("CLAUDE.md").unlink()
    ws.path("init.sh").unlink()

    h = Harness(ws)
    big_ask = (
        "Build todo.py, a CLI todo app, with all 12 features in "
        "feature_list.json: add, list, done, delete, count, search, priority, "
        "priority ordering, due dates, clear-done, missing-id errors, usage "
        "text. Keep going until everything works."
    )

    # ---- session 1: builds until the context ceiling kills it ----
    # cut mid-token, the way a dying stream really truncates a file
    half_of_f06 = todo_source(6)[: todo_source(6).index("def cmd_search") + 12]
    s1 = ScriptedModel(
        [
            Turn(
                text="Starting with add and list.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(2)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="Now done, delete, count.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(5)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="Adding search next.",
                tool="write_file",
                args={"path": "todo.py", "content": half_of_f06},
            ),
        ]
    )
    agent1 = Agent(s1, builtin_tools(ws), ws, budget=CEILING, keep_last=999)
    agent1.maybe_compact = lambda *a, **k: None  # naive harness: no compaction

    snaps = {}

    def on_event1(event, msg):
        t = total_tokens(agent1.context)
        if event == "user":
            snaps["ask"] = h.snap("ask", agent1, loop_node="context")
        if event == "tool_result" and "5 passed" in (msg.content if msg else ""):
            _mark(ws, "F01", True), _mark(ws, "F02", True), _mark(ws, "F03", True)
            _mark(ws, "F04", True), _mark(ws, "F05", True)
            snaps["filling"] = h.snap("filling", agent1, loop_node="tool")
        if t > CEILING and event == "tool_result":
            snaps["dead"] = h.snap("dead", agent1, loop_node="model")
            raise ContextFull

    died = False
    try:
        agent1.run(big_ask, on_event=on_event1)
    except ContextFull:
        died = True

    # ---- session 2: fresh context, nothing to read, guesses wrong ----
    s2 = ScriptedModel(
        [
            Turn(
                text="Fresh session. I see a todo.py already exists, so the "
                "earlier work must be nearly done -- I assume features 1-9 "
                "are complete and I should continue with due dates (F09)."
            )
        ]
    )
    agent2 = Agent(s2, builtin_tools(ws), ws, budget=CEILING)
    agent2.run("Continue building todo.py.")
    snaps["guess"] = h.snap("guess", agent2, loop_node="model")

    # ---- session 3: fresh context, declares victory at 5/12 ----
    s3 = ScriptedModel(
        [
            Turn(
                text="The tests I glanced at looked healthy. The project is "
                "complete! todo.py ships all requested features."
            )
        ]
    )
    agent3 = Agent(s3, builtin_tools(ws), ws, budget=CEILING)
    agent3.run("Wrap up todo.py.")
    snaps["victory"] = h.snap("victory", agent3, loop_node="done")

    # ---- frames ----
    draw_frame(
        snaps["ask"],
        FIGURES / "step-01.png",
        TITLE,
        right="files",
        note="One giant ask, one model. The disk: the user's 12-item feature list, the tests, nothing else.",
    )
    draw_frame(
        snaps["filling"],
        FIGURES / "step-02.png",
        TITLE,
        right="files",
        note="The session builds for real: 5 features green. But every write also fills the one context it has.",
    )
    draw_frame(
        snaps["dead"],
        FIGURES / "step-03.png",
        TITLE,
        right="files",
        note="The ceiling. The session dies mid-feature-6: todo.py is half-written and does not even compile.",
    )
    draw_frame(
        snaps["guess"],
        FIGURES / "step-04.png",
        TITLE,
        right="files",
        note="Session 2 wakes with an empty column and nothing to read. It guesses the state -- wrong (5/12, not 9).",
    )
    draw_frame(
        snaps["victory"],
        FIGURES / "step-05.png",
        TITLE,
        right="files",
        note='Session 3: "The project is complete!" The board says 5/12. Both failures are named in Anthropic\'s harness paper.',
    )

    # ---- oracle ----
    assert died, "the naive session must really die at the ceiling"
    assert snaps["dead"].total_tokens > CEILING
    compiles = subprocess.run(
        [PYTHON, "-m", "py_compile", str(ws.path("todo.py"))], capture_output=True
    )
    assert compiles.returncode != 0, "the half-written todo.py should not compile"
    passing = sum(1 for f in snaps["victory"].features if f["passes"])
    assert passing == 5, f"the board must show 5/12, got {passing}"
    assert "complete" in agent3.done_text.lower()
    assert "assume" in agent2.done_text.lower()
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {figs}"
    print(
        f"{len(figs)} figures, died at {snaps['dead'].total_tokens}/{CEILING} tok, "
        f"todo.py truncated (py_compile rc={compiles.returncode}), board 5/12. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
