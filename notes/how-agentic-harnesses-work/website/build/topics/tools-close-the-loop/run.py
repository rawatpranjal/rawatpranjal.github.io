"""Tools close the loop: a text-only model cannot act, tools give it hands.

A real (scripted-model) session adds the search command, feature F06, to
todo.py. Six frames: the fresh ask, the model's first tool call, the harness
executing the read, a real write landing real code on disk, a real test run
reporting the real consequence, and the model's closing text -- one full
model -> tool -> result -> model cycle, twice over.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace, todo_source  # noqa: E402
from harness import Agent, Harness, ScriptedModel, Turn, builtin_tools  # noqa: E402
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Tools close the loop"
BUDGET = 1000


def main():
    clear(FIGURES)
    ws = build_workspace(stage=5)
    tools = builtin_tools(ws)
    tool_names = ", ".join(tools.names())

    model = ScriptedModel(
        [
            Turn(
                text="Let me look at the current file before adding search.",
                tool="read_file",
                args={"path": "todo.py"},
            ),
            Turn(
                text="Adding the search command now.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(6)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="search is in: todo.py now implements F06, "
                "6 of 12 feature tests are green."
            ),
        ]
    )
    agent = Agent(model, tools, ws, budget=BUDGET)
    h = Harness(ws)

    snaps = {}
    tool_use_n = 0
    tool_result_n = 0

    def on_event(event, msg):
        nonlocal tool_use_n, tool_result_n
        if event == "user":
            snaps["fresh"] = h.snap("fresh", agent, loop_node="context")
        elif event == "assistant:tool_use":
            tool_use_n += 1
            if tool_use_n == 1:
                snaps["first_call"] = h.snap("first_call", agent, loop_node="hooks")
        elif event == "tool_result":
            tool_result_n += 1
            if tool_result_n == 1:
                snaps["read_result"] = h.snap("read_result", agent, loop_node="tool")
            elif tool_result_n == 2:
                snaps["write_result"] = h.snap("write_result", agent, loop_node="tool")
            elif tool_result_n == 3:
                snaps["tests_result"] = h.snap(
                    "tests_result", agent, loop_node="context"
                )
        elif event == "assistant:text":
            snaps["done"] = h.snap("done", agent, loop_node="done")

    agent.run(
        "Add a search command to todo.py (feature F06): "
        "todo search TERM filters tasks by substring.",
        on_event=on_event,
    )

    draw_frame(
        snaps["fresh"],
        FIGURES / "step-01.png",
        TITLE,
        right="loop",
        note=f"Fresh context, just the ask, but a tool belt now sits beside the model: {tool_names}.",
    )
    draw_frame(
        snaps["first_call"],
        FIGURES / "step-02.png",
        TITLE,
        right="loop",
        note='The model\'s first move is not text, it is a tool call: read_file("todo.py").',
    )
    draw_frame(
        snaps["read_result"],
        FIGURES / "step-03.png",
        TITLE,
        right="loop",
        note="The harness, not the model, executes the read; a real tool_result (green) enters the column.",
    )
    draw_frame(
        snaps["write_result"],
        FIGURES / "step-04.png",
        TITLE,
        right="files",
        note="Second cycle: write_file lands the real F06 code, todo.py on disk turns amber.",
    )
    draw_frame(
        snaps["tests_result"],
        FIGURES / "step-05.png",
        TITLE,
        right="loop",
        note="Third cycle: run_tests executes for real, and the verbatim result is how the model sees consequences.",
    )
    draw_frame(
        snaps["done"],
        FIGURES / "step-06.png",
        TITLE,
        right="loop",
        note="The model answers in text and the loop closes: model, tool, result, model -- an agent is born.",
    )

    # ---- oracle ----
    assert ws.files()["todo.py"] == todo_source(6)
    run_tests_msgs = [m for m in agent.context if m.tool_name == "run_tests"]
    assert run_tests_msgs and "6 passed" in run_tests_msgs[-1].content
    assert tools.call_counts["read_file"] >= 1
    assert tools.call_counts["write_file"] >= 1
    assert tools.call_counts["run_tests"] >= 1
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, todo.py == todo_source(6), "
        f"tests: {run_tests_msgs[-1].content.strip()!r}. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
