"""The agent loop: one full ORIENT -> ACT -> CHECK -> RECORD lap.

A real (scripted-model) session picks up feature F07 (priority levels).
It reads progress.md (ORIENT), writes a broken implementation (ACT), runs
the real test suite and gets a real red result (CHECK), then -- without the
failed attempt vanishing from context -- writes the real fix (ACT again),
reruns the suite for a real green result (CHECK), and records the win with
a real feature-list flip and a real git commit (RECORD).

Deviation: the brief's literal `.replace('"high", "normal", "low"',
'"high", "low"')` drops "normal" from the validation tuple, but test_f07
never exercises "normal" so that change does not actually fail any test.
This script instead drops "high" (`'"normal", "low"'`), which the existing
test_f07_priority assertion (`run("priority", "1", "high")` must land as
"high") genuinely catches: 6 passed/6 failed before the fix, 7 passed/5
failed after. Same shape of bug (one accepted level lost), a level the
test suite actually checks.
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
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    builtin_tools,
    mark_feature,
    read_features,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The agent loop"
BUDGET = 2800
LOOP_LABELS = {"model": "ORIENT", "hooks": "ACT", "tool": "CHECK", "context": "RECORD"}

BROKEN_F07 = todo_source(7).replace('"high", "normal", "low"', '"normal", "low"')


def main():
    clear(FIGURES)
    ws = build_workspace(stage=6, memory=True)
    tools = builtin_tools(ws)

    model = ScriptedModel(
        [
            Turn(
                text="Checking progress before touching code.",
                tool="read_file",
                args={"path": "progress.md"},
            ),
            Turn(
                text="Building priority levels next: high, normal, low.",
                tool="write_file",
                args={"path": "todo.py", "content": BROKEN_F07},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="The validation tuple lost a level; rewriting it in full.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(7)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="F07 is green and committed. One lap of the loop done."),
        ]
    )
    agent = Agent(model, tools, ws, budget=BUDGET)
    h = Harness(ws)

    snaps = {}
    first_failure_content = {}
    write_n = 0
    test_n = 0

    def on_event(event, msg):
        nonlocal write_n, test_n
        if event == "tool_result" and msg.tool_name == "read_file":
            snaps["orient"] = h.snap("orient", agent, loop_node="model")
        elif event == "tool_result" and msg.tool_name == "write_file":
            write_n += 1
            key = "act1" if write_n == 1 else "act2"
            snaps[key] = h.snap(key, agent, loop_node="hooks")
        elif event == "tool_result" and msg.tool_name == "run_tests":
            test_n += 1
            if test_n == 1:
                first_failure_content["text"] = msg.content
                snaps["check1"] = h.snap("check1", agent, loop_node="tool")
                snaps["retry"] = h.snap("retry", agent, loop_node="tool")
            else:
                snaps["check2"] = h.snap("check2", agent, loop_node="tool")
                mark_feature(ws, "F07", True)
                ws.git_commit("F07: priority levels")
                snaps["record"] = h.snap("record", agent, loop_node="context")
        elif event == "assistant:text":
            snaps["done"] = h.snap("done", agent, loop_node="done")

    agent.run("Add priority levels (F07) to todo.py.", on_event=on_event)

    draw_frame(
        snaps["orient"],
        FIGURES / "step-01.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="ORIENT: the model reads progress.md before touching any code.",
    )
    draw_frame(
        snaps["act1"],
        FIGURES / "step-02.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="ACT: it writes a priority handler, but the validation tuple lost a level.",
    )
    draw_frame(
        snaps["check1"],
        FIGURES / "step-03.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="CHECK: run_tests really fails, 6 passed 6 failed, and that red line stays verbatim in context.",
    )
    draw_frame(
        snaps["retry"],
        FIGURES / "step-04.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="The failed attempt is not erased. The model's own mistake sits in the column for the next lap.",
    )
    draw_frame(
        snaps["act2"],
        FIGURES / "step-05.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="ACT again: the full, correct priority validation is written back to todo.py.",
    )
    draw_frame(
        snaps["check2"],
        FIGURES / "step-06.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="CHECK again: run_tests really passes now, 7 of 12 features green.",
    )
    draw_frame(
        snaps["record"],
        FIGURES / "step-07.png",
        TITLE,
        right="files",
        loop_labels=LOOP_LABELS,
        note="RECORD: feature_list.json flips F07 to passing and a real git commit lands on disk.",
    )
    draw_frame(
        snaps["done"],
        FIGURES / "step-08.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note="One lap closes: orient, act, check, record, and the cost is sitting right there on the gauge.",
    )

    # ---- oracle ----
    assert "failed" in first_failure_content["text"]
    assert "failed" in snaps["check1"].context[-1].content
    run_tests_msgs = [
        m
        for m in agent.context
        if m.tool_name == "run_tests" and m.kind == "tool_result"
    ]
    assert len(run_tests_msgs) == 2
    assert "failed" in run_tests_msgs[0].content
    assert "7 passed" in run_tests_msgs[1].content
    assert any(first_failure_content["text"] in m.content for m in agent.context), (
        "the first failure must still be sitting in context at the end"
    )
    features = {f["id"]: f["passes"] for f in read_features(ws)}
    assert features["F07"] is True
    commit_msgs = [msg for _, msg in ws.git_log()]
    assert any("F07" in m for m in commit_msgs)
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 8, f"expected 8 figures, got {figs}"
    print(
        f"{len(figs)} figures, first run {run_tests_msgs[0].content.strip()!r}, "
        f"second run {run_tests_msgs[1].content.strip()!r}, F07 passing, "
        f"commit log has {len(commit_msgs)} commits. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
