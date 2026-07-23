"""The Ralph loop: agentic laziness meets a driver that will not take "done"
for an answer.

A stop hook named "ralph" really re-reads feature_list.json from disk on
every stop attempt. The scripted model tries to quit three times before all
12 features are real; each time the hook bounces it back with the honest
count and the next unfinished feature id. The model only gets to actually
stop once the oracle, not its own words, says 12/12.

Source: Anthropic's Ralph loop, the "are you REALLY done" recitation.
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
    HookDecision,
    Hooks,
    ScriptedModel,
    Turn,
    builtin_tools,
    mark_feature,
    passing_count,
    read_features,
)
from harnessviz import clear, draw_burndown, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The Ralph loop"


def main():
    clear(FIGURES)
    ws = build_workspace(stage=9, memory=True)

    ralph_calls = {"n": 0}
    history: list[tuple[str, int]] = []

    def ralph_hook(text: str):
        ralph_calls["n"] += 1
        n = ralph_calls["n"]
        feats = read_features(ws)
        k = passing_count(feats)
        history.append((f"iter {n}", k))
        if k < 12:
            unfinished = next(f["id"] for f in feats if not f.get("passes"))
            return HookDecision(
                action="continue_",
                text=(
                    f"Ralph check {n}: feature_list.json shows {k}/12, you are "
                    f"not done. Unfinished: {unfinished}. Keep working."
                ),
            )
        return None  # 12/12 for real: let the stop go through

    hooks = Hooks()
    hooks.on("stop", "ralph", ralph_hook)

    model = ScriptedModel(
        [
            Turn(text="DONE. The todo app is finished."),
            Turn(
                text="Implementing F10.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(10)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="Done now."),
            Turn(
                text="Implementing F11.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(11)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="Done."),
            Turn(
                text="Implementing F12.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(12)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="COMPLETE: all 12 features pass."),
        ]
    )
    agent = Agent(model, builtin_tools(ws), ws, hooks=hooks, budget=20000)
    h = Harness(ws)
    snaps = {}

    def on_event(event, msg):
        if event == "assistant:text" and msg.content.startswith("DONE."):
            snaps["step01"] = h.snap("premature-done", agent, loop_node="model")
        if event == "hook:stop:continue" and "step02" not in snaps:
            snaps["step02"] = h.snap("first-bounce", agent, loop_node="hooks")
        if event == "tool_result":
            if "10 passed" in msg.content:
                mark_feature(ws, "F10", True)
                ws.git_commit("F10: clear-done")
            elif "11 passed" in msg.content:
                mark_feature(ws, "F11", True)
                ws.git_commit("F11: missing-id errors")
                snaps["step03"] = h.snap("mid-grind", agent, loop_node="tool")
            elif "12 passed" in msg.content:
                mark_feature(ws, "F12", True)
                ws.git_commit("F12: usage text")

    agent.run(
        "Build all 12 features in feature_list.json. Stop only when they really pass.",
        on_event=on_event,
    )
    snaps["step04"] = h.snap("finish", agent, loop_node="done")

    draw_frame(
        snaps["step01"],
        FIGURES / "step-01.png",
        TITLE,
        right="files",
        note="The model says done. The disk says 9/12 passing.",
    )
    draw_frame(
        snaps["step02"],
        FIGURES / "step-02.png",
        TITLE,
        right="files",
        note=(
            "The driver recites the feature list back, a budget of "
            "iterations, not faith."
        ),
    )
    draw_frame(
        snaps["step03"],
        FIGURES / "step-03.png",
        TITLE,
        right="files",
        note=(
            "F10 and F11 land for real. Two boxes flip green, two new "
            "commits appear on the log."
        ),
    )
    draw_frame(
        snaps["step04"],
        FIGURES / "step-04.png",
        TITLE,
        right="files",
        note=(
            "The completion promise fires only when the oracle agrees: "
            "12/12, read from disk."
        ),
    )
    draw_burndown(
        history,
        12,
        FIGURES / "step-05.png",
        TITLE,
        note="The loop that beats laziness is a while loop plus an honest file.",
    )

    # ---- oracle ----
    continues = sum(
        1
        for e in agent.hooks.fired
        if e["hook"] == "ralph" and e["action"] == "continue_"
    )
    assert continues == 3, f"ralph must bounce exactly 3 times, got {continues}"
    rc, passed, failed, summary = ws.run_tests()
    assert rc == 0 and passed == 12 and failed == 0, summary
    assert history == [
        ("iter 1", 9),
        ("iter 2", 10),
        ("iter 3", 11),
        ("iter 4", 12),
    ], history
    assert all(history[i][1] < history[i + 1][1] for i in range(len(history) - 1))
    assert "COMPLETE" in agent.done_text, agent.done_text
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {figs}"
    print(
        f"{len(figs)} figures, ralph bounced {continues} times, final board "
        f"{passed}/12, done_text={agent.done_text!r}. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
