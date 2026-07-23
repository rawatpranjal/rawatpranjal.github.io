"""One call, one context: a model call is a pure function of the column.

Four frames of a real (scripted-model) call: the context column is the entire
input, the assistant block is the entire output, and the workspace on disk is
untouched -- proven by diffing the real file tree before and after.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace  # noqa: E402
from harness import Agent, Harness, ScriptedModel, Turn, builtin_tools  # noqa: E402
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "One call, one context"


def main():
    clear(FIGURES)
    ws = build_workspace(stage=5)
    files_before = ws.files()

    model = ScriptedModel(
        [
            Turn(
                text="A priority flag needs three changes: a cmd_priority handler, "
                "a validity check on the level, and a sort key in cmd_list. "
                "I would start with the handler."
            )
        ]
    )
    agent = Agent(model, builtin_tools(ws), ws, budget=400)
    h = Harness(ws)

    # frame 1: before anything -- the column holds only the system prompt
    h.snap("empty", agent, loop_node="context")

    beats = {}

    def on_event(event, msg):
        if event in ("user", "assistant:text"):
            beats[event] = h.snap(
                event, agent, loop_node="context" if event == "user" else "model"
            )

    agent.run("How would you add a priority flag to todo add?", on_event=on_event)
    h.snap("frozen", agent, loop_node="done")

    files_after = ws.files()

    draw_frame(
        h.snapshots[0],
        FIGURES / "step-01.png",
        TITLE,
        note="Before anything happens: the column holds one block, the system prompt.",
        right="loop",
    )
    draw_frame(
        h.snapshots[1],
        FIGURES / "step-02.png",
        TITLE,
        note="The user's message is appended. The column IS the model's entire universe.",
        right="loop",
    )
    draw_frame(
        h.snapshots[2],
        FIGURES / "step-03.png",
        TITLE,
        note="The model answers in text. Look at the disk on other frames: nothing out there changed.",
        right="loop",
    )
    draw_frame(
        h.snapshots[3],
        FIGURES / "step-04.png",
        TITLE,
        note="A model call is a pure function of the column. To act on the world, it needs hands.",
        right="files",
    )

    # ---- oracle ----
    assert files_before == files_after, "a bare model call must not touch the disk"
    assert len(h.snapshots[3].context) == len(h.snapshots[0].context) + 2
    assert h.snapshots[3].total_tokens > h.snapshots[0].total_tokens
    assert model.contexts_seen, "the model really received the context"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"
    print(
        f"{len(figs)} figures, disk untouched ({len(files_before)} files), "
        f"context {h.snapshots[0].total_tokens}->{h.snapshots[3].total_tokens} tok. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
