"""Memory lives on disk: a session ends and its context dies, but four
channels survive: git history, progress.md, feature_list.json, AGENTS.md.

Session A finishes F06 for real (writes todo.py, runs the tests, marks the
feature, appends a progress line that names a dead end, appends a gotcha,
commits). Session B is a brand-new Agent -- an empty context -- that boots by
reading those four channels back off disk. It does not remember anything;
the repo does.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import FAILED_APPROACH_LINE, build_workspace, todo_source  # noqa: E402
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
TITLE = "Memory lives on disk"

GOTCHA_LINE = (
    "- search matches substrings case-insensitively; forgetting .lower() on "
    "both sides of the compare breaks test_f06_search."
)


def main():
    clear(FIGURES)
    ws = build_workspace(stage=5, memory=True)
    tools = builtin_tools(ws)
    h = Harness(ws)

    # ---- session A: finishes F06, then the context ends ----
    modelA = ScriptedModel(
        [
            Turn(
                text="Implementing search: F06 needs a cmd_search handler.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(6)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="F06 verified end-to-end, 6/12 passing. Updating progress, "
                "noting a gotcha, and committing before this context goes away."
            ),
        ]
    )
    agentA = Agent(modelA, tools, ws, budget=3000, name="session-A")
    agentA.run("Continue building todo.py: implement F06 (search).")

    # step-01: the session's final context, full of work, about to be destroyed
    h.snap("session-a-final", agentA, loop_node="done")
    draw_frame(
        h.snapshots[0],
        FIGURES / "step-01.png",
        TITLE,
        note="Session A just finished F06. This column is about to be destroyed.",
        right="loop",
    )

    # ---- four channels, lit one per frame, each snap right after its write ----
    ws.git_commit("F06: search command")
    h.snap("git", agentA, loop_node="done")
    draw_frame(
        h.snapshots[1],
        FIGURES / "step-02.png",
        TITLE,
        note="Channel 1: git history. The code and green tests are now a real commit, forever.",
        right="files",
    )

    new_progress = ws.read("progress.md") + "\n" + FAILED_APPROACH_LINE + "\n"
    ws.write("progress.md", new_progress)
    h.snap("progress", agentA, loop_node="done")
    draw_frame(
        h.snapshots[2],
        FIGURES / "step-03.png",
        TITLE,
        note="Channel 2: progress.md, append-only. The failed approach is logged so no future session retries it.",
        right="files",
        show_content=("progress.md",),
    )

    mark_feature(ws, "F06", True)
    h.snap("feature", agentA, loop_node="done")
    draw_frame(
        h.snapshots[3],
        FIGURES / "step-04.png",
        TITLE,
        note="Channel 3: feature_list.json, one flag flipped. The board now really reads 6/12.",
        right="files",
    )

    new_agents = ws.read("AGENTS.md") + GOTCHA_LINE + "\n"
    ws.write("AGENTS.md", new_agents)
    h.snap("agents", agentA, loop_node="done")
    draw_frame(
        h.snapshots[4],
        FIGURES / "step-05.png",
        TITLE,
        note="Channel 4: AGENTS.md, the gotcha. The next session inherits this the moment it reads the file.",
        right="files",
        show_content=("AGENTS.md",),
    )

    # ---- session B: a brand-new Agent, fresh context, boots from disk ----
    modelB = ScriptedModel(
        [
            Turn(
                text="Fresh session. Reading progress.md, AGENTS.md, "
                "feature_list.json, and git log before touching anything.",
                tool="read_file",
                args={"path": "progress.md"},
            ),
            Turn(tool="read_file", args={"path": "AGENTS.md"}),
            Turn(tool="read_file", args={"path": "feature_list.json"}),
            Turn(tool="bash", args={"command": "git log"}),
            Turn(
                text="F06 landed, one dead end noted, one gotcha noted, 6/12 "
                "passing. Continuing from F07."
            ),
        ]
    )
    agentB = Agent(modelB, builtin_tools(ws), ws, budget=3000, name="session-B")
    agentB.run("Continue building todo.py.")
    h.snap("session-b-booted", agentB, loop_node="done")
    draw_frame(
        h.snapshots[5],
        FIGURES / "step-06.png",
        TITLE,
        note="The agent did not get smarter. The repo did. The next agent just reads it.",
        right="loop",
    )

    # ---- oracle ----
    contextB_text = "\n".join(m.content for m in agentB.context)
    assert FAILED_APPROACH_LINE in contextB_text, (
        "session B must read back the failed-approach line"
    )
    assert GOTCHA_LINE in contextB_text, "session B must read back the gotcha"
    commits = ws.git_log()
    assert len(commits) == 2, (
        f"expected 2 commits total (1 scaffold + 1 F06), got {commits}"
    )
    features = read_features(ws)
    f06 = next(f for f in features if f["id"] == "F06")
    assert f06["passes"] is True, "F06 must pass on disk"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, git log grew to {len(commits)} commits, "
        f"F06 passes={f06['passes']}, session B context carries both memory lines. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
