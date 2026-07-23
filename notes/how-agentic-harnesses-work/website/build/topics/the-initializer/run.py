"""The initializer: session 1 is a DIFFERENT prompt. It builds the
environment, not the app.

Anthropic's initializer-agent pattern: the first session writes the feature
list, the run instructions, the plan, the log, and the tests -- then commits.
It writes zero lines of app code. Every future session wakes into a world
this session built, not a blank directory.

Source: Anthropic, "Effective harnesses for long-running agents" (2025).
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import (  # noqa: E402
    CLAUDE_MD,
    INIT_SH,
    PROGRESS_SEED,
    TESTS_SRC,
    feature_list_json,
)
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    Workspace,
    builtin_tools,
    read_features,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The initializer"
ASK = (
    "Build todo.py, a CLI todo app with add, list, done, delete, priorities, "
    "due dates, search."
)


def main():
    clear(FIGURES)
    ws = Workspace()  # bare: nothing on disk, not even git
    tools = builtin_tools(ws)
    h = Harness(ws)

    model = ScriptedModel(
        [
            Turn(
                text="Session 1 builds the environment, not the app. Writing "
                "the feature list first, as JSON: models mangle markdown "
                "structure far more than they mangle JSON.",
                tool="write_file",
                args={"path": "feature_list.json", "content": feature_list_json()},
            ),
            Turn(
                text="Next, how to run this project -- written once so no "
                "future session has to rediscover it.",
                tool="write_file",
                args={"path": "init.sh", "content": INIT_SH},
            ),
            Turn(
                text="Now the plan and the log: progress.md starts the "
                "append-only history, CLAUDE.md carries the standing rules.",
                tool="write_file",
                args={"path": "progress.md", "content": PROGRESS_SEED},
            ),
            Turn(
                tool="write_file",
                args={"path": "CLAUDE.md", "content": CLAUDE_MD},
            ),
            Turn(
                text="Tests before app. Writing the oracle now so it exists "
                "before a single line of todo.py does, and so it cannot be "
                "quietly loosened later.",
                tool="write_file",
                args={"path": "tests/test_todo.py", "content": TESTS_SRC},
            ),
            Turn(
                text="Environment ready: feature list, run instructions, "
                "plan, log, and tests are on disk. No app code -- that is "
                "for the next session."
            ),
        ]
    )
    agent = Agent(model, tools, ws, budget=6000, name="initializer")

    snaps = {}

    def on_event(event, msg):
        if event == "user":
            snaps["ask"] = h.snap("ask", agent, loop_node="context")
        if event == "tool_result":
            if msg.content.startswith("wrote feature_list.json"):
                snaps["features"] = h.snap("features", agent, loop_node="tool")
            elif msg.content.startswith("wrote init.sh"):
                snaps["init"] = h.snap("init", agent, loop_node="tool")
            elif msg.content.startswith("wrote CLAUDE.md"):
                snaps["plan_log"] = h.snap("plan_log", agent, loop_node="tool")
            elif msg.content.startswith("wrote tests/test_todo.py"):
                snaps["tests"] = h.snap("tests", agent, loop_node="tool")

    agent.run(ASK, on_event=on_event)

    # the wrap-up: git for real, no app code involved
    ws.git_init()
    ws.git_commit("initializer: environment ready")
    snaps["git"] = h.snap("git", agent, loop_node="done")

    # ---- frames ----
    draw_frame(
        snaps["ask"],
        FIGURES / "step-01.png",
        TITLE,
        note="Nothing exists yet. The disk is empty; only the ask is in the column.",
        right="files",
    )
    draw_frame(
        snaps["features"],
        FIGURES / "step-02.png",
        TITLE,
        note="feature_list.json, 12 items, all red. JSON beats markdown here: "
        "models mangle markdown structure, not JSON schema fields.",
        right="files",
        show_content=("feature_list.json",),
    )
    draw_frame(
        snaps["init"],
        FIGURES / "step-03.png",
        TITLE,
        note="init.sh: how to run the project, written once, so no session re-derives it.",
        right="files",
        show_content=("init.sh",),
    )
    draw_frame(
        snaps["plan_log"],
        FIGURES / "step-04.png",
        TITLE,
        note="progress.md (the log) and CLAUDE.md (the plan and its rules) land together.",
        right="files",
        show_content=("progress.md", "CLAUDE.md"),
    )
    draw_frame(
        snaps["tests"],
        FIGURES / "step-05.png",
        TITLE,
        note="tests/test_todo.py: the oracle is part of the environment, written before the app, editing it forbidden.",
        right="files",
        show_content=("tests/test_todo.py",),
    )
    draw_frame(
        snaps["git"],
        FIGURES / "step-06.png",
        TITLE,
        note="First commit. The initializer wrote no app code at all -- it built the world every future session wakes up in.",
        right="files",
    )

    # ---- oracle ----
    assert not ws.exists("todo.py"), "the initializer must write zero app code"
    features = read_features(ws)
    assert len(features) == 12 and all(f["passes"] is False for f in features), (
        "feature_list.json must have 12 items, all unpassed"
    )
    commits = ws.git_log()
    assert len(commits) == 1, f"expected exactly 1 commit, got {commits}"
    rc, passed, failed, summary = ws.run_tests()
    # todo.py does not exist, so the suite cannot even import -- the honest
    # starting truth is 0 passed and a real, immediate failure.
    assert rc != 0 and passed == 0, f"expected a real failure, 0 passed, got {summary}"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, todo.py absent, 12/12 features red, "
        f"{len(commits)} commit, tests {summary}. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
