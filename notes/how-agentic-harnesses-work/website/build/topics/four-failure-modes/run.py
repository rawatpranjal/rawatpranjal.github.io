"""Four failure modes: Anthropic's effective-harnesses table, staged and fixed.

Four scenes, each two frames: a failure that really happens on disk, then a
fix that really repairs it. A fresh build_workspace per scene, cleaned up
after. The one scripted part in every scene is the model; everything it
touches (files, git, hooks, tests) is real.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace, run_one_test, todo_source  # noqa: E402
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
    total_tokens,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Four failure modes"


class OnceBounce:
    """A stop hook that bounces exactly once, then lets the session end."""

    def __init__(self, text: str):
        self.text = text
        self.fired = False

    def __call__(self, text: str):
        if not self.fired:
            self.fired = True
            return HookDecision("continue_", text=self.text)
        return HookDecision()


# ---- scene 1: premature victory ---------------------------------------------


def scene1():
    ws = build_workspace(stage=5)
    tools = builtin_tools(ws)
    h = Harness(ws)

    model_a = ScriptedModel([Turn(text="All done! The todo app is complete.")])
    agent_a = Agent(model_a, tools, ws, budget=800, name="session-a")
    agent_a.run("Status check: is todo.py finished?")
    snap1 = h.snap("victory-claim", agent_a, loop_node="done")
    draw_frame(
        snap1,
        FIGURES / "step-01.png",
        TITLE,
        right="files",
        note="The session declares victory in text. The board underneath still "
        "reads 5 of 12 features passing -- the claim and the disk disagree.",
    )

    hooks = Hooks()
    once = OnceBounce("feature_list.json: 7 features still failing")
    hooks.on("stop", "feature-gate", once)
    model_b = ScriptedModel(
        [
            Turn(text="All done! The todo app is complete."),
            Turn(text="Understood: 7 features are still failing. Correcting course."),
        ]
    )
    agent_b = Agent(model_b, tools, ws, hooks=hooks, budget=800, name="session-b")
    agent_b.run("Status check: is todo.py finished?")
    snap2 = h.snap("bounced", agent_b, loop_node="done")
    draw_frame(
        snap2,
        FIGURES / "step-02.png",
        TITLE,
        right="files",
        note="Same disk, same 7 red boxes -- unarguable. A stop hook reads the "
        "feature list, not the claim, and bounces the session for real.",
    )

    return {
        "passing": passing_count(read_features(ws)),
        "claim_text": agent_a.done_text,
        "bounced": any(e["action"] == "continue_" for e in agent_b.hooks.fired),
        "ws": ws,
    }


# ---- scene 2: dirty state -----------------------------------------------------


def scene2():
    ws = build_workspace(stage=7)
    broken = todo_source(7).replace("save(items)", "save(item)", 1)
    ws.write("todo.py", broken)
    dirty_before = ws.git_dirty()

    tools = builtin_tools(ws)
    h = Harness(ws)

    empty_agent = Agent(ScriptedModel([]), tools, ws, budget=1500, name="session-2")
    snap3 = h.snap("dirty", empty_agent, loop_node="none")
    draw_frame(
        snap3,
        FIGURES / "step-03.png",
        TITLE,
        right="files",
        note="The previous session died leaving broken code and no note. This "
        "session wakes to a clean-looking stage 7 that does not actually run.",
    )

    model = ScriptedModel(
        [
            Turn(
                text="Smoke test before touching anything.",
                tool="run_tests",
                args={},
            ),
            Turn(
                text="save(item) typo in the header. Restoring the clean "
                "stage-7 source.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(7)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="7 of 12 passing again. Committing the clean state."),
        ]
    )
    agent2 = Agent(model, tools, ws, budget=1500, name="session-2")
    run_results = []

    def on_event(event, msg):
        if event == "tool_result" and msg.tool_name == "run_tests":
            m = re.search(r"(\d+) passed, (\d+) failed", msg.content)
            run_results.append((int(m.group(1)), int(m.group(2))))

    agent2.run("Continue building todo.py.", on_event=on_event)
    ws.git_commit("fix: repair todo.py, restore stage 7 save(items)")
    dirty_after = ws.git_dirty()
    snap4 = h.snap("fixed", agent2, loop_node="done")
    draw_frame(
        snap4,
        FIGURES / "step-04.png",
        TITLE,
        right="files",
        note="Smoke test at the start, commit at the end. The new commit dot "
        "is the only thing the next session inherits -- state stays clean.",
    )

    return {
        "dirty_before": dirty_before,
        "dirty_after": dirty_after,
        "run_results": run_results,
        "ws": ws,
    }


# ---- scene 3: fake done --------------------------------------------------------


def scene3():
    ws = build_workspace(stage=8)
    mark_feature(ws, "F09", True)

    tools = builtin_tools(ws)
    h = Harness(ws)

    model_claim = ScriptedModel(
        [Turn(text="F09 due dates: implemented and marked passing.")]
    )
    agent_claim = Agent(model_claim, tools, ws, budget=800, name="fake-done")
    agent_claim.run("Status on F09?")
    snap5 = h.snap("fake-green", agent_claim, loop_node="done")
    feats_at_claim = read_features(ws)
    f09_marked = next(f for f in feats_at_claim if f["id"] == "F09")["passes"]
    ok5, tail5 = run_one_test(ws, "F09")
    draw_frame(
        snap5,
        FIGURES / "step-05.png",
        TITLE,
        right="files",
        note=f"The board shows F09 green. run_one_test disagrees: {tail5}. "
        "The box is green, the truth is red.",
    )

    mark_feature(ws, "F09", False)
    ws.write("todo.py", todo_source(9))
    ok6, tail6 = run_one_test(ws, "F09")
    mark_feature(ws, "F09", True)

    model_verified = ScriptedModel(
        [
            Turn(
                text=f"Reverted the claim, implemented due dates for real, "
                f"verified end-to-end: {tail6}. Now marking F09 passing."
            )
        ]
    )
    agent_verified = Agent(model_verified, tools, ws, budget=800, name="verified-done")
    agent_verified.run("Status on F09?")
    snap6 = h.snap("verified", agent_verified, loop_node="done")
    feats_final = read_features(ws)
    f09_final = next(f for f in feats_final if f["id"] == "F09")["passes"]
    draw_frame(
        snap6,
        FIGURES / "step-06.png",
        TITLE,
        right="files",
        note="Verification before marking: implement, rerun the oracle, only "
        "then flip the box -- exactly what a human user would check.",
    )

    return {
        "f09_marked_at_claim": f09_marked,
        "ok5": ok5,
        "ok6": ok6,
        "f09_final": f09_final,
        "ws": ws,
    }


# ---- scene 4: wasted warmup ----------------------------------------------------


def scene4():
    ws = build_workspace(stage=9)
    ws.path("init.sh").unlink()

    tools = builtin_tools(ws)
    h = Harness(ws)

    model_warm = ScriptedModel(
        [
            Turn(
                text="No instructions file here. Let's look around.",
                tool="bash",
                args={"command": "ls"},
            ),
            Turn(tool="read_file", args={"path": "tests/test_todo.py"}),
            Turn(tool="bash", args={"command": "python todo.py list"}),
            Turn(tool="run_tests", args={}),
            Turn(
                text="OK, now I understand the project. 9 of 12 features are "
                "already passing."
            ),
        ]
    )
    agent_warm = Agent(model_warm, tools, ws, budget=4000, name="warmup")
    ready = {}

    def on_event_warm(event, msg):
        if event == "tool_result" and msg.tool_name == "run_tests":
            ready["warm"] = total_tokens(agent_warm.context)

    agent_warm.run("Continue building todo.py.", on_event=on_event_warm)
    snap7 = h.snap("warmup-done", agent_warm, loop_node="done")
    draw_frame(
        snap7,
        FIGURES / "step-07.png",
        TITLE,
        right="none",
        note=f"{ready['warm']} tokens burned re-discovering how to run the "
        "project (ls, read the tests, run it by hand) before any real work.",
    )

    ws.write(
        "init.sh",
        "# how to run this project -- read me first, don't rediscover it\n"
        "python -m unittest discover -s tests -v\n",
    )

    model_fix = ScriptedModel(
        [
            Turn(
                text="Reading init.sh first.",
                tool="read_file",
                args={"path": "init.sh"},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="9 of 12 features passing, ready to continue."),
        ]
    )
    agent_fix = Agent(model_fix, tools, ws, budget=4000, name="fixed")
    ready_fix = {}

    def on_event_fix(event, msg):
        if event == "tool_result" and msg.tool_name == "run_tests":
            ready_fix["fixed"] = total_tokens(agent_fix.context)

    agent_fix.run("Continue building todo.py.", on_event=on_event_fix)
    snap8 = h.snap("fixed-done", agent_fix, loop_node="done")
    draw_frame(
        snap8,
        FIGURES / "step-08.png",
        TITLE,
        right="none",
        note=f"{ready_fix['fixed']} tokens to reach the same point: init.sh "
        "said how to run the tests, so the session just ran them.",
    )

    return {"warm_tokens": ready["warm"], "fixed_tokens": ready_fix["fixed"], "ws": ws}


def main():
    clear(FIGURES)

    r1 = scene1()
    r2 = scene2()
    r3 = scene3()
    r4 = scene4()

    # ---- oracle: scene 1, premature victory ----
    assert r1["passing"] == 5, r1["passing"]
    assert "complete" in r1["claim_text"].lower()
    assert r1["bounced"], "the stop hook must really have bounced the session"

    # ---- oracle: scene 2, dirty state ----
    assert r2["dirty_before"] is True
    assert r2["dirty_after"] is False
    assert len(r2["run_results"]) == 2
    assert r2["run_results"][0][0] == 0, r2["run_results"]
    assert r2["run_results"][1][0] == 7, r2["run_results"]

    # ---- oracle: scene 3, fake done ----
    assert r3["f09_marked_at_claim"] is True
    assert r3["ok5"] is False
    assert r3["ok6"] is True
    assert r3["f09_final"] is True

    # ---- oracle: scene 4, wasted warmup ----
    assert r4["warm_tokens"] > r4["fixed_tokens"], (
        r4["warm_tokens"],
        r4["fixed_tokens"],
    )

    for r in (r1, r2, r3, r4):
        r["ws"].cleanup()

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 8, f"expected 8 figures, got {figs}"
    print(
        f"{len(figs)} figures. victory 5/12 + bounced, dirty True->False "
        f"(0->7 passed), fake F09 True/False->True/True, warmup "
        f"{r4['warm_tokens']}->{r4['fixed_tokens']} tok. All checks passed."
    )


if __name__ == "__main__":
    main()
