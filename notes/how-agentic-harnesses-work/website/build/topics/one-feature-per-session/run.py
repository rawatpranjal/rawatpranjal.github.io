"""One feature per session: the get-your-bearings ritual, run for real.

Session S7 wakes into a repo a previous session left dirty (todo.py is
genuinely broken). It gets its bearings, picks the single highest-priority
unfinished feature, confirms the app is broken before touching anything,
repairs it, implements the one feature, verifies end-to-end, flips exactly
one flag, then commits and logs -- leaving the repo mergeable for whoever
reads it next.

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

from fixtures import build_workspace, run_one_test, todo_source  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    builtin_tools,
    mark_feature,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "One feature per session"
PROGRESS_LINE = "2026-01-07 S7: F07 done, tests 7/12."


def main():
    clear(FIGURES)
    ws = build_workspace(stage=6, memory=True, git=True)
    # a previous session left todo.py genuinely broken, uncommitted
    broken = todo_source(6).replace("items.append", "itemsappend", 1)
    ws.write("todo.py", broken)
    commits_before = len(ws.git_log())

    tools = builtin_tools(ws)
    h = Harness(ws)

    model = ScriptedModel(
        [
            Turn(text="Getting my bearings.", tool="bash", args={"command": "pwd"}),
            Turn(tool="read_file", args={"path": "progress.md"}),
            Turn(
                text="Checking feature_list.json for the next priority.",
                tool="read_file",
                args={"path": "feature_list.json"},
            ),
            Turn(
                text="F01-F06 are marked done. F07 (priority levels) is the "
                "next unfinished feature -- one feature this session, not many.",
                tool="run_tests",
                args={},
            ),
            Turn(
                text="The smoke test fails. The app is broken before I've "
                "changed a line -- fixing that first.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(6)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="Base app confirmed working again. Implementing F07: priority levels.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(7)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text="F07 shipped: priority levels are in, 7/12 features "
                "passing. Committing and logging."
            ),
        ]
    )
    agent = Agent(model, tools, ws, budget=6000, name="S7")

    n = [0]
    extra = {}

    def on_event(event, msg):
        if event == "tool_result":
            n[0] += 1
            if n[0] == 2:  # pwd + progress.md read
                h.snap("bearings", agent, loop_node="tool")
            elif n[0] == 3:  # feature_list.json read
                h.snap("priority", agent, loop_node="tool")
            elif n[0] == 4:  # first run_tests: really fails
                h.snap("smoke-fail", agent, loop_node="tool")
            elif n[0] == 6:  # run_tests after the fix: really 6 passed
                h.snap("repaired", agent, loop_node="tool")
            elif n[0] == 7:  # todo.py written with F07
                h.snap("implemented", agent, loop_node="tool")
            elif n[0] == 8:  # run_tests after F07: really 7 passed
                h.snap("verify-tests", agent, loop_node="tool")
                ok, tail = run_one_test(ws, "F07")
                extra["run_one_test"] = tail
                mark_feature(ws, "F07", True)
                h.snap("flip", agent, loop_node="tool")
        if event == "assistant:text":
            ws.git_commit("F07: priority levels")
            ws.write(
                "progress.md", ws.read("progress.md") + "\n" + PROGRESS_LINE + "\n"
            )
            h.snap("wrap-up", agent, loop_node="done")

    agent.run("Continue building todo.py.", on_event=on_event)

    snaps = h.snapshots

    # ---- frames ----
    draw_frame(
        snaps[0],
        FIGURES / "step-01.png",
        TITLE,
        note="Step one of the ritual: where am I, what happened before. pwd, then read progress.md.",
        right="files",
        show_content=("progress.md",),
    )
    draw_frame(
        snaps[1],
        FIGURES / "step-02.png",
        TITLE,
        note="feature_list.json read. F01-F06 done; F07 is next. ONE feature per session, not many.",
        right="files",
    )
    draw_frame(
        snaps[2],
        FIGURES / "step-03.png",
        TITLE,
        note="Check the app basically works BEFORE building anything new. It does not: the smoke test really fails.",
        right="loop",
    )
    draw_frame(
        snaps[3],
        FIGURES / "step-04.png",
        TITLE,
        note="Fix first: clean todo.py restores the baseline. Tests really run green again (6/12). Repair the world before adding to it.",
        right="files",
    )
    draw_frame(
        snaps[4],
        FIGURES / "step-05.png",
        TITLE,
        note="Implement the one feature: F07, priority levels. todo.py changes again.",
        right="files",
    )
    draw_frame(
        snaps[5],
        FIGURES / "step-06.png",
        TITLE,
        note=f"Verify end-to-end: run_one_test(F07) -- {extra['run_one_test']}. run_tests also confirms 7/12 in context.",
        right="loop",
    )
    draw_frame(
        snaps[6],
        FIGURES / "step-07.png",
        TITLE,
        note="Flip exactly one field: F07 red to green, and only after end-to-end verification. Never edit a description.",
        right="files",
    )
    draw_frame(
        snaps[7],
        FIGURES / "step-08.png",
        TITLE,
        note="Commit, log, done. The session leaves the repo mergeable; the next session starts clean.",
        right="files",
        show_content=("progress.md",),
    )

    # ---- oracle ----
    run_tests_results = [
        m.content
        for m in agent.context
        if m.tool_name == "run_tests" and m.kind == "tool_result"
    ]
    assert len(run_tests_results) == 3, run_tests_results
    assert run_tests_results[0].startswith("ERROR"), (
        "the first run_tests must really fail"
    )
    assert "6 passed" in run_tests_results[1], run_tests_results[1]
    assert "7 passed" in run_tests_results[2], run_tests_results[2]

    feats_step02 = {f["id"]: f["passes"] for f in snaps[1].features}
    feats_step07 = {f["id"]: f["passes"] for f in snaps[6].features}
    changed = [fid for fid in feats_step02 if feats_step02[fid] != feats_step07[fid]]
    assert changed == ["F07"], f"expected only F07 to flip, got {changed}"

    commits_after = len(ws.git_log())
    assert commits_after == commits_before + 1, (
        f"expected git log to grow by 1, {commits_before} -> {commits_after}"
    )
    assert ws.read("progress.md").rstrip().splitlines()[-1] == PROGRESS_LINE

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 8, f"expected 8 figures, got {figs}"
    print(
        f"{len(figs)} figures, run_tests really failed then 6 then 7 passed, "
        f"only F07 flipped, git log {commits_before}->{commits_after}, "
        "progress.md ends with the S7 line. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
