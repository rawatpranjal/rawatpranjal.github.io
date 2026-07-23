"""Executor vs verifier: the producer never grades its own work.

An executor claims a feature is done and marks it green while its own test
genuinely fails. Only artifacts (a diff, real tool output) cross to a second,
fresh agent with zero shared context, and that verifier reruns the oracle
instead of reading the claim -- catching the fake green for real.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace, run_one_test, todo_source, unified_diff  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    builtin_tools,
    mark_feature,
    read_features,
)
from harnessviz import clear, draw_diff_verdict, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Executor vs verifier"

CLAIM = "F10 clear-done implemented and verified, all good."


def f10_passes(ws) -> bool:
    return next(f for f in read_features(ws) if f["id"] == "F10")["passes"]


def _clean_pycache(ws):
    """run_one_test (unlike the run_tests tool) does not set
    PYTHONDONTWRITEBYTECODE, so it leaves __pycache__ dirs on disk. Sweep
    them so the files panel photographs the project, not python's cache."""
    for p in ws.root.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


def main():
    clear(FIGURES)
    ws = build_workspace(stage=9)
    tools = builtin_tools(ws)
    h = Harness(ws)

    src9 = todo_source(9)
    src10 = todo_source(10)
    # A realistic clear-done bug: filters the right list, then saves the
    # wrong one -- the completed task is never actually dropped.
    old_block = (
        "def cmd_clear_done(args):\n"
        '    items = [i for i in load() if not i["done"]]\n'
        "    save(items)"
    )
    new_block = (
        "def cmd_clear_done(args):\n"
        '    items = [i for i in load() if not i["done"]]\n'
        "    save(load())"
    )
    assert src10.count(old_block) == 1
    broken = src10.replace(old_block, new_block, 1)

    # ---- step 1: the executor finishes, confident, and marks green ----
    model_exec1 = ScriptedModel(
        [
            Turn(
                text="Implementing F10: clear-done.",
                tool="write_file",
                args={"path": "todo.py", "content": broken},
            ),
            Turn(text=CLAIM),
        ]
    )
    agent_exec1 = Agent(model_exec1, tools, ws, budget=1500, name="executor")
    agent_exec1.run("Implement F10: todo clear-done.")
    mark_feature(ws, "F10", True)
    passes_after_claim = f10_passes(ws)
    ok_fake, _tail_fake = run_one_test(ws, "F10")
    _clean_pycache(ws)

    snap1 = h.snap("executor-claim", agent_exec1, loop_node="done")
    draw_frame(
        snap1,
        FIGURES / "step-01.png",
        TITLE,
        right="files",
        note="A confident claim and a green box. The executor never reruns "
        "the oracle on its own work.",
    )

    # ---- step 2: only artifacts cross the firewall ----
    diff = unified_diff(src9, broken, "todo.py")
    draw_diff_verdict(
        diff,
        FIGURES / "step-02.png",
        TITLE,
        note="Only artifacts cross: the diff, the files, the tests. The "
        "executor's narrative stays behind.",
    )

    # ---- step 3: a fresh verifier, its own agent, zero shared context ----
    tools_verify = builtin_tools(ws)
    model_verify = ScriptedModel(
        [
            Turn(tool="run_tests", args={}),
            Turn(
                text="VERDICT: FAIL. clear-done never removes the completed "
                "task, test_f10 fails."
            ),
        ]
    )
    agent_verify = Agent(model_verify, tools_verify, ws, budget=1500, name="verifier")
    verify_evidence = {}

    def on_event_verify(event, msg):
        if event == "tool_result" and msg.tool_name == "run_tests":
            verify_evidence["line"] = msg.content

    agent_verify.run(diff, on_event=on_event_verify)
    snap3 = h.snap("verifier-verdict", agent_verify, loop_node="done")
    draw_frame(
        snap3,
        FIGURES / "step-03.png",
        TITLE,
        right="loop",
        note="Fresh eyes, zero shared context. It reruns the oracle instead "
        "of reading the claim.",
    )

    # ---- step 4: the fake green is caught, the mark reverts ----
    mark_feature(ws, "F10", False)
    passes_after_revert = f10_passes(ws)
    draw_diff_verdict(
        diff,
        FIGURES / "step-04.png",
        TITLE,
        verdict="FAIL",
        evidence=verify_evidence["line"],
        note="The fake green is caught. The box flips back.",
    )

    # ---- step 5: retry, this time with the verifier's evidence in context ----
    feedback = agent_verify.done_text
    model_exec2 = ScriptedModel(
        [
            Turn(
                text="Fixing F10 based on verifier feedback.",
                tool="write_file",
                args={"path": "todo.py", "content": src10},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="F10 clear-done fixed. 10 of 12 tests passing."),
        ]
    )
    agent_exec2 = Agent(model_exec2, tools, ws, budget=1500, name="executor-2")
    agent_exec2.run(f"Verifier feedback: {feedback}\n\nFix F10 clear-done.")
    ok_real, _tail_real = run_one_test(ws, "F10")
    _clean_pycache(ws)
    mark_feature(ws, "F10", True)
    passes_final = f10_passes(ws)

    snap5 = h.snap("executor2-done", agent_exec2, loop_node="done")
    draw_frame(
        snap5,
        FIGURES / "step-05.png",
        TITLE,
        right="files",
        note="Session 2 carries the failure in context, writes the real "
        "fix, and this time the test genuinely passes before anyone marks "
        "the box.",
    )

    # ---- step 6: the box flips, and it was never the author who flipped it ----
    rc_final, passed_final, failed_final, summary_final = ws.run_tests()
    draw_diff_verdict(
        unified_diff(src9, src10, "todo.py"),
        FIGURES / "step-06.png",
        TITLE,
        verdict="PASS",
        evidence=f"test_f10_clear_done: OK ({summary_final})",
        note="Only now does the box flip, and it was never the author who flipped it.",
    )

    # ---- oracle ----
    assert passes_after_claim is True, "the fake claim must really be marked green"
    assert ok_fake is False, "the broken write's own test must really fail"
    assert passes_after_revert is False, "the mark must really revert on FAIL"
    assert ok_real is True, "the corrected write's test must really pass"
    assert passes_final is True
    assert rc_final != 0, "only 10 of 12 stages exist, the suite is not fully green"
    assert passed_final == 10, f"expected 10 passing, got {summary_final}"
    assert CLAIM not in [m.content for m in agent_verify.context], (
        "the verifier's context must never contain the executor's narrative claim"
    )

    ws.cleanup()

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures. fake claim marked green while F10 test failed "
        f"({ok_fake}), verifier caught it and reverted the mark, retry passed "
        f"({summary_final}), claim string never reached the verifier. "
        "All checks passed."
    )


if __name__ == "__main__":
    main()
