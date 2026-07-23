"""Hooks own the boundary: five real hooks mirroring a real Claude Code
settings.json, wired around a scripted model that tries a destructive command,
a real write, a premature stop, and a real test run.

Every block, inject, and bounce below is a real HookDecision returned by a
real callback and really enforced by Agent.run: the deny genuinely prevents
tools.execute from ever being called (proven by a call-count assert on the
bash tool), the stop hook genuinely bounces the model back into the loop, and
the final "done" text is what the model says only after the hook lets it go.
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
    passing_count,
    read_features,
)
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Hooks own the boundary"


def main():
    clear(FIGURES)
    ws = build_workspace(stage=8)  # F01-F08 already passing, F09-F12 left

    tools = builtin_tools(ws)
    hooks = Hooks()

    # 1. session_start "orient" -- injects a directive on every fresh session.
    def orient():
        return HookDecision(
            action="inject",
            text="SessionStart: read CLAUDE.md and feature_list.json before acting.",
        )

    hooks.on("session_start", "orient", orient)

    # 2. user_prompt_submit "goal-reminder" -- mirrors the real goal-reminder.sh:
    # fires only when the prompt names /goal.
    def goal_reminder(prompt):
        if "/goal" in prompt:
            return HookDecision(
                action="inject",
                text="goal-reminder: /goal sets a completion condition a fast "
                "model checks every turn, keep going until it holds.",
            )
        return HookDecision()

    hooks.on("user_prompt_submit", "goal-reminder", goal_reminder)

    # 3. pre_tool_use "protect" -- denies destructive bash and secret writes.
    def protect(tool, args):
        if tool == "bash" and "rm -rf" in args.get("command", ""):
            return HookDecision(
                action="deny", reason="protect: rm -rf is destructive, blocked"
            )
        if tool == "write_file" and args.get("path") == ".env":
            return HookDecision(
                action="deny", reason="protect: .env is protected, write blocked"
            )
        return HookDecision()

    hooks.on("pre_tool_use", "protect", protect)

    # 4. post_tool_use "lint" -- injects a check straight into the tool result.
    def lint(tool, result):
        if tool == "write_file":
            return HookDecision(
                action="inject", text="lint: line length ok, imports sorted"
            )
        return HookDecision()

    hooks.on("post_tool_use", "lint", lint)

    # 5. stop "not-done" -- bounces once via a closure counter, then lets go,
    # so the run terminates instead of looping forever on an unfinished board.
    state = {"fired": False}

    def not_done(text):
        if not state["fired"] and passing_count(read_features(ws)) < 12:
            state["fired"] = True
            return HookDecision(
                action="continue_",
                text="Stop hook: feature_list.json shows unfinished work, keep going.",
            )
        return HookDecision()

    hooks.on("stop", "not-done", not_done)

    model = ScriptedModel(
        [
            Turn(
                text="Reading CLAUDE.md to orient before touching anything.",
                tool="read_file",
                args={"path": "CLAUDE.md"},
            ),
            Turn(
                text="Clearing stale test artifacts first.",
                tool="bash",
                args={"command": "rm -rf tests"},
            ),
            Turn(
                text="Implementing F09: attach a due date to a task.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(9)},
            ),
            Turn(text="Stopping here."),
            Turn(text="Verifying end to end.", tool="run_tests", args={}),
            Turn(text="F09 done for real."),
        ]
    )
    agent = Agent(model, tools, ws, hooks=hooks, budget=4000)
    h = Harness(ws)
    registered = hooks.registered()
    snaps = {}
    tool_result_n = {"n": 0}

    def on_event(event, msg):
        if event == "user":
            snaps["01"] = h.snap("step01", agent, loop_node="context")
        elif event == "tool_result":
            tool_result_n["n"] += 1
            if tool_result_n["n"] == 1:
                snaps["02"] = h.snap("step02", agent, loop_node="tool")
            elif tool_result_n["n"] == 2:
                snaps["04"] = h.snap("step04", agent, loop_node="tool")
            elif tool_result_n["n"] == 3:
                snaps["06"] = h.snap("step06", agent, loop_node="tool")
        elif event == "hook:pre_tool_use:deny":
            snaps["03"] = h.snap("step03", agent, loop_node="hooks")
        elif event == "hook:stop:continue":
            snaps["05"] = h.snap("step05", agent, loop_node="hooks")

    agent.run(
        "/goal finish F09: attach due dates and verify end to end.", on_event=on_event
    )
    snaps["07"] = h.snap("step07", agent, loop_node="done")

    notes = {
        "01": "Before the model ever speaks: SessionStart injects a directive and "
        "UserPromptSubmit injects a goal reminder (the prompt has /goal). Both "
        "amber blocks sit in the column first.",
        "02": "read_file passes PreToolUse clean, an allow flag, then the read "
        "really happens.",
        "03": "bash rm -rf tests hits PreToolUse and is denied for real, BLOCKED "
        "enters the column as a tool result. The command never touched the disk.",
        "04": "write_file succeeds. PostToolUse's lint hook injects its check "
        "straight into the tool result.",
        "05": "The model tries to stop at 9 of 12 features. The stop hook fires "
        "continue_, a bounce message pushes it back to work.",
        "06": "run_tests really runs the suite, 9 passed comes back as a real "
        "tool result, not a narrated number.",
        "07": "Hooks are the deterministic skeleton around a stochastic model. "
        "Prose can be ignored, a gate cannot.",
    }
    for key in ("01", "02", "03", "04", "05", "06", "07"):
        draw_frame(
            snaps[key],
            FIGURES / f"step-{key}.png",
            TITLE,
            note=notes[key],
            right="rail",
            registered_hooks=registered,
        )

    # ---- oracle ----
    assert ws.path("tests/test_todo.py").exists(), (
        "the rm must have been really blocked"
    )
    assert tools.call_counts.get("bash", 0) == 0, (
        "the bash tool's underlying subprocess must never run: the deny happens "
        "before tools.execute"
    )
    denies = [e for e in agent.hooks.fired if e["action"] == "deny"]
    assert len(denies) == 1, denies
    assert denies[0]["event"] == "pre_tool_use" and denies[0]["hook"] == "protect", (
        denies
    )
    continues = [
        e
        for e in agent.hooks.fired
        if e["event"] == "stop" and e["action"] == "continue_"
    ]
    assert len(continues) == 1, continues
    assert agent.done_text == "F09 done for real.", agent.done_text
    assert any("BLOCKED by hook" in m.content for m in agent.context), (
        "the denial message must be in the final context"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {figs}"
    print(
        f"{len(figs)} figures, 1 deny (protect), 1 stop continue_, "
        f"done_text={agent.done_text!r}, bash call_count=0. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
