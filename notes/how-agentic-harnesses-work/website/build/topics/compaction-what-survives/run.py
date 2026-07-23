"""Compaction, what survives: the context outgrows its budget, compaction
summarizes it, and a constraint dropped by the summary gets violated for
real. Pinning the constraint (instead of hoping the summary keeps it) is
what actually survives; a pre_tool_use guard is what actually prevents it.

Scenario A (the hazard): the constraint rides in as an ordinary message.
Real reads fill the context, one more real read pushes it over budget,
maybe_compact really fires, and the squeeze really drops the constraint
into an unread middle. With no guard left standing, the model's next write
really changes a feature description on disk.

Scenario B (the fix): same flow, but the constraint is pinned, so it
survives the same real compaction, and a pre_tool_use hook that really
diffs descriptions blocks the same edit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    Harness,
    HookDecision,
    Hooks,
    Message,
    ScriptedModel,
    Turn,
    builtin_tools,
)
from harnessviz import clear, draw_frame, draw_token_ledger  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Compaction, what survives"
BUDGET = 1000
KEEP_LAST = 3
CONSTRAINT = "Constraint: never change feature descriptions in feature_list.json."
# the budget the agent is given for the later, unrelated write attempt, so that
# write does not trigger a second compaction and muddy the one we're studying.
POST_DEMO_BUDGET = 5000


def build_flow(ws, pinned: bool, with_guard: bool):
    """Constraint message, then real work that fills the context, then one
    more real read that pushes past budget and fires a real compaction."""
    tools = builtin_tools(ws)
    hooks = Hooks()

    def checkpoint_flush(**_):
        line = "checkpoint: flushed to disk before the squeeze\n"
        prev = ws.read("progress.md") if ws.exists("progress.md") else ""
        ws.write("progress.md", prev + line)
        return HookDecision()

    hooks.on("pre_compact", "checkpoint-flush", checkpoint_flush)

    if with_guard:
        original = {
            f["id"]: f["description"] for f in json.loads(ws.read("feature_list.json"))
        }

        def constraint_guard(tool, args):
            if tool == "write_file" and args.get("path") == "feature_list.json":
                try:
                    new_feats = json.loads(args.get("content", ""))
                except (TypeError, ValueError):
                    return HookDecision()
                for f in new_feats:
                    if original.get(f["id"]) != f["description"]:
                        return HookDecision(
                            action="deny",
                            reason=f"constraint-guard: {f['id']} description "
                            "changed, write blocked",
                        )
            return HookDecision()

        hooks.on("pre_tool_use", "constraint-guard", constraint_guard)

    agent = Agent(
        ScriptedModel([]), tools, ws, hooks=hooks, budget=BUDGET, keep_last=KEEP_LAST
    )
    h = Harness(ws)
    timeline = []

    def record(event, msg):
        timeline.append((event, h.snap(event, agent)))

    agent.append(
        Message("user", "text", CONSTRAINT, pinned=pinned),
        on_event=record,
        event="constraint",
    )

    # real work on F10, several real reads, staying under budget.
    agent.model = ScriptedModel(
        [
            Turn(
                text="Continuing F10: clear-done.",
                tool="read_file",
                args={"path": "CLAUDE.md"},
            ),
            Turn(
                text="Reviewing the current implementation before adding clear-done.",
                tool="read_file",
                args={"path": "todo.py"},
            ),
            Turn(text="Checking current test status.", tool="run_tests", args={}),
            Turn(
                text="F10 spec looks straightforward, will implement clear-done next session."
            ),
        ]
    )
    agent.run(
        "Work on F10: clear-done. Read the plan, the code, and check tests first.",
        on_event=record,
    )
    near_full = timeline[-1][1]

    # one more real read: this is the append that pushes past budget and
    # fires maybe_compact for real, inside the loop.
    agent.model = ScriptedModel(
        [
            Turn(
                text="Double-checking the feature spec before writing clear-done.",
                tool="read_file",
                args={"path": "feature_list.json"},
            ),
            Turn(text="F10 implemented and verified."),
        ]
    )
    agent.run("Continue: double check the spec then wrap up F10.", on_event=record)

    pre_compact_snap = next(s for e, s in timeline if e == "pre_compact")
    compact_snap = next(s for e, s in timeline if e == "compact")
    return agent, h, timeline, near_full, pre_compact_snap, compact_snap


def main():
    clear(FIGURES)

    # ---- Scenario A: the hazard ----
    ws_a = build_workspace(stage=9)
    agent_a, h_a, timeline_a, near_full_a, pre_a, post_a = build_flow(
        ws_a, pinned=False, with_guard=False
    )

    draw_frame(
        near_full_a,
        FIGURES / "step-01.png",
        TITLE,
        right="none",
        note=f"Real reads fill the context for real: {near_full_a.total_tokens}/{BUDGET} tok. "
        "The constraint rode in as an ordinary, un-pinned user message, no different "
        "from any other turn.",
    )
    draw_frame(
        pre_a,
        FIGURES / "step-02.png",
        TITLE,
        right="files",
        note="One more real read pushes past budget. PreCompact fires for real: "
        "checkpoint-flush writes progress.md before the squeeze.",
    )
    draw_frame(
        post_a,
        FIGURES / "step-03.png",
        TITLE,
        right="none",
        note=f"After compaction: {pre_a.total_tokens} tok collapse to "
        f"{post_a.total_tokens} tok, head plus one summary plus tail.",
    )

    compact_idx = next(i for i, (e, _) in enumerate(timeline_a) if e == "compact")
    series = [(e[:10], s.total_tokens) for e, s in timeline_a]
    draw_token_ledger(
        series,
        BUDGET,
        FIGURES / "step-04.png",
        TITLE,
        note=f"Real token counts per event: {pre_a.total_tokens} tok right before the "
        f"cliff, {post_a.total_tokens} right after.",
        mark={compact_idx: "compact"},
    )

    original_json = ws_a.read("feature_list.json")
    violated_json = original_json.replace(
        "todo add TEXT saves a task", "todo add stores an item"
    )
    assert violated_json != original_json, "the replace must really change the JSON"

    agent_a.budget = POST_DEMO_BUDGET
    agent_a.model = ScriptedModel(
        [
            Turn(
                text="Cleaning up the feature wording for clarity.",
                tool="write_file",
                args={"path": "feature_list.json", "content": violated_json},
            ),
            Turn(text="Descriptions tidied up."),
        ]
    )
    agent_a.run("Also tidy up the feature descriptions while you're in there.")
    step05 = h_a.snap("step05", agent_a)
    draw_frame(
        step05,
        FIGURES / "step-05.png",
        TITLE,
        right="files",
        note="The constraint was in the dropped middle: the summary kept the gist of "
        "the work but lost the rule. No guard is left standing, the write succeeds.",
    )

    # ---- Scenario B: the fix ----
    ws_b = build_workspace(stage=9)
    agent_b, h_b, _timeline_b, _near_full_b, _pre_b, post_b = build_flow(
        ws_b, pinned=True, with_guard=True
    )

    agent_b.budget = POST_DEMO_BUDGET
    agent_b.model = ScriptedModel(
        [
            Turn(
                text="Cleaning up the feature wording for clarity.",
                tool="write_file",
                args={"path": "feature_list.json", "content": violated_json},
            ),
            Turn(text="The guard blocked that edit, descriptions stay as written."),
        ]
    )
    agent_b.run("Also tidy up the feature descriptions while you're in there.")
    step06 = h_b.snap("step06", agent_b)
    draw_frame(
        step06,
        FIGURES / "step-06.png",
        TITLE,
        right="files",
        note="Pin what must survive, gate what must never happen: the pinned constraint "
        "is still in the column, constraint-guard denies the same edit.",
    )

    # ---- oracle: scenario A ----
    assert agent_a.compactions == 1, agent_a.compactions
    assert post_a.total_tokens < pre_a.total_tokens
    assert not any(CONSTRAINT in m.content for m in post_a.context), (
        "the constraint must be genuinely absent right after compaction"
    )
    final_a = json.loads(ws_a.read("feature_list.json"))
    f01_a = next(f for f in final_a if f["id"] == "F01")
    assert f01_a["description"] != "todo add TEXT saves a task to todos.json", (
        "the description must really have changed on disk"
    )

    # ---- oracle: scenario B ----
    assert agent_b.compactions == 1, agent_b.compactions
    assert any(m.pinned and CONSTRAINT in m.content for m in post_b.context), (
        "the pinned constraint must be in the post-compaction context"
    )
    final_b = json.loads(ws_b.read("feature_list.json"))
    f01_b = next(f for f in final_b if f["id"] == "F01")
    assert f01_b["description"] == "todo add TEXT saves a task to todos.json", (
        "the guard must really have kept the description unchanged"
    )
    denies_b = [e for e in agent_b.hooks.fired if e["action"] == "deny"]
    assert any(e["hook"] == "constraint-guard" for e in denies_b), denies_b

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, A: 1 compaction {pre_a.total_tokens}->{post_a.total_tokens} "
        "tok, description changed on disk; B: pinned constraint survives, guard denied "
        "the same edit. All checks passed."
    )
    ws_a.cleanup()
    ws_b.cleanup()


if __name__ == "__main__":
    main()
