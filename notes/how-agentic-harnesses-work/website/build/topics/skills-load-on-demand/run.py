"""Skills load on demand: progressive disclosure as real context economics.

Three tiers, three real costs. Every session pays for a one-line index of
every skill. A task that matches one pulls in its full body. The body's own
references load only if the task actually follows them there. Every token
count on these frames is measured from real Message objects, never typed by
hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import SKILL_SPOKE, SKILLS, build_workspace  # noqa: E402
from harness import Agent, Harness, Message, ScriptedModel, SkillLibrary, builtin_tools  # noqa: E402
from harnessviz import clear, draw_frame, draw_skill_tiers  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Skills load on demand"


def main():
    clear(FIGURES)
    # Just the one skill this ask matches, so the index stays a genuine
    # one-line-per-skill tax rather than padded with an irrelevant second entry.
    ws = build_workspace(stage=7, skills=False)
    ws.write(".skills/cli-conventions/SKILL.md", SKILLS["cli-conventions"][1])
    ws.write(f".skills/{SKILL_SPOKE[0]}", SKILL_SPOKE[1])
    skills_lib = SkillLibrary(ws)

    agent = Agent(
        ScriptedModel([]), builtin_tools(ws), ws, skills=skills_lib, budget=800
    )
    h = Harness(ws)

    # step 1: a fresh session -- the index is the one thing every session pays.
    index_msg = skills_lib.index_message()
    agent.append(index_msg)
    snap1 = h.snap("index", agent, loop_node="context")

    # step 2: the ask matches the cli-conventions index line -- nothing else
    # has loaded yet.
    user_prompt = "Add the due command following our CLI house rules."
    agent.append(Message("user", "text", user_prompt, pinned=True))
    snap2 = h.snap("ask", agent, loop_node="context")

    # step 3: the match triggers the full body to load, one jump.
    body_msg = skills_lib.load("cli-conventions")
    agent.append(body_msg)
    snap3 = h.snap("body", agent, loop_node="context")

    # step 4: the body names references/exit-codes.md, so that real spoke
    # file loads too -- only because it was actually followed.
    spoke_text = ws.read(".skills/cli-conventions/references/exit-codes.md")
    spoke_msg = Message("system", "skill_body", spoke_text)
    agent.append(spoke_msg)
    snap4 = h.snap("spoke", agent, loop_node="context")

    draw_frame(
        snap1,
        FIGURES / "step-01.png",
        TITLE,
        right="none",
        note=f"Fresh session: the skill index loads, {index_msg.tokens} tokens for every skill's one-line description.",
    )
    draw_frame(
        snap2,
        FIGURES / "step-02.png",
        TITLE,
        right="none",
        note="The ask matches the cli-conventions index line, that match is the trigger, nothing else has loaded.",
    )
    draw_frame(
        snap3,
        FIGURES / "step-03.png",
        TITLE,
        right="none",
        note=f"The match pulls in the full body: {body_msg.tokens} tokens land in one jump.",
    )
    draw_frame(
        snap4,
        FIGURES / "step-04.png",
        TITLE,
        right="none",
        note=f"The body names references/exit-codes.md, so the harness follows it: {spoke_msg.tokens} more tokens, only because it was named.",
    )

    measured = [
        (
            "tier 1: the index",
            "one line per skill, every session pays it",
            index_msg.tokens,
        ),
        ("tier 2: the body", "SKILL.md, loaded when the task matches", body_msg.tokens),
        ("tier 3: the spokes", "references/*, loaded only when followed", -1),
    ]
    draw_skill_tiers(
        measured,
        FIGURES / "step-05.png",
        TITLE,
        hot=1,
        note="Three tiers, three real costs: a flat tax, a task-triggered load, and an opt-in the index can't see.",
    )

    # ---- oracle ----
    assert index_msg.tokens < 60, index_msg.tokens
    assert body_msg.tokens > 5 * index_msg.tokens, (body_msg.tokens, index_msg.tokens)
    assert "dispatch" in body_msg.content
    assert (
        snap1.total_tokens
        < snap2.total_tokens
        < snap3.total_tokens
        < snap4.total_tokens
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {figs}"
    print(
        f"{len(figs)} figures, index {index_msg.tokens} tok, body {body_msg.tokens} tok "
        f"({body_msg.tokens / index_msg.tokens:.1f}x index), spoke {spoke_msg.tokens} tok. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
