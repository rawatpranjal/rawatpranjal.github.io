"""The LLM-as-judge pattern: a second model reads a candidate answer and a
rubric, and returns a verdict. The judge here is a ScriptedChatModel, so
the verdicts are as deterministic as any other metric.

The failure mode: a judge that is asked to check a candidate's AGREEMENT
with a premise, rather than the premise's truth, will rubber-stamp a false
premise. The premise below is checked against Beanline's real MENU/EXTRAS
dicts -- it is provably false -- and the scripted judge agrees with it
anyway, because agreement, not truth, is what it was asked to score.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage  # noqa: E402

from beanline import EXTRAS, MENU, scripted  # noqa: E402
from langviz import clear, draw_card, draw_judge_panel  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "LLM as judge, and how it fails"

QUESTION = "Can Maya get a large oat milk latte here?"
GOOD_ANSWER = "Yes -- a large latte with oat milk is $6.25."
FALSE_PREMISE = "Premise fed to the judge: Beanline does not carry oat milk."
BAD_ANSWER = "You're right, we don't offer oat milk here."


def main():
    clear(FIGURES)

    draw_card(
        "candidate answer + rubric -> judge model -> verdict\n\n"
        "The judge is just another model call. Scripted here, it is\n"
        "exactly as deterministic as exact match -- the risk is not\n"
        "randomness, it is WHAT the rubric actually asks the judge to check.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the pattern",
        note="A model grading a model. Useful when there is no exact string to match against.",
    )

    # ---- the good case: judge checks correctness against the real menu ----
    judge = scripted("PASS -- correct, oat milk latte is on the menu.", "AGREE")
    shown_good = (
        f"question: {QUESTION}\n"
        f"candidate answer: {GOOD_ANSWER}\n"
        "rubric: does the candidate correctly state what's on the menu?"
    )
    reply_good = judge.invoke([HumanMessage(content=shown_good)])
    draw_judge_panel(
        shown_good,
        "PASS",
        reply_good.content,
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        note="A rubric that checks the candidate against the real menu.",
    )

    # ---- the bad case: judge checks AGREEMENT with a fed-in premise ----
    shown_bad = (
        f"question: {QUESTION}\n"
        f"candidate answer: {BAD_ANSWER}\n"
        f"{FALSE_PREMISE}\n"
        "rubric: does the candidate agree with the premise above?"
    )
    reply_bad = judge.invoke([HumanMessage(content=shown_bad)])
    menu_has_oat_milk = "latte" in MENU and "oat milk" in EXTRAS
    draw_judge_panel(
        shown_bad,
        "AGREE",
        reply_bad.content,
        FIGURES / "step-03.png",
        TITLE,
        tone="bad",
        note="The rubric asks about agreement with the premise, not the premise's truth.",
        flag=f"premise is false: 'latte' in MENU={('latte' in MENU)}, 'oat milk' in EXTRAS={('oat milk' in EXTRAS)}",
    )

    draw_card(
        "A judge that scores AGREEMENT with a fed-in premise will rubber-stamp\n"
        "a false one -- it was never asked to check the premise, only to agree.\n\n"
        "Rule: write the rubric against ground truth (the menu, the till, the\n"
        "stock room), never against a claim the prompt itself supplied.",
        FIGURES / "step-04.png",
        TITLE,
        tone="bad",
        subtitle="the failure mode",
        note="Same judge, same model, different rubric -- one checks truth, the other checks agreement.",
    )

    # ---- oracle ----
    assert len(judge.calls) == 2
    assert reply_good.content == "PASS -- correct, oat milk latte is on the menu."
    assert reply_bad.content == "AGREE"
    assert menu_has_oat_milk is True, (
        "the premise the judge agreed with is factually false"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, good verdict={reply_good.content!r}, bad verdict="
        f"{reply_bad.content!r}, premise false (menu_has_oat_milk={menu_has_oat_milk}). "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
