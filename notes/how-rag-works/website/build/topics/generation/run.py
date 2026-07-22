"""Generation: the model writes a grounded answer, with a citation -- or refuses.

Drives the real toy pipeline for a covered question (case A) and a genuinely
uncovered one (case B). Both answers come from the real generate_answer(), the
real cite-or-refuse gate -- nothing here is staged.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import DEMO_QUERY, build_pipeline  # noqa: E402
from ragviz import (  # noqa: E402
    INK,
    PANEL,
    PAPER,
    Answer,
    clear,
    draw_retrieval_table,
    generate_answer,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIGURES = HERE / "figures"

GREEN = "#6bbf8a"
RED = "#e06b9c"

UNCOVERED_QUERY = "what is the weather in tokyo tomorrow"


def draw_answer_card(
    answer: Answer, path: Path, title: str, border: str, note: str = ""
):
    """A dark answer card, styled to match draw_prompt_assembly's card (PANEL
    fill, monospace text) -- border color signals grounded (green) vs refused
    (red)."""
    lines = textwrap.wrap(answer.text, width=58) or [answer.text]
    fig, ax = plt.subplots(figsize=(9.0, 1.2 + 0.4 * len(lines)), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(lines) - 1.1, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, -len(lines) - 0.85),
            9.9,
            len(lines) + 1.6,
            boxstyle="round,pad=0.02",
            facecolor=PANEL,
            edgecolor=border,
            linewidth=1.8,
        )
    )
    for i, line in enumerate(lines):
        ax.text(0.3, -i, line, fontsize=11, family="monospace", color=INK, va="center")
    fig.text(0.015, 0.94, title, fontsize=11, fontweight="bold", color=INK, ha="left")
    if note:
        fig.text(
            0.015, -len(lines) - 1.0 + 0.05, note, fontsize=7.5, color=INK, ha="left"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    # Case A: a question the corpus covers.
    hits = index.search(DEMO_QUERY, k=3)
    scores = [s for _, s in hits]
    draw_retrieval_table(
        hits,
        FIGURES / "step-01.png",
        "Case A: a question the corpus covers",
        query=DEMO_QUERY,
        note="Real top-3 chunks, real cosine similarity -- what the model gets to read.",
    )
    answer_a = generate_answer(DEMO_QUERY, [c for c, _ in hits], scores)
    draw_answer_card(
        answer_a,
        FIGURES / "step-02.png",
        "Grounded: answered, with a citation",
        border=GREEN,
        note="The top chunk clears the refusal bar, so the model writes an answer and cites it.",
    )

    # Case B: a question the corpus genuinely does not cover.
    weak_hits = index.search(UNCOVERED_QUERY, k=3)
    weak_scores = [s for _, s in weak_hits]
    draw_retrieval_table(
        weak_hits,
        FIGURES / "step-03.png",
        "Case B: a question the corpus doesn't cover",
        query=UNCOVERED_QUERY,
        note="Every score is near zero -- nothing here is worth citing.",
    )
    answer_b = generate_answer(UNCOVERED_QUERY, [c for c, _ in weak_hits], weak_scores)
    draw_answer_card(
        answer_b,
        FIGURES / "step-04.png",
        "Refused: not in the corpus",
        border=RED,
        note="Below the refusal threshold, the model declines instead of guessing.",
    )

    # The oracle: real checks against the real run, not assumed outcomes.
    assert not answer_a.refused, "the demo query should be answerable from the corpus"
    assert "[1]" in answer_a.text, "a grounded answer should carry a citation"
    assert answer_b.refused is True, "a genuinely uncovered query should refuse"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures, case A: {answer_a.text!r}, "
        f"case B refused={answer_b.refused}. All checks passed."
    )


if __name__ == "__main__":
    main()
