"""The problem: without retrieval, a model can't see the corpus and guesses.

Contrasts a plain language model (no access to Aurora Cloud's documents) with a
retrieval-augmented one. The "with retrieval" half is a real run against the real
corpus -- real search, a real score, a real cite-or-refuse check -- nothing here is
staged.
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
    NEW_COLOR,
    PANEL,
    PAPER,
    REFUSAL_THRESHOLD,
    Answer,
    clear,
    draw_retrieval_table,
    draw_system,
    generate_answer,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIGURES = HERE / "figures"


def draw_answer_card(
    answer: Answer, source_title: str, path: Path, title: str, note: str = ""
):
    """A dark answer card: the grounded text plus its [1] citation. One-off,
    styled to match draw_prompt_assembly's card (PANEL fill, monospace text)."""
    lines = textwrap.wrap(answer.text, width=58) or [answer.text]
    n = len(lines) + 1  # +1 reserves a row for the citation line
    fig, ax = plt.subplots(figsize=(9.0, 1.1 + 0.34 * n), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(-n - 0.5, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, -n - 0.25),
            9.9,
            n + 1.05,
            boxstyle="round,pad=0.02",
            facecolor=PANEL,
            edgecolor=NEW_COLOR,
            linewidth=1.6,
        )
    )
    for i, line in enumerate(lines):
        ax.text(0.3, -i, line, fontsize=11, family="monospace", color=INK, va="center")
    ax.text(
        0.3,
        -len(lines) - 0.5,
        f"[1] {source_title}",
        fontsize=9,
        family="monospace",
        color=NEW_COLOR,
        va="center",
        fontweight="bold",
    )
    fig.text(0.015, 0.95, title, fontsize=11, fontweight="bold", color=INK, ha="left")
    if note:
        fig.text(0.015, 0.03, note, fontsize=7.5, color=INK, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def main():
    clear(FIGURES)

    draw_system(
        FIGURES / "step-01.png",
        stages=["question", "language model alone", "an answer, maybe wrong"],
        arrows=["?", "guesses"],
        title="Without retrieval",
        note="The model never sees Aurora Cloud's documents. It answers from memory, or none at all.",
    )

    draw_system(
        FIGURES / "step-02.png",
        stages=["question", "corpus", "retrieved chunks", "grounded answer"],
        arrows=["search", "cite", "write"],
        title="With retrieval",
        note="Now the answer is built from real documents, not memory.",
    )

    chunks, index = build_pipeline()
    hits = index.search(DEMO_QUERY, k=3)
    top_chunks = [c for c, _ in hits]
    scores = [s for _, s in hits]
    draw_retrieval_table(
        hits,
        FIGURES / "step-03.png",
        "What retrieval finds",
        query=DEMO_QUERY,
        note="Real cosine similarity, real corpus -- this is what the fix looks up before answering.",
    )

    answer = generate_answer(DEMO_QUERY, top_chunks, scores)
    draw_answer_card(
        answer,
        top_chunks[0].doc_title,
        FIGURES / "step-04.png",
        "The grounded answer",
        note="Built from the chunk above. [1] links the claim back to its source.",
    )

    # The oracle: real checks against the real run, not assumed outcomes.
    assert not answer.refused, "the demo query should be answerable from the corpus"
    assert "[1]" in answer.text, "a grounded answer should carry a citation"
    assert scores[0] > REFUSAL_THRESHOLD, "the top hit should clear the refusal bar"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures, top hit {top_chunks[0].doc_id} @ {scores[0]:.3f}, "
        f"answer: {answer.text!r}. All checks passed."
    )


if __name__ == "__main__":
    main()
