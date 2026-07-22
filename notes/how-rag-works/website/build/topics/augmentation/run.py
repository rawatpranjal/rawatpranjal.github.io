"""Augmentation: the retrieved chunks become one literal prompt string.

Drives the real pipeline against the real corpus. The prompt rendered here is
the exact string assemble_prompt() built -- not a paraphrase of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import DEMO_QUERY, build_pipeline  # noqa: E402
from ragviz import (  # noqa: E402
    assemble_prompt,
    clear,
    draw_prompt_assembly,
    draw_retrieval_table,
    draw_system,
)

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    hits = index.search(DEMO_QUERY, k=3)
    draw_retrieval_table(
        hits,
        FIGURES / "step-01.png",
        "What got retrieved",
        query=DEMO_QUERY,
        note="The same top-3 chunks retrieval always returns for this query -- now they need to reach the model.",
    )

    # note left blank: draw_prompt_assembly's note anchor sits far below the
    # rendered card, and bbox_inches="tight" then stretches the saved PNG to
    # reach it, leaving a huge blank gap (pre-existing in ragviz.py, not ours to touch).
    prompt = assemble_prompt(DEMO_QUERY, [c for c, _ in hits])
    draw_prompt_assembly(
        prompt,
        FIGURES / "step-02.png",
        "The assembled prompt",
    )

    draw_system(
        FIGURES / "step-03.png",
        stages=["retrieved chunks", "assemble_prompt()", "prompt with [n] markers"],
        arrows=["stitch", "number"],
        title="From chunks to one prompt string",
        note="This literal string, not the chunks themselves, is what the model actually sees.",
    )

    # The oracle: the everyday facts the deck states, checked against the real prompt.
    assert "[1]" in prompt, "chunk 1 should be tagged [1]"
    assert "[2]" in prompt, "chunk 2 should be tagged [2]"
    assert "[3]" in prompt, "chunk 3 should be tagged [3]"
    assert DEMO_QUERY in prompt, "the literal query text should appear in the prompt"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(
        f"{len(figs)} figures, prompt is {len(prompt)} chars with 3 [n] markers. All checks passed."
    )


if __name__ == "__main__":
    main()
