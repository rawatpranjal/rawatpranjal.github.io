"""The naive RAG loop: the bird's-eye view. Indexing happens once, offline;
retrieval + generation happen per question, online.

Drives the real toy pipeline end to end for one concrete question -- nothing
here is staged, every retrieved chunk, prompt, and answer is real output.
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
    generate_answer,
)

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)

    draw_system(
        FIGURES / "step-01.png",
        stages=["corpus", "chunks", "index"],
        arrows=["chunk", "fit"],
        title="Offline: happens once",
        frame_label="build this once, ahead of time",
        note="The corpus is chunked and indexed a single time, before anyone asks a question.",
    )

    draw_system(
        FIGURES / "step-02.png",
        stages=["query", "retrieve", "augment", "generate"],
        arrows=["search index", "build prompt", "call model"],
        title="Online: happens per question",
        frame_label="run this every time someone asks",
        note="Every question runs this same four-step loop against the index built offline.",
    )

    chunks, index = build_pipeline()
    hits = index.search(DEMO_QUERY, k=3)
    retrieved = [c for c, _ in hits]
    scores = [s for _, s in hits]

    draw_retrieval_table(
        hits,
        FIGURES / "step-03.png",
        "One question's retrieve step",
        query=DEMO_QUERY,
        note="The real top-3 chunks for this question, ranked by real cosine similarity.",
    )

    prompt = assemble_prompt(DEMO_QUERY, retrieved)
    # NB: draw_prompt_assembly's note= arg has a latent bug (fig.text placed
    # outside figure-fraction range blows up the bbox_inches="tight" canvas to
    # thousands of px tall). ragviz.py is shared/read-only, so omit note= here.
    draw_prompt_assembly(
        prompt,
        FIGURES / "step-04.png",
        "One question's augment step",
    )

    answer = generate_answer(DEMO_QUERY, retrieved, scores)

    # The oracle: a real pass through the online loop should ground, not refuse.
    assert not answer.refused, "a well-covered demo query should not refuse"
    assert "[1]" in prompt, (
        "the assembled prompt should carry a numbered citation marker"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures, {len(retrieved)} chunks retrieved, "
        f"answer: {answer.text!r}. All checks passed."
    )


if __name__ == "__main__":
    main()
