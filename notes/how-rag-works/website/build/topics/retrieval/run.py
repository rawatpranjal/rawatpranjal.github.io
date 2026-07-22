"""Retrieval: embed the query, pull the top-k nearest chunks, real scores.

Drives the real TF-IDF index against the real corpus. Nothing here is staged --
every score in the figures is a real cosine similarity from a real search.
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
from ragviz import clear, draw_embedding_space, draw_retrieval_table, draw_system  # noqa: E402

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    draw_system(
        FIGURES / "step-01.png",
        stages=["corpus", "chunks", "TF-IDF index"],
        arrows=["chunk", "fit"],
        title="What retrieval searches over",
        note="48 documents, chunked, turned into TF-IDF vectors once. This step already happened.",
    )

    hits = index.search(DEMO_QUERY, k=5)
    draw_retrieval_table(
        hits,
        FIGURES / "step-02.png",
        "Top-5 nearest chunks",
        query=DEMO_QUERY,
        note="Ranked by real cosine similarity between the query vector and every chunk vector.",
    )

    draw_embedding_space(
        index,
        FIGURES / "step-03.png",
        "Where the query landed",
        highlight_query=DEMO_QUERY,
        note="The query (star) lands nearest to hr documents (blue), even off from the main clump -- that's why hr docs win.",
    )

    weak_query = "what is the weather in tokyo tomorrow"
    weak_hits = index.search(weak_query, k=5)
    draw_retrieval_table(
        weak_hits,
        FIGURES / "step-04.png",
        "A query the corpus doesn't cover",
        query=weak_query,
        note="Every score is low. Retrieval still returns its top-5 -- it never refuses on its own.",
    )

    # The oracle: the everyday facts the deck states, checked against real search.
    assert hits[0][0].doc_id in ("hr-01", "hr-02"), (
        "the PTO query should hit a PTO doc first"
    )
    assert hits[0][1] > 0.3, "a real on-topic match should score well above zero"
    assert weak_hits[0][1] < hits[0][1], (
        "an uncovered query should score lower than a covered one"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures, top hit {hits[0][0].doc_id} @ {hits[0][1]:.3f}. All checks passed."
    )


if __name__ == "__main__":
    main()
