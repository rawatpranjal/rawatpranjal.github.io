"""Embeddings as geometry: a chunk of text becomes a point in space.

Drives the real TF-IDF index against the real corpus. The 2D projection is a
real TruncatedSVD fit on the real TF-IDF matrix -- nothing here is staged.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import DEMO_QUERY, DEMO_QUERY_OFFTOPIC, build_pipeline  # noqa: E402
from ragviz import clear, draw_embedding_space, draw_system  # noqa: E402

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)

    draw_system(
        FIGURES / "step-01.png",
        stages=["chunk text", "TF-IDF vector", "point in space"],
        arrows=["vectorize", "project"],
        title="From words to a point in space",
    )

    chunks, index = build_pipeline()

    draw_embedding_space(
        index,
        FIGURES / "step-02.png",
        "The embedding space",
        highlight_query=DEMO_QUERY,
        note="The query (star) lands nearest to hr documents (blue), even off from the main clump -- that's why hr documents win.",
    )

    draw_embedding_space(
        index,
        FIGURES / "step-03.png",
        "A different question, a different neighborhood",
        highlight_query=DEMO_QUERY_OFFTOPIC,
        note="This question lands among the offtopic pages instead.",
    )

    # The oracle: the everyday facts the deck states, checked against a real index.
    assert index.search(DEMO_QUERY, k=1)[0][0].tag == "hr", (
        "the PTO query should land in the hr cluster"
    )
    assert index.search(DEMO_QUERY_OFFTOPIC, k=1)[0][0].tag == "offtopic", (
        "the knife-sharpening query should land in the offtopic cluster"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(f"{len(figs)} figures, {len(chunks)} chunks indexed. All checks passed.")


if __name__ == "__main__":
    main()
