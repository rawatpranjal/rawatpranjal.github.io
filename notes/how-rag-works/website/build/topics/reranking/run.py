"""Reranking: a second, more careful pass over the candidates retrieval already found.

Drives the real TF-IDF index and the real keyword-overlap rerank() against the real
corpus. Nothing here is staged -- every score in the figures is a real number from a
real search, and the before/after order is whatever the two functions actually produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_pipeline  # noqa: E402
from ragviz import clear, draw_retrieval_table, draw_system, rerank  # noqa: E402

FIGURES = HERE / "figures"

# Chosen empirically: tried the near-duplicate-pair queries first (PTO days, expense
# threshold, failover lag) -- rerank can never reorder those two docs against each
# other, because the one word that differs between them (15 vs 20, 60 vs 30 ...) never
# appears in the query, so the tie always breaks back to whichever the first pass
# already had on top. This query, tried next, is a real query with a real top-1 swap.
QUERY = "on-call handoff day"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    initial = index.search(QUERY, k=5)
    draw_retrieval_table(
        initial,
        FIGURES / "step-01.png",
        "Before reranking",
        query=QUERY,
        note="TF-IDF cosine similarity. The Onboarding Checklist edges out the actual on-call doc.",
    )

    reranked = rerank(QUERY, initial)
    draw_retrieval_table(
        reranked,
        FIGURES / "step-02.png",
        "After reranking",
        query=QUERY,
        note="Keyword-overlap rescoring. The On-call Rotation Schedule jumps to #1.",
    )

    draw_system(
        FIGURES / "step-03.png",
        stages=["candidates", "rerank()", "reordered candidates"],
        arrows=["top-5", "re-score"],
        title="A second, more careful pass",
        note="Same 5 candidates in, same 5 candidates out -- only the order changes.",
    )

    # The oracle: the everyday claim the deck makes, checked against real search.
    ids_before = [c.id for c, _ in initial]
    ids_after = [c.id for c, _ in reranked]
    assert ids_before != ids_after, (
        "expected this query to visibly reorder the candidates"
    )
    assert initial[0][0].id == "hr-08-c0", (
        "TF-IDF's top hit should be the Onboarding Checklist"
    )
    assert reranked[0][0].id == "eng-09-c0", (
        "reranking should promote the On-call Rotation Schedule to #1"
    )
    assert set(ids_before) == set(ids_after), (
        "reranking reorders candidates, it doesn't fetch new ones"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(
        f"{len(figs)} figures. Before: #1 {initial[0][0].id}. "
        f"After: #1 {reranked[0][0].id}. Order changed: {ids_before != ids_after}."
    )


if __name__ == "__main__":
    main()
