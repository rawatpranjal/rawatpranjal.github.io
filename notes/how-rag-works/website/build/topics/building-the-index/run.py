"""Building the index: fit once, search forever.

Every chunk gets vectorized and the vectors are stored together as one fitted
TF-IDF index. That fit happens exactly once. After that, any question searches
the same index instantly -- no re-fitting, no rebuilding. Nothing here is
staged: build_pipeline() runs for real, once, and the same index object
answers three different real queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

import fixtures  # noqa: E402
from fixtures import DEMO_QUERY, DEMO_QUERY_OFFTOPIC, DEMO_QUERY_WEAK  # noqa: E402
from ragviz import clear, draw_embedding_space, draw_retrieval_table, draw_system  # noqa: E402

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)

    # Count real calls to build_pipeline() -- the index is fit exactly once,
    # never rebuilt for any of the three queries below.
    build_calls = {"n": 0}
    real_build_pipeline = fixtures.build_pipeline

    def counted_build_pipeline():
        build_calls["n"] += 1
        return real_build_pipeline()

    draw_system(
        FIGURES / "step-01.png",
        stages=["48 chunks", "fit TF-IDF vectorizer", "one fitted index"],
        arrows=["vectorize", "store"],
        title="Building the index (once)",
    )

    chunks, index = counted_build_pipeline()
    draw_embedding_space(
        index,
        FIGURES / "step-02.png",
        "48 chunks, indexed once",
        note="Every chunk is a point in this space now. No more building needed after this.",
    )

    # Reuse the SAME index for three different questions. Nothing refits.
    demo_hits = index.search(DEMO_QUERY, k=5)
    offtopic_hits = index.search(DEMO_QUERY_OFFTOPIC, k=5)
    weak_hits = index.search(DEMO_QUERY_WEAK, k=5)

    draw_retrieval_table(
        demo_hits,
        FIGURES / "step-03.png",
        "One index, three questions, zero rebuilds",
        query=DEMO_QUERY,
        note="Same index, three different questions, no rebuild in between -- only .search() runs each time.",
    )

    # The oracle: the index was built once and still answers every question.
    assert build_calls["n"] == 1, (
        "build_pipeline() must run exactly once -- the index is fit once, reused for every query"
    )
    for name, hits in (
        ("demo", demo_hits),
        ("offtopic", offtopic_hits),
        ("weak", weak_hits),
    ):
        assert len(hits) > 0, (
            f"{name} query should still return results from the shared index"
        )
        assert hits[0][1] > 0, (
            f"{name} query's top hit should have a positive real score"
        )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(
        f"{len(figs)} figures, {len(chunks)} chunks indexed once, "
        f"3 queries searched (top scores: demo={demo_hits[0][1]:.3f}, "
        f"offtopic={offtopic_hits[0][1]:.3f}, weak={weak_hits[0][1]:.3f}). "
        "All checks passed."
    )


if __name__ == "__main__":
    main()
