"""Hybrid search: dense (embeddings) and sparse (BM25) rank chunks differently.
Fusing the two rankings (reciprocal rank fusion) is more robust than either alone.

Every score here is real: a real TF-IDF cosine similarity, a real pure-numpy BM25
score, and a real RRF fusion of the two -- nothing is staged for the picture.

Candidates tried (dense top-5 doc_id order vs sparse top-5 doc_id order), before
picking a query:
  - "how much does aurora vault cost"        -> both rankings agree at #1 (prod-03),
    mild reshuffling further down. Not the most interesting case.
  - "how many days of pto do employees get"  -> dense and sparse nearly tie between
    the two near-duplicate PTO docs (hr-01/hr-02); a real disagreement but a small one.
  - "what happens when the database goes down" -> same story: eng-01/eng-02 swap
    places, but it's a near-duplicate tie, not a semantic vs. keyword split.
  - "aurora notify escalation to on-call"    -> the sharpest disagreement: dense's
    #1 is the *pricing* doc (prod-12), because every Aurora Notify doc scores
    similarly on broad topical similarity. BM25's #1 is the actual Escalation
    Policy (prod-06), because it exact-matches "escalation" and "on-call". Dense
    buries the escalation doc at #4. This is the query used below.
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
from ragviz import bm25_search, clear, draw_retrieval_table, draw_system, rrf_fuse  # noqa: E402

FIGURES = HERE / "figures"

QUERY = "aurora notify escalation to on-call"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    dense = index.search(QUERY, k=5)
    draw_retrieval_table(
        dense,
        FIGURES / "step-01.png",
        "Dense search (embeddings)",
        query=QUERY,
        note="Cosine similarity over TF-IDF vectors. Every Aurora Notify doc scores "
        "similarly -- the pricing doc edges out the actual escalation policy.",
    )

    sparse = bm25_search(chunks, QUERY, k=5)
    draw_retrieval_table(
        sparse,
        FIGURES / "step-02.png",
        "Sparse search (BM25, keyword overlap)",
        query=QUERY,
        note='BM25 rewards exact term matches -- "escalation" and "on-call" pull the '
        "Escalation Policy doc straight to #1.",
    )

    fused = rrf_fuse(dense, sparse)
    draw_retrieval_table(
        fused[:5],
        FIGURES / "step-03.png",
        "Fused (reciprocal rank fusion)",
        query=QUERY,
        score_label="rrf score",
        note="Combines rank, not raw score, from both lists (k=60). Docs strong in "
        "either ranking rise; docs weak in both sink.",
    )

    draw_system(
        FIGURES / "step-04.png",
        stages=["dense ranking", "sparse ranking", "fused ranking"],
        arrows=["rank", "rank"],
        title="Combining two rankings",
        note="RRF only ever sees rank position, so one system's raw score scale "
        "can never dominate the other's.",
    )

    # The oracle: a real, checked property of RRF, not an assumed one. The fused
    # top-1 must appear somewhere in the top-5 of BOTH input rankings -- RRF can
    # only promote a chunk that at least one retriever already surfaced.
    dense_ids = [c.doc_id for c, _ in dense]
    sparse_ids = [c.doc_id for c, _ in sparse]
    fused_top1_id = fused[0][0].doc_id
    assert fused_top1_id in dense_ids, "fused #1 should appear in dense's top-5"
    assert fused_top1_id in sparse_ids, "fused #1 should appear in sparse's top-5"

    # And the disagreement that motivated picking this query: the two rankings
    # really do put different docs at #1.
    assert dense_ids[0] != sparse_ids[0], (
        "this query was chosen because dense and sparse disagree at #1"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures. dense#1={dense_ids[0]} sparse#1={sparse_ids[0]} "
        f"fused#1={fused_top1_id}. All checks passed."
    )


if __name__ == "__main__":
    main()
