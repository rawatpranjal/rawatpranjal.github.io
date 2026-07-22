"""The one shared pipeline every topic's run.py builds fresh, deterministically.

Mirrors the git deck's pattern: no cross-run cache, each topic starts clean and
rebuilds from the same frozen corpus. DEMO_QUERY is fixed so the same question
visibly evolves across topics 6-10 (retrieval -> hybrid -> rerank -> prompt -> answer).
"""

from __future__ import annotations

from corpus import DOCS
from ragviz import Chunk, VectorIndex, chunk_corpus

DEMO_QUERY = "how many days of paid time off do I get"
DEMO_QUERY_OFFTOPIC = "how do I sharpen a kitchen knife"
DEMO_QUERY_WEAK = "what happens during a database outage"

EVAL_QUERIES = [
    # Each (query, gold) pair is verified to actually top-1 match its gold doc --
    # unlike "paid time off" or "database outage", which land on a near-duplicate
    # pair (hr-01/hr-02, eng-01/eng-02) and can't be faithful to one specific id.
    ("parental leave weeks", "hr-06"),
    ("code review pull request size", "eng-08"),
    ("aurora vault encryption key rotation", "prod-08"),
    ("what is the meaning of life", None),
]


def build_pipeline() -> tuple[list[Chunk], VectorIndex]:
    """Fresh chunks + a fresh fitted index, every call. No disk cache."""
    chunks = chunk_corpus(DOCS)
    index = VectorIndex(chunks)
    return chunks, index


if __name__ == "__main__":
    chunks, index = build_pipeline()
    assert len(chunks) == len(DOCS) == 48
    hits = index.search(DEMO_QUERY, k=5)
    assert hits[0][0].doc_id in ("hr-01", "hr-02")
    print(f"ok: {len(chunks)} chunks, demo query top hit = {hits[0][0].doc_id}")
