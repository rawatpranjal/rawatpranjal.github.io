"""Chunking the corpus: splitting documents into overlapping passages.

Confirms that most of Aurora Cloud's 48 short docs collapse to a single chunk
under the default window (size=40, overlap=10), then shrinks the window until
a real document splits for real, and proves the overlap mechanism against
that document's actual words -- nothing here is staged.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from corpus import DOCS  # noqa: E402
from ragviz import chunk_corpus, clear, draw_retrieval_table, draw_system  # noqa: E402

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)

    # ---- step 1: under the default window, most docs stay whole -----------
    default_chunks = chunk_corpus(DOCS, size=40, overlap=10)
    by_doc_default = defaultdict(list)
    for c in default_chunks:
        by_doc_default[c.doc_id].append(c)
    n_single = sum(1 for cs in by_doc_default.values() if len(cs) == 1)
    assert n_single / len(DOCS) >= 0.9, (
        "most docs should reduce to exactly one chunk under the default window"
    )

    # ---- step 2: shrink the window until a real doc splits for real -------
    small_chunks = chunk_corpus(DOCS, size=15, overlap=5)
    by_doc_small = defaultdict(list)
    for c in small_chunks:
        by_doc_small[c.doc_id].append(c)
    multi = [(doc_id, cs) for doc_id, cs in by_doc_small.items() if len(cs) >= 2]
    assert multi, "a size=15 window should split at least one document"

    # pick the doc with the most pieces; break ties by corpus order
    doc_order = {d.id: i for i, d in enumerate(DOCS)}
    doc_id, doc_chunks = max(multi, key=lambda kv: (len(kv[1]), -doc_order[kv[0]]))
    doc = next(d for d in DOCS if d.id == doc_id)
    n_words = len(doc.text.split())

    draw_system(
        FIGURES / "step-01.png",
        stages=["one long document", "chunk(size=15, overlap=5)", "N chunks"],
        arrows=["split", f"{len(doc_chunks)}x"],
        title="Splitting a document into chunks",
        note=(
            f'"{doc.title}" is {n_words} words: whole under size=40, but split '
            f"into {len(doc_chunks)} overlapping pieces under size=15."
        ),
    )

    overlap_preview = (
        "chunk 1 ends '... "
        + " ".join(doc_chunks[0].text.split()[-5:])
        + "' -- chunk 2 begins '"
        + " ".join(doc_chunks[1].text.split()[:5])
        + " ...' (the shared 5-word overlap)"
    )
    draw_retrieval_table(
        [(chunk, i) for i, chunk in enumerate(doc_chunks)],
        FIGURES / "step-02.png",
        "One document's chunks, in order",
        query=doc.title,
        note=overlap_preview,
    )

    draw_system(
        FIGURES / "step-03.png",
        stages=[doc.title, "size=40, overlap=10", "size=15, overlap=5"],
        arrows=["1 chunk", f"{len(doc_chunks)} chunks"],
        title="Same document, two window sizes",
        note="A wide window keeps a short doc whole. A narrow one forces real splits.",
    )

    # ---- oracle: the split and the overlap mechanism are both real --------
    assert len(doc_chunks) >= 2, f"{doc_id} should have split into 2+ chunks"
    for i in range(len(doc_chunks) - 1):
        cur_tail = doc_chunks[i].text.split()[-5:]
        next_head = doc_chunks[i + 1].text.split()[:5]
        assert cur_tail == next_head, (
            f"chunk {i} and {i + 1} of {doc_id} should share a 5-word overlap"
        )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(
        f"{len(figs)} figures. default window: {n_single}/{len(DOCS)} docs -> "
        f"1 chunk. {doc_id} ({n_words} words) -> {len(doc_chunks)} chunks under "
        "size=15, overlap verified. All checks passed."
    )


if __name__ == "__main__":
    main()
