"""Chroma, for real: the same corpus, the same query, a real vector store.

Every earlier topic ran a hand-built TF-IDF index (build/lib/ragviz.py). This coda
swaps that toy index for chromadb, a real embedded vector store, and swaps the toy
word-window chunker for chonkie, a real chunking library. Same 48-doc Aurora Cloud
corpus, same demo query as the retrieval topic -- nothing here is staged.

Requires the dedicated coda venv (chromadb, chonkie, matplotlib, scikit-learn --
see build/requirements-coda.txt), not the deck's main pipeline venv:

    /Users/pranjal/.venvs/rag-coda/bin/python topics/chroma-real/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

import numpy as np
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chonkie import TokenChunker

from corpus import DOCS  # noqa: E402
from ragviz import Chunk, clear, draw_retrieval_table, draw_system  # noqa: E402

FIGURES = HERE / "figures"
DEMO_QUERY = "how many days of paid time off do I get"  # same query as topics/retrieval


def main():
    clear(FIGURES)

    draw_system(
        FIGURES / "step-01.png",
        stages=["48 docs (same corpus)", "chromadb.Client()", "coll.query()"],
        arrows=["add()", "top-k"],
        title="The toy index, swapped for a real vector store",
        note="Same 48-document Aurora Cloud corpus. A real chromadb collection replaces the hand-built TF-IDF index.",
    )

    # ---- real chonkie: chunk one document, deterministic -------------------
    target = next(d for d in DOCS if d.id == "hr-01")
    chunker = TokenChunker(tokenizer="character", chunk_size=90, chunk_overlap=20)
    real_chunks = chunker.chunk(target.text)
    real_chunks_again = chunker.chunk(target.text)  # run twice, compare

    display_chunks = [
        Chunk(f"{target.id}-real-c{i}", target.id, target.title, target.tag, c.text)
        for i, c in enumerate(real_chunks)
    ]
    draw_retrieval_table(
        [(c, float(len(c.text))) for c in display_chunks],
        FIGURES / "step-02.png",
        "Chonkie's real chunks of one document",
        query=target.title,
        score_label="chars",
        highlight_top=False,
        note=(
            f"Real chonkie.TokenChunker output, chunk_size=90 chars, overlap=20 -- "
            f"{len(real_chunks)} chunks with real character-offset boundaries, not a mockup."
        ),
    )

    # ---- real chromadb: add the same corpus, query it -----------------------
    client = chromadb.Client()
    coll = client.create_collection("aurora-cloud-real")
    coll.add(ids=[d.id for d in DOCS], documents=[d.text for d in DOCS])

    result = coll.query(query_texts=[DEMO_QUERY], n_results=5)
    result_again = coll.query(
        query_texts=[DEMO_QUERY], n_results=5
    )  # run twice, compare

    hit_ids = result["ids"][0]
    hit_dists = result["distances"][0]
    doc_by_id = {d.id: d for d in DOCS}
    display_hits = [
        (
            Chunk(
                f"{hid}-c0",
                hid,
                doc_by_id[hid].title,
                doc_by_id[hid].tag,
                doc_by_id[hid].text,
            ),
            dist,
        )
        for hid, dist in zip(hit_ids, hit_dists)
    ]
    draw_retrieval_table(
        display_hits,
        FIGURES / "step-03.png",
        "Chroma's real top-5, same query",
        query=DEMO_QUERY,
        score_label="l2 dist",
        highlight_top=True,
        note="Real chromadb .query() against the real corpus -- lower distance is closer. Same demo query as the toy retrieval topic.",
    )

    # ---- oracle: real chromadb + real chonkie, checked, not assumed --------
    assert isinstance(coll, chromadb.api.models.Collection.Collection), (
        "coll must be a real chromadb Collection, not a stub"
    )

    # Independently recompute the ranking: embed the corpus and the query with
    # chroma's own default embedding function, but score with plain numpy squared
    # L2 (matching the collection's "l2" hnsw space) instead of trusting chroma's
    # internal search -- this is the "recompute independently and compare".
    embed = embedding_functions.DefaultEmbeddingFunction()
    doc_vecs = np.array(embed([d.text for d in DOCS]))
    query_vec = np.array(embed([DEMO_QUERY]))[0]
    sq_l2 = ((doc_vecs - query_vec) ** 2).sum(axis=1)
    expected_order = np.argsort(sq_l2)[:5]
    expected_ids = [DOCS[i].id for i in expected_order]
    expected_dists = [float(sq_l2[i]) for i in expected_order]

    assert hit_ids == expected_ids, (
        "chroma's returned top-5 doc ids must match an independent numpy recompute"
    )
    assert np.allclose(hit_dists, expected_dists, atol=1e-4), (
        "chroma's returned distances must match the independent recompute"
    )
    assert hit_ids[0] in ("hr-01", "hr-02"), (
        "the PTO query should still hit a PTO doc first, same as the toy TF-IDF pipeline"
    )
    assert hit_ids == result_again["ids"][0], (
        "two runs of the same query against the same collection must return identical ids"
    )
    assert hit_dists == result_again["distances"][0], (
        "two runs must return identical distances -- the embedding function is deterministic"
    )

    assert [c.text for c in real_chunks] == [c.text for c in real_chunks_again], (
        "chonkie chunking must be deterministic across runs"
    )
    assert len(real_chunks) == 3, (
        f"expected 3 chunks of {target.id}, got {len(real_chunks)}"
    )
    for c in real_chunks:
        assert target.text[c.start_index : c.end_index] == c.text, (
            "each chunk's text must match its own start/end offsets into the source doc"
        )
    for i in range(len(real_chunks) - 1):
        overlap = real_chunks[i].end_index - real_chunks[i + 1].start_index
        assert overlap == 20, (
            f"expected a 20-char overlap between chunks {i} and {i + 1}"
        )
        assert real_chunks[i].text[-overlap:] == real_chunks[i + 1].text[:overlap], (
            "the overlapping span must be the literal shared text, not just matching lengths"
        )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"

    print(
        f"{len(figs)} figures. chroma: top hit {hit_ids[0]} @ l2={hit_dists[0]:.4f} "
        f"(matches independent numpy recompute, repeat query identical). "
        f"chonkie: {target.id} -> {len(real_chunks)} real chunks, boundaries verified, "
        "repeat chunk identical. All checks passed."
    )


if __name__ == "__main__":
    main()
