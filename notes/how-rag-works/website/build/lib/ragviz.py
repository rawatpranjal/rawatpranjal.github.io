"""Run a real toy RAG pipeline, snapshot the true state, draw it.

Every figure in this collection is a photograph of a retrieval pipeline that
actually ran against the real corpus (lib/corpus.py), never a hand-drawn mockup.
Embeddings are real TF-IDF vectors, similarity scores are real cosine similarities,
BM25 scores are computed for real, and the agent trace is a real multi-step search
loop against the real index. Nothing is faked for the picture.

Ported from the git deck's gitviz.py: the box/arrow/label drawing primitives,
_wrap(), the dark palette, and draw_system() are domain-agnostic and reused as-is.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from corpus import Doc

# Dark palette, matching the deck's design tokens (bg #16171a, accent #7cb0de).
INK = "#dbe3ee"
MUTED = "#7d8894"
PAPER = "#16171a"
PANEL = "#1c1f26"
NODE_FILL = "#1d2733"
NODE_EDGE = "#7cb0de"
LANE_COLORS = ["#7cb0de", "#e8834a", "#b19cf5", "#6bbf8a", "#e06b9c", "#7aa2f0"]
NEW_COLOR = "#e8a13c"


# ---- chunking -------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    tag: str
    text: str


def chunk_corpus(docs: list[Doc], size: int = 40, overlap: int = 10) -> list[Chunk]:
    """Fixed word-window chunker. Most of these short docs are one chunk; a
    long doc would split into overlapping windows of `size` words."""
    chunks: list[Chunk] = []
    for doc in docs:
        words = doc.text.split()
        if len(words) <= size:
            chunks.append(Chunk(f"{doc.id}-c0", doc.id, doc.title, doc.tag, doc.text))
            continue
        start, i = 0, 0
        while start < len(words):
            window = words[start : start + size]
            chunks.append(
                Chunk(f"{doc.id}-c{i}", doc.id, doc.title, doc.tag, " ".join(window))
            )
            start += size - overlap
            i += 1
    return chunks


# ---- dense retrieval (TF-IDF + cosine) ------------------------------------


class VectorIndex:
    """Real TF-IDF vectors over the chunk texts, real cosine similarity search."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix)[0]
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def embedding_2d(self) -> np.ndarray:
        """A real 2D projection of the TF-IDF space, for the embedding-space figure."""
        svd = TruncatedSVD(n_components=2, random_state=0)
        return svd.fit_transform(self.matrix)


# ---- sparse retrieval (BM25) -----------------------------------------------


def bm25_search(
    chunks: list[Chunk], query: str, k: int = 5, k1: float = 1.5, b: float = 0.75
) -> list[tuple[Chunk, float]]:
    """A small pure-numpy BM25, independent of the TF-IDF index, for hybrid fusion."""
    tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    docs_tokens = [tokenize(c.text) for c in chunks]
    q_tokens = tokenize(query)
    doc_lens = np.array([len(t) for t in docs_tokens])
    avgdl = doc_lens.mean() if len(doc_lens) else 1.0

    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    n = len(chunks)

    scores = np.zeros(n)
    for term in q_tokens:
        if term not in df:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, tokens in enumerate(docs_tokens):
            tf = tokens.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_lens[i] / avgdl)
            scores[i] += idf * (tf * (k1 + 1)) / denom

    order = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def rrf_fuse(
    dense: list[tuple[Chunk, float]], sparse: list[tuple[Chunk, float]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """Reciprocal rank fusion: combine two rankings by rank, not raw score."""
    rank_score: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for rank, (chunk, _) in enumerate(dense):
        rank_score[chunk.id] = rank_score.get(chunk.id, 0) + 1 / (k + rank + 1)
        chunk_by_id[chunk.id] = chunk
    for rank, (chunk, _) in enumerate(sparse):
        rank_score[chunk.id] = rank_score.get(chunk.id, 0) + 1 / (k + rank + 1)
        chunk_by_id[chunk.id] = chunk
    ranked = sorted(rank_score.items(), key=lambda kv: -kv[1])
    return [(chunk_by_id[cid], score) for cid, score in ranked]


# ---- reranking --------------------------------------------------------------


def rerank(
    query: str, candidates: list[tuple[Chunk, float]]
) -> list[tuple[Chunk, float]]:
    """A deterministic keyword-overlap rescorer: counts exact query-word hits in
    each chunk, breaking ties by the original score. Real enough to demote a
    near-duplicate that scores similarly but overlaps the query less exactly."""
    q_words = set(re.findall(r"[a-z0-9]+", query.lower()))

    def overlap_score(chunk: Chunk) -> float:
        c_words = re.findall(r"[a-z0-9]+", chunk.text.lower())
        hits = sum(1 for w in c_words if w in q_words)
        return hits / (len(c_words) ** 0.5)

    rescored = [(c, overlap_score(c), orig) for c, orig in candidates]
    rescored.sort(key=lambda t: (-t[1], -t[2]))
    return [(c, score) for c, score, _ in rescored]


# ---- augmentation + generation -----------------------------------------------


def assemble_prompt(query: str, chunks: list[Chunk]) -> str:
    lines = [f"Answer using only the excerpts below. Question: {query}", ""]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({c.doc_title}) {c.text}")
    lines.append("")
    lines.append("Cite each fact as [n]. If the excerpts don't cover it, say so.")
    return "\n".join(lines)


@dataclass
class Answer:
    text: str
    citations: list[int]
    refused: bool


REFUSAL_THRESHOLD = 0.12


def generate_answer(query: str, chunks: list[Chunk], scores: list[float]) -> Answer:
    """Template-based, not an LLM call: stitches the top chunk's text behind a
    [1] citation. Refuses when the best real similarity is below threshold,
    the same cite-or-refuse gate a real RAG system needs."""
    if not scores or scores[0] < REFUSAL_THRESHOLD:
        return Answer(
            text=f'The corpus doesn\'t cover "{query}" well enough to answer.',
            citations=[],
            refused=True,
        )
    top = chunks[0]
    sentence = top.text.split(".")[0].strip()
    return Answer(text=f"{sentence}. [1]", citations=[1], refused=False)


# ---- agentic tool-use loop ---------------------------------------------------


@dataclass
class ToolCall:
    query: str
    top_chunks: list[tuple[Chunk, float]]


@dataclass
class AgentTrace:
    question: str
    calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""


def run_agent(question: str, index: VectorIndex, k: int = 3) -> AgentTrace:
    """A scripted, deterministic multi-hop loop: search, check confidence, and
    only reformulate + search again if the first call came back weak."""
    trace = AgentTrace(question=question)
    first = index.search(question, k=k)
    trace.calls.append(ToolCall(question, first))

    if first[0][1] < 0.2:
        # Weak first hit: reformulate by dropping stopword-like filler and
        # retrying once, the way an agent would after a bad tool result.
        reformulated = " ".join(w for w in question.split() if len(w) > 3)
        second = index.search(reformulated, k=k)
        trace.calls.append(ToolCall(reformulated, second))
        best = second
    else:
        best = first

    top_chunk, top_score = best[0]
    if top_score < REFUSAL_THRESHOLD:
        trace.final_answer = "Not covered in the corpus."
    else:
        trace.final_answer = f"{top_chunk.text.split('.')[0].strip()}. [1]"
    return trace


# ---- evaluation ---------------------------------------------------------------


def eval_retrieval(
    index: VectorIndex, queries_with_gold: list[tuple[str, str]], k: int = 5
) -> list[dict]:
    """queries_with_gold: (query, expected doc_id). Returns one row per query with
    real recall@k and a faithfulness pass/fail against the real top hit."""
    rows = []
    for query, gold_doc_id in queries_with_gold:
        hits = index.search(query, k=k)
        hit_doc_ids = [c.doc_id for c, _ in hits]
        recall = 1.0 if gold_doc_id in hit_doc_ids else 0.0
        top_chunk, top_score = hits[0]
        faithful = top_score >= REFUSAL_THRESHOLD and top_chunk.doc_id == gold_doc_id
        rows.append(
            {
                "query": query,
                "gold": gold_doc_id,
                "top_hit": top_chunk.doc_id,
                "top_score": round(top_score, 3),
                f"recall@{k}": recall,
                "faithful": faithful,
            }
        )
    return rows


# ---- shared drawing primitives (ported from gitviz.py, domain-agnostic) -----


def _wrap(text: str, width_in: float, fontsize: float = 7.5) -> list[str]:
    chars = max(20, int((width_in - 0.2) * 72 / (0.52 * fontsize)))
    return textwrap.wrap(text, width=chars) or [text]


def draw_system(
    path: Path,
    stages: list[str],
    arrows: list[str],
    title: str,
    frame_label: str = "",
    note: str = "",
):
    """A labeled-stages-with-arrows schematic. Ported verbatim from gitviz.py."""
    n = len(stages)
    fig, ax = plt.subplots(figsize=(2.6 * n + 0.6, 3.0), facecolor=PAPER)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if frame_label:
        ax.add_patch(
            FancyBboxPatch(
                (0.04, 0.10),
                n - 0.08,
                0.66,
                boxstyle="round,pad=0.01",
                facecolor=PANEL,
                edgecolor=MUTED,
                linewidth=1.0,
                linestyle="--",
            )
        )
        ax.text(0.12, 0.70, frame_label, fontsize=8.5, color=MUTED, fontweight="bold")

    for i, label in enumerate(stages):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        ax.add_patch(
            FancyBboxPatch(
                (i + 0.14, 0.34),
                0.72,
                0.20,
                boxstyle="round,pad=0.01",
                facecolor=PAPER,
                edgecolor=color,
                linewidth=1.6,
            )
        )
        ax.text(
            i + 0.5,
            0.44,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            family="monospace",
            color=INK,
        )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (i + 0.88, 0.44),
                    (i + 1.12, 0.44),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color=INK,
                    linewidth=1.3,
                )
            )
            if i < len(arrows):
                ax.text(
                    i + 1.0,
                    0.55,
                    arrows[i],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    family="monospace",
                    color=INK,
                )

    fig.text(
        0.015, 0.955, title, fontsize=10.5, fontweight="bold", color=INK, ha="left"
    )
    if note:
        fig.text(0.015, 0.04, note, fontsize=7.5, color=INK, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


# ---- RAG-specific figures ----------------------------------------------------


def draw_embedding_space(
    index: VectorIndex,
    path: Path,
    title: str,
    highlight_query: str | None = None,
    note: str = "",
):
    """A real 2D TruncatedSVD projection of the TF-IDF space, colored by tag."""
    coords = index.embedding_2d()
    tag_colors = {
        "hr": LANE_COLORS[0],
        "product": LANE_COLORS[1],
        "eng": LANE_COLORS[2],
        "offtopic": MUTED,
    }
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    y_span = coords[:, 1].max() - coords[:, 1].min() if len(coords) else 1.0
    label_offset = max(y_span * 0.05, 0.02)
    for (x, y), chunk in zip(coords, index.chunks):
        ax.scatter(
            x,
            y,
            s=60,
            color=tag_colors.get(chunk.tag, INK),
            edgecolor=PAPER,
            linewidth=0.5,
            zorder=2,
        )
    if highlight_query:
        qvec = index.vectorizer.transform([highlight_query])
        svd = TruncatedSVD(n_components=2, random_state=0)
        svd.fit(index.matrix)
        qxy = svd.transform(qvec)[0]
        ax.scatter(
            qxy[0],
            qxy[1],
            s=180,
            marker="*",
            color=NEW_COLOR,
            edgecolor=INK,
            linewidth=1,
            zorder=3,
        )
        ax.text(
            qxy[0],
            qxy[1] + label_offset,
            "query",
            color=NEW_COLOR,
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t
        )
        for t, c in tag_colors.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        facecolor=PANEL,
        edgecolor=MUTED,
        labelcolor=INK,
        fontsize=8,
    )
    fig.text(0.02, 0.965, title, fontsize=12, fontweight="bold", color=INK, ha="left")
    if note:
        fig.text(0.02, 0.02, note, fontsize=8, color=INK, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def draw_retrieval_table(
    results: list[tuple[Chunk, float]],
    path: Path,
    title: str,
    query: str = "",
    score_label: str = "score",
    highlight_top: bool = True,
    note: str = "",
):
    """A ranked table of chunks, each row a real hit with its real score."""
    rows = len(results)
    fig, ax = plt.subplots(figsize=(9.5, 0.85 + 0.62 * rows), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(-rows - 0.3, 1.3)
    ax.axis("off")

    if query:
        ax.text(
            0.1, 0.9, f'query: "{query}"', fontsize=9, family="monospace", color=MUTED
        )

    for i, (chunk, score) in enumerate(results):
        y = -i
        color = NEW_COLOR if (highlight_top and i == 0) else INK
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y - 0.26),
                9.9,
                0.5,
                boxstyle="round,pad=0.02",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.4 if i == 0 else 0.8,
            )
        )
        ax.text(
            0.2,
            y,
            f"#{i + 1}",
            fontsize=9,
            family="monospace",
            color=color,
            va="center",
            fontweight="bold",
        )
        ax.text(0.75, y, chunk.doc_title, fontsize=9.5, color=color, va="center")
        ax.text(
            9.75,
            y,
            f"{score:.3f}",
            fontsize=9,
            family="monospace",
            color=color,
            va="center",
            ha="right",
        )

    fig.text(
        0.01,
        1 - 0.16 / (0.85 + 0.62 * rows),
        title,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    if note:
        fig.text(0.01, 0.02, note, fontsize=8, color=INK, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def draw_prompt_assembly(prompt: str, path: Path, title: str, note: str = ""):
    """The literal assembled prompt string, as a monospace card.

    Text is wrapped to the box's actual usable width (not the figure's outer
    width) and every wrapped row is drawn -- a long excerpt becomes several
    rows rather than overflowing the card or silently losing text.
    """
    source_lines = prompt.splitlines() or [" "]
    # Usable width in inches: box spans x=[0.05, 9.95] of a 10-wide axes on a
    # 9.0-inch figure, text starts at x=0.25, so ~8.6 data-units of margin
    # remain -- convert that to inches at the figure's actual width.
    usable_width_in = (9.95 - 0.25) / 10 * 9.0
    rows: list[tuple[str, str]] = []  # (text, color) one entry per rendered row
    for line in source_lines:
        color = NEW_COLOR if line.startswith("[") else INK
        wrapped = _wrap(line, usable_width_in, fontsize=8.5) or [line]
        for w in wrapped:
            rows.append((w, color))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.0, 0.9 + 0.32 * n), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(-n - 0.5, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.05, -n - 0.15),
            9.9,
            n + 0.9,
            boxstyle="round,pad=0.02",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=1.0,
        )
    )
    for i, (text, color) in enumerate(rows):
        ax.text(
            0.25,
            -i,
            text,
            fontsize=8.5,
            family="monospace",
            color=color,
            va="center",
        )
    fig.text(0.015, 0.94, title, fontsize=11, fontweight="bold", color=INK, ha="left")
    if note:
        # Axes data coordinates (already spanning [-n-0.5, 1.0]), not figure
        # fraction -- fig.text here previously took a raw data-space offset as
        # a [0,1] fraction, which is always out of range and, combined with
        # bbox_inches="tight", inflated the saved canvas to include the gap.
        ax.text(0.25, -n - 0.35, note, fontsize=7.5, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def draw_agent_trace(trace: AgentTrace, path: Path, title: str, note: str = ""):
    """Each tool call as a stacked step: query in, top results out."""
    n = len(trace.calls)
    fig, ax = plt.subplots(figsize=(9.0, 1.3 * n + 1.6), facecolor=PAPER)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3 * n - 0.6, 1.0)
    ax.axis("off")
    ax.text(
        0.1,
        0.7,
        f'question: "{trace.question}"',
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )

    for i, call in enumerate(trace.calls):
        y0 = -1.3 * i - 0.4
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y0 - 1.0),
                9.9,
                1.15,
                boxstyle="round,pad=0.02",
                facecolor=PANEL,
                edgecolor=LANE_COLORS[i % len(LANE_COLORS)],
                linewidth=1.3,
            )
        )
        ax.text(
            0.25,
            y0 - 0.15,
            f'search_corpus("{call.query}")',
            fontsize=8.5,
            family="monospace",
            color=LANE_COLORS[i % len(LANE_COLORS)],
            fontweight="bold",
        )
        top_chunk, top_score = call.top_chunks[0]
        ax.text(
            0.25,
            y0 - 0.55,
            f"-> top: {top_chunk.doc_title}  (score {top_score:.3f})",
            fontsize=8.0,
            family="monospace",
            color=INK,
        )
    y_final = -1.3 * n - 0.2
    ax.text(
        0.25,
        y_final,
        f"finish: {trace.final_answer}",
        fontsize=9,
        color=NEW_COLOR,
        fontweight="bold",
    )

    fig.text(0.015, 0.94, title, fontsize=11, fontweight="bold", color=INK, ha="left")
    if note:
        # Axes data coordinates, not figure fraction -- see draw_prompt_assembly.
        ax.text(0.25, -1.3 * n - 0.55, note, fontsize=7.5, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def draw_eval_results(rows: list[dict], path: Path, title: str, note: str = ""):
    """The eval table: one row per query, real recall@k and faithfulness."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.0, 0.9 + 0.55 * n), facecolor=PAPER)
    ax.set_xlim(0, 12)
    ax.set_ylim(-n - 0.3, 1.0)
    ax.axis("off")
    recall_key = next(k for k in rows[0] if k.startswith("recall@"))

    for i, row in enumerate(rows):
        y = -i
        ok = row["faithful"] and row[recall_key] == 1.0
        color = "#6bbf8a" if ok else "#e06b9c"
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y - 0.24),
                11.9,
                0.46,
                boxstyle="round,pad=0.02",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.text(0.2, y, row["query"], fontsize=8.5, color=INK, va="center")
        ax.text(
            7.5,
            y,
            f"{recall_key}={row[recall_key]:.0f}",
            fontsize=8,
            family="monospace",
            color=color,
            va="center",
        )
        ax.text(
            9.5,
            y,
            "faithful" if row["faithful"] else "unfaithful",
            fontsize=8,
            family="monospace",
            color=color,
            va="center",
        )
        ax.text(
            11.8,
            y,
            "PASS" if ok else "FAIL",
            fontsize=8.5,
            family="monospace",
            color=color,
            va="center",
            ha="right",
            fontweight="bold",
        )

    fig.text(
        0.01,
        1 - 0.2 / (0.9 + 0.55 * n),
        title,
        fontsize=11,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    if note:
        # Axes data coordinates, not figure fraction -- see draw_prompt_assembly.
        ax.text(0.2, -n - 0.15, note, fontsize=8, color=MUTED, va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def clear(*dirs: Path):
    """run.py regenerates figures/ from scratch, as the schema requires."""
    import shutil

    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


if __name__ == "__main__":
    import tempfile

    from corpus import DOCS

    chunks = chunk_corpus(DOCS)
    assert len(chunks) == len(DOCS), "these short docs should chunk 1:1"

    index = VectorIndex(chunks)
    hits = index.search("how many days of paid time off do I get", k=5)
    assert hits[0][0].doc_id in ("hr-01", "hr-02"), "PTO query should hit the PTO doc"
    assert hits[0][1] > 0, "a real match should have positive cosine similarity"

    offtopic_hits = index.search("how do I sharpen a kitchen knife", k=3)
    assert offtopic_hits[0][0].tag == "offtopic", (
        "an offtopic query should hit offtopic"
    )

    sparse_hits = bm25_search(chunks, "database outage failover", k=5)
    assert sparse_hits[0][0].doc_id in ("eng-01", "eng-02"), (
        "BM25 should surface the runbook"
    )

    fused = rrf_fuse(hits, sparse_hits)
    assert len(fused) > 0

    reranked = rerank("how many days of paid time off", hits)
    assert reranked[0][0].tag == "hr"

    prompt = assemble_prompt("PTO policy", [c for c, _ in hits[:2]])
    assert "[1]" in prompt and "[2]" in prompt

    scores = [s for _, s in hits]
    answer = generate_answer("PTO policy", [c for c, _ in hits], scores)
    assert not answer.refused

    bad_answer = generate_answer("what is the meaning of life", chunks, [0.0])
    assert bad_answer.refused, "a low-similarity query should refuse"

    trace = run_agent("what happens during a database outage", index)
    assert trace.final_answer

    eval_rows = eval_retrieval(
        index,
        [
            ("parental leave weeks", "hr-06"),
            ("code review pull request size", "eng-08"),
        ],
    )
    assert all("faithful" in r for r in eval_rows)
    assert all(r["faithful"] for r in eval_rows), (
        "these two queries are verified to top-1 match their gold doc cleanly"
    )

    out = Path(tempfile.mkdtemp(prefix="ragviz-check-"))
    draw_embedding_space(
        index, out / "embed.png", "Embedding space", highlight_query="PTO policy"
    )
    draw_retrieval_table(hits, out / "retrieval.png", "Retrieval", query="PTO policy")
    draw_prompt_assembly(prompt, out / "prompt.png", "Prompt assembly")
    draw_agent_trace(trace, out / "agent.png", "Agent trace")
    draw_eval_results(eval_rows, out / "eval.png", "Eval results")
    draw_system(
        out / "system.png",
        ["corpus", "chunks", "index"],
        ["chunk", "embed"],
        "Naive RAG pipeline",
    )
    for name in ("embed", "retrieval", "prompt", "agent", "eval", "system"):
        assert (out / f"{name}.png").exists()

    print(f"ok: {len(chunks)} chunks indexed, 6 figures rendered to {out}")
