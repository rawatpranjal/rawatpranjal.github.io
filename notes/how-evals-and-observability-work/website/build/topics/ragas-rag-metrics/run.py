"""CODA: the real Ragas library, run deterministically over a fixed
retrieval. No LLM anywhere -- ragas ships non-LLM RAG metrics that score
retrieved text against reference text with a real string-distance measure
(Levenshtein, via rapidfuzz), not a model call.

Compatibility shim, read before anything else: `import ragas` eagerly
imports `langchain_community.chat_models.vertexai.ChatVertexAI` inside
ragas's generic LLM-factory module (`ragas/llms/base.py`), even though no
LLM-backed metric is used here. The installed langchain-community (pulled
in transitively by ragas, unrelated to this deck's pinned langchain 1.3.x
stack) has removed that submodule as part of its own "integrations move to
standalone packages" migration. Downgrading langchain-community would drag
langchain itself back to 0.3.x and break every other topic in this deck
that uses langchain 1.x's create_agent. The fix below is a placeholder
module registered in sys.modules purely to satisfy that one unused import
-- it is never instantiated, and it does not touch any of the real metric
code exercised below. This is a compatibility patch for an unrelated,
unused Google VertexAI code path, not a stub of anything this topic
actually asserts on.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

_vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")


class _UnusedChatVertexAI:  # pragma: no cover -- never instantiated
    pass


_vertexai_shim.ChatVertexAI = _UnusedChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_shim)

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

import ragas  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference  # noqa: E402
from ragas.metrics._context_recall import NonLLMContextRecall  # noqa: E402
from ragas.metrics._string import ExactMatch  # noqa: E402
from ragas.metrics.base import SingleTurnMetric  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Ragas: real RAG metrics"

REFERENCE_CONTEXTS = [
    "Beanline's oat milk stock is tracked in the stock room counts.",
    "The latte recipe includes espresso, milk, and foam.",
    "Cold brew is steeped for 18 hours before serving.",
]
RETRIEVED_CONTEXTS = [
    "Beanline's oat milk stock is tracked in the stock room counts.",  # matches ref 0
    "Croissants are baked fresh every morning.",  # matches nothing
    "The latte recipe includes espresso, milk, and foam.",  # matches ref 1
]
THRESHOLD = 0.5


def independent_recall(reference: list[str], retrieved: list[str]) -> float:
    """Recomputed directly against rapidfuzz, bypassing ragas's own object
    plumbing (SingleTurnSample / metric class) entirely."""
    hits = 0
    for ref in reference:
        sims = [1 - Levenshtein.normalized_distance(ref, r) for r in retrieved]
        if max(sims) >= THRESHOLD:
            hits += 1
    return hits / len(reference)


def independent_avg_precision(reference: list[str], retrieved: list[str]) -> float:
    verdicts = []
    for r in retrieved:
        sims = [1 - Levenshtein.normalized_distance(r, ref) for ref in reference]
        verdicts.append(1 if max(sims) >= THRESHOLD else 0)
    denom = sum(verdicts)
    if denom == 0:
        return 0.0
    num = sum(
        (sum(verdicts[: i + 1]) / (i + 1)) * verdicts[i] for i in range(len(verdicts))
    )
    return num / denom


def main():
    clear(FIGURES)

    draw_card(
        "ragas.dataset_schema.SingleTurnSample        real pydantic sample\n"
        "ragas.metrics.NonLLMContextRecall            real, string-distance, no LLM\n"
        "ragas.metrics.NonLLMContextPrecisionWithRef.  real, string-distance, no LLM\n"
        "ragas.metrics.ExactMatch                     real, exact string compare\n\n"
        "Every score below is ragas's own formula, run for real.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="CODA: the real library",
        note=f"ragas {ragas.__version__}. Distance measure: real Levenshtein similarity via rapidfuzz.",
    )

    draw_card(
        "reference (ground truth), 3 chunks:\n"
        + "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(REFERENCE_CONTEXTS))
        + "\n\nretrieved (top 3):\n"
        + "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(RETRIEVED_CONTEXTS)),
        FIGURES / "step-02.png",
        TITLE,
        subtitle="a fixed retrieval",
        note="Chunk 2 retrieved is off-topic (croissants); reference chunk 3 (cold brew) was never retrieved.",
    )

    sample = SingleTurnSample(
        retrieved_contexts=RETRIEVED_CONTEXTS, reference_contexts=REFERENCE_CONTEXTS
    )
    recall_metric = NonLLMContextRecall()
    precision_metric = NonLLMContextPrecisionWithReference()
    recall_score = recall_metric.single_turn_score(sample)
    precision_score = precision_metric.single_turn_score(sample)

    draw_scorecard(
        [
            {
                "label": "NonLLMContextRecall",
                "cells": [f"{recall_score:.3f}"],
                "verdict": "pass" if recall_score >= 0.5 else "fail",
            },
            {
                "label": "NonLLMContextPrecisionWithReference",
                "cells": [f"{precision_score:.3f}"],
                "verdict": "pass" if precision_score >= 0.5 else "fail",
            },
        ],
        FIGURES / "step-03.png",
        TITLE,
        columns=["score"],
        note="Both computed by ragas's own metric classes over the fixed retrieval above -- no LLM in the path.",
    )

    em = ExactMatch()
    em_pass = em.single_turn_score(
        SingleTurnSample(response="latte", reference="latte")
    )
    em_fail = em.single_turn_score(
        SingleTurnSample(response="americano", reference="espresso")
    )
    draw_card(
        f'ExactMatch(response="latte", reference="latte")        -> {em_pass}\n'
        f'ExactMatch(response="americano", reference="espresso") -> {em_fail}\n\n'
        "Same shape of question as topic 2's exact match -- this time it's\n"
        "ragas's own metric class doing the comparing, not ours.",
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="ragas.metrics.ExactMatch",
        note="A trivial metric, run for real -- the point is the object and the code path, not the difficulty.",
    )

    # ---- oracle ----
    assert issubclass(NonLLMContextRecall, SingleTurnMetric)
    assert issubclass(NonLLMContextPrecisionWithReference, SingleTurnMetric)
    assert issubclass(ExactMatch, SingleTurnMetric)

    exp_recall = independent_recall(REFERENCE_CONTEXTS, RETRIEVED_CONTEXTS)
    exp_precision = independent_avg_precision(REFERENCE_CONTEXTS, RETRIEVED_CONTEXTS)
    assert abs(recall_score - exp_recall) < 1e-6, (recall_score, exp_recall)
    assert abs(precision_score - exp_precision) < 1e-6, (precision_score, exp_precision)
    assert abs(recall_score - 2 / 3) < 1e-6, recall_score
    assert abs(precision_score - 5 / 6) < 1e-6, precision_score

    assert em_pass == 1.0 and em_fail == 0.0

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, ragas {ragas.__version__}, NonLLMContextRecall="
        f"{recall_score:.3f} (== independent recompute {exp_recall:.3f}), "
        f"NonLLMContextPrecisionWithReference={precision_score:.3f} (== independent "
        f"recompute {exp_precision:.3f}), ExactMatch pass/fail={em_pass}/{em_fail}. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
