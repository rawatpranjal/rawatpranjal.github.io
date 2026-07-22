"""Agentic RAG: an agent decides whether to search again, real trace.

Drives the real run_agent() loop against the real corpus. Nothing here is staged --
the tool calls, the scores, and the decision to reformulate are all real.
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
from ragviz import clear, draw_agent_trace, draw_system, run_agent  # noqa: E402

FIGURES = HERE / "figures"

# DEMO_QUERY_WEAK ("what happens during a database outage") scores 0.212 -- just
# above the 0.2 reformulation threshold, so the agent is confident on the first
# try and never makes a second call. Tried alongside three other vague/awkward
# phrasings; this one actually triggers the two-call path.
QUESTION = "what's up with the database when it just kind of goes down"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    trace = run_agent(QUESTION, index, k=3)

    draw_agent_trace(
        trace,
        FIGURES / "step-01.png",
        "The agent's tool-use trace",
    )

    draw_system(
        FIGURES / "step-02.png",
        stages=["question", "search_corpus", "confident?", "answer"],
        arrows=["call", "check", "yes"],
        title="The agent's decision loop",
        note="If the first search is weak, it reformulates and searches again before answering.",
    )

    # The oracle: the agent always finishes with an answer, and if it reformulated,
    # the reformulation actually changed the query it sent.
    assert trace.final_answer, "the agent must always produce a final answer"
    if len(trace.calls) == 2:
        assert trace.calls[0].query != trace.calls[1].query, (
            "a reformulation that doesn't change the query isn't a reformulation"
        )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {figs}"

    print(
        f"{len(figs)} figures, {len(trace.calls)} tool call(s), "
        f"final: {trace.final_answer!r}. All checks passed."
    )


if __name__ == "__main__":
    main()
