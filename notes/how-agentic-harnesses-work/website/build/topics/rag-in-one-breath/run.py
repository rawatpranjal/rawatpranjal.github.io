"""RAG, in one breath: retrieval is just one more way to fill the column.

Deliberately brief -- a sibling deck covers RAG end to end. Here the model
needs one fact from a real four-file docs/ corpus it has never seen; a real
TF-IDF search finds the right file, and the top hit becomes one more message
in the context.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import DOCS, build_workspace, mini_search  # noqa: E402
from harness import Agent, Harness, Message, ScriptedModel, Turn, builtin_tools  # noqa: E402
from harnessviz import clear, draw_frame  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "RAG, in one breath"
QUERY = "how should priorities order list"


def main():
    clear(FIGURES)
    ws = build_workspace(stage=9, docs=True)

    model = ScriptedModel(
        [
            Turn(
                text="[1] says list orders high priority first, then normal, "
                "then low, so I'll sort todo.py's list on that field."
            )
        ]
    )
    agent = Agent(model, builtin_tools(ws), ws, budget=600)
    h = Harness(ws)

    # step 1: the corpus sits on disk; the model's context holds nothing of it.
    snap1 = h.snap("corpus", agent, loop_node="context")

    agent.append(Message("user", "text", QUERY, pinned=True))

    # step 2: a real TF-IDF search; the top hit becomes a numbered excerpt.
    hits = mini_search(QUERY, DOCS, k=2)
    all_scores = dict(mini_search(QUERY, DOCS, k=len(DOCS)))
    top_path, top_score = hits[0]
    lunch_score = all_scores["docs/team-lunch.md"]
    excerpt = Message("system", "memory", f"[1] {top_path}: {DOCS[top_path]}")
    agent.append(excerpt)
    snap2 = h.snap("retrieved", agent, loop_node="context")

    # step 3: the model answers, citing the excerpt.
    turn = model.complete(agent.context)
    agent.append(Message("assistant", "text", turn.text))
    snap3 = h.snap("answered", agent, loop_node="done")

    draw_frame(
        snap1,
        FIGURES / "step-01.png",
        TITLE,
        right="files",
        note="A four-file corpus on disk, one of them off-topic, a corpus the model has never seen.",
    )
    draw_frame(
        snap2,
        FIGURES / "step-02.png",
        TITLE,
        right="none",
        note=f"A real TF-IDF search scores {top_path} at {top_score:.2f}, team-lunch.md scores {lunch_score:.2f}, near zero.",
    )
    draw_frame(
        snap3,
        FIGURES / "step-03.png",
        TITLE,
        right="none",
        note="The answer cites [1]. Retrieval is just another way to fill the column, the full story is its own deck.",
    )

    # ---- oracle ----
    assert hits[0][0] == "docs/priority-spec.md", hits
    assert "docs/team-lunch.md" not in [p for p, _ in hits], hits
    assert "[1]" in turn.text
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {figs}"
    print(
        f"{len(figs)} figures, top hit {top_path} @ {top_score:.2f}, "
        f"team-lunch.md @ {lunch_score:.2f}. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
