"""Evaluation: a fixed question set, real recall@k, real faithfulness.

Drives the real TF-IDF index against the real corpus with the deck's frozen
EVAL_QUERIES set. Nothing here is staged -- every PASS/FAIL in the figures comes
from a real search against the real index.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import EVAL_QUERIES, build_pipeline  # noqa: E402
from ragviz import clear, draw_eval_results, draw_system, eval_retrieval  # noqa: E402

FIGURES = HERE / "figures"


def main():
    clear(FIGURES)
    chunks, index = build_pipeline()

    rows = eval_retrieval(index, EVAL_QUERIES, k=5)

    draw_eval_results(
        rows,
        FIGURES / "step-01.png",
        "Evaluation: recall@5 and faithfulness",
        note=(
            "Three questions have a real answer in the corpus. One doesn't -- and a "
            "FAIL there is the system correctly admitting it, not a bug."
        ),
    )

    draw_system(
        FIGURES / "step-02.png",
        stages=["fixed question set", "search each", "check recall@k + faithfulness"],
        arrows=["run", "score"],
        title="How you actually know it's working",
    )

    # The oracle: the answerable questions really do pass, the 1 unanswerable
    # one really does fail -- both checked against real search, not asserted.
    for row in rows:
        if row["gold"] is not None:
            assert row["recall@5"] == 1.0 and row["faithful"] is True, (
                f"expected a real pass for {row['query']!r}, got {row}"
            )
        else:
            assert row["recall@5"] == 0.0, (
                f"the unanswerable question should score recall@5=0, got {row}"
            )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {figs}"

    n_answerable = sum(1 for r in rows if r["gold"] is not None)
    print(
        f"{len(figs)} figures, {n_answerable}/{n_answerable} answerable queries passed. All checks passed."
    )


if __name__ == "__main__":
    main()
