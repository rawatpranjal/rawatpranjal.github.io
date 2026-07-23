"""Three offline metrics, in increasing sophistication: exact match on a
scripted prediction stream, recall@k over a fixed retrieval, and a
deterministic illustration of why surface metrics like BLEU/ROUGE mislead
-- two answers that mean the same thing but share almost no tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_card, draw_test_set  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Offline metrics: exact match, recall, and the BLEU trap"

CASES = [
    ("c1", "one large oat milk latte", "latte"),
    ("c2", "a cappuccino and a croissant to go", "cappuccino"),
    ("c3", "medium cold brew with vanilla", "cold brew"),
    ("c4", "two small espressos, one with an extra shot", "espresso"),
    ("c5", "a small mocha and a cookie", "mocha"),
]
# case 4's prediction is deliberately wrong -- an americano was never on the menu.
PREDICTIONS = ["latte", "cappuccino", "cold brew", "americano", "mocha"]

RELEVANT_DOCS = {"latte-doc", "milk-doc", "oat-doc"}
RETRIEVED_TOP3 = ["latte-doc", "unrelated-doc", "milk-doc"]

REFERENCE = "A large oat milk latte is $6.25."
CANDIDATE = "Comes to $6.25 total for the oversized oat-milk coffee you ordered."


def _tokens(text: str) -> set[str]:
    return {w.strip(".,!?").lower() for w in text.split()}


def token_overlap(a: str, b: str) -> float:
    """A crude proxy for what BLEU/ROUGE actually measure: shared surface
    tokens over the union, nothing about meaning."""
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb)


def extract_price(text: str) -> str | None:
    import re

    m = re.search(r"\$\d+\.\d{2}", text)
    return m.group(0) if m else None


def main():
    clear(FIGURES)

    # ---- exact match over a scripted prediction stream ----
    model = scripted(*PREDICTIONS)
    predicted = [
        model.invoke([HumanMessage(content=text)]).content for _, text, _ in CASES
    ]
    hits = [p == e for p, (_, _, e) in zip(predicted, CASES)]
    exact_match = sum(hits) / len(CASES)

    draw_card(
        "exact match     did the prediction equal the expected string, verbatim\n"
        "recall@k        of the truly relevant items, how many did retrieval surface\n"
        "BLEU / ROUGE     surface n-gram overlap with a reference -- not meaning",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="three offline metrics, three different questions",
        note="All three are pure functions of (prediction, expected). No model is in the scoring path.",
    )

    draw_test_set(
        [
            {
                "id": cid,
                "input": text,
                "expected": expected,
                "verdict": "pass" if h else "fail",
            }
            for (cid, text, expected), h in zip(CASES, hits)
        ],
        FIGURES / "step-02.png",
        TITLE,
        note=f"Exact match: {sum(hits)}/{len(CASES)} = {exact_match:.0%}. c4 predicted 'americano' -- not on the menu.",
    )

    # ---- recall@k over a fixed retrieval ----
    hit_set = set(RETRIEVED_TOP3) & RELEVANT_DOCS
    recall_at_3 = len(hit_set) / len(RELEVANT_DOCS)
    draw_card(
        f"relevant (ground truth): {sorted(RELEVANT_DOCS)}\n"
        f"retrieved, top 3:        {RETRIEVED_TOP3}\n"
        f"hit:                     {sorted(hit_set)}\n\n"
        f"recall@3 = {len(hit_set)}/{len(RELEVANT_DOCS)} = {recall_at_3:.3f}",
        FIGURES / "step-03.png",
        TITLE,
        subtitle="recall@k over a fixed retrieval",
        note="A retrieval quality metric: of what actually mattered, how much did the top-k surface.",
    )

    # ---- the BLEU/ROUGE trap ----
    overlap = token_overlap(REFERENCE, CANDIDATE)
    price_ref, price_cand = extract_price(REFERENCE), extract_price(CANDIDATE)
    draw_card(
        f'reference:  "{REFERENCE}"\ncandidate:  "{CANDIDATE}"',
        FIGURES / "step-04.png",
        TITLE,
        subtitle="two answers, the same meaning, very different words",
        note="A human reading both would call these the same answer.",
    )

    draw_test_set(
        [
            {
                "id": "surface",
                "input": "token overlap (BLEU/ROUGE proxy)",
                "expected": f"{overlap:.3f}",
                "verdict": "fail",
            },
            {
                "id": "targeted",
                "input": "price extracted from each ($ regex)",
                "expected": f"{price_ref} == {price_cand}",
                "verdict": "pass",
            },
        ],
        FIGURES / "step-05.png",
        TITLE,
        note="Surface overlap scores this pair as almost unrelated. A metric that targets what matters does not.",
    )

    draw_card(
        "BLEU/ROUGE grade word choice, not correctness.\n"
        "A reworded, equally correct answer can score near zero.\n"
        "A wrong answer that happens to reuse the reference's words can score high.\n\n"
        "Measure the thing you actually care about, not its surface form.",
        FIGURES / "step-06.png",
        TITLE,
        tone="bad",
        subtitle="why surface metrics mislead",
        note="This is the deterministic version of a well-known failure mode -- no judge, no LLM, just token sets.",
    )

    # ---- oracle ----
    assert len(model.calls) == 5
    assert predicted == PREDICTIONS
    assert hits == [True, True, True, False, True]
    assert exact_match == 4 / 5

    assert hit_set == {"latte-doc", "milk-doc"}
    assert recall_at_3 == 2 / 3

    assert overlap < 0.1, f"expected a low surface overlap, got {overlap}"
    assert price_ref == price_cand == "$6.25"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, exact_match={exact_match:.2f} (4/5), recall@3={recall_at_3:.3f} "
        f"(2/3), token_overlap={overlap:.3f} (low) vs price match={price_ref}=={price_cand}. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
