"""A test set is a frozen list of (input, expected) pairs, checked into the
repo like code. The flipbook builds Beanline's five-case test set row by
row; each expected answer is a real pydantic Order, constructed for real,
not a string that merely looks like one.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from beanline import Order, OrderItem, total  # noqa: E402
from langviz import clear, draw_card, draw_test_set  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "What is a test set?"


def build_cases() -> list[dict]:
    """Five fixed orders. Each `expected` is a real Order, parsed by the
    real pydantic model -- if any of these raised, the build would fail
    before a single figure got drawn."""
    specs = [
        (
            "c1",
            "one large oat milk latte",
            Order(items=[OrderItem(drink="latte", size="large", extras=["oat milk"])]),
        ),
        (
            "c2",
            "a cappuccino and a croissant to go",
            Order(
                items=[OrderItem(drink="cappuccino", size="medium")],
                food=["croissant"],
                to_go=True,
            ),
        ),
        (
            "c3",
            "medium cold brew with vanilla",
            Order(
                items=[OrderItem(drink="cold brew", size="medium", extras=["vanilla"])]
            ),
        ),
        (
            "c4",
            "two small espressos, one with an extra shot",
            Order(
                items=[
                    OrderItem(drink="espresso", size="small"),
                    OrderItem(drink="espresso", size="small", extras=["extra shot"]),
                ]
            ),
        ),
        (
            "c5",
            "a small mocha and a cookie",
            Order(items=[OrderItem(drink="mocha", size="small")], food=["cookie"]),
        ),
    ]
    return [
        {"id": cid, "input": text, "expected": repr(order), "order": order}
        for cid, text, order in specs
    ]


def main():
    clear(FIGURES)
    cases = build_cases()

    for i in range(1, len(cases) + 1):
        draw_test_set(
            [
                {"id": c["id"], "input": c["input"], "expected": c["expected"]}
                for c in cases[:i]
            ],
            FIGURES / f"step-0{i}.png",
            TITLE,
            note="A test set is fixed inputs paired with known-correct answers, checked into the repo.",
        )

    draw_test_set(
        [
            {
                "id": c["id"],
                "input": c["input"],
                "expected": c["expected"],
                "verdict": "pass",
            }
            for c in cases
        ],
        FIGURES / "step-06.png",
        TITLE,
        note="Every expected Order parsed via the real pydantic model -- that's the PASS.",
    )

    totals = [total(c["order"]) for c in cases]
    draw_card(
        f"N = {len(cases)} fixed cases\n"
        f"totals: {', '.join(f'${t:.2f}' for t in totals)}\n\n"
        "Same 5 cases, byte-identical, every single run.\n"
        "Nothing here depends on what any model says.",
        FIGURES / "step-07.png",
        TITLE,
        tone="good",
        subtitle="the eval's ground truth",
        note="The test set is the one part of an eval that must never be scripted -- it's already fixed.",
    )

    # ---- oracle ----
    assert len(cases) == 5
    for c in cases:
        assert isinstance(c["order"], Order), (
            f"{c['id']}: expected must be a real Order"
        )
    assert totals == [6.25, 8.00, 5.25, 7.00, 7.75], totals

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {len(cases)} cases, all expected Orders parsed via real "
        f"pydantic, totals={totals}. All checks passed."
    )


if __name__ == "__main__":
    main()
