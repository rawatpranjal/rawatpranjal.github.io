"""CODA: the real DeepEval library, run deterministically. No stub, no
custom re-implementation of its scoring -- deepeval's own LLMTestCase,
its own built-in ExactMatchMetric (a non-LLM metric shipped by the
library itself), and its own assert_test all execute for real against the
same five Beanline cases and scripted predictions from offline-metrics.

Telemetry is opted out before deepeval is imported. Nothing in the
asserted path calls a hosted LLM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import HumanMessage  # noqa: E402

from deepeval import assert_test  # noqa: E402
from deepeval.metrics import BaseMetric, ExactMatchMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_card, draw_test_set  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "DeepEval, for real"

CASES = [
    ("c1", "one large oat milk latte", "latte"),
    ("c2", "a cappuccino and a croissant to go", "cappuccino"),
    ("c3", "medium cold brew with vanilla", "cold brew"),
    ("c4", "two small espressos, one with an extra shot", "espresso"),
    ("c5", "a small mocha and a cookie", "mocha"),
]
PREDICTIONS = ["latte", "cappuccino", "cold brew", "americano", "mocha"]  # c4 wrong


def main():
    clear(FIGURES)

    draw_card(
        "deepeval.test_case.LLMTestCase       real pydantic test case object\n"
        "deepeval.metrics.ExactMatchMetric    real, built-in, non-LLM metric\n"
        "deepeval.assert_test                 real assertion helper\n\n"
        "No custom scoring code, no mock -- the library's own objects run.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="CODA: the real library",
        note=f"deepeval {__import__('deepeval').__version__}, telemetry opted out before import.",
    )

    model = scripted(*PREDICTIONS)
    test_cases = []
    metrics = []
    for (cid, text, expected), _ in zip(CASES, PREDICTIONS):
        actual = model.invoke([HumanMessage(content=text)]).content
        tc = LLMTestCase(input=text, actual_output=actual, expected_output=expected)
        metric = ExactMatchMetric()
        metric.measure(tc)
        test_cases.append(tc)
        metrics.append(metric)

    draw_test_set(
        [
            {
                "id": cid,
                "input": tc.input,
                "expected": f"{tc.expected_output!r} (deepeval score={m.score})",
                "verdict": "pass" if m.is_successful() else "fail",
            }
            for (cid, _, _), tc, m in zip(CASES, test_cases, metrics)
        ],
        FIGURES / "step-02.png",
        TITLE,
        note="Every score and verdict above came from deepeval's own ExactMatchMetric.measure().",
    )

    pass_rate = sum(m.score for m in metrics) / len(metrics)

    # assert_test: passes silently on a match, raises for real on a mismatch
    assert_test(test_cases[0], [ExactMatchMetric()])
    caught = None
    try:
        assert_test(test_cases[3], [ExactMatchMetric()])
    except AssertionError as e:
        caught = str(e)

    draw_card(
        f"assert_test(case c1, [ExactMatchMetric()])   -> no exception (score 1.0)\n"
        f"assert_test(case c4, [ExactMatchMetric()])   -> raised AssertionError\n\n"
        f"caught: {caught[:180] if caught else '(nothing raised -- would be a bug)'}",
        FIGURES / "step-03.png",
        TITLE,
        tone="bad" if caught else "neutral",
        subtitle="deepeval's own assert_test, live",
        note="The exception text above is deepeval's own, not ours.",
    )

    draw_card(
        f"deepeval pass rate: {sum(1 for m in metrics if m.is_successful())}/{len(metrics)} "
        f"= {pass_rate:.0%}\n\n"
        "Same numbers as the hand-rolled exact-match topic -- computed this\n"
        "time by deepeval's real, non-LLM ExactMatchMetric and assert_test.",
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="real library, real numbers",
        note="deepeval.metrics.ExactMatchMetric is a genuine subclass of deepeval.metrics.BaseMetric.",
    )

    # ---- oracle ----
    assert issubclass(ExactMatchMetric, BaseMetric)
    assert all(isinstance(m, ExactMatchMetric) for m in metrics)
    assert all(isinstance(tc, LLMTestCase) for tc in test_cases)
    scores = [m.score for m in metrics]
    assert scores == [1.0, 1.0, 1.0, 0.0, 1.0], scores
    assert pass_rate == 0.8
    assert caught is not None, "assert_test must raise on the mismatched case"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, deepeval {__import__('deepeval').__version__}, "
        f"scores={scores}, pass_rate={pass_rate:.0%}, assert_test raised on c4. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
