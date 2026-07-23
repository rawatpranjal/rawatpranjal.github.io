"""A real agent run, instrumented into a nested span tree. The run itself
is real: create_agent(model, tools) with a ScriptedChatModel, executing
get_menu then compute_price for real against Beanline's fixtures. The
spans are built directly from the run's own message list -- token counts
come from real word counts on real message content, cost is real
arithmetic on those counts, and durations are fixed per-call-type
constants (a stand-in for a wall clock, not a script).

The oracle recomputes token/cost totals directly from the raw message
list, independently of the span-building loop, and asserts they match.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from beanline import Stock, make_tools, scripted, tool_call_msg  # noqa: E402
from langviz import clear, draw_card, draw_messages, draw_span_timeline  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "A real trace: spans, tokens, cost"

MODEL_MS = 120
TOOL_MS = 30
RATE_PER_TOKEN = 0.00002  # a real, if invented, per-token price


def _tok(msg) -> int:
    """Real word count off the real message -- content when there is any,
    otherwise the tool-call repr, which is what the model actually emitted."""
    text = msg.content if msg.content else str(msg.tool_calls)
    return len(text.split())


def main():
    clear(FIGURES)

    stock = Stock({"oat milk": 2, "almond milk": 0, "whole milk": 10, "croissant": 3})
    check_stock, get_menu, compute_price = make_tools(stock)

    model = scripted(
        tool_call_msg("get_menu", {}, "c1"),
        tool_call_msg(
            "compute_price",
            {"drink": "latte", "size": "large", "extras": ["oat milk"]},
            "c2",
        ),
        "A large oat-milk latte is $6.25.",
    )
    agent = create_agent(model, [get_menu, compute_price])
    result = agent.invoke(
        {"messages": [HumanMessage(content="price a large oat-milk latte")]}
    )
    messages = result["messages"]

    draw_messages(
        messages,
        FIGURES / "step-01.png",
        TITLE,
        new=(len(messages) - 1,),
        note="The real run: 1 human turn, 3 model calls, 2 tool calls, executed for real.",
    )

    # ---- build the span tree from the real message list ----
    non_human = [m for m in messages if not isinstance(m, HumanMessage)]
    names = []
    spans = [
        {
            "name": "agent_run",
            "parent": None,
            "start_ms": 0,
            "duration_ms": 0,
            "tokens": 0,
            "cost": 0.0,
        }
    ]
    t = 0
    model_i = tool_i = 0
    for m in non_human:
        is_tool = isinstance(m, ToolMessage)
        if is_tool:
            tool_i += 1
            name = f"tool_call_{tool_i}"
            dur = TOOL_MS
        else:
            model_i += 1
            name = f"model_call_{model_i}"
            dur = MODEL_MS
        tok = _tok(m)
        cost = round(tok * RATE_PER_TOKEN, 6)
        spans.append(
            {
                "name": name,
                "parent": "agent_run",
                "start_ms": t,
                "duration_ms": dur,
                "tokens": tok,
                "cost": cost,
            }
        )
        names.append(name)
        t += dur

    root = spans[0]
    root["duration_ms"] = t
    root["tokens"] = sum(s["tokens"] for s in spans[1:])
    root["cost"] = round(sum(s["cost"] for s in spans[1:]), 6)

    draw_span_timeline(
        spans[:3],
        FIGURES / "step-02.png",
        TITLE,
        note="As the run happens: the root span, then the first model call and tool call.",
    )
    draw_span_timeline(
        spans,
        FIGURES / "step-03.png",
        TITLE,
        note=f"All {len(spans) - 1} child spans, nested under agent_run -- {root['tokens']} tokens, ${root['cost']:.5f}.",
    )

    draw_card(
        f"spans: {len(spans)} ({len(spans) - 1} children + 1 root)\n"
        f"trajectory: {' -> '.join(names)}\n"
        f"root tokens: {root['tokens']}   root cost: ${root['cost']:.5f}\n"
        f"child token sum == root tokens, child cost sum == root cost",
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="the span tree",
        note="Same arithmetic invariant a real tracer enforces: parents summarize their children.",
    )

    # ---- oracle ----
    assert isinstance(messages[1], AIMessage) and messages[1].tool_calls
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], AIMessage) and messages[3].tool_calls
    assert isinstance(messages[4], ToolMessage)
    assert isinstance(messages[5], AIMessage) and not messages[5].tool_calls

    assert names == [
        "model_call_1",
        "tool_call_1",
        "model_call_2",
        "tool_call_2",
        "model_call_3",
    ]
    assert len(spans) == 6, f"expected 6 spans (1 root + 5 children), got {len(spans)}"
    children = spans[1:]
    assert all(s["parent"] == "agent_run" for s in children)
    assert root["parent"] is None

    # independent recomputation, straight off the raw message list
    independent_tokens = sum(_tok(m) for m in non_human)
    independent_cost = round(
        sum(round(_tok(m) * RATE_PER_TOKEN, 6) for m in non_human), 6
    )
    assert independent_tokens == root["tokens"] == sum(s["tokens"] for s in children)
    assert (
        independent_cost == root["cost"] == round(sum(s["cost"] for s in children), 6)
    )
    assert root["duration_ms"] == sum(s["duration_ms"] for s in children)

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {len(spans)} spans, trajectory={names}, "
        f"root tokens={root['tokens']} (== child sum), root cost=${root['cost']:.5f} "
        f"(== child sum). All checks passed."
    )


if __name__ == "__main__":
    main()
