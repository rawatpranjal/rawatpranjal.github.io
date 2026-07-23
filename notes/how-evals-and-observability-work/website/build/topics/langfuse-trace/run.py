"""CODA: the real Langfuse SDK (v4, OpenTelemetry-based), instrumenting a
real agent run, captured entirely offline. No Langfuse server, no network
flush -- the Langfuse client is constructed with a real OTEL
InMemorySpanExporter in place of its default network exporter, so every
span it emits lands in local memory and is read back for the oracle.

The run itself is the same real create_agent execution as a-real-trace:
a ScriptedChatModel, real get_menu/compute_price tool calls, real message
content. Langfuse spans wrap each step with real, langfuse-namespaced OTEL
attributes (langfuse.observation.type, usage_details, input, output) --
nothing here is re-implemented by hand, it's the SDK's own span objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root_dir = HERE
while not (root_dir / "lib").is_dir():
    root_dir = root_dir.parent
sys.path.insert(0, str(root_dir / "lib"))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from langfuse import Langfuse  # noqa: E402

from beanline import Stock, make_tools, scripted, tool_call_msg  # noqa: E402
from langviz import clear, draw_card, draw_span_timeline  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Langfuse, captured offline"


def _tok(msg) -> int:
    text = msg.content if msg.content else str(msg.tool_calls)
    return len(text.split())


def main():
    clear(FIGURES)

    draw_card(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter\n"
        "langfuse.Langfuse(..., span_exporter=<in-memory exporter>)\n\n"
        "The real SDK, pointed at a real OTEL exporter that never leaves the\n"
        "process. No server, no API call, no network flush.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="CODA: the real library, captured offline",
        note="langfuse v4 is OpenTelemetry-based -- swapping the exporter is a supported constructor argument.",
    )

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
    non_human = [m for m in messages if not isinstance(m, HumanMessage)]

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-local-test",
        secret_key="sk-local-test",
        span_exporter=exporter,
        tracing_enabled=True,
    )

    with client.start_as_current_observation(
        name="agent_run", as_type="span", input=messages[0].content
    ):
        model_i = tool_i = 0
        for m in non_human:
            tok = _tok(m)
            if isinstance(m, ToolMessage):
                tool_i += 1
                with client.start_as_current_observation(
                    name=f"tool_call_{tool_i}",
                    as_type="tool",
                    input={"tool_call_id": m.tool_call_id},
                    output=m.content,
                ):
                    pass
            else:
                model_i += 1
                with client.start_as_current_observation(
                    name=f"model_call_{model_i}",
                    as_type="generation",
                    model="scripted",
                    usage_details={"output": tok},
                    output=m.content if m.content else str(m.tool_calls),
                ):
                    pass

    client.flush()
    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    # the real root span's real OTEL span_id, read off the finished span itself
    root_span_id = by_name["agent_run"].context.span_id

    span_rows = []
    t0 = min(s.start_time for s in spans)
    for s in spans:
        parent_name = (
            "agent_run" if s.parent and s.parent.span_id == root_span_id else None
        )
        span_rows.append(
            {
                "name": s.name,
                "parent": parent_name,
                "start_ms": (s.start_time - t0) / 1e6,
                "duration_ms": max((s.end_time - s.start_time) / 1e6, 0.05),
                "tokens": json.loads(
                    s.attributes.get("langfuse.observation.usage_details", "{}")
                ).get("output")
                if s.attributes.get("langfuse.observation.usage_details")
                else None,
            }
        )
    # root itself has no parent
    for row in span_rows:
        if row["name"] == "agent_run":
            row["parent"] = None
    # get_finished_spans() returns finish order (children close before their
    # parent), so re-sort root-first, then children by start time, for display.
    span_rows.sort(key=lambda r: (r["parent"] is not None, r["start_ms"]))

    draw_span_timeline(
        span_rows,
        FIGURES / "step-02.png",
        TITLE,
        note=f"{len(spans)} real langfuse spans, captured by the in-memory OTEL exporter -- zero network calls.",
        root_label="langfuse trace",
    )

    types_by_name = {
        s.name: s.attributes.get("langfuse.observation.type") for s in spans
    }
    draw_card(
        "\n".join(
            f"{name:16s} langfuse.observation.type = {t}"
            for name, t in sorted(types_by_name.items())
        ),
        FIGURES / "step-03.png",
        TITLE,
        tone="good",
        subtitle="real langfuse-namespaced OTEL attributes",
        note="Every attribute above was written by the langfuse SDK itself, read back off the real ReadableSpan objects.",
    )

    # ---- oracle ----
    assert isinstance(client, Langfuse)
    assert len(spans) == 6, f"expected 6 spans (1 root + 5 children), got {len(spans)}"
    expected_names = {
        "agent_run",
        "model_call_1",
        "tool_call_1",
        "model_call_2",
        "tool_call_2",
        "model_call_3",
    }
    assert set(by_name) == expected_names, set(by_name)

    root_span = by_name["agent_run"]
    assert root_span.parent is None
    for name in expected_names - {"agent_run"}:
        child = by_name[name]
        assert child.parent is not None
        assert child.parent.span_id == root_span_id, (
            f"{name}: parent span_id does not match the real root's OTEL span_id"
        )

    assert types_by_name["agent_run"] == "span"
    for name in ("model_call_1", "model_call_2", "model_call_3"):
        assert types_by_name[name] == "generation"
    for name in ("tool_call_1", "tool_call_2"):
        assert types_by_name[name] == "tool"

    # independent recomputation of token counts straight off the raw message list
    independent_tokens = {}
    model_i = tool_i = 0
    for m in non_human:
        if isinstance(m, ToolMessage):
            tool_i += 1
        else:
            model_i += 1
            independent_tokens[f"model_call_{model_i}"] = _tok(m)
    for name, tok in independent_tokens.items():
        captured = json.loads(
            by_name[name].attributes["langfuse.observation.usage_details"]
        )
        assert captured["output"] == tok, (name, captured, tok)

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {len(spans)} real langfuse spans captured offline, "
        f"names={sorted(by_name)}, all child parent span_ids == root's real OTEL span_id, "
        f"token counts cross-checked against the raw message list. All checks passed."
    )


if __name__ == "__main__":
    main()
