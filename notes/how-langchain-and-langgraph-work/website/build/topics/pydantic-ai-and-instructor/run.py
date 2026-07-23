"""Pydantic AI and Instructor: the lighter workhorses.

LangChain/LangGraph earn their keep on durable, multi-step, multi-call
work. Most jobs are one call, and two much lighter tools do that job with
far less machinery:

  * Pydantic AI -- a typed single-agent framework. `Agent(model,
    output_type=SomeModel)` returns a real, validated instance of that
    model. One pip install, no graph.
  * Instructor -- patches ANY OpenAI-compatible client so `.create()`
    returns a validated Pydantic object instead of raw JSON text.

Two real, deterministic runs:

  (a) A real `pydantic_ai.Agent`, driven by `FunctionModel` (ships with
      the library, no network -- the same "script the model, run the
      real framework" idiom as this deck's ScriptedChatModel). The
      Agent really executes; the typed Order it returns is real.
  (b) Real `instructor`, patching a real OpenAI-compatible client
      pointed at a local ollama `qwen2.5:0.5b`, temperature 0, a fixed
      seed. A 494M model is not reliable on every schema -- probed here,
      not assumed: a 3-field OrderItem with a `list[str]` extras field
      corrupts under this model (it emits extras as a bare string, or
      invents an extra never mentioned in the prompt, failing pydantic
      validation on 2 of 3 prompts tried). Trimmed to a 2-field schema
      (drink, size, no list), the same model and mode gives a valid
      object on every prompt tried and an IDENTICAL object across 5
      repeats of the same prompt -- that determinism, not the model's
      semantic judgment, is what the oracle below asserts.

Beanline's own OrderItem/Order (build/lib/beanline.py) are not imported:
beanline.py hard-imports langchain_core at module scope, and this coda's
venv deliberately excludes langchain/langgraph -- the whole point of the
coda is the job done WITHOUT them. The Order model below mirrors
Beanline's shape instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_messages, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The lighter workhorses"

OLLAMA_MODEL = "qwen2.5:0.5b"


# ---- the two schemas -------------------------------------------------------


class Order(BaseModel):
    """Mirrors Beanline's OrderItem. Fully deterministic here (FunctionModel
    supplies the args), so the list field is no risk."""

    drink: str = Field(description="one of the menu drinks")
    size: str = Field(default="medium", description="small, medium or large")
    extras: list[str] = Field(default_factory=list, description="extras, e.g. oat milk")


class SimpleOrder(BaseModel):
    """Trimmed for a 494M local model: no list field, the shape it hits
    reliably (see the probe below)."""

    drink: str = Field(description="the drink name, e.g. latte, cappuccino, espresso")
    size: str = Field(description="small, medium, or large")


# ---- (a) real pydantic_ai.Agent, scripted via FunctionModel ---------------


def run_pydantic_ai():
    calls: list = []

    def take_order(messages, info: AgentInfo) -> ModelResponse:
        calls.append(list(messages))
        tool_name = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool_name,
                    args={"drink": "latte", "size": "large", "extras": ["oat milk"]},
                )
            ]
        )

    agent = Agent(
        FunctionModel(take_order),
        output_type=Order,
        system_prompt="Extract Maya's drink order.",
    )
    result = agent.run_sync("large oat milk latte for Maya")
    return agent, result, calls


# ---- (b) real instructor, patched OpenAI client -> local ollama -----------


def instructor_client() -> instructor.Instructor:
    return instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON,
    )


def extract(client, model, prompt: str):
    """One real instructor call. Returns (order_or_None, error_name_or_None)."""
    try:
        order = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Extract the coffee order as JSON."},
                {"role": "user", "content": prompt},
            ],
            response_model=model,
            temperature=0,
            seed=42,
            max_retries=2,
        )
        return order, None
    except Exception as e:  # instructor's own retry-exhausted exception
        return None, type(e).__name__


def main():
    clear(FIGURES)

    # ---- frame 1: the three framework choices, same fixed job -----------
    draw_scorecard(
        [
            {
                "label": "raw SDK (openai/anthropic)",
                "cells": ["hand-parsed", "no", "0 extra"],
            },
            {
                "label": "Pydantic AI",
                "cells": ["Agent(output_type=Order)", "no", "1 (pydantic-ai-slim)"],
            },
            {
                "label": "LangGraph",
                "cells": ["node returns typed state", "yes", "2 (langchain+langgraph)"],
            },
        ],
        FIGURES / "step-01.png",
        TITLE,
        columns=["typed output", "durable multi-step", "extra installs"],
        note="Same fixed job -- price one order. Pick the lightest tool that does it.",
    )

    # ---- frame 2: real pydantic_ai.Agent, real typed Order --------------
    agent, result, calls = run_pydantic_ai()
    draw_messages(
        [
            {"role": "system", "text": "Extract Maya's drink order."},
            {"role": "human", "text": "large oat milk latte for Maya"},
        ],
        FIGURES / "step-02.png",
        TITLE,
        right_text=json.dumps(result.output.model_dump(), indent=2),
        right_title="Order (real Agent.run_sync, FunctionModel)",
        note="A real pydantic_ai.Agent ran; result.output is a real, validated Order instance.",
    )

    # ---- frame 3: real instructor, prose -> validated object -------------
    client = instructor_client()
    prompt = "a large latte"
    simple_order, simple_err = extract(client, SimpleOrder, prompt)
    draw_messages(
        [{"role": "human", "text": prompt}],
        FIGURES / "step-03.png",
        TITLE,
        right_text=json.dumps(simple_order.model_dump(), indent=2)
        if simple_order
        else f"FAILED: {simple_err}",
        right_title="SimpleOrder (instructor + qwen2.5:0.5b, temp=0)",
        note="instructor.from_openai(...) patches a real client; .create() returns a validated object, not text.",
    )

    # ---- frame 4: schema shape decides reliability on a 494M model -------
    order_probe_prompts = [
        "a large latte",
        "a small espresso",
        "a large oat milk latte",
    ]
    order_results = [extract(client, Order, p) for p in order_probe_prompts]
    order_fail_n = sum(1 for _, err in order_results if err is not None)

    repeats = [extract(client, SimpleOrder, prompt) for _ in range(5)]
    repeat_orders = [o for o, _ in repeats]
    all_valid = all(o is not None for o in repeat_orders)
    all_identical = all_valid and all(
        o.model_dump() == repeat_orders[0].model_dump() for o in repeat_orders
    )

    draw_scorecard(
        [
            {
                "label": "OrderItem (drink+size+extras:list)",
                "cells": [f"{order_fail_n}/{len(order_probe_prompts)} failed", "-"],
                "verdict": "fail" if order_fail_n else "pass",
            },
            {
                "label": "SimpleOrder (drink+size, no list)",
                "cells": [
                    "0/5 failed" if all_valid else "some failed",
                    "5/5 identical" if all_identical else "not identical",
                ],
                "verdict": "pass" if all_valid and all_identical else "fail",
            },
        ],
        FIGURES / "step-04.png",
        TITLE,
        columns=["validation", f"repeat {prompt!r} x5 @ temp=0"],
        note="A 494M model is not reliable on every schema. Trim to the shape it actually hits, then trust temp=0.",
    )

    # ---- oracle ----
    assert type(result).__name__ == "AgentRunResult"
    assert isinstance(result.output, Order), (
        "the Agent must really return a typed Order"
    )
    assert result.output.drink == "latte" and result.output.extras == ["oat milk"]
    assert len(calls) == 1, "FunctionModel must have been invoked exactly once"
    seen_parts = [type(p).__name__ for p in calls[0][0].parts]
    assert seen_parts == ["SystemPromptPart", "UserPromptPart"], (
        "the model must see the real system + user prompt parts"
    )

    assert simple_order is not None, f"instructor call failed: {simple_err}"
    assert isinstance(simple_order, SimpleOrder), (
        "instructor must return a validated SimpleOrder instance, not raw JSON"
    )

    assert all_valid, (
        "temp-0 instructor extraction must produce a valid object every repeat"
    )
    assert all_identical, (
        "temp-0 + fixed seed must be deterministic: 5 repeats of the same prompt must match"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures. pydantic_ai.Agent -> real Order{result.output.model_dump()}. "
        f"instructor -> real SimpleOrder{simple_order.model_dump()}. "
        f"OrderItem(extras:list) failed {order_fail_n}/{len(order_probe_prompts)} prompts on qwen2.5:0.5b; "
        f"SimpleOrder was valid and identical across all 5 repeats. All checks passed."
    )


if __name__ == "__main__":
    main()
