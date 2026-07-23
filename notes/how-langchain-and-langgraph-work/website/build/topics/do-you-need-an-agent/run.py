"""Do you need an agent at all? One fixed task, three contenders.

Pricing Maya's usual -- a large oat-milk latte -- is a well-specified task:
one drink, one size, one extra, one right answer. Three ways to solve it,
all landing on $6.25: (a) plain python, zero model calls; (b) one
structured call, a scripted model emits the OrderItem JSON and python
prices it; (c) a full create_agent loop, three model calls and two real
tool calls to reach the same number. Model call counts come from
len(model.calls) on the real ScriptedChatModel, not a claim -- there is no
model object at all for (a).

The flip: an ambiguous request ("something cold, not too sweet, whatever's
cheapest") breaks plain python for real (OrderItem's drink field takes any
string, so price() dies on a real KeyError, not a staged one), the
structured call interprets the prose into a menu item, and only when the
answer must also depend on live stock does the agent's tool use earn its
keep.
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
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.output_parsers import PydanticOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from beanline import (  # noqa: E402
    EXTRAS,
    MENU,
    SIZES,
    OrderItem,
    Stock,
    make_tools,
    price,
    scripted,
    tool_call_msg,
)
from langviz import clear, draw_card, draw_messages, draw_pipeline, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Do you need an agent at all?"

USUAL = OrderItem(drink="latte", size="large", extras=["oat milk"])
AMBIGUOUS = "something cold, not too sweet, whatever's cheapest"


def main():
    clear(FIGURES)

    stock = Stock(
        {
            "oat milk": 2,
            "almond milk": 0,
            "whole milk": 10,
            "croissant": 3,
            "cold brew": 6,
        }
    )
    check_stock, get_menu, compute_price = make_tools(stock)
    parser = PydanticOutputParser(pydantic_object=OrderItem)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Extract the drink order. " + parser.get_format_instructions()
            ),
            ("human", "{input}"),
        ]
    )

    # ---- frame 1: the task and the ladder ----
    draw_card(
        "Maya's usual: a large oat-milk latte.\n"
        "One drink, one size, one extra. One right answer.\n\n"
        "  (a) plain python         code\n"
        "  (b) one structured call  code + 1 model call\n"
        "  (c) a full agent         code + 3 model calls, 2 tool calls\n\n"
        "All three must land on the same price.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="a fixed, well-specified task -- three contenders",
        note="The ladder: plain code, one structured call, a full agent loop.",
    )

    # ---- frame 2: contender (a) -- plain python, 0 model calls ----
    price_a = price(USUAL)
    draw_card(
        "price(OrderItem(\n"
        '    drink="latte", size="large", extras=["oat milk"]\n'
        "))\n"
        f"-> ${price_a:.2f}\n\n"
        "no prompt, no parser, no model object.\n"
        "0 model calls -- there is nothing to call.",
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        subtitle="contender (a): plain python",
        note="Real arithmetic over MENU/SIZES/EXTRAS. Fastest, cheapest, zero calls.",
    )

    # ---- frame 3: contender (b) -- one structured call ----
    model_b = scripted('{"drink": "latte", "size": "large", "extras": ["oat milk"]}')
    chain_b = prompt | model_b | parser
    item_b = chain_b.invoke({"input": "large oat milk latte"})
    price_b = price(item_b)
    draw_pipeline(
        [
            {
                "label": "prose",
                "type_label": "str",
                "payload": '"large oat milk latte"',
                "state": "done",
            },
            {
                "label": "model",
                "type_label": "ScriptedChatModel",
                "payload": f"drink={item_b.drink!r}",
                "state": "done",
            },
            {
                "label": "parser",
                "type_label": "OrderItem",
                "payload": f"size={item_b.size} extras={item_b.extras}",
                "state": "done",
            },
            {
                "label": "price()",
                "type_label": "float",
                "payload": f"${price_b:.2f}",
                "state": "active",
            },
        ],
        FIGURES / "step-03.png",
        TITLE,
        note="1 model call turns prose into a typed OrderItem; python prices it, same as (a).",
    )

    # ---- frame 4: contender (c) -- the full agent trace ----
    model_c = scripted(
        tool_call_msg("get_menu", {}, "c1"),
        tool_call_msg(
            "compute_price",
            {"drink": "latte", "size": "large", "extras": ["oat milk"]},
            "c2",
        ),
        "A large oat-milk latte is $6.25.",
    )
    agent_c = create_agent(model_c, [get_menu, compute_price])
    result_c = agent_c.invoke(
        {
            "messages": [
                HumanMessage(content="price Maya's usual: a large oat-milk latte")
            ]
        }
    )
    tool_msgs_c = [m for m in result_c["messages"] if isinstance(m, ToolMessage)]
    price_c = float(
        tool_msgs_c[1].content.lstrip("$")
    )  # real number off the real tool reply
    draw_messages(
        result_c["messages"],
        FIGURES / "step-04.png",
        TITLE,
        new=(len(result_c["messages"]) - 1,),
        note="3 model calls, 2 real tool calls -- get_menu then compute_price -- same $6.25 answer.",
    )

    # ---- frame 5: the scorecard ----
    draw_scorecard(
        [
            {
                "label": "(a) plain python",
                "cells": [f"${price_a:.2f}", "0", "0"],
                "verdict": "pass",
            },
            {
                "label": "(b) one structured call",
                "cells": [f"${price_b:.2f}", "1", "0"],
                "verdict": "pass",
            },
            {
                "label": "(c) full agent",
                "cells": [f"${price_c:.2f}", "3", "2"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-05.png",
        TITLE,
        columns=["answer", "model calls", "tool calls"],
        note="Same answer, three costs: 0, 1, 3 calls. On a fixed task, autonomy is pure overhead.",
    )

    # ---- frame 6: the flip -- an ambiguous request ----
    err_name = ""
    try:
        price(OrderItem(drink=AMBIGUOUS))
    except (KeyError, ValueError) as e:
        err_name = type(e).__name__

    chain_b2 = (
        prompt
        | scripted('{"drink": "cold brew", "size": "medium", "extras": []}')
        | parser
    )
    item_b2 = chain_b2.invoke({"input": AMBIGUOUS})

    model_c2 = scripted(
        tool_call_msg("check_stock", {"item": "cold brew"}, "c3"),
        "Cold brew's in stock -- 6 left, and it's the cheapest cold option at $4.25.",
    )
    agent_c2 = create_agent(model_c2, [check_stock, compute_price])
    result_c2 = agent_c2.invoke({"messages": [HumanMessage(content=AMBIGUOUS)]})
    tool_msgs_c2 = [m for m in result_c2["messages"] if isinstance(m, ToolMessage)]

    draw_scorecard(
        [
            {
                "label": "(a) plain python",
                "cells": [err_name, "0 calls"],
                "verdict": "fail",
            },
            {
                "label": "(b) one structured call",
                "cells": [item_b2.drink, "1 call"],
                "verdict": "pass",
            },
            {
                "label": "(c) full agent",
                "cells": [tool_msgs_c2[0].content, "2 calls"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        columns=["response", "cost"],
        note=(
            "Ambiguous input: (a) can't parse prose, (b) interprets it as cold brew, "
            "(c) also checks the real fridge. Climb the ladder only when forced."
        ),
    )

    # ---- oracle ----
    assert (
        price_a
        == price(USUAL)
        == round(MENU["latte"] + SIZES["large"] + EXTRAS["oat milk"], 2)
    )
    assert price_a == 6.25

    assert len(model_b.calls) == 1
    assert isinstance(item_b, OrderItem) and item_b.drink == "latte"
    assert price_b == 6.25

    assert len(model_c.calls) == 3
    assert len(tool_msgs_c) == 2
    assert tool_msgs_c[0].content == get_menu.invoke({})
    assert tool_msgs_c[1].content == compute_price.invoke(
        {"drink": "latte", "size": "large", "extras": ["oat milk"]}
    )
    assert tool_msgs_c[1].content == "$6.25"
    assert "6.25" in result_c["messages"][-1].content

    assert err_name in ("KeyError", "ValueError")
    assert item_b2.drink == "cold brew" and "cold brew" in MENU

    assert len(model_c2.calls) == 2
    assert len(tool_msgs_c2) == 1
    assert (
        tool_msgs_c2[0].content == check_stock.invoke({"item": "cold brew"}) == "6 left"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, fixed task: (a) {price_a:.2f}/0 calls, (b) {price_b:.2f}/1 call, "
        f"(c) {price_c:.2f}/3 calls+2 tools. "
        f"Ambiguous: python raised {err_name}, structured call read {item_b2.drink!r}, "
        f"agent checked stock -> {tool_msgs_c2[0].content!r}. All checks passed."
    )


if __name__ == "__main__":
    main()
