"""Tool calling: the model never touches the fridge.

Maya asks whether there's oat milk today. The model cannot know, so it
emits a tool_call. The harness executes the real check_stock function
against the real Stock (2 left), sends the count back as a ToolMessage
whose tool_call_id matches the request, and the model's final answer
quotes the number it was GIVEN, not one it invented. The oracle re-runs
the tool independently and compares.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from beanline import Stock, make_tools, scripted, tool_call_msg  # noqa: E402
from langviz import clear, draw_messages, draw_tool_wire  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Tools are round trips"


def world_lines(stock: Stock, changed: str | None = None) -> list[str]:
    return [
        ("*" if item == changed else "") + f"{item}: {n}"
        for item, n in stock.counts.items()
    ]


def main():
    clear(FIGURES)

    stock = Stock()
    check_stock, get_menu, compute_price = make_tools(stock)

    schema = {
        "check_stock": "item: str -> how many units are left",
        "get_menu": "-> the full menu board",
        "compute_price": "drink, size, extras -> price",
    }

    draw_tool_wire(
        1,
        FIGURES / "step-01.png",
        TITLE,
        schema=schema,
        note="Binding shows the model the tools' names and typed args. Nothing has run.",
    )

    system = SystemMessage(content="You are the barista at Beanline Coffee.")
    human = HumanMessage(content="do you have oat milk today?")
    model = scripted(
        tool_call_msg("check_stock", {"item": "oat milk"}, "call_oat_1"),
        "Yep -- 2 oat milks left, Maya. Want one in a latte?",
    ).bind_tools([check_stock, get_menu, compute_price])

    ai1 = model.invoke([system, human])
    call = ai1.tool_calls[0]

    draw_tool_wire(
        2,
        FIGURES / "step-02.png",
        TITLE,
        schema=schema,
        call={"name": call["name"], "args": call["args"], "id": call["id"]},
        note="The model does not know the fridge. It emits a REQUEST: check_stock({'item': 'oat milk'}).",
    )

    result_content = check_stock.invoke(call["args"])
    draw_tool_wire(
        3,
        FIGURES / "step-03.png",
        TITLE,
        schema=schema,
        call={"name": call["name"], "args": call["args"], "id": call["id"]},
        world=world_lines(stock, changed="oat milk"),
        note="YOUR code opens the fridge: the real function reads the real count. The model is not here.",
    )

    tool_msg = ToolMessage(content=result_content, tool_call_id=call["id"])
    draw_tool_wire(
        4,
        FIGURES / "step-04.png",
        TITLE,
        schema=schema,
        call={"name": call["name"], "args": call["args"], "id": call["id"]},
        result={"content": result_content, "tool_call_id": tool_msg.tool_call_id},
        world=world_lines(stock),
        note="The count travels back as a ToolMessage. Its tool_call_id matches the request: a closed loop.",
    )

    final = model.invoke([system, human, ai1, tool_msg])
    draw_messages(
        [system, human, ai1, tool_msg, final],
        FIGURES / "step-05.png",
        TITLE,
        new=(4,),
        note="Second model call, tool result in context: the answer quotes the number it was GIVEN.",
    )

    # ---- oracle ----
    assert isinstance(ai1, AIMessage) and ai1.tool_calls
    assert call["name"] == "check_stock" and call["args"] == {"item": "oat milk"}
    assert result_content == check_stock.invoke(call["args"]) == "2 left", (
        "the ToolMessage content must really come from executing the function"
    )
    assert tool_msg.tool_call_id == call["id"]
    assert stock.check("oat milk") == 2
    assert "2" in final.content
    assert [type(m) for m in model.calls[1]] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    ], "the model's second context must hold the full round trip"
    assert model.bound_tools == [check_stock, get_menu, compute_price]
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, round trip closed: tool_call id {call['id']!r} "
        f"matched, real count 2 quoted in the final answer. All checks passed."
    )


if __name__ == "__main__":
    main()
