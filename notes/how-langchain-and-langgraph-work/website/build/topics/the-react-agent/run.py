"""The agent is a two-node cycle: create_agent compiles the whole ReAct loop.

Maya asks a question no single call answers: "what's the cheapest large
drink if I add an extra shot?" langchain.agents.create_agent(model, tools)
is the current canonical constructor -- create_react_agent is deprecated in
langgraph 1.x. The routing (tool_calls present -> tools node, absent ->
END) is the library's real prebuilt graph, not ours: the model decides
three times and the tools node really executes get_menu and compute_price
against beanline's real MENU/SIZES/EXTRAS, cycling model -> tools -> model
until the model stops asking for tools.
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
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from beanline import EXTRAS, MENU, SIZES, Stock, make_tools, scripted, tool_call_msg  # noqa: E402
from langviz import clear, draw_card, draw_graph, draw_messages  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The agent is a two-node cycle"

QUESTION = "what's the cheapest large drink if I add an extra shot?"

POSITIONS = {
    "__start__": (8, 30),
    "model": (32, 30),
    "tools": (66, 30),
    "__end__": (92, 30),
}


def main():
    clear(FIGURES)

    check_stock, get_menu, compute_price = make_tools(Stock())
    model = scripted(
        tool_call_msg("get_menu", {}, "c1"),
        tool_call_msg(
            "compute_price",
            {"drink": "espresso", "size": "large", "extras": ["extra shot"]},
            "c2",
        ),
        "cheapest large with an extra shot: espresso, $5.00.",
    )
    agent = create_agent(model, [get_menu, compute_price])
    graph = agent.get_graph()
    nodes = list(graph.nodes)

    result = agent.invoke({"messages": [HumanMessage(content=QUESTION)]})
    human, ai1, tool1, ai2, tool2, ai3 = result["messages"]

    # ---- frame 1: the prebuilt shape ----
    draw_graph(
        graph,
        POSITIONS,
        FIGURES / "step-01.png",
        TITLE,
        note="This compiled graph is the same model-decides / tools-execute loop you drove by hand in the last two topics.",
    )

    # ---- frame 2: lap 1, model decides get_menu ----
    draw_graph(
        graph,
        POSITIONS,
        FIGURES / "step-02.png",
        TITLE,
        active="model",
        taken_edges=(("__start__", "model"),),
        edge_label=("__start__", "model", "invoke"),
        state_rows=[
            {
                "channel": "model call 1",
                "value": "tool_call: get_menu()",
                "reducer": None,
                "delta": None,
                "changed": True,
            },
        ],
        note="Lap 1: model call 1 sees only the question, so it emits tool_call get_menu -- it cannot price anything yet.",
    )

    # ---- frame 3: tools node runs get_menu for real ----
    menu_fragment = tool1.content.split(". sizes:")[0]
    draw_graph(
        graph,
        POSITIONS,
        FIGURES / "step-03.png",
        TITLE,
        active="tools",
        visited=("model",),
        taken_edges=(("__start__", "model"), ("model", "tools")),
        edge_label=("model", "tools", "get_menu"),
        state_rows=[
            {
                "channel": "ToolMessage",
                "value": tool1.content,
                "reducer": None,
                "delta": None,
                "changed": True,
            },
        ],
        note=f'The tools node really ran get_menu(): "{menu_fragment}" -- straight out of beanline.menu_board().',
    )

    # ---- frame 4: lap 2, model decides compute_price ----
    draw_graph(
        graph,
        POSITIONS,
        FIGURES / "step-04.png",
        TITLE,
        active="model",
        visited=("model", "tools"),
        taken_edges=(("__start__", "model"), ("model", "tools"), ("tools", "model")),
        edge_label=("tools", "model", "get_menu ok"),
        state_rows=[
            {
                "channel": "model call 2",
                "value": "tool_call: compute_price(...)",
                "reducer": None,
                "delta": None,
                "changed": True,
            },
        ],
        note="Lap 2: with the real menu now in context, model call 2 emits tool_call compute_price(drink='espresso', size='large', extras=['extra shot']).",
    )

    # ---- frame 5: the real arithmetic ----
    espresso, large, extra_shot = MENU["espresso"], SIZES["large"], EXTRAS["extra shot"]
    arithmetic = (
        f"MENU['espresso']       = ${espresso:.2f}\n"
        f"SIZES['large']         = ${large:.2f}\n"
        f"EXTRAS['extra shot']   = ${extra_shot:.2f}\n"
        "----------------------------------\n"
        f"total                  = ${espresso + large + extra_shot:.2f}\n\n"
        "compute_price(drink='espresso', size='large', extras=['extra shot'])\n"
        f"-> {tool2.content!r}"
    )
    draw_card(
        arithmetic,
        FIGURES / "step-05.png",
        TITLE,
        tone="good",
        subtitle="compute_price: real arithmetic over MENU/SIZES/EXTRAS, never model arithmetic",
        note="The ToolMessage content is the same $5.00 you get calling compute_price directly.",
    )

    # ---- frame 6: lap 3, no tool_calls -> END ----
    draw_graph(
        graph,
        POSITIONS,
        FIGURES / "step-06.png",
        TITLE,
        active="model",
        visited=("model", "tools"),
        taken_edges=(
            ("__start__", "model"),
            ("model", "tools"),
            ("tools", "model"),
            ("model", "__end__"),
        ),
        edge_label=("model", "__end__", "no tool_calls"),
        note="Lap 3: model call 3 has both tool results in context and emits no tool_calls -- the conditional edge routes to END.",
    )

    # ---- frame 7: the whole transcript is just messages ----
    draw_messages(
        result["messages"],
        FIGURES / "step-07.png",
        TITLE,
        new=(5,),
        note="The whole ReAct transcript -- 1 human, 3 ai, 2 tool -- is just a list of messages, nothing more.",
    )

    # ---- oracle ----
    call1 = ai1.tool_calls[0]
    call2 = ai2.tool_calls[0]
    assert tool1.content == get_menu.invoke(call1["args"]), (
        "ToolMessage 1 must equal independently re-invoking the named tool with the recorded args"
    )
    assert tool2.content == compute_price.invoke(call2["args"]), (
        "ToolMessage 2 must equal independently re-invoking the named tool with the recorded args"
    )
    assert isinstance(ai3, AIMessage) and not ai3.tool_calls
    assert "5.00" in ai3.content
    assert round(espresso + large + extra_shot, 2) == 5.00
    assert "model" in nodes and "tools" in nodes, nodes
    assert len(model.calls) == 3
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, nodes={nodes}, 2 ToolMessages matched independent "
        f"re-invocation, final answer {ai3.content!r} confirms "
        f"{espresso:.2f}+{large:.2f}+{extra_shot:.2f}={espresso + large + extra_shot:.2f}, "
        f"{len(model.calls)} model calls. All checks passed."
    )


if __name__ == "__main__":
    main()
