"""Prompt templates: the prompt is code, the menu is data.

The barista script has holes: {menu}, {name}, {input}. ChatPromptTemplate
fills them at invoke time from the real MENU. Then the shop raises the
latte price and the SAME template, re-invoked, renders the new number,
and model.calls proves the new price really reached the model. The oracle
pins both renders and the input_variables contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.prompt_values import ChatPromptValue  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

import beanline  # noqa: E402
from beanline import menu_board, scripted  # noqa: E402
from langviz import clear, draw_card, draw_messages, draw_pipeline  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Prompts are functions"

TEMPLATE_SRC = (
    "ChatPromptTemplate.from_messages([\n"
    '  ("system",\n'
    '   "You are the barista at\n'
    "    Beanline Coffee. Serving {name}.\n"
    '    Menu: {menu}"),\n'
    '  ("human", "{input}"),\n'
    "])"
)


def main():
    clear(FIGURES)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the barista at Beanline Coffee. Serving {name}. Menu: {menu}",
            ),
            ("human", "{input}"),
        ]
    )

    draw_card(
        TEMPLATE_SRC,
        FIGURES / "step-01.png",
        TITLE,
        subtitle="three holes: {menu}, {name}, {input}",
        note="The barista script with holes. Nothing rendered yet -- this is code, not a prompt.",
    )

    values = {"menu": menu_board(), "name": "Maya", "input": "how much is a latte?"}
    pv = prompt.invoke(values)

    draw_pipeline(
        [
            {
                "label": "values",
                "type_label": "dict",
                "payload": '{"menu": menu_board(), "name": "Maya", "input": "how much is a latte?"}',
                "state": "done",
            },
            {
                "label": "prompt.invoke",
                "type_label": "ChatPromptTemplate",
                "payload": "fills {menu} {name} {input}",
                "state": "active",
            },
            {
                "label": "rendered",
                "type_label": "ChatPromptValue",
                "payload": "[system, human] with the real $4.50 menu inside",
                "state": "done",
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        note="Invoke the template like a function: real MENU data flows into the holes.",
    )

    model = scripted("A latte is $4.50, Maya.", "A latte is $4.75 now, Maya.")
    model.invoke(pv)
    draw_messages(
        pv.to_messages(),
        FIGURES / "step-03.png",
        TITLE,
        new=(0, 1),
        note="The rendered messages, as the model received them: latte $4.50, straight from MENU.",
    )

    # the shop raises the price; the template does not change
    beanline.MENU["latte"] = 4.75
    pv2 = prompt.invoke(
        {"menu": menu_board(), "name": "Maya", "input": "how much is a latte?"}
    )
    model.invoke(pv2)

    draw_card(
        'MENU["latte"] = 4.75\n\n'
        "template: UNCHANGED\n"
        "render 1: ...latte $4.50...\n"
        "render 2: ...latte $4.75...",
        FIGURES / "step-04.png",
        TITLE,
        tone="neutral",
        subtitle="same template, new data",
        note="Tuesday: the shop raises the latte price. Nobody edits a prompt string.",
    )

    draw_messages(
        pv2.to_messages(),
        FIGURES / "step-05.png",
        TITLE,
        new=(0,),
        right_text=(
            "prompt.input_variables\n"
            "  -> ['input', 'menu', 'name']\n"
            "\n"
            "a template declares what it\n"
            "needs; invoke() must supply it\n"
            "\n"
            "the template is code,\n"
            "the menu is data"
        ),
        right_title="the contract",
        note="Re-invoked: the model demonstrably sees $4.75. The template never changed.",
    )

    # ---- oracle ----
    rendered1 = pv.to_messages()[0].content
    rendered2 = pv2.to_messages()[0].content
    assert isinstance(pv, ChatPromptValue)
    assert "latte $4.50" in rendered1
    assert "latte $4.75" in rendered2
    assert set(prompt.input_variables) == {"menu", "name", "input"}
    assert len(model.calls) == 2
    assert "latte $4.75" in model.calls[1][0].content, (
        "the new price must really have reached the model"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {len(figs)}"
    beanline.MENU["latte"] = 4.50  # restore for good hygiene
    print(
        f"{len(figs)} figures, render 1 carried $4.50, render 2 carried $4.75, "
        f"both photographed from model.calls. All checks passed."
    )


if __name__ == "__main__":
    main()
