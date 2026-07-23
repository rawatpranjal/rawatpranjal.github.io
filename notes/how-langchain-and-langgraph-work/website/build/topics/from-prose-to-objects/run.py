"""Structured output: the LLM's job ends at the object boundary.

Maya's fuzzy prose order becomes a typed pydantic Order via
PydanticOutputParser. The format instructions really travel inside the
prompt, the model's JSON really parses into an Order, and the price is
real arithmetic on real menu data: $6.25 for the drink, $9.75 with the
croissant. A mangled scripted reply shows the boundary is enforced, not
hoped for: the parser raises OutputParserException, it does not limp.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.exceptions import OutputParserException  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402
from langchain_core.output_parsers import PydanticOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from beanline import EXTRAS, FOOD, MENU, SIZES, Order, scripted, total  # noqa: E402
from langviz import clear, draw_card, draw_messages, draw_pipeline  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "From prose to objects"

ORDER_JSON = (
    '{"items": [{"drink": "latte", "size": "large", "extras": ["oat milk"]}], '
    '"food": ["croissant"], "to_go": true}'
)


def main():
    clear(FIGURES)

    draw_card(
        '"large oat milk latte and a\n croissant, to go please"',
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the input: human prose, fuzzy and fine",
        note="Maya orders like a person. The till needs fields, not vibes.",
    )

    parser = PydanticOutputParser(pydantic_object=Order)

    draw_card(
        "class OrderItem(BaseModel):\n"
        "    drink: str\n"
        '    size: str = "medium"\n'
        "    extras: list[str] = []\n"
        "\n"
        "class Order(BaseModel):\n"
        "    items: list[OrderItem]\n"
        "    food: list[str] = []\n"
        "    to_go: bool = False",
        FIGURES / "step-02.png",
        TITLE,
        subtitle="the schema: what the till can actually process",
        note="A pydantic model is the contract. The parser will enforce it.",
    )

    # the format instructions contain literal JSON braces, so they enter as a
    # concrete SystemMessage (passed through untemplated), not a template hole
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Extract the order. " + parser.get_format_instructions()
            ),
            ("human", "{input}"),
        ]
    )

    model = scripted(ORDER_JSON)
    chain = prompt | model | parser
    order = chain.invoke(
        {"input": "large oat milk latte and a croissant, to go please"}
    )

    draw_messages(
        model.calls[0],
        FIGURES / "step-03.png",
        TITLE,
        new=(0,),
        note="parser.get_format_instructions() rides INSIDE the real prompt: the schema is told to the model.",
    )

    draw_pipeline(
        [
            {
                "label": "prose",
                "type_label": "str",
                "payload": '"large oat milk latte..."',
                "state": "done",
            },
            {
                "label": "model",
                "type_label": "AIMessage",
                "payload": ORDER_JSON[:48] + "...",
                "state": "done",
            },
            {
                "label": "parser",
                "type_label": "Order",
                "payload": "items=[latte/large/oat] food=[croissant] to_go=True",
                "state": "active",
            },
        ],
        FIGURES / "step-04.png",
        TITLE,
        note="The model emits JSON; the parser turns it into a typed Order object. The LLM's job ends here.",
    )

    t = total(order)
    draw_card(
        "latte            4.50\n"
        "  large         +1.00\n"
        "  oat milk      +0.75\n"
        "croissant        3.50\n"
        "                -----\n"
        f"total           ${t:.2f}",
        FIGURES / "step-05.png",
        TITLE,
        tone="good",
        subtitle="real arithmetic, zero model involvement",
        note="total(order) is plain python over MENU/SIZES/EXTRAS/FOOD. Models interpret; tills add.",
    )

    # the dishonest path: the model chats instead of emitting the schema
    bad_model = scripted("Sure! One large oat milk latte -- $6.25.")
    bad_chain = prompt | bad_model | parser
    err_name = ""
    try:
        bad_chain.invoke({"input": "same again please"})
    except OutputParserException as e:
        err_name = type(e).__name__

    draw_card(
        "model reply (prose, no JSON):\n"
        '  "Sure! One large oat milk\n'
        '   latte -- $6.25."\n'
        "\n"
        f"parser: raised {err_name}\n"
        "\n"
        "no Order object was produced.\n"
        "the boundary is enforced, not hoped for.",
        FIGURES / "step-06.png",
        TITLE,
        tone="bad",
        subtitle="garbage in, exception out",
        note="A malformed reply does not limp into the till as a half-order. It raises, loudly, at the boundary.",
    )

    # ---- oracle ----
    assert isinstance(order, Order)
    assert order.items[0].drink == "latte" and order.items[0].size == "large"
    assert order.items[0].extras == ["oat milk"] and order.to_go is True
    assert order.food == ["croissant"]
    expected = MENU["latte"] + SIZES["large"] + EXTRAS["oat milk"] + FOOD["croissant"]
    assert t == round(expected, 2) == 9.75, (
        "the total must recompute from real menu data"
    )
    assert parser.get_format_instructions() in model.calls[0][0].content, (
        "the format instructions must really have reached the model"
    )
    assert err_name == "OutputParserException", "the mangled reply must really raise"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, Order parsed, total ${t:.2f} recomputed from MENU, "
        f"mangled JSON raised {err_name}. All checks passed."
    )


if __name__ == "__main__":
    main()
