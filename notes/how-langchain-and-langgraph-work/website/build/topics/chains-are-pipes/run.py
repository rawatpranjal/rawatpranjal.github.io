"""Runnables and LCEL: the pipe changes the payload's type.

The greeter chain prompt | model | StrOutputParser() runs for real, and
the flipbook photographs the payload at each hop: dict, ChatPromptValue,
AIMessage, str. The oracle recomputes the chain hop by hop and asserts
the composed run equals the hop-wise run, then .batch()es two customers
through the same pipe and checks order.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompt_values import ChatPromptValue  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from beanline import menu_board, scripted  # noqa: E402
from langviz import clear, draw_pipeline  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Chains are pipes"

MOCHA_REPLY = "One mocha coming up, Maya -- $5.00."
BEN_REPLY = "A cold brew for Ben -- $4.25."


def stages(upto: int, payloads: list[str]) -> list[dict]:
    """The four hops of the greeter chain, lit up to hop `upto`."""
    labels = [
        ("input", "dict"),
        ("prompt", "ChatPromptValue"),
        ("model", "AIMessage"),
        ("parser", "str"),
    ]
    out = []
    for i, (label, type_label) in enumerate(labels):
        state = "done" if i < upto else ("active" if i == upto else "pending")
        out.append(
            {
                "label": label,
                "type_label": type_label,
                "payload": payloads[i] if i <= upto else "",
                "state": state,
            }
        )
    return out


def main():
    clear(FIGURES)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are the barista at Beanline Coffee. Menu: {menu}"),
            ("human", "{name} says: {input}"),
        ]
    )
    parser = StrOutputParser()

    values = {"menu": menu_board(), "name": "Maya", "input": "something chocolatey?"}
    payloads = [
        '{"name": "Maya", "input": "something chocolatey?"}',
        "[system: barista + menu..., human: Maya says...]",
        f'AIMessage("{MOCHA_REPLY}")',
        f'"{MOCHA_REPLY}"',
    ]

    for i, note in enumerate(
        [
            "The order slip enters the counter: a plain dict.",
            "Hop 1, the prompt station: the dict becomes rendered, typed messages.",
            "Hop 2, the model station: the messages become an AIMessage.",
            "Hop 3, the parser station: the message becomes a plain string.",
        ]
    ):
        draw_pipeline(
            stages(i, payloads),
            FIGURES / f"step-0{i + 1}.png",
            TITLE,
            note=note,
        )

    # run it for real, composed
    model = scripted(MOCHA_REPLY)
    chain = prompt | model | parser
    answer = chain.invoke(values)

    done = stages(3, payloads)
    done[3]["state"] = "done"
    draw_pipeline(
        done,
        FIGURES / "step-05.png",
        TITLE,
        note="chain = prompt | model | parser; chain.invoke(...) ran the whole counter in one call.",
    )

    # batch: two customers, one pipe
    model_b = scripted(MOCHA_REPLY, BEN_REPLY)
    chain_b = prompt | model_b | parser
    pair = chain_b.batch(
        [
            values,
            {"menu": menu_board(), "name": "Ben", "input": "something cold?"},
        ]
    )

    draw_pipeline(
        [
            {
                "label": "maya + ben",
                "type_label": "2 dicts",
                "payload": ".batch([maya, ben])",
                "state": "done",
            },
            {
                "label": "the same pipe",
                "type_label": "prompt | model | parser",
                "payload": "",
                "state": "active",
            },
            {
                "label": "2 answers",
                "type_label": "list[str]",
                "payload": f'["{pair[0]}", "{pair[1]}"]',
                "state": "done",
            },
        ],
        FIGURES / "step-06.png",
        TITLE,
        note="Same pipe, two payloads: .batch keeps input order. Every stage answers .invoke, .batch, .stream.",
    )

    # ---- oracle ----
    m2 = scripted(MOCHA_REPLY)
    hop1 = prompt.invoke(values)
    hop2 = (prompt | m2).invoke(values)
    assert isinstance(hop1, ChatPromptValue)
    assert isinstance(hop2, AIMessage)
    assert parser.invoke(hop2) == answer, (
        "hop-wise recomputation must equal the composed run"
    )
    assert isinstance(answer, str) and answer == MOCHA_REPLY
    assert model.calls[0] == hop1.to_messages(), (
        "the composed chain must have shown the model exactly the rendered prompt"
    )
    assert pair == [MOCHA_REPLY, BEN_REPLY], "batch must preserve input order"
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, composed run == hop-wise run, batch of 2 in order. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
