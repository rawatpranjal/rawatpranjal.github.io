"""The conversation is a typed list.

Beanline gets a persona via a SystemMessage, Maya says hello via a
HumanMessage, and the reply comes back not as a string but as an AIMessage
object that appends straight onto the same list. The list IS the
conversation: model.calls photographs exactly which messages crossed to
the model, and the oracle checks the roles, the order, and the types.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_messages  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The conversation is a typed list"

PERSONA = (
    "You are the barista at Beanline Coffee. Friendly, brief, always "
    "quote exact prices from the menu."
)


def main():
    clear(FIGURES)

    model = scripted(
        "Hi Maya! The mocha's been popular all morning -- $5.00. "
        "Or your usual latte, $4.50."
    )

    convo: list = []

    draw_messages(
        convo,
        FIGURES / "step-01.png",
        TITLE,
        right_text=(
            "SystemMessage   who the bot is\n"
            "HumanMessage    what the user said\n"
            "AIMessage       what the model said\n"
            "ToolMessage     what a tool returned\n"
            "\n"
            "convo: list[BaseMessage] = []"
        ),
        right_title="the four message types",
        note="Not strings: typed objects with a role, content, and metadata.",
    )

    convo.append(SystemMessage(content=PERSONA))
    draw_messages(
        convo,
        FIGURES / "step-02.png",
        TITLE,
        new=(0,),
        note="The persona drops in as a SystemMessage: instructions, not dialogue.",
    )

    convo.append(HumanMessage(content="hi! what's good today?"))
    draw_messages(
        convo,
        FIGURES / "step-03.png",
        TITLE,
        new=(1,),
        note="Maya speaks: a HumanMessage. The list now holds two roles in order.",
    )

    reply = model.invoke(convo)
    draw_messages(
        convo,
        FIGURES / "step-04.png",
        TITLE,
        right_text=(
            "reply = model.invoke(convo)\n"
            "\n"
            "type(reply)  ->  AIMessage\n"
            'reply.type   ->  "ai"\n'
            "\n"
            "# the WHOLE list crossed over;\n"
            "# the model saw both messages"
        ),
        right_title="model.invoke(list) -> AIMessage",
        note="The whole list crosses to the model in one call. The return value is a message, not a string.",
    )

    convo.append(reply)
    draw_messages(
        convo,
        FIGURES / "step-05.png",
        TITLE,
        new=(2,),
        note="The reply appends straight back onto the list. The list is the conversation.",
    )

    # ---- oracle ----
    assert isinstance(reply, AIMessage) and reply.type == "ai"
    assert "$5.00" in reply.content
    assert len(model.calls) == 1
    assert [type(m) for m in model.calls[0]] == [SystemMessage, HumanMessage], (
        "the model must have really received the persona and the greeting"
    )
    assert model.calls[0][0].content == PERSONA
    assert [m.type for m in convo] == ["system", "human", "ai"]
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, model really saw [system, human], "
        f"convo ends [system, human, ai]. All checks passed."
    )


if __name__ == "__main__":
    main()
