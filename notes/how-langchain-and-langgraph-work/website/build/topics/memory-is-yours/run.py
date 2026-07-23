"""Chat history lives client-side: the model is stateless, the shop keeps the notebook.

Turn 1: "I'm Maya -- I'll have a latte." Turn 2: "actually, make it a
large." Run turn 2 twice. Bare chain: the model's actual received context
is one message and "it" is unresolvable. Wrapped in
RunnableWithMessageHistory keyed session_id="maya": the model demonstrably
receives turn 1 too. A second session "ben" stays empty: isolation is real.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.chat_history import InMemoryChatMessageHistory  # noqa: E402
from langchain_core.prompts import (  # noqa: E402
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_dual_messages, draw_messages  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Memory is yours, not the model's"

TURN_1 = "hi, I'm Maya -- I'll have a latte."
TURN_2 = "actually, make it a large."


def main():
    clear(FIGURES)

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are the barista at Beanline Coffee."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    model = scripted(
        "One latte for Maya, coming up -- $4.50.",
        "You got it, Maya -- one LARGE latte, $5.50.",
    )
    with_memory = RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = {"configurable": {"session_id": "maya"}}

    # turn 1, through the wrapper: lands in the "maya" history box
    with_memory.invoke({"input": TURN_1}, config=cfg)

    draw_messages(
        list(store["maya"].messages),
        FIGURES / "step-01.png",
        TITLE,
        new=(0, 1),
        right_text=(
            "store = {\n"
            '  "maya": [human, ai],\n'
            "}\n"
            "\n"
            "# a shelf under the counter.\n"
            "# the model keeps NOTHING."
        ),
        right_title='the history store, session "maya"',
        note="Turn 1 lands and is stored client-side, keyed by session. The model already forgot it.",
    )

    # turn 2, path A: bare model, no history wrapper
    amnesiac = scripted("Make WHAT large? I don't have an order for you.")
    bare_ctx = prompt.invoke({"history": [], "input": TURN_2}).to_messages()
    amnesiac.invoke(bare_ctx)

    # turn 2, path B: through the wrapper, same session
    with_memory.invoke({"input": TURN_2}, config=cfg)

    amnesia_ctx = amnesiac.calls[0]
    memory_ctx = model.calls[1]

    draw_dual_messages(
        amnesia_ctx,
        memory_ctx,
        FIGURES / "step-02.png",
        TITLE,
        left_label="bare chain: what the model saw",
        right_label='with history ("maya"): what the model saw',
        left_verdict="bad",
        right_verdict="ok",
        note='Same turn-2 code, different memory owner. Left: "it" is unresolvable. Right: turn 1 is present.',
    )

    draw_messages(
        memory_ctx,
        FIGURES / "step-03.png",
        TITLE,
        new=(1, 2),
        note="The wrapper spliced the stored turn-1 exchange into the real prompt. That is all memory is.",
    )

    get_session_history("ben")  # ben walks in; his box exists and is empty

    draw_dual_messages(
        list(store["maya"].messages),
        list(store["ben"].messages),
        FIGURES / "step-04.png",
        TITLE,
        left_label='store["maya"]: 4 messages',
        right_label='store["ben"]: empty',
        left_verdict="ok",
        right_verdict="ok",
        note="Two customers, two shelves. Sessions isolate because the KEY isolates, not the model.",
    )

    # ---- oracle ----
    assert len(amnesia_ctx) == 2, "the bare path must have shown the model no history"
    assert not any(TURN_1 in str(m.content) for m in amnesia_ctx)
    assert any("latte" in str(m.content) for m in memory_ctx), (
        "turn 1 must really be inside the wrapped path's context"
    )
    assert len(memory_ctx) >= 4
    assert len(store["maya"].messages) == 4
    assert len(store["ben"].messages) == 0
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, bare context {len(amnesia_ctx)} msgs vs wrapped "
        f"{len(memory_ctx)} msgs, maya=4 stored, ben=0. All checks passed."
    )


if __name__ == "__main__":
    main()
