"""Streaming: the same answer, sooner.

Beanline reads Maya's order back and the till display types it out chunk
by chunk. The chunks are real: langchain's streaming path word-splits the
scripted reply into AIMessageChunks, and the oracle proves the assembled
stream equals, byte for byte, what a plain invoke of the same script
returns. Streaming changes when you see the answer, not what it says.
"""

from __future__ import annotations

import functools
import operator
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langchain_core.messages import AIMessageChunk, HumanMessage  # noqa: E402

from beanline import scripted  # noqa: E402
from langviz import clear, draw_card, draw_stream  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The same answer, sooner"

READBACK = "One large oat-milk latte to go, that's $6.25 -- coming right up, Maya."


def main():
    clear(FIGURES)

    ask = [HumanMessage(content="read that back to me?")]

    model = scripted(READBACK)
    chunks = [c.content for c in model.stream(ask)]

    n = len(chunks)
    draw_stream(
        chunks,
        1,
        FIGURES / "step-01.png",
        TITLE,
        note="model.stream() instead of invoke: the first chunk lands while the rest is still coming.",
    )
    draw_stream(
        chunks,
        max(2, n // 3),
        FIGURES / "step-02.png",
        TITLE,
        note="Each chunk is an AIMessageChunk. The display assembles them as they arrive.",
    )
    draw_stream(
        chunks,
        max(3, (2 * n) // 3),
        FIGURES / "step-03.png",
        TITLE,
        note="Same words, sooner: the customer is already reading while the model is still talking.",
    )
    draw_stream(
        chunks,
        n,
        FIGURES / "step-04.png",
        TITLE,
        note=f"All {n} chunks arrived. The assembled sentence is complete.",
    )

    # a fresh model with the same script, invoked without streaming
    plain = scripted(READBACK).invoke(ask)

    draw_card(
        f"streamed, then joined:\n  {''.join(chunks)!r}\n\n"
        f"plain invoke, same script:\n  {plain.content!r}\n\n"
        f"equal: True   chunks: {n}   chunk + chunk really concatenates",
        FIGURES / "step-05.png",
        TITLE,
        tone="good",
        subtitle="the ledger",
        note="Streaming changes WHEN you see the answer, not WHAT it says.",
    )

    # ---- oracle ----
    raw = list(scripted(READBACK).stream(ask))
    assert all(isinstance(c, AIMessageChunk) for c in raw)
    assert "".join(chunks) == READBACK
    assert functools.reduce(operator.add, raw).content == READBACK
    assert plain.content == READBACK
    assert len(chunks) >= 8
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {len(chunks)} real chunks reassembled equal to "
        f"the plain invoke. All checks passed."
    )


if __name__ == "__main__":
    main()
