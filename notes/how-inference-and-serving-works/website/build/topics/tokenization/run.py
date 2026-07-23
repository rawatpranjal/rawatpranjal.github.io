"""Tokenization: gpt2's byte-pair tokenizer splits a real sentence into a
real list of token ids. The flipbook photographs the string, the token
pieces, and the ids underneath. The oracle round-trips encode -> decode
through two independent tokenizer entry points and pins down the exact
id list and length for the fixed running sentence.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from gpt2util import SENTENCE, load  # noqa: E402
from langviz import clear, draw_card, draw_tokens  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Tokenization"


def main():
    clear(FIGURES)
    tok, _ = load()

    draw_card(
        f'"{SENTENCE}"\n\nA plain python string. The model never sees this.',
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the running example",
        note="Every topic in this deck completes this exact sentence, one real gpt2 token at a time.",
    )

    pieces = tok.tokenize(SENTENCE)
    draw_tokens(
        pieces,
        FIGURES / "step-02.png",
        TITLE,
        note=f"tok.tokenize(text) -> {len(pieces)} byte-pair pieces. 'Ġ' marks a leading space.",
    )

    ids = tok.encode(SENTENCE)
    draw_tokens(
        pieces,
        FIGURES / "step-03.png",
        TITLE,
        ids=ids,
        note="tok.encode(text) -> the real integer id for each piece, gpt2's fixed 50257-token vocab.",
    )

    decoded = tok.decode(ids)
    draw_card(
        f'tok.encode(text) -> {ids}\n\ntok.decode(ids) -> "{decoded}"',
        FIGURES / "step-04.png",
        TITLE,
        tone="good",
        subtitle="round trip",
        note="decode(encode(text)) reconstructs the original string exactly.",
    )

    # ---- oracle ----
    ids_via_call = tok(SENTENCE)["input_ids"]  # independent entry point
    assert ids == ids_via_call, "encode() and the tokenizer __call__ must agree"
    assert tok.decode(tok.encode(SENTENCE)) == SENTENCE, "round trip must be exact"
    assert ids == [464, 3797, 3332, 319, 262, 2603], f"unexpected id list: {ids}"
    assert len(ids) == 6
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {len(ids)} tokens, round trip exact, ids == {ids}. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
