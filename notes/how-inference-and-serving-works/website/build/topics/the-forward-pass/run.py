"""The forward pass: one real call to gpt2 turns a token strip into
next-token logits, and greedy argmax turns those logits into one more
token. The flipbook photographs the context, the raw logits, and the
appended greedy token. The oracle recomputes the forward pass twice and
checks the argmax id agrees with itself and with a from-scratch call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from gpt2util import forward, load  # noqa: E402
from langviz import clear, draw_card, draw_tokens  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The forward pass"
CONTEXT = "The cat sat on the"  # SENTENCE minus its last word, the true target


def main():
    clear(FIGURES)
    tok, model = load()

    ctx_ids = tok(CONTEXT, return_tensors="pt")["input_ids"]
    ctx_pieces = tok.tokenize(CONTEXT)

    draw_tokens(
        ctx_pieces,
        FIGURES / "step-01.png",
        TITLE,
        note=f'input_ids, shape {tuple(ctx_ids.shape)}: "{CONTEXT}"',
    )

    draw_card(
        "logits = model(input_ids).logits\n"
        f"logits.shape -> (1, {ctx_ids.shape[-1]}, 50257)\n\n"
        "one real number per vocab entry, for every position.\n"
        "only the LAST position's row decides the next token.",
        FIGURES / "step-02.png",
        TITLE,
        subtitle="one forward pass",
        note="A single matrix multiply chain, run once, on CPU, no sampling yet.",
    )

    next_id, logits, _ = forward(model, ctx_ids)
    next_piece = tok.decode([next_id])
    top_logit = float(logits[0, next_id].item())

    draw_card(
        f"logits[:, -1, :].argmax() -> id {next_id}\n"
        f'tok.decode([{next_id}]) -> "{next_piece}"\n'
        f"logit value -> {top_logit:.3f}",
        FIGURES / "step-03.png",
        TITLE,
        tone="good",
        subtitle="greedy argmax",
        note="No randomness: argmax always picks this same id for this same context.",
    )

    draw_tokens(
        ctx_pieces + [next_piece],
        FIGURES / "step-04.png",
        TITLE,
        new=(len(ctx_pieces),),
        note=f'"{CONTEXT}" + greedy token -> "{CONTEXT}{next_piece}"',
    )

    # ---- oracle ----
    next_id_2, logits_2, _ = forward(model, ctx_ids)  # rerun, same context
    assert next_id_2 == next_id, "greedy decoding must be deterministic across runs"
    assert torch.allclose(logits, logits_2), "logits must be identical across runs"

    # independent from-scratch recomputation
    with torch.no_grad():
        raw = model(input_ids=ctx_ids).logits
    fresh_argmax = int(torch.argmax(raw[:, -1, :], dim=-1).item())
    assert fresh_argmax == next_id, "a fresh model(...) call must agree with forward()"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {len(figs)}"
    print(
        f'{len(figs)} figures, greedy token id {next_id} ("{next_piece}"), '
        f"deterministic across 2 reruns and a fresh call. All checks passed."
    )


if __name__ == "__main__":
    main()
