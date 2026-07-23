"""Batching: two prompts of different lengths, padded to the same width and
run through gpt2 in ONE forward pass with an attention mask. The flipbook
photographs the padded batch and the per-row next-token pick. The oracle
proves batching doesn't change the answer: each batched row's logits equal
that same prompt run alone, and the pad positions are the ones the mask
marks as 0.
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

from gpt2util import load  # noqa: E402
from langviz import clear, draw_card, draw_tokens  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Batching"
PROMPT_A = "The cat sat on the"
PROMPT_B = "A dog ran across the busy"


def main():
    clear(FIGURES)
    tok, model = load()

    batch = tok([PROMPT_A, PROMPT_B], return_tensors="pt", padding=True)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    pad_id = tok.pad_token_id

    row_a_pieces = [
        tok.decode([i]) if i != pad_id else "<pad>" for i in ids[0].tolist()
    ]
    row_b_pieces = [
        tok.decode([i]) if i != pad_id else "<pad>" for i in ids[1].tolist()
    ]
    n_pad_a = int((ids[0] == pad_id).sum().item())
    n_pad_b = int((ids[1] == pad_id).sum().item())

    draw_tokens(
        row_a_pieces,
        FIGURES / "step-01.png",
        TITLE,
        new=tuple(i for i, t in enumerate(row_a_pieces) if t == "<pad>"),
        note=f'row 0: "{PROMPT_A}", right-padded with {n_pad_a} <pad> token(s) to match row 1.',
    )
    draw_tokens(
        row_b_pieces,
        FIGURES / "step-02.png",
        TITLE,
        new=tuple(i for i, t in enumerate(row_b_pieces) if t == "<pad>"),
        note=f'row 1: "{PROMPT_B}", {n_pad_b} pad token(s) -- the batch shape is {tuple(ids.shape)}.',
    )

    with torch.no_grad():
        batch_out = model(input_ids=ids, attention_mask=mask)
    batch_logits = batch_out.logits  # (2, T, vocab)

    # each row's next token comes from ITS OWN last real (non-pad) position
    real_len_a = int(mask[0].sum().item())
    real_len_b = int(mask[1].sum().item())
    logits_a = batch_logits[0, real_len_a - 1, :]
    logits_b = batch_logits[1, real_len_b - 1, :]
    next_a = tok.decode([int(torch.argmax(logits_a).item())])
    next_b = tok.decode([int(torch.argmax(logits_b).item())])

    draw_card(
        f"attention_mask row 0: {mask[0].tolist()}\n"
        f"attention_mask row 1: {mask[1].tolist()}\n\n"
        f'greedy next for row 0 (real pos {real_len_a - 1}): "{next_a}"\n'
        f'greedy next for row 1 (real pos {real_len_b - 1}): "{next_b}"',
        FIGURES / "step-03.png",
        TITLE,
        tone="good",
        subtitle="one forward pass, two prompts",
        note="The mask tells attention which positions are real; pad positions are excluded.",
    )

    # ---- oracle ----
    ids_a_alone = tok(PROMPT_A, return_tensors="pt")["input_ids"]
    ids_b_alone = tok(PROMPT_B, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits_a_alone = model(input_ids=ids_a_alone).logits[0, -1, :]
        logits_b_alone = model(input_ids=ids_b_alone).logits[0, -1, :]

    assert torch.allclose(logits_a, logits_a_alone, atol=1e-3), (
        "batched row 0's logits must equal running prompt A alone"
    )
    assert torch.allclose(logits_b, logits_b_alone, atol=1e-3), (
        "batched row 1's logits must equal running prompt B alone"
    )
    assert mask[0].tolist().count(0) == n_pad_a
    assert mask[1].tolist().count(0) == n_pad_b
    assert n_pad_a != n_pad_b or PROMPT_A == PROMPT_B, (
        "prompts of different length must pad differently"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, batch shape {tuple(ids.shape)}, "
        f"row logits match single-prompt runs (row0 pad={n_pad_a}, row1 pad={n_pad_b}). "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
