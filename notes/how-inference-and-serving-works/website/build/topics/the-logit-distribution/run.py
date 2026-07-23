"""The logit distribution: a softmax over gpt2's real 50257-entry vocab
turns raw logits into a probability distribution, and the flipbook zooms
into its top-k. The oracle checks the distribution is a real distribution
(sums to 1), that its argmax matches the forward pass, and that the top
probability is a valid probability.
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
from langviz import clear, draw_card, draw_topk  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The logit distribution"
CONTEXT = "The cat sat on the"
K = 8


def main():
    clear(FIGURES)
    tok, model = load()

    ctx_ids = tok(CONTEXT, return_tensors="pt")["input_ids"]
    next_id, logits, _ = forward(model, ctx_ids)

    probs = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probs, K)
    items = [
        (tok.decode([int(i)]), float(p)) for p, i in zip(topk.values, topk.indices)
    ]
    picked_idx = int(topk.indices.tolist().index(next_id))

    draw_card(
        f"logits: shape {tuple(logits.shape)}, raw scores, unbounded\n"
        "probs = softmax(logits)\n"
        f"probs: shape {tuple(probs.shape)}, sums to 1.0, each in [0, 1]",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="one softmax, over the whole vocab",
        note="50257 real probabilities, one per gpt2 vocab entry, computed from the real logits.",
    )

    draw_topk(
        items,
        FIGURES / "step-02.png",
        TITLE,
        note=f'top-{K} of 50257, given context "{CONTEXT}"',
    )

    draw_topk(
        items,
        FIGURES / "step-03.png",
        TITLE,
        picked=picked_idx,
        note="the greedy pick (argmax) is the tallest bar -- same id the forward-pass topic found.",
    )

    # ---- oracle ----
    total = float(probs.sum().item())
    assert abs(total - 1.0) < 1e-4, f"softmax must sum to 1.0, got {total}"
    assert int(torch.argmax(probs).item()) == next_id, (
        "distribution argmax must match forward pass"
    )
    top_prob = float(probs.max().item())
    assert 0.0 <= top_prob <= 1.0
    assert items[picked_idx][1] == top_prob

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, probs sum to {total:.4f}, top prob {top_prob:.4f}, "
        f"argmax matches forward pass. All checks passed."
    )


if __name__ == "__main__":
    main()
