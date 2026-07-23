"""Sampling: the SAME real logits, decoded three ways. Greedy always takes
the argmax. Temperature reshapes the distribution before sampling (higher
temperature spreads mass, raising entropy). Top-p keeps the smallest
prefix of the sorted distribution whose cumulative mass reaches p, then
renormalizes. The oracle checks greedy == argmax, that a hotter
temperature really raises entropy, and that top-p's kept set shrinks (and
its kept mass stays >= p) as p drops.
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
from langviz import clear, draw_sampling  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Sampling"
CONTEXT = "The cat sat on the"
TOP_N = 6  # bars shown per panel (real distribution, just truncated for drawing)


def entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def top_p_set(probs: torch.Tensor, p: float) -> tuple[list[int], float]:
    """Smallest sorted prefix whose cumulative mass reaches p."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=0)
    n_keep = int((cum < p).sum().item()) + 1  # first index where cum >= p
    kept = sorted_idx[:n_keep].tolist()
    kept_mass = float(cum[n_keep - 1].item())
    return kept, kept_mass


def panel_items(tok, probs: torch.Tensor, idxs) -> list[tuple[str, float]]:
    return [(tok.decode([i]), float(probs[i])) for i in idxs[:TOP_N]]


def main():
    clear(FIGURES)
    tok, model = load()

    ctx_ids = tok(CONTEXT, return_tensors="pt")["input_ids"]
    greedy_id, logits, _ = forward(model, ctx_ids)
    probs = torch.softmax(logits[0], dim=-1)

    # greedy
    greedy_idx = int(torch.argmax(probs).item())
    order = torch.argsort(probs, descending=True).tolist()

    # temperature: reshape logits, then softmax
    t_lo, t_hi = 0.7, 1.6
    probs_lo = torch.softmax(logits[0] / t_lo, dim=-1)
    probs_hi = torch.softmax(logits[0] / t_hi, dim=-1)
    h_lo, h_hi = entropy(probs_lo), entropy(probs_hi)
    order_lo = torch.argsort(probs_lo, descending=True).tolist()
    order_hi = torch.argsort(probs_hi, descending=True).tolist()

    draw_sampling(
        [
            {
                "label": "greedy",
                "items": panel_items(tok, probs, order),
                "picked": 0,
                "stat": "argmax, always",
            },
            {
                "label": f"temperature={t_lo}",
                "items": panel_items(tok, probs_lo, order_lo),
                "picked": 0,
                "stat": f"H={h_lo:.2f} nats",
            },
            {
                "label": f"temperature={t_hi}",
                "items": panel_items(tok, probs_hi, order_hi),
                "picked": 0,
                "stat": f"H={h_hi:.2f} nats",
            },
        ],
        FIGURES / "step-01.png",
        TITLE,
        note="Same logits, one softmax per temperature: dividing by a hotter T flattens the distribution.",
    )

    # top-p: two thresholds on the SAME (temperature=1) distribution
    p_hi, p_lo = 0.9, 0.5
    kept_hi, mass_hi = top_p_set(probs, p_hi)
    kept_lo, mass_lo = top_p_set(probs, p_lo)

    draw_sampling(
        [
            {
                "label": "full distribution",
                "items": panel_items(tok, probs, order),
                "picked": 0,
                "stat": f"{TOP_N} of 50257 shown",
            },
            {
                "label": f"top-p={p_hi}",
                "items": panel_items(tok, probs, kept_hi),
                "picked": 0,
                "stat": f"kept {len(kept_hi)}, mass={mass_hi:.3f}",
            },
            {
                "label": f"top-p={p_lo}",
                "items": panel_items(tok, probs, kept_lo),
                "picked": 0,
                "stat": f"kept {len(kept_lo)}, mass={mass_lo:.3f}",
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        note="Sorted mass, truncated at p: a smaller p keeps a smaller, more confident set.",
    )

    # ---- oracle ----
    assert greedy_idx == greedy_id, "greedy panel must equal the forward-pass argmax"
    assert h_hi > h_lo, f"hotter temperature must raise entropy: {h_hi} vs {h_lo}"
    assert mass_hi >= p_hi - 1e-6 and mass_lo >= p_lo - 1e-6, "kept mass must reach p"
    assert len(kept_lo) <= len(kept_hi), (
        "a smaller p must keep a set no larger than a bigger p's"
    )
    assert len(kept_lo) < len(kept_hi), (
        "a strictly smaller p must shrink the kept set here"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, greedy==argmax, H({t_hi})={h_hi:.2f} > H({t_lo})={h_lo:.2f}, "
        f"top-p kept {len(kept_hi)}->{len(kept_lo)} tokens as p dropped {p_hi}->{p_lo}. "
        f"All checks passed."
    )


if __name__ == "__main__":
    main()
