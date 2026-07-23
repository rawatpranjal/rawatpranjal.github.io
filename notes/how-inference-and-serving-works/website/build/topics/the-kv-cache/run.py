"""The KV cache: reusing past keys/values lets each new step attend to the
whole prefix without recomputing it. The flipbook photographs the cache
grid growing one column per generated token. The oracle proves the cache
is a speedup, not a different answer: at every step, next-token logits
computed WITH the cache equal logits computed by a full recompute WITHOUT
any cache, and the cache's sequence dimension grows by exactly 1 each step.
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
from langviz import clear, draw_card, draw_kv_cache  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "The KV cache"
CONTEXT = "The cat sat on the"
N_STEPS = 4  # generated tokens after the prefix


def main():
    clear(FIGURES)
    tok, model = load()
    n_layers = model.config.n_layer

    ctx_ids = tok(CONTEXT, return_tensors="pt")["input_ids"]

    draw_card(
        f"past_key_values: one (key, value) pair per layer x {n_layers} layers\n"
        "each holds every token's key/value the model has already seen\n\n"
        "step 1 (the prefix): no cache yet, so it computes all positions.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="what gets cached",
        note=f'context: "{CONTEXT}", {ctx_ids.shape[-1]} tokens, {n_layers} layers to fill.',
    )

    # ---- generate with the cache, recording logits + cache shape each step ----
    cur_ids = ctx_ids
    past = None
    with_cache_logits = []
    seq_lens = []
    generated_pieces = []
    for step in range(N_STEPS + 1):  # +1: the prefix forward pass itself
        next_id, logits, past = forward(model, cur_ids, past_key_values=past)
        with_cache_logits.append(logits)
        seq_lens.append(past.get_seq_length())
        draw_kv_cache(
            n_layers,
            seq_lens[-1],
            FIGURES / f"step-0{step + 2}.png",
            TITLE,
            note=f"after step {step}: cache holds {seq_lens[-1]} positions x {n_layers} layers.",
        )
        if step > 0:
            generated_pieces.append(tok.decode([next_id]))
        cur_ids = torch.tensor([[next_id]])  # only the new token goes in next time

    draw_card(
        f'greedy continuation: "{CONTEXT}" + "{"".join(generated_pieces)}"\n\n'
        f"{N_STEPS} generation steps, each a forward pass over exactly 1 new token\n"
        f"plus the growing cache -- never a full recompute of the prefix.",
        FIGURES / f"step-0{N_STEPS + 3}.png",
        TITLE,
        tone="good",
        subtitle="what the cache bought",
        note="Verified against a from-scratch, no-cache recompute below.",
    )

    # ---- oracle ----
    # cache sequence length must grow by exactly 1 each step
    for i in range(1, len(seq_lens)):
        assert seq_lens[i] == seq_lens[i - 1] + 1, (
            f"cache seq length must grow by 1 each step, got {seq_lens[i - 1]} -> {seq_lens[i]}"
        )
    assert seq_lens[0] == ctx_ids.shape[-1]

    # WITH cache vs WITHOUT cache: full recompute of the growing sequence must
    # give the identical next-token logits at every step.
    full_ids = ctx_ids
    for step in range(N_STEPS + 1):
        with torch.no_grad():
            no_cache_logits = model(input_ids=full_ids, use_cache=False).logits[
                :, -1, :
            ]
        assert torch.allclose(no_cache_logits, with_cache_logits[step], atol=1e-3), (
            f"step {step}: cached and uncached logits must match (the cache is a speedup, not a different answer)"
        )
        next_id_check = int(torch.argmax(no_cache_logits, dim=-1).item())
        full_ids = torch.cat([full_ids, torch.tensor([[next_id_check]])], dim=-1)

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == N_STEPS + 3, f"expected {N_STEPS + 3} figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, cache grew {seq_lens[0]}->{seq_lens[-1]} by exactly 1/step, "
        f"cached logits == uncached recompute at all {N_STEPS + 1} steps. All checks passed."
    )


if __name__ == "__main__":
    main()
