"""Why vLLM is fast: a concept topic, drawn not run. The naive server keeps
one contiguous KV-cache block per request (wasted, over-allocated memory);
vLLM's PagedAttention slices the cache into fixed-size blocks shared across
requests like an OS page table, and continuous batching admits a new
request into a running batch the instant a GPU slot frees up rather than
waiting for the whole batch to finish. No gpt2 call here -- there is no
serving engine in this deck to benchmark, so the oracle only pins down the
figure count, never a fabricated speed number.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Why vLLM is fast"


def main():
    clear(FIGURES)

    draw_card(
        "naive serving: one contiguous KV-cache allocation per request,\n"
        "sized for the longest sequence the server might see.\n\n"
        "most of that block sits empty for most requests.\n"
        "a batch only starts once, and only ends once, together.",
        FIGURES / "step-01.png",
        TITLE,
        tone="bad",
        subtitle="the naive server",
        note="Fixed-size, over-provisioned, and static-batched: memory and requests both wait on the slowest member.",
    )

    draw_card(
        "PagedAttention: the KV cache is sliced into fixed-size BLOCKS,\n"
        "allocated on demand, like OS virtual-memory pages.\n\n"
        "continuous batching: a finished request's slot is handed to the\n"
        "next queued request immediately, not at the next batch boundary.",
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        subtitle="vLLM's two ideas",
        note="Both ideas raise GPU utilization by shrinking wasted memory and wasted idle time, not by changing the math.",
    )

    draw_card(
        "same forward pass, same logits, same greedy/sampled token\n"
        "vLLM changes THROUGHPUT (requests/sec, GPU $ per token),\n"
        "never the model's answer for a given prompt and settings.",
        FIGURES / "step-03.png",
        TITLE,
        subtitle="what does NOT change",
        note="This deck's oracles never assert a wall-clock number for vLLM: no vLLM install here to measure honestly.",
    )

    # ---- oracle ----
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, concept-only (no vLLM install, no fabricated numbers). All checks passed."
    )


if __name__ == "__main__":
    main()
