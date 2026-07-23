"""vLLM, the throughput path (CODA): a concept topic, drawn not run.
Ollama is the right tool for one process serving one or a few users --
exactly what the two prior real calls in this deck did. The moment
request volume grows past what one GPU's naive batching can absorb, the
paged-attention + continuous-batching ideas from why-vllm-is-fast stop
being trivia and become the throughput path teams reach for. No vLLM
install here, so the oracle only pins down the figure count.
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
TITLE = "vLLM, the throughput path"


def main():
    clear(FIGURES)

    draw_card(
        "this deck's two real serving calls (Ollama, litellm) both talked\n"
        "to ONE process, serving requests roughly one at a time.\n\n"
        "that is the right choice for a laptop, a demo, or a single user --\n"
        "and the wrong choice once real concurrent traffic shows up.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="where Ollama tops out",
        note="Ollama serves you well until throughput, not correctness, becomes the constraint.",
    )

    draw_card(
        "vLLM: an inference server built around PagedAttention + continuous\n"
        "batching, purpose-built to push many concurrent requests through\n"
        "one GPU at high utilization.\n\n"
        "same OpenAI-shaped call surface litellm already showed --\n"
        "point api_base at a vLLM server instead of ollama, nothing else changes.",
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        subtitle="the outgrow path",
        note="The serving engine changes underneath; the call shape from litellm-one-shape does not.",
    )

    # ---- oracle ----
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, concept-only (no vLLM install, no fabricated numbers). All checks passed."
    )


if __name__ == "__main__":
    main()
