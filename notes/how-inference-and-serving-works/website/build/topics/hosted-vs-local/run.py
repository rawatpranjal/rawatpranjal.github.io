"""Hosted vs local: a concept topic, drawn not run. Every topic so far ran
gpt2 on this laptop's CPU -- one point on a wider spectrum. A hosted API
call trades control and privacy for someone else's GPUs and ops team; a
local model trades convenience for owning the weights, the latency, and
the failure modes. Neither is universally right, so no gpt2 call and no
benchmark here -- the oracle only pins down the figure count.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Hosted vs local"


def main():
    clear(FIGURES)

    draw_card(
        "hosted API (OpenAI, Anthropic, a cloud endpoint):\n"
        "  a network call to someone else's GPUs and ops team.\n\n"
        "local model (this deck's gpt2, or an Ollama model):\n"
        "  a process call to weights you loaded onto your own machine.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="two ways to run inference",
        note="Every prior topic in this deck used the local path: gpt2, in-process, on CPU.",
    )

    draw_scorecard(
        [
            {"label": "latency", "cells": ["network + queue", "local compute only"]},
            {"label": "cost", "cells": ["per-token, metered", "your hardware, sunk"]},
            {
                "label": "privacy",
                "cells": ["prompt leaves the box", "prompt never leaves"],
            },
            {
                "label": "control",
                "cells": ["their model, their schedule", "your weights, your version"],
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        columns=["hosted API", "local model"],
        note="Neither column wins outright: the next two topics run BOTH a real local model and a real hosted-style call.",
    )

    # ---- oracle ----
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, concept-only comparison (no benchmark claimed). All checks passed."
    )


if __name__ == "__main__":
    main()
