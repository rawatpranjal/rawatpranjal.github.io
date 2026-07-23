"""Ollama, for real (CODA): a real HTTP call to a real local server. This
is the deck's second serving path, alongside the raw in-process gpt2 calls
every earlier topic made. `import ollama` talks to http://localhost:11434,
where a real qwen2.5:0.5b is loaded. The flipbook photographs the real
prompt and the real completion. The oracle makes the SAME temperature=0
call twice and asserts the returned text is byte-identical -- real
determinism, over a real network call, not a mock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ollama

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Ollama, for real"
MODEL = "qwen2.5:0.5b"
PROMPT = "Complete this sentence in five words or fewer: The cat sat on the"


def main():
    clear(FIGURES)

    draw_card(
        f"ollama.generate(\n"
        f'    model="{MODEL}",\n'
        f'    prompt="{PROMPT}",\n'
        '    options={"temperature": 0},\n'
        ")\n\n"
        "-> a real HTTP POST to http://localhost:11434/api/generate",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the real call",
        note="Same shape as every serving API: a prompt in, a completion out, over the network (here, localhost).",
    )

    resp1 = ollama.generate(model=MODEL, prompt=PROMPT, options={"temperature": 0})
    text1 = resp1["response"]

    draw_card(
        f'prompt: "{PROMPT}"\n\ncompletion: "{text1.strip()}"',
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        subtitle="run 1",
        note=f"model: {MODEL}, served by the real ollama daemon on this machine.",
    )

    resp2 = ollama.generate(model=MODEL, prompt=PROMPT, options={"temperature": 0})
    text2 = resp2["response"]

    draw_card(
        f'run 1: "{text1.strip()}"\n\nrun 2: "{text2.strip()}"\n\nidentical: {text1 == text2}',
        FIGURES / "step-03.png",
        TITLE,
        tone="good" if text1 == text2 else "bad",
        subtitle="run 2, same prompt, temperature=0",
        note="Two real network round trips to a real model server, checked byte for byte.",
    )

    # ---- oracle ----
    assert isinstance(text1, str) and len(text1.strip()) > 0, (
        "response must be non-empty"
    )
    assert text1 == text2, "temperature=0 must be byte-identical across two real calls"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, real ollama call to {MODEL}, response {len(text1)} chars, "
        f"2 runs byte-identical. All checks passed."
    )


if __name__ == "__main__":
    main()
