"""litellm, one shape (CODA): the same completion call, unified. litellm's
`completion()` wraps ollama's HTTP server behind the OpenAI-shaped request
and response every hosted provider also speaks, so the SAME call signature
that would hit OpenAI or Anthropic hits the local qwen2.5:0.5b here. The
flipbook photographs the one call shape in front of the local provider.
The oracle makes a real request, checks the response is genuinely
OpenAI-shaped (choices[0].message.content), non-empty, and that two
temperature=0 calls return identical text.
"""

from __future__ import annotations

import sys
from pathlib import Path

from litellm import completion

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from langviz import clear, draw_card  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "litellm, one shape"
MODEL = "ollama/qwen2.5:0.5b"
API_BASE = "http://localhost:11434"
QUESTION = "Complete this sentence in five words or fewer: The cat sat on the"


def call():
    return completion(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        temperature=0,
        api_base=API_BASE,
    )


def main():
    clear(FIGURES)

    draw_card(
        "completion(\n"
        f'    model="{MODEL}",\n'
        '    messages=[{"role": "user", "content": ...}],\n'
        "    temperature=0,\n"
        f'    api_base="{API_BASE}",\n'
        ")\n\n"
        'swap model="ollama/..." for "openai/gpt-4o" or "anthropic/claude-..."\n'
        "and this exact call hits a hosted provider instead.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="one call shape, any provider",
        note="litellm normalizes every provider behind the OpenAI chat-completion shape.",
    )

    r1 = call()
    text1 = r1.choices[0].message.content

    draw_card(
        f'r.choices[0].message.content ->\n"{text1.strip()}"',
        FIGURES / "step-02.png",
        TITLE,
        tone="good",
        subtitle="run 1, real response",
        note=f"model={MODEL}, routed through litellm to the real local ollama server.",
    )

    r2 = call()
    text2 = r2.choices[0].message.content

    draw_card(
        f'run 1: "{text1.strip()}"\n\nrun 2: "{text2.strip()}"\n\nidentical: {text1 == text2}',
        FIGURES / "step-03.png",
        TITLE,
        tone="good" if text1 == text2 else "bad",
        subtitle="run 2, same call, temperature=0",
        note="Two real calls through litellm's real HTTP path to the real local model.",
    )

    # ---- oracle ----
    assert hasattr(r1, "choices") and len(r1.choices) > 0, (
        "response must be OpenAI-shaped: .choices"
    )
    assert hasattr(r1.choices[0], "message") and hasattr(
        r1.choices[0].message, "content"
    ), "response must be OpenAI-shaped: .choices[0].message.content"
    assert isinstance(text1, str) and len(text1.strip()) > 0, (
        "response content must be non-empty"
    )
    assert text1 == text2, "temperature=0 must be byte-identical across two real calls"

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, real litellm call to {MODEL}, OpenAI-shaped response, "
        f"2 runs byte-identical. All checks passed."
    )


if __name__ == "__main__":
    main()
