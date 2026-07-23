"""Assemble the evals deck.

The narrative scaffold (scaffold.qmd) is hand-authored: the design-system header,
the conceptual SVG slides, the part dividers. Wherever a real-pipeline flipbook
belongs, the scaffold carries a one-line marker:

    <!--FB:topic|Title-->

This script expands each marker into a section divider plus one auto-animate slide
per figure in assets/<topic>/ (sorted), so stepping the deck flipbooks the true
evolution of real eval/observability code running at Beanline Coffee. Everything
else in the scaffold passes through unchanged. Output is evals.qmd, which
quarto renders to the folder deck.

    /Users/pranjal/.venvs/evals-deck/bin/python build/make_deck.py [--render]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent  # notes/how-evals-and-observability-work/website
ASSETS = DECK / "assets"

MARKER = re.compile(r"^\s*<!--FB:([a-z0-9-]+)\|(.+?)-->\s*$")


def flipbook(topic: str, title: str) -> str:
    frames = sorted((ASSETS / topic).glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames for topic {topic!r} in {ASSETS / topic}")
    out = [f"\n# {title}\n"]
    for f in frames:
        # auto-animate tweens between consecutive same-title slides, so each
        # figure morphs into the next rather than cutting.
        out.append(
            f"\n## {title} {{auto-animate=true}}\n\n"
            f'![](assets/{topic}/{f.name}){{fig-align="center"}}\n'
        )
    return "".join(out)


def main() -> None:
    scaffold = (DECK / "scaffold.qmd").read_text().splitlines(keepends=True)
    out, n_fb, n_frames = [], 0, 0
    for line in scaffold:
        m = MARKER.match(line)
        if m:
            block = flipbook(m.group(1), m.group(2))
            out.append(block)
            n_fb += 1
            n_frames += block.count("auto-animate=true")
        else:
            out.append(line)
    (DECK / "evals.qmd").write_text("".join(out))
    print(f"wrote evals.qmd: {n_fb} flipbooks, {n_frames} figure slides")

    if "--render" in sys.argv:
        proc = subprocess.run(
            ["quarto", "render", "evals.qmd", "--to", "revealjs"],
            cwd=DECK,
            capture_output=True,
            text=True,
        )
        sys.stderr.write(proc.stdout[-1500:] + proc.stderr[-1500:])
        if proc.returncode != 0:
            raise SystemExit("quarto render failed")
        print("rendered evals.html")


if __name__ == "__main__":
    main()
