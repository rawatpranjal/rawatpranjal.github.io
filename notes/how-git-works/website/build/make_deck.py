"""Assemble the monster git deck.

The narrative scaffold (scaffold.qmd) is hand-authored: the design-system header,
the legend, the conceptual SVG slides, the part dividers. Wherever a real-repo
flipbook belongs, the scaffold carries a one-line marker:

    <!--FB:topic-->            (title taken from the topic's README H1)
    <!--FB:topic|My Title-->   (explicit title)

This script expands each marker into a section divider plus one auto-animate slide
per figure in assets/<topic>/ (sorted), so stepping the deck flipbooks the true
evolution of a real repository. Everything else in the scaffold passes through
unchanged. Output is git.qmd, which quarto renders to the folder deck.

    ~/.local/bin/python build/make_deck.py [--render]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent  # notes/how-git-works
ASSETS = DECK / "assets"
TUT_SRC = Path("/Users/pranjal/Code/applied-science-git/git-and-github")

MARKER = re.compile(r"^\s*<!--FB:([a-z0-9-]+)(?:\|(.+?))?-->\s*$")


def readme_title(topic: str) -> str:
    """The topic's own README H1, so titles stay in the source's voice."""
    for readme in TUT_SRC.glob(f"*/{topic}/README.md"):
        return readme.read_text().splitlines()[0].lstrip("# ").strip()
    return topic.replace("-", " ").capitalize()


def flipbook(topic: str, title: str | None) -> str:
    title = title or readme_title(topic)
    frames = sorted((ASSETS / topic).glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames for topic {topic!r} in {ASSETS / topic}")
    out = [f"\n# {title}\n"]
    for f in frames:
        # auto-animate tweens between consecutive same-title slides, so the graph
        # grows rather than cuts. The command is drawn into each figure already.
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
    (DECK / "git.qmd").write_text("".join(out))
    print(f"wrote git.qmd: {n_fb} flipbooks, {n_frames} figure slides")

    if "--render" in sys.argv:
        proc = subprocess.run(
            ["quarto", "render", "git.qmd", "--to", "revealjs"],
            cwd=DECK,
            capture_output=True,
            text=True,
        )
        sys.stderr.write(proc.stdout[-1500:] + proc.stderr[-1500:])
        if proc.returncode != 0:
            raise SystemExit("quarto render failed")
        print("rendered git.html")


if __name__ == "__main__":
    main()
