"""Run every topic's run.py, copy its figures/ into assets/<topic>/.

Fails loudly if any topic errors or produces no figures, before make_deck.py ever
gets a chance to silently expand an empty flipbook.

    ~/.local/bin/python build/regen_all.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
PYTHON = "/Users/pranjal/.local/bin/python"


def main() -> None:
    topics = sorted(p for p in (HERE / "topics").iterdir() if (p / "run.py").exists())
    if not topics:
        raise SystemExit("no topics found under build/topics/")

    for topic_dir in topics:
        topic = topic_dir.name
        proc = subprocess.run(
            [PYTHON, "run.py"], cwd=topic_dir, capture_output=True, text=True
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(f"{topic}: run.py failed")

        figures = sorted((topic_dir / "figures").glob("*.png"))
        if not figures:
            raise SystemExit(f"{topic}: run.py produced no figures")

        dest = ASSETS / topic
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for f in figures:
            shutil.copy2(f, dest / f.name)

        print(f"{topic}: {len(figures)} figures -- {proc.stdout.strip()}")

    print(f"ok: {len(topics)} topics regenerated")


if __name__ == "__main__":
    main()
