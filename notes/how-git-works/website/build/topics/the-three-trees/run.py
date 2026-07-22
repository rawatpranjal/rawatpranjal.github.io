"""The three trees: the working tree, the index, and HEAD.

Edits one file and reads the same file back from all three places after every
step, so the figures show the actual divergence git is tracking rather than a
drawing of it. `git status` is nothing more than a report on these differences.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw_trees, read_trees  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

FILE = "recipe.md"


def status(box: Sandbox) -> str:
    """The porcelain status code, which is exactly a summary of the two gaps."""
    # The two status columns are staged-vs-HEAD and working-vs-staged, and the
    # first column can be a space. Stripping the leading space destroys the
    # distinction, so only the trailing newline comes off.
    return (
        subprocess.run(
            ["git", "status", "--porcelain", FILE],
            cwd=box.paths["alice"],
            capture_output=True,
            text=True,
        ).stdout.rstrip("\n")
        or "(clean)"
    )


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice",))
    rows = []

    def capture(step, title, command, note, expect_status):
        trees = read_trees(box, "alice", FILE)
        got = status(box)
        assert got == expect_status, (
            f"{title}: expected status {expect_status!r}, git said {got!r}"
        )
        draw_trees(trees, FIGURES / f"trees-{step:02d}.png", title, command, note)
        rows.append(
            {
                "step": step,
                "title": title,
                "command": command,
                "working_tree": (trees["working tree"] or "")
                .strip()
                .replace("\n", " / "),
                "index": (trees["index"] or "").strip().replace("\n", " / "),
                "head": (trees["HEAD"] or "").strip().replace("\n", " / "),
                "git_status": got,
            }
        )
        return trees

    # 1. A clean starting point. All three trees agree.
    box.commit("alice", FILE, "eggs\nflour\n", "add eggs and flour")
    capture(
        1,
        "A clean working tree",
        "git commit -m 'add eggs and flour'",
        "All three agree, so git status has nothing to report.",
        "(clean)",
    )

    # 2. Edit the file. Only the working tree moves.
    box.write("alice", FILE, "eggs\nflour\nmilk\n")
    capture(
        2,
        "After editing the file",
        f"edit {FILE}",
        "The working tree moved. The index and HEAD did not. This gap is 'not staged for commit'.",
        " M " + FILE,
    )

    # 3. Stage it. The index catches up. HEAD does not.
    box.git("alice", f"add {FILE}")
    capture(
        3,
        "After git add",
        f"git add {FILE}",
        "git add copies the working tree into the index. That is its whole job.",
        "M  " + FILE,
    )

    # 4. Commit. HEAD catches up, and all three agree again.
    box.git("alice", "commit -m add-milk")
    capture(
        4,
        "After git commit",
        "git commit -m 'add milk'",
        "git commit turns whatever is in the index into a new commit. The working tree was never consulted.",
        "(clean)",
    )

    # The point of the whole tutorial: commit takes the INDEX, not the file on
    # disk. Stage one version, then edit again, and the commit captures the
    # staged version, leaving your newer edit uncommitted.
    box.write("alice", FILE, "eggs\nflour\nmilk\nSTAGED\n")
    box.git("alice", f"add {FILE}")
    box.write("alice", FILE, "eggs\nflour\nmilk\nSTAGED\nEDITED-AFTER-STAGING\n")
    trees = capture(
        5,
        "Edited again after staging",
        f"git add {FILE}; edit {FILE}",
        "Three different versions at once. This is legal, and it is what the index is for.",
        "MM " + FILE,
    )
    assert trees["working tree"] != trees["index"] != trees["HEAD"], (
        "all three trees genuinely differ here"
    )

    box.git("alice", "commit -m commit-the-staged-version")
    committed = read_trees(box, "alice", FILE)
    assert "EDITED-AFTER-STAGING" not in committed["HEAD"], (
        "the commit captured the staged version, not the newer edit on disk"
    )
    assert "EDITED-AFTER-STAGING" in committed["working tree"], (
        "the newer edit is still sitting in the working tree, uncommitted"
    )
    capture(
        6,
        "After committing the staged version",
        "git commit -m 'commit the staged version'",
        "The commit holds the STAGED line. The later edit is still uncommitted on disk.",
        " M " + FILE,
    )

    with (TABLES / "the-three-trees.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step",
                "title",
                "command",
                "working_tree",
                "index",
                "head",
                "git_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} figures, 1 table. Checks passed.")
    box.cleanup()


if __name__ == "__main__":
    main()
