"""What a commit is: a snapshot, a parent link, and a hash of both.

Builds a real three-commit history, then reads the commit objects back out of
git with plumbing so the figures and the tables show what git actually stored,
not what a diagram author remembered.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():  # walk up to the collection root
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, render_split  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice",))

    box.commit("alice", "recipe.md", "eggs\n", "add eggs")
    box.snap(
        "one commit", note="A commit with no parent. This is the root of the history."
    )

    box.commit("alice", "recipe.md", "eggs\nflour\n", "add flour")
    box.snap(
        "two commits",
        note="The new commit records the previous one as its parent. The arrow runs backwards in time.",
    )

    box.commit("alice", "recipe.md", "eggs\nflour\nmilk\n", "add milk")
    final = box.snap(
        "three commits",
        note="A history is a chain of snapshots, each one pointing back at the one it came from.",
    )

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    # What git actually stored. `cat-file -p` prints the raw commit object.
    alice = box.paths["alice"]
    rows = []
    for sha in sorted(final.repos["alice"].commits, key=lambda s: _depth(alice, s)):
        raw = subprocess.run(
            ["git", "cat-file", "-p", sha], cwd=alice, capture_output=True, text=True
        ).stdout
        fields = {"tree": "", "parent": ""}
        for line in raw.splitlines():
            for key in fields:
                if line.startswith(key + " "):
                    fields[key] = line.split()[1][:7]
        rows.append(
            {
                "commit": sha,
                "tree": fields["tree"],
                "parent": fields["parent"] or "(none, this is the root)",
                "subject": final.repos["alice"].commits[sha].subject,
            }
        )

    with (TABLES / "commit-objects.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["commit", "tree", "parent", "subject"])
        writer.writeheader()
        writer.writerows(rows)

    # The oracle. Every claim the README makes about this history, checked.
    commits = final.repos["alice"].commits
    assert len(commits) == 3, "three commits were made, so three exist"

    roots = [c for c in commits.values() if not c.parents]
    assert len(roots) == 1, "exactly one commit has no parent"
    assert all(len(c.parents) == 1 for c in commits.values() if c.parents), (
        "an ordinary commit has exactly one parent"
    )
    assert len({r["tree"] for r in rows}) == 3, (
        "each commit points at a different tree, because the content changed each time"
    )

    # The hash covers the parent, not just the content. Same file, different
    # parent, therefore a different commit. This is why history is tamper-evident.
    box.git("alice", "switch --detach " + roots[0].sha)
    box.write("alice", "recipe.md", "eggs\nflour\nmilk\n")
    box.git("alice", "add recipe.md")
    box.git("alice", "commit -m same-content-different-parent")
    replay = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=alice,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tip = max(commits, key=lambda s: _depth(alice, s))
    same_tree = [_tree_of(alice, replay), _tree_of(alice, tip)]
    assert same_tree[0] == same_tree[1], (
        "identical working tree, so identical tree object"
    )
    assert replay != tip, (
        "identical content but a different parent gives a different commit hash"
    )

    with (TABLES / "hash-covers-the-parent.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["commit", "tree", "parent", "note"])
        writer.writerow(
            [tip, same_tree[1], "the second commit", "built up one file at a time"]
        )
        writer.writerow(
            [
                replay,
                same_tree[0],
                "the root commit",
                "same final file, replayed onto a different parent",
            ]
        )

    print(
        f"3 commits, {len(list(FIGURES.glob('*.png')))} figures, 3 tables. Checks passed."
    )
    box.cleanup()


def _tree_of(repo: Path, rev: str) -> str:
    """The tree object a commit points at: the snapshot itself."""
    return subprocess.run(
        ["git", "rev-parse", "--short", f"{rev}^{{tree}}"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _depth(repo: Path, sha: str) -> int:
    out = subprocess.run(
        ["git", "rev-list", "--count", sha], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    return int(out or 0)


if __name__ == "__main__":
    main()
