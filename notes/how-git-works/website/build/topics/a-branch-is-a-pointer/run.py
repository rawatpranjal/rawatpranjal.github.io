"""A branch is a pointer: a file holding one hash, and nothing else.

Creates branches, switches between them, detaches HEAD, and deletes a branch,
reading the ref files straight off disk at every step. The claim that a branch
is 41 bytes rather than a copy of the project is checked, not asserted.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, render_split  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"


def ref_file(box: Sandbox, branch: str) -> Path:
    return box.paths["alice"] / ".git" / "refs" / "heads" / branch


def main():
    clear(FIGURES, TABLES)
    # The reflog view stays on, so the commit stranded by the branch deletion
    # at the end is still visible. It really is still on disk.
    box = Sandbox(people=("alice",), show_unreachable=True)

    box.commit("alice", "recipe.md", "eggs\n", "add eggs")
    box.commit("alice", "recipe.md", "eggs\nflour\n", "add flour")
    box.snap(
        "two commits on main",
        note="main is a pointer at the newest commit. HEAD says which pointer you are moving.",
    )

    # Creating a branch writes one small file. It copies nothing.
    before = len(box.read("alice").commits)
    box.git("alice", "branch feature")
    after = box.read("alice").commits
    assert len(after) == before, "creating a branch creates no commit"

    on_disk = ref_file(box, "feature").read_text().strip()
    assert len(on_disk) == 40, "a branch is one 40-character hash on disk"
    assert on_disk.startswith(after[box.read("alice").branches["main"]].sha), (
        "the new branch points at exactly the commit you were standing on"
    )
    box.snap(
        "git branch feature",
        note=f"No new commit. Just a new file, .git/refs/heads/feature, containing {on_disk[:12]}...",
    )

    # Switching moves HEAD. It does not move any branch.
    box.git("alice", "switch feature")
    box.snap(
        "git switch feature",
        note="Only HEAD moved. Both branches still point at the same commit.",
    )

    # Committing moves the branch HEAD is attached to, and only that one.
    main_before = box.read("alice").branches["main"]
    box.commit("alice", "sauce.md", "butter\n", "start the sauce")
    state = box.read("alice")
    assert state.branches["main"] == main_before, (
        "committing on feature must not move main"
    )
    assert state.branches["feature"] != main_before, "feature moved forward"
    box.snap(
        "commit on feature",
        note="feature moved forward because HEAD was attached to it. main did not move.",
    )

    # Detached HEAD: pointing HEAD straight at a commit, attached to no branch.
    box.git("alice", "switch --detach " + main_before)
    detached = box.read("alice")
    assert detached.head is None, "HEAD is attached to no branch now"
    assert detached.head_sha == main_before, "HEAD points straight at a commit"
    box.snap(
        "git switch --detach",
        note="HEAD points at a commit with no branch in between. A commit made here would belong to nothing.",
    )

    # Deleting a branch removes the pointer. The commits it pointed at survive,
    # unreferenced, exactly where they were.
    box.git("alice", "switch main")
    tip_of_feature = box.read("alice").branches["feature"]
    box.git("alice", "branch -D feature")
    after_delete = box.read("alice")
    assert "feature" not in after_delete.branches, "the pointer is gone"
    assert tip_of_feature in after_delete.commits, (
        "the commit itself survives the deletion of the branch that pointed at it"
    )
    assert not ref_file(box, "feature").exists(), "the ref file is gone from disk"
    box.snap(
        "git branch -D feature",
        note=f"The pointer is gone. The commit {tip_of_feature} is still on disk, just unreachable by name.",
    )

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    with (TABLES / "what-a-branch-costs.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question", "answer"])
        writer.writerow(["commits before git branch feature", before])
        writer.writerow(["commits after git branch feature", len(after)])
        writer.writerow(["bytes written to create the branch", len(on_disk) + 1])
        writer.writerow(["what those bytes contain", "one 40-character commit hash"])
        writer.writerow(["files copied", 0])

    print(f"{len(list(FIGURES.glob('*.png')))} figures, 2 tables. Checks passed.")
    box.cleanup()


if __name__ == "__main__":
    main()
