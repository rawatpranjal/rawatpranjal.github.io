"""What goes wrong when two agents share one working directory, and the fix.

Part one: two agents in one checkout. Agent A leaves uncommitted work, Agent B
runs a routine `git reset --hard` before starting its own task, and Agent A's
work is gone. Part two: the same two agents, a worktree each, and Agent B's
reset touches only Agent B.

The three trees in each figure are read out of real git with `git show`, so the
before and after are exactly what git held on disk.
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

from lib.gitviz import Sandbox, clear, draw_trees, read_trees  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

BASE = "timeout = 30\n"
A_WORK = "timeout = 30\nretries = 5\n"
B_MESS = "timeout = 999\n"


def main() -> None:
    clear(FIGURES, TABLES)

    # ---- PART 1: one shared checkout, the hazard ----
    shared = Sandbox(people=("repo",), local=True)
    shared.commit("repo", "config.py", BASE, "set defaults")

    # Agent A edits a file and has not committed. The change is only on disk.
    shared.write("repo", "config.py", A_WORK)
    before = read_trees(shared, "repo", "config.py")
    draw_trees(
        before,
        FIGURES / "step-01.png",
        "One directory: Agent A has uncommitted work",
        command="# agent A edited config.py, has not committed",
        note="The new line lives only in the working tree. The index and the last "
        "commit still hold the old file. Nothing has saved Agent A's change.",
    )

    # Agent B, in the SAME directory, cleans up before its own task.
    shared.git("repo", "reset --hard HEAD")
    after = read_trees(shared, "repo", "config.py")
    draw_trees(
        after,
        FIGURES / "step-02.png",
        "Agent B ran git reset --hard: Agent A's work is gone",
        command="git reset --hard HEAD",
        note="reset --hard forces the working tree to match HEAD, discarding every "
        "uncommitted change in the directory, including another agent's.",
    )

    # ---- PART 2: a worktree each, the fix ----
    box = Sandbox(people=("dev",))
    box.commit("dev", "config.py", BASE, "set defaults")
    box.git("dev", "push -u origin main")
    for key, branch in (("agent-a", "agent-a"), ("agent-b", "agent-b")):
        box.git("dev", f"worktree add {box.root / key} -b {branch}")
        box.paths[key] = box.root / key

    # Agent A's good, uncommitted work, in its own worktree.
    box.write("agent-a", "config.py", A_WORK)
    # Agent B makes a mess in ITS worktree and resets --hard to undo it.
    box.write("agent-b", "config.py", B_MESS)
    box.git("agent-b", "reset --hard HEAD")

    survivor = read_trees(box, "agent-a", "config.py")
    b_tree = read_trees(box, "agent-b", "config.py")
    draw_trees(
        survivor,
        FIGURES / "step-03.png",
        "A worktree each: Agent B's reset never touches Agent A",
        command="# in agent-b's worktree: git reset --hard HEAD",
        note="Agent B's reset only cleaned Agent B's working tree. Agent A's "
        "uncommitted work, in a separate worktree, is exactly as it was.",
    )

    # ---- table: the whole story in two rows ----
    with (TABLES / "hazard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setup", "agent A's uncommitted work after agent B's reset --hard"])
        w.writerow(["one shared checkout", "lost"])
        w.writerow(["one worktree per agent", "survives"])

    # ---- ASSERTIONS ----
    # (A) before the reset, the shared working tree really held Agent A's change
    assert before["working tree"] == A_WORK
    assert before["HEAD"] == BASE, "it was never committed"
    assert before["index"] == BASE, "it was never staged"

    # (B) after Agent B's reset in the shared checkout, Agent A's work is gone
    assert after["working tree"] == BASE, "reset --hard discarded the uncommitted edit"
    assert after["working tree"] != before["working tree"], "the file changed under A"

    # (C) with a worktree each, Agent A's uncommitted work survives Agent B's reset
    assert survivor["working tree"] == A_WORK, "A's work is untouched in its worktree"
    assert b_tree["working tree"] == BASE, "B's reset cleaned only B's own worktree"

    # (D) the figures were produced
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {[f.name for f in figs]}"

    shared.cleanup()
    box.cleanup()
    print("the-shared-checkout-hazard: ok, 3 figures, all assertions passed")


if __name__ == "__main__":
    main()
