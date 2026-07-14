"""How an agent should stage, reconcile, and land its work. Real repos, asserted.

Three things separate a safe agent commit from a dangerous one, and two of them
are shown here on real repositories:

  scoped staging   `git add -A` in a shared tree sweeps in another agent's
                   unrelated work. Staging your own paths by name does not.
  a moving main    while the agent worked, main moved. Comparing only to the
                   branch point misses it; fetching and rebasing onto the new
                   main reveals and resolves it before the pull request.

The third, the human review gate on a pull request, is a GitHub policy rather
than a local git fact, so it lives in the README and the best-practices table.
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

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"


def files_in_head(box: Sandbox, who: str) -> set[str]:
    """Every path in the HEAD commit's tree, read straight out of git."""
    out = subprocess.run(
        ["git", "ls-tree", "--name-only", "-r", "HEAD"],
        cwd=box.paths[who],
        capture_output=True,
        text=True,
    ).stdout
    return set(out.split())


def scoped_staging() -> dict[str, set[str]]:
    """Same situation, two staging commands. What ends up in the commit differs."""
    result = {}
    for strategy in ("add -A", "add my_feature.py"):
        box = Sandbox(people=("agent",), local=True)
        box.commit("agent", "app.py", "print('app')\n", "start the app")
        # Another agent's unrelated, uncommitted file is sitting in the directory.
        box.write("agent", "other_agent_wip.py", "# half-finished, not mine\n")
        # This agent's own new file.
        box.write("agent", "my_feature.py", "def feature():\n    return 1\n")
        box.git("agent", strategy)
        box.git("agent", "commit -m add-my-feature")
        result[strategy] = files_in_head(box, "agent")
        box.cleanup()
    return result


def moving_main() -> tuple[Sandbox, str, str, str]:
    """Agent branches from main, main moves, agent reconciles before its PR."""
    box = Sandbox(people=("agent", "teammate"))
    box.commit("agent", "app.py", "print('app')\n", "start the app")
    box.git("agent", "push -u origin main")
    box.git("teammate", "pull")
    branch_point = box.read("agent").branches["main"]

    # The agent starts a fix on a branch off the current main.
    box.git("agent", "switch -c agent-fix")
    box.commit("agent", "fix.py", "def fix():\n    return 1\n", "add the fix")

    # Meanwhile a teammate lands their own work on main. main has moved.
    box.commit(
        "teammate", "feature.py", "def feature():\n    return 2\n", "add feature"
    )
    box.git("teammate", "push")

    # The agent fetches, and now sees that origin/main is ahead of its branch point.
    box.git("agent", "fetch origin")
    new_main = box.read("github").branches["main"]
    box.snap(
        "main moved while the agent worked",
        note="The agent's branch is based on the old main. origin/main is now "
        "ahead. Opening a pull request now would compare against a main that "
        "moved on. Fetch first, then reconcile.",
    )

    # The agent rebases its branch onto the new main before opening the PR.
    box.git("agent", "rebase origin/main")
    box.snap(
        "the agent rebases onto the new main, then opens its PR",
        note="The fix is replayed on top of the teammate's work. The branch is now "
        "a clean, up-to-date proposal a human can review and merge.",
    )
    fix_tip = box.read("agent").branches["agent-fix"]
    return box, branch_point, new_main, fix_tip


def is_ancestor(box: Sandbox, who: str, maybe_ancestor: str, tip: str) -> bool:
    proc = box.git(who, f"merge-base --is-ancestor {maybe_ancestor} {tip}", check=False)
    return proc.returncode == 0


BEST_PRACTICES = [
    (
        "give each agent its own git worktree",
        "separate HEAD, index, and working directory, so parallel edits never collide",
        "Claude Code docs",
    ),
    (
        "branch each worktree from origin/HEAD",
        "start from a clean tree that matches the remote",
        "Claude Code docs",
    ),
    (
        "stage explicit paths, never git add -A in a shared tree",
        "avoid sweeping another agent's unrelated work into your commit",
        "this tutorial",
    ),
    (
        "diff against the current origin/main, not your branch point",
        "main moves while you work, so reconcile the real delta before the PR",
        "this tutorial",
    ),
    (
        "open a pull request, require human review, do not push to main",
        "keep unreviewed agent code off the branch you deploy from",
        "Claude Code, GitHub",
    ),
    (
        "enforce branch protection, the requester cannot self-approve",
        "make the review a real gate rather than a suggestion",
        "GitHub docs",
    ),
    (
        "give the agent a check it can run, tests or a build",
        "it self-corrects instead of stopping at looks-done",
        "Claude Code docs",
    ),
    (
        "have a fresh-context agent review the diff",
        "an unbiased reader catches what the author is blind to",
        "Claude Code docs",
    ),
    (
        "never trust an agent's I-committed or I-did-not, read git log and status",
        "agents mis-report, so verify what is on disk",
        "this tutorial",
    ),
]


def main() -> None:
    clear(FIGURES, TABLES)

    # ---- demo 1: scoped staging ----
    staged = scoped_staging()
    with (TABLES / "scoped-add.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["staging command", "files in the resulting commit"])
        for strat, files in staged.items():
            w.writerow([f"git {strat}", ", ".join(sorted(files))])

    # ---- demo 2: a moving main ----
    box, branch_point, new_main, fix_tip = moving_main()
    render(box, FIGURES, TABLES, mode="team", repos=("github", "agent", "teammate"))

    # before-rebase ancestry was captured inside moving_main via the two snaps;
    # re-check the end state here for the assertions.
    contains_new_main = is_ancestor(box, "agent", new_main, "agent-fix")

    # ---- best-practices table (from a generated CSV, per the house rule) ----
    with (TABLES / "best-practices.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["practice", "why", "source"])
        w.writerows(BEST_PRACTICES)

    # ---- ASSERTIONS ----
    # (A) git add -A swept the other agent's file into the commit; scoped did not
    assert "other_agent_wip.py" in staged["add -A"], "add -A absorbs foreign work"
    assert "my_feature.py" in staged["add -A"]
    assert "other_agent_wip.py" not in staged["add my_feature.py"], (
        "scoped staging commits only the named path"
    )
    assert "my_feature.py" in staged["add my_feature.py"]

    # (B) main really moved: the new main is not the agent's branch point
    assert new_main != branch_point, "the teammate's push advanced main"

    # (C) after fetch + rebase, the agent's branch is built on the new main
    assert contains_new_main, "the rebased branch now contains the new main"
    assert is_ancestor(box, "agent", branch_point, new_main), (
        "the new main descends from the old one"
    )

    # (D) figures and the two demo tables exist
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, f"expected 2 figures, got {[f.name for f in figs]}"
    assert (TABLES / "scoped-add.csv").exists()
    assert (TABLES / "best-practices.csv").exists()

    box.cleanup()
    print("agent-commit-and-review-hygiene: ok, 2 figures, all assertions passed")


if __name__ == "__main__":
    main()
