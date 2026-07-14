"""The two kinds of merge, decided by one question: did main move?

Builds three sandboxes that start from an identical history.

  ff       main never moved, so `git merge` slides the pointer forward and
           writes no commit at all.
  noff     the same history, merged with --no-ff, which forces a merge commit
           that a fast-forward would not have created.
  diverged main moved too, so a fast-forward is impossible. `--ff-only` is run
           first and is expected to fail, then a plain merge builds a commit
           with two parents.

Every claim the README makes is checked here. The fast-forward asserts the
commit count did not change and that no commit has two parents. The three-way
merge asserts a two-parent commit now exists and the count went up by exactly
one. The refusal asserts a non-zero exit code and a history that did not move.
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

from lib.gitviz import Sandbox, clear, draw, layout  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"


def out(box: Sandbox, args: str) -> str:
    """Run a read-only git command and hand back its stdout, stripped."""
    return box.git("alice", args, record=False, check=False).stdout.strip()


def short(box: Sandbox, rev: str) -> str:
    return out(box, f"rev-parse --short {rev}")


def parents(state, sha: str) -> list[str]:
    return state.commits[sha].parents


def log_graph(box: Sandbox) -> list[str]:
    """git's own drawing of the history, captured verbatim.

    The renderer in lib/gitviz.py places a commit by its depth, so when a merge
    commit's second parent is already an ancestor of its first (which is exactly
    the --no-ff case) the extra edge lies flat along the chain and cannot be
    seen. git log --graph puts that lineage in its own column instead, so the
    merged branch shows up as a bubble. Both views are of the same DAG.
    """
    return [line for line in out(box, "log --graph --oneline").splitlines() if line]


def verdict(proc) -> str:
    """The line where git names what it just did, taken from git's own output."""
    text = (proc.stdout or "") + (proc.stderr or "")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("Fast-forward", "Merge made", "fatal:", "Already up")):
            return line
    return text.strip().splitlines()[0]


def merge_facts(box: Sandbox) -> dict[str, str]:
    """The one test that decides which kind of merge you are about to get.

    A fast-forward is possible exactly when the current branch is already an
    ancestor of the branch being merged in, which is the same as saying the
    merge base of the two IS the current branch's tip. Then there is nothing to
    combine, and git can just slide the pointer.
    """
    base_full = out(box, "merge-base main feature")
    base = short(box, base_full)
    main_tip = short(box, "main")
    feature_tip = short(box, "feature")
    is_ancestor = box.git(
        "alice", "merge-base --is-ancestor main feature", record=False, check=False
    )
    return {
        "main_tip": main_tip,
        "feature_tip": feature_tip,
        "merge_base": base,
        "merge_base_is_main_tip": "yes" if base == main_tip else "no",
        "fast_forward_possible": "yes" if is_ancestor.returncode == 0 else "no",
    }


def build_common(box: Sandbox):
    """The history every scenario starts from: two commits, then a feature commit.

    Content and author dates are fixed, so this produces byte-identical commit
    hashes in every sandbox. That is content-addressing, not a coincidence.
    """
    box.commit("alice", "recipe.md", "eggs\n", "add eggs")
    box.commit("alice", "recipe.md", "eggs\nflour\n", "add flour")
    box.git("alice", "switch -c feature")
    box.commit("alice", "sauce.md", "butter\n", "start the sauce")
    box.commit("alice", "sauce.md", "butter\ncream\n", "thicken the sauce")
    box.git("alice", "switch main")


def main():
    clear(FIGURES, TABLES)
    kinds: list[dict] = []
    tests: list[dict] = []

    # ---- 1. The fast-forward -------------------------------------------
    ff = Sandbox(people=("alice",))
    build_common(ff)
    ff_start = ff.snap(
        "main has not moved since the branch",
        note="feature is two steps ahead of main. main is still an ancestor of feature.",
    )
    ff_test = merge_facts(ff)
    assert ff_test["fast_forward_possible"] == "yes", (
        "main is an ancestor of feature, so a fast-forward is available"
    )
    assert ff_test["merge_base"] == ff_test["main_tip"], (
        "the merge base of main and feature IS main's tip: nothing to combine"
    )
    tests.append({"scenario": "main did not move", **ff_test})

    before = len(ff_start.repos["alice"].commits)
    merging = ff.git("alice", "merge feature", check=False)
    assert merging.returncode == 0, "the fast-forward merge succeeds"
    ff_done = ff.snap(
        "git merge feature",
        note="Fast-forward. No commit was written. The main pointer slid onto feature's commit.",
    )
    ff_state = ff_done.repos["alice"]

    # The claims the README makes about this figure.
    assert len(ff_state.commits) == before, (
        "a fast-forward creates NO commit: the count is unchanged"
    )
    assert all(len(c.parents) <= 1 for c in ff_state.commits.values()), (
        "no commit in a fast-forwarded history has two parents"
    )
    assert ff_state.branches["main"] == ff_state.branches["feature"], (
        "main now points at exactly the commit feature points at"
    )
    assert ff_state.branches["main"] == ff_test["feature_tip"], (
        "and that commit is the one feature already had, not a new one"
    )
    assert not ff_done.highlight, "nothing is new, so nothing is drawn as new"
    assert "Fast-forward" in merging.stdout, (
        "git itself calls it a fast-forward, in so many words"
    )

    kinds.append(
        {
            "scenario": "main did not move",
            "command": "git merge feature",
            "exit_code": merging.returncode,
            "commits_before": before,
            "commits_after": len(ff_state.commits),
            "merge_commit_created": "no",
            "parents_of_main_tip": len(parents(ff_state, ff_state.branches["main"])),
            "main_tip_after": ff_state.branches["main"],
            "git_said": verdict(merging),
        }
    )

    # git's own view: a fast-forwarded history is a straight line. There is no
    # bubble, because there is no merge commit to open one.
    ff_graph = log_graph(ff)
    assert all(line.startswith("* ") for line in ff_graph), (
        "every line of the fast-forwarded log is a plain commit: no fork, no bubble"
    )
    assert len(ff_graph) == len(ff_state.commits), "and one line per commit, no extras"
    graphs = [{"scenario": "main did not move", "line": line} for line in ff_graph]

    ff_snaps = list(ff.snapshots)
    ff_xs, ff_ys = layout(ff_snaps)
    for i, snap in enumerate(ff_snaps, start=1):
        draw(snap, ff_xs, ff_ys, FIGURES / f"ff-{i:02d}.png", mode="solo")
    ff.cleanup()

    # ---- 2. The same history, merged with --no-ff -----------------------
    noff = Sandbox(people=("alice",))
    build_common(noff)
    noff_start = noff.snap(
        "the identical starting point, rebuilt",
        note="Same content, same hashes. A commit is named by a hash of what is in it.",
    )
    noff_state0 = noff_start.repos["alice"]

    # Same content in, same hashes out. Worth checking, because it is the
    # reason the two scenarios below are genuinely comparable.
    assert set(noff_state0.commits) == set(ff_start.repos["alice"].commits), (
        "identical content and dates give identical commit hashes"
    )

    before_noff = len(noff_state0.commits)
    noff_merge = noff.git(
        "alice", "merge --no-ff feature -m merge-feature", check=False
    )
    assert noff_merge.returncode == 0, "--no-ff merge succeeds"
    noff_done = noff.snap(
        "git merge --no-ff feature",
        note="A fast-forward was possible. --no-ff refused it and wrote a merge commit anyway.",
    )
    noff_final = noff_done.repos["alice"]
    noff_tip = noff_final.branches["main"]

    assert len(noff_final.commits) == before_noff + 1, (
        "--no-ff writes exactly one new commit where a fast-forward wrote none"
    )
    assert len(parents(noff_final, noff_tip)) == 2, (
        "and that commit has two parents, even though one line of history was empty"
    )
    assert noff_final.branches["feature"] in parents(noff_final, noff_tip), (
        "one parent is the tip of feature"
    )
    assert ff_test["main_tip"] in parents(noff_final, noff_tip), (
        "the other parent is where main was standing"
    )
    assert noff_final.branches["feature"] != noff_tip, (
        "main advanced onto the merge commit, feature did not move"
    )

    kinds.append(
        {
            "scenario": "main did not move, --no-ff",
            "command": "git merge --no-ff feature",
            "exit_code": noff_merge.returncode,
            "commits_before": before_noff,
            "commits_after": len(noff_final.commits),
            "merge_commit_created": "yes",
            "parents_of_main_tip": len(parents(noff_final, noff_tip)),
            "main_tip_after": noff_tip,
            "git_said": verdict(noff_merge),
        }
    )

    # This is what --no-ff actually buys. The merge commit opens a bubble, so the
    # two commits of the branch are still legible as one unit of work, which a
    # fast-forward would have dissolved into the trunk.
    noff_graph = log_graph(noff)
    assert any(line.strip() == "|\\" for line in noff_graph), (
        "--no-ff opens a fork in git's own graph, where the fast-forward opened none"
    )
    assert any(line.strip() == "|/" for line in noff_graph), (
        "and closes it again at the commit the branch started from"
    )
    graphs += [
        {"scenario": "main did not move, --no-ff", "line": line} for line in noff_graph
    ]

    noff_snaps = list(noff.snapshots)
    noff_xs, noff_ys = layout(noff_snaps)
    for i, snap in enumerate(noff_snaps, start=1):
        draw(snap, noff_xs, noff_ys, FIGURES / f"noff-{i:02d}.png", mode="solo")
    noff.cleanup()

    # ---- 3. main moved too, so the histories diverged -------------------
    div = Sandbox(people=("alice",))
    build_common(div)
    # Two more commits on main, touching a different file from feature's, so the
    # merge below is clean. Conflicts are a separate tutorial.
    div.commit("alice", "pantry.md", "salt\n", "stock the pantry")
    div.commit("alice", "pantry.md", "salt\npepper\n", "more pantry")
    div_start = div.snap(
        "main moved as well: the histories diverged",
        note="Neither branch is an ancestor of the other now. There is nothing to slide forward onto.",
    )
    div_state0 = div_start.repos["alice"]
    div_test = merge_facts(div)
    assert div_test["fast_forward_possible"] == "no", (
        "main is no longer an ancestor of feature"
    )
    assert div_test["merge_base"] != div_test["main_tip"], (
        "the merge base is the old shared commit, not main's tip"
    )
    tests.append({"scenario": "main moved too", **div_test})

    before_div = len(div_state0.commits)
    old_main = div_test["main_tip"]
    feature_tip = div_test["feature_tip"]

    # Ask for a fast-forward that cannot happen. Git refuses rather than
    # silently doing something else.
    refused = div.git("alice", "merge --ff-only feature", check=False)
    after_refusal = div.read("alice")
    assert refused.returncode != 0, (
        "--ff-only exits non-zero when it cannot fast-forward"
    )
    assert len(after_refusal.commits) == before_div, "the refusal changed nothing"
    assert after_refusal.branches["main"] == old_main, "main did not move"
    assert not after_refusal.dirty, "and the working tree was left alone"
    refusal_said = verdict(refused)
    assert refusal_said.startswith("fatal:"), "git refuses out loud, it does not guess"

    kinds.append(
        {
            "scenario": "main moved too, --ff-only",
            "command": "git merge --ff-only feature",
            "exit_code": refused.returncode,
            "commits_before": before_div,
            "commits_after": len(after_refusal.commits),
            "merge_commit_created": "no",
            "parents_of_main_tip": len(
                parents(after_refusal, after_refusal.branches["main"])
            ),
            "main_tip_after": after_refusal.branches["main"],
            "git_said": refusal_said,
        }
    )

    # Now the merge git actually has to do: build a new commit from two lines.
    three_way = div.git("alice", "merge feature -m merge-feature", check=False)
    assert three_way.returncode == 0, "the three-way merge succeeds, with no conflict"
    div_done = div.snap(
        "git merge feature",
        note="A new commit with TWO parents: the tip of main and the tip of feature.",
    )
    div_final = div_done.repos["alice"]
    tip = div_final.branches["main"]

    assert len(div_final.commits) == before_div + 1, (
        "the three-way merge writes exactly one new commit"
    )
    two_parent = [s for s, c in div_final.commits.items() if len(c.parents) == 2]
    assert two_parent == [tip], (
        "exactly one commit has two parents, and it is the new tip of main"
    )
    assert set(parents(div_final, tip)) == {old_main, feature_tip}, (
        "its parents are where main was and where feature is"
    )
    assert div_final.branches["feature"] == feature_tip, (
        "merging into main does not move feature"
    )
    assert div_done.highlight == {tip}, "the merge commit is the only new object"
    # The merge commit's snapshot really does hold both sides' work.
    assert out(div, "show HEAD:sauce.md") == "butter\ncream", (
        "feature's file, at feature's version, is in the merge commit's snapshot"
    )
    assert out(div, "show HEAD:pantry.md") == "salt\npepper", (
        "and so is main's file, which feature never saw"
    )

    kinds.append(
        {
            "scenario": "main moved too",
            "command": "git merge feature",
            "exit_code": three_way.returncode,
            "commits_before": before_div,
            "commits_after": len(div_final.commits),
            "merge_commit_created": "yes",
            "parents_of_main_tip": len(parents(div_final, tip)),
            "main_tip_after": tip,
            "git_said": verdict(three_way),
        }
    )

    div_graph = log_graph(div)
    assert any(line.strip() == "|\\" for line in div_graph), (
        "the three-way merge forks in git's graph too"
    )
    assert any(line.startswith("* |") for line in div_graph), (
        "and unlike the --no-ff case, main's own lane carries commits of its own"
    )
    graphs += [{"scenario": "main moved too", "line": line} for line in div_graph]

    div_snaps = list(div.snapshots)
    div_xs, div_ys = layout(div_snaps)
    for i, snap in enumerate(div_snaps, start=1):
        draw(snap, div_xs, div_ys, FIGURES / f"diverged-{i:02d}.png", mode="solo")
    div.cleanup()

    # ---- the tables the README quotes from ------------------------------
    with (TABLES / "merge-kinds.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(kinds[0]))
        writer.writeheader()
        writer.writerows(kinds)

    with (TABLES / "the-fast-forward-test.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)

    with (TABLES / "log-graph.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "line"])
        writer.writeheader()
        writer.writerows(graphs)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(f"{len(figs)} figures, 3 tables. All merge claims checked and passing.")


if __name__ == "__main__":
    main()
