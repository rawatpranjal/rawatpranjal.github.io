"""What a pull request is: the git operations underneath a GitHub proposal.

A pull request (PR) is a GitHub feature, not a git command. There is no
`git pull-request`. A PR is a proposal, "please merge my feature branch into
main", opened so people can review and discuss before the merge lands. This
script builds the real git operations a PR wraps and reads the true state back
after each one. The three GitHub actions (opening the PR, clicking Merge,
clicking Delete branch) are browser actions, not commands, so the README
narrates them in prose. Every git fact drawn in the figures is real.

The story runs in team mode, a GitHub panel on top of Alice's clone:

  1. a base commit already on GitHub, Alice cloned it
  2. Alice branches and commits locally, so GitHub has never heard of the branch
  3. git push -u origin feature: the branch appears on GitHub, Alice opens a PR
  4. git push again, same branch: GitHub's feature branch moves forward and the
     open PR updates itself. No second PR, no second branch.
  5. GitHub merges the PR: main moves to contain feature's commits, server-side
  6. Alice pulls the merged main and deletes her local feature branch

The assertions at the end check every claim the README makes: the second push
moves the same branch and creates no new one, the branch exists on GitHub after
the push, main contains feature's commits after the merge, and the local branch
is gone after deletion while its commits live on inside main.
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

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"


def gh(box: Sandbox):
    return box.read("github")


def alice(box: Sandbox):
    return box.read("alice")


def github_merges_pr(
    box: Sandbox,
    feature_ref: str = "origin/feature",
    message: str = "Merge-pull-request-#1-from-feature-into-main",
):
    """Model GitHub's server-side merge of the pull request.

    Clicking Merge on GitHub does not run on Alice's machine. GitHub already has
    every object and performs the merge itself, then advances main. A bare repo
    has no working tree to merge in, so this uses a throwaway clone standing in
    for GitHub's own server-side worktree: it merges feature into main with a
    merge commit (what the default Merge button writes) and pushes main back.
    The clone is never registered with the sandbox, so it is never drawn. Only
    GitHub's main moving forward shows up in the figures.
    """
    scratch = box.root / "github-server-worktree"
    box.git(
        "github", f"clone {box.paths['github']} {scratch}", cwd=box.root, record=False
    )
    box.git("github", "config user.name github", cwd=scratch, record=False)
    box.git("github", "config user.email github@example.com", cwd=scratch, record=False)
    box.git("github", "switch main", cwd=scratch, record=False)
    box.git(
        "github", f"merge --no-ff {feature_ref} -m {message}", cwd=scratch, record=False
    )
    box.git("github", "push origin main", cwd=scratch, record=False)


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice",))
    rows: list[dict] = []

    def capture(stage: str):
        g, a = gh(box), alice(box)
        rows.append(
            {
                "stage": stage,
                "github_main_tip": g.branches.get("main", ""),
                "github_feature_tip": g.branches.get("feature", "(none)"),
                "github_branches": " ".join(sorted(g.branches)),
                "alice_main_tip": a.branches.get("main", ""),
                "alice_has_local_feature": "yes" if "feature" in a.branches else "no",
            }
        )

    # 1. A base commit, already pushed to GitHub. Alice's clone and GitHub agree.
    box.commit("alice", "app.py", "def rate():\n    return 0\n", "initial app")
    box.git("alice", "push -u origin main")
    box.snap(
        "A base commit lives on GitHub. Alice has it too.",
        note="The starting line. One commit on main, both on GitHub and in Alice's clone. No feature branch anywhere yet.",
    )
    capture("base on GitHub")

    # 2. Alice starts a feature branch and commits, on her machine only.
    box.git("alice", "switch -c feature")
    box.commit("alice", "app.py", "def rate():\n    return 42\n", "set-the-rate")
    box.snap(
        "Alice starts a feature branch, locally",
        note="feature lives only on Alice's machine. GitHub still shows just main. A branch is local until you push it.",
    )
    capture("feature committed locally")

    # 3. Alice pushes the branch. Now it exists on GitHub, and she opens the PR.
    box.git("alice", "push -u origin feature")
    box.snap(
        "git push -u origin feature",
        note="The feature branch now exists on GitHub. In the browser Alice opens a pull request, feature into main. Opening a PR runs no git command and moves no branch.",
    )
    capture("feature pushed, PR opened")
    assert "feature" in gh(box).branches, (
        "the feature branch exists on GitHub after the push"
    )

    # The state the 'no new branch' claim is measured against.
    gh_before = gh(box)
    branches_before = set(gh_before.branches)
    feature_tip_before = gh_before.branches["feature"]

    # 4. A reviewer asks for a change. Alice commits again and pushes the SAME
    #    branch. The open PR updates itself. She does not open a second PR.
    box.commit(
        "alice",
        "app.py",
        "def rate():\n    return 43  # reviewer asked: 42 was a placeholder\n",
        "address-review",
    )
    box.git("alice", "push")
    box.snap(
        "git push (again, the same branch)",
        note="No new pull request, no new branch. GitHub's feature branch moved forward and the open PR now shows two commits. Pushing to the branch updates the PR automatically.",
    )
    capture("second commit pushed")

    gh_after = gh(box)
    assert set(gh_after.branches) == branches_before, (
        "pushing again created no new branch on GitHub"
    )
    assert len(gh_after.branches) == 2, (
        "GitHub still has exactly two branches, main and feature"
    )
    assert gh_after.branches["feature"] != feature_tip_before, (
        "the feature branch tip moved forward"
    )
    feature_tip_final = gh_after.branches["feature"]

    # 5. The PR is approved. GitHub merges it into main, server-side. main moves.
    github_merges_pr(box)
    box.snap(
        "GitHub merges the pull request into main",
        note="Clicking Merge runs on GitHub, not on Alice's machine. main now contains both of feature's commits. Alice's clone has not fetched, so it still shows the old main.",
    )
    capture("PR merged on GitHub")

    merged = gh(box)
    is_anc = box.git(
        "github",
        f"merge-base --is-ancestor {feature_tip_final} main",
        record=False,
        check=False,
    )
    assert is_anc.returncode == 0, (
        "after the merge, GitHub's main contains feature's commits"
    )
    assert alice(box).branches["main"] != merged.branches["main"], (
        "Alice has not pulled, so her main still lags GitHub's"
    )

    # 6. Alice syncs the merged main and deletes her finished local branch.
    box.git("alice", "switch main")
    box.git("alice", "pull")
    box.git("alice", "branch -d feature")
    box.snap(
        "git switch main; git pull; git branch -d feature",
        note="Alice pulls the merged main and deletes her local feature branch. Its commits live on inside main. The copy on GitHub stays until someone clicks Delete branch.",
    )
    capture("Alice pulled and deleted local feature")

    final_alice = alice(box)
    assert "feature" not in final_alice.branches, "Alice's local feature branch is gone"
    anc_local = box.git(
        "alice",
        f"merge-base --is-ancestor {feature_tip_final} main",
        record=False,
        check=False,
    )
    assert anc_local.returncode == 0, "but feature's commits are in Alice's main"
    assert "feature" in gh(box).branches, (
        "the branch still exists on GitHub: deleting a local branch does not delete the remote one"
    )

    render(box, FIGURES, TABLES, mode="team")

    with (TABLES / "pr-lifecycle.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, 2 tables. All pull-request claims checked and passing."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
