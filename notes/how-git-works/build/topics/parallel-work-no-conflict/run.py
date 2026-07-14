"""Two people, disjoint files, one clean merge each: divergence is not conflict.

The everyday case of collaboration, built from real git. Alice and Bob both
clone the same repository, branch from the same commit, and edit different
files: Alice touches loader.py, Bob touches model.py. Both push their branch
and both land on main. Git combines the two lines of work on its own, with no
question asked, because the changes never overlap.

The point the figures make is that two branches growing out of one commit is
the expected shape of parallel work, not a warning sign. A conflict needs two
edits to the same lines (see 04-collaboration/same-line-conflict). Here the
edits are to different files, so there is nothing to resolve.

The story runs in team mode, one panel per repository (GitHub on top, then
Alice, then Bob):

  1. a base project on GitHub with loader.py and model.py, both cloned
  2. Alice branches fix-loader and rewrites loader.py, on her machine only
  3. Bob branches tune-model and rewrites model.py, on his machine only
  4. both push their branch: GitHub now shows the two branches forking from
     the shared base commit. This fork is divergence, the normal state.
  5. GitHub merges Alice's pull request: main had not moved, so a fast-forward
  6. GitHub merges Bob's pull request: main has moved, so a real three-way
     merge, and git combines the two disjoint files with no conflict
  7. both pull the integrated main: everyone holds both changes, main ahead

Every claim the README makes is asserted from the real repositories: the two
branches share the base as their common ancestor yet have different tips; each
merge returns exit code 0 and writes no conflict markers; the final main holds
both loader.py's and model.py's changes; and main's history grew to carry both
contributions. A wrong diagram fails the run.
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

# The base project both people clone. Two files, so two people can each own one.
BASE_LOADER = "def load(path):\n    f = open(path)\n    return f.read()\n"
BASE_MODEL = "def predict(x):\n    return 0.0\n"

# Alice hardens the loader with a context manager. She never touches model.py.
ALICE_LOADER = "def load(path):\n    with open(path) as f:\n        return f.read()\n"

# Bob makes the model return a real prediction. He never touches loader.py.
BOB_MODEL = "def predict(x):\n    return 1.0 if x > 0 else 0.0\n"

MARKERS = ("<<<<<<<", ">>>>>>>")


def gsha(box: Sandbox, ref: str) -> str:
    return box.git(
        "github", f"rev-parse {ref}", record=False, check=False
    ).stdout.strip()


def gcount(box: Sandbox, ref: str = "main") -> int:
    out = box.git("github", f"rev-list --count {ref}", record=False, check=False).stdout
    return int(out.strip() or 0)


def is_ancestor(box: Sandbox, older: str, newer: str) -> bool:
    return (
        box.git(
            "github",
            f"merge-base --is-ancestor {older} {newer}",
            record=False,
            check=False,
        ).returncode
        == 0
    )


def read_file(box: Sandbox, who: str, name: str) -> str | None:
    p = box.paths[who] / name
    return p.read_text() if p.exists() else None


def has_markers(text: str | None) -> bool:
    return text is not None and any(m in text for m in MARKERS)


def tree_has_markers(box: Sandbox, who: str) -> bool:
    """True if any tracked-looking file in someone's working tree holds markers."""
    base = box.paths[who]
    for p in base.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        try:
            if has_markers(p.read_text()):
                return True
        except (UnicodeDecodeError, OSError):
            continue
    return False


def github_merges(box: Sandbox, feature_ref: str, message: str):
    """Model GitHub clicking Merge on a pull request, server-side.

    Clicking Merge runs on GitHub, not on anyone's laptop. GitHub already holds
    every object, so it merges the branch into main itself and advances main. A
    bare repo has no working tree to merge in, so this drives a throwaway clone
    standing in for GitHub's own server-side worktree. The clone is never
    registered with the sandbox, so it is never drawn; only GitHub's main moving
    shows up in the figures. Default merge behaviour: a fast-forward when main
    has not moved, a merge commit when it has. Returns the merge process so the
    caller can read its exit code and output.
    """
    scratch = box.root / "gh-worktree"
    if not scratch.exists():
        box.git(
            "github",
            f"clone {box.paths['github']} {scratch}",
            cwd=box.root,
            record=False,
        )
        box.git("github", "config user.name github", cwd=scratch, record=False)
        box.git(
            "github", "config user.email github@example.com", cwd=scratch, record=False
        )
    box.git("github", "switch main", cwd=scratch, record=False)
    proc = box.git(
        "github",
        f"merge {feature_ref} -m {message}",
        cwd=scratch,
        record=False,
        check=False,
    )
    box.git("github", "push origin main", cwd=scratch, record=False)
    return proc, scratch


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice", "bob"))
    rows: list[dict] = []

    def capture(stage: str):
        g, a, b = box.read("github"), box.read("alice"), box.read("bob")
        rows.append(
            {
                "stage": stage,
                "github_main": g.branches.get("main", ""),
                "github_branches": " ".join(sorted(g.branches)),
                "alice_head": a.head or "(detached)",
                "alice_branches": " ".join(sorted(a.branches)),
                "bob_head": b.head or "(detached)",
                "bob_branches": " ".join(sorted(b.branches)),
            }
        )

    # ---- 1. a base project on GitHub, both people cloned ----------------
    box.write("alice", "loader.py", BASE_LOADER, record=False)
    box.write("alice", "model.py", BASE_MODEL, record=False)
    box.git("alice", "add loader.py model.py")
    box.git("alice", "commit -m project-skeleton")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    s1 = box.snap(
        "A shared project, cloned by both people",
        note="One base commit holds loader.py and model.py. Alice pushed it, Bob pulled it. GitHub main, both local mains and both origin/main caches all point at the same commit. This is the common ancestor everything below grows from.",
    )
    base_full = gsha(box, "main")
    count_start = gcount(box, "main")
    capture("base cloned by both")

    # ---- 2. Alice branches and rewrites loader.py, locally --------------
    box.git("alice", "switch -c fix-loader")
    box.commit("alice", "loader.py", ALICE_LOADER, "use-a-context-manager")
    s2 = box.snap(
        "Alice branches and edits loader.py",
        note="git switch -c fix-loader made a new branch off the base and moved HEAD onto it. Alice rewrote loader.py and committed. The branch and the commit live only on Alice's disk. GitHub and Bob have heard nothing.",
    )
    capture("alice committed locally")

    # ---- 3. Bob branches and rewrites model.py, locally -----------------
    box.git("bob", "switch -c tune-model")
    box.commit("bob", "model.py", BOB_MODEL, "return-a-real-prediction")
    s3 = box.snap(
        "Bob branches and edits model.py",
        note="Bob did the same thing to a different file. Both branches now start from the same base commit and each adds one commit, to loader.py and to model.py respectively. Their histories have diverged, and nothing is wrong: this is what parallel work looks like.",
    )
    capture("bob committed locally")

    gh3 = box.read("github")
    assert "fix-loader" not in gh3.branches and "tune-model" not in gh3.branches, (
        "a branch is local until pushed: GitHub knows neither branch yet"
    )
    assert set(gh3.branches) == {"main"}, "GitHub still has only main at this point"

    # ---- 4. both push their branch: GitHub sees the fork ----------------
    box.git("alice", "push -u origin fix-loader")
    box.git("bob", "push -u origin tune-model")
    s4 = box.snap(
        "Both push: two branches, one ancestor",
        note="git push -u origin <branch> put each branch on GitHub. The GitHub panel now shows fix-loader and tune-model both growing out of the base commit. Two branches from a common ancestor is divergence. It is the normal state of collaboration, not a conflict.",
    )
    loader_commit = gsha(box, "fix-loader")
    model_commit = gsha(box, "tune-model")
    capture("both branches pushed")

    gh4 = box.read("github")
    assert "fix-loader" in gh4.branches and "tune-model" in gh4.branches, (
        "both branches now exist on GitHub"
    )

    # ---- 5. GitHub merges Alice's PR: a fast-forward --------------------
    proc1, _ = github_merges(
        box, "origin/fix-loader", "Merge-pull-request-1-fix-loader"
    )
    s5 = box.snap(
        "GitHub merges Alice's pull request",
        note="main had not moved since Alice branched, so this is a fast-forward: git slides main up onto Alice's commit and writes no merge commit. GitHub main now carries the loader.py change. Alice and Bob have not pulled, so their mains still sit on the base.",
    )
    tip_pr1 = gsha(box, "main")
    count_pr1 = gcount(box, "main")
    capture("alice PR merged")

    # ---- 6. GitHub merges Bob's PR: a real three-way merge -------------
    proc2, scratch = github_merges(
        box, "origin/tune-model", "Merge-pull-request-2-tune-model"
    )
    s6 = box.snap(
        "GitHub merges Bob's pull request",
        note="main has moved since Bob branched, so his branch and main have diverged. This is a genuine three-way merge. Git combines loader.py and model.py on its own, with no conflict, because the two changes touch different files. It records the join as one merge commit with two parents.",
    )
    tip_pr2 = gsha(box, "main")
    count_pr2 = gcount(box, "main")
    capture("bob PR merged")

    # ---- 7. both pull the integrated main -------------------------------
    box.git("alice", "switch main")
    box.git("alice", "pull")
    box.git("bob", "switch main")
    box.git("bob", "pull")
    s7 = box.snap(
        "Both pull: everyone holds both changes",
        note="git switch main; git pull brought the integrated main down to each person. Alice's loader change and Bob's model change now sit together on every copy of main, which is ahead of the base commit all three started from.",
    )
    capture("both pulled the integrated main")

    render(box, FIGURES, TABLES, mode="team")

    # ================= ASSERTIONS: every README claim ===================

    # (A) The two branches share the base as their common ancestor, and have
    #     genuinely diverged: different tips, and neither reaches the other.
    mb = box.git(
        "github", "merge-base fix-loader tune-model", record=False, check=False
    ).stdout.strip()
    assert mb == base_full, (
        "the two branches share exactly the base commit as their common ancestor"
    )
    assert loader_commit != model_commit, "the two branch tips are different commits"
    assert not is_ancestor(box, "fix-loader", "tune-model"), (
        "fix-loader is not an ancestor of tune-model"
    )
    assert not is_ancestor(box, "tune-model", "fix-loader"), (
        "tune-model is not an ancestor of fix-loader"
    )
    assert gcount(box, "fix-loader") == 2 and gcount(box, "tune-model") == 2, (
        "each branch is the base plus exactly one commit"
    )

    # (B) Alice's merge is a clean fast-forward: exit 0, no marker, no commit.
    assert proc1.returncode == 0, "Alice's merge succeeded"
    assert "Fast-forward" in proc1.stdout, "Alice's merge was a fast-forward"
    assert count_pr1 == count_start + 1, (
        "the fast-forward added one commit to main, Alice's, and no merge commit"
    )

    # (C) Bob's merge is a clean three-way merge: exit 0, no marker, one merge
    #     commit with two parents. This is the heart of the tutorial.
    assert proc2.returncode == 0, "Bob's merge succeeded with no conflict"
    assert "Fast-forward" not in proc2.stdout, (
        "Bob's merge was a real merge, not a fast-forward"
    )
    gh6 = box.read("github")
    merge_commit = gh6.branches["main"]
    assert len(gh6.commits[merge_commit].parents) == 2, (
        "the merge commit ties two lines of history together with two parents"
    )

    # (D) No conflict markers anywhere: not in the merged file, not in any tree.
    assert not tree_has_markers(box, "alice"), "Alice's working tree holds no markers"
    assert not tree_has_markers(box, "bob"), "Bob's working tree holds no markers"

    # (E) Final main contains BOTH loader.py's and model.py's changes.
    alice_loader = read_file(box, "alice", "loader.py")
    alice_model = read_file(box, "alice", "model.py")
    bob_loader = read_file(box, "bob", "loader.py")
    bob_model = read_file(box, "bob", "model.py")
    assert alice_loader == ALICE_LOADER == bob_loader, (
        "the final loader.py is Alice's version, on both machines"
    )
    assert alice_model == BOB_MODEL == bob_model, (
        "the final model.py is Bob's version, on both machines"
    )
    assert "with open(path)" in alice_loader, "Alice's loader change is present"
    assert "1.0 if x > 0" in bob_model, "Bob's model change is present"
    assert is_ancestor(box, loader_commit, "main"), (
        "Alice's commit is in the history of the integrated main"
    )
    assert is_ancestor(box, model_commit, "main"), (
        "Bob's commit is in the history of the integrated main"
    )

    # (F) main's history grew to carry both contributions.
    assert count_pr2 > count_start, "main is ahead of where it started"
    assert count_pr2 == 4, (
        "main went from the base to base plus loader plus model plus the merge commit"
    )
    assert count_pr2 - count_start == 3, (
        "the growth is the two contributions and the one merge commit that joined them"
    )

    # (G) Everyone converged: all three mains equal, and ahead of the base.
    a7, b7, g7 = box.read("alice"), box.read("bob"), box.read("github")
    assert a7.branches["main"] == b7.branches["main"] == g7.branches["main"], (
        "after the pulls, all three mains point at the same integrated commit"
    )
    assert a7.branches["main"] != base_full, "and that commit is ahead of the base"

    # ================= TABLES the README quotes =========================

    with (TABLES / "branch-divergence.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "base_commit",
                "alice_branch_tip",
                "bob_branch_tip",
                "common_ancestor",
                "tips_differ",
                "neither_is_ancestor",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "base_commit": base_full[:7],
                "alice_branch_tip": loader_commit[:7],
                "bob_branch_tip": model_commit[:7],
                "common_ancestor": mb[:7],
                "tips_differ": "yes",
                "neither_is_ancestor": "yes",
            }
        )

    strat1 = "fast-forward" if "Fast-forward" in proc1.stdout else "three-way merge"
    strat2 = "fast-forward" if "Fast-forward" in proc2.stdout else "three-way merge"
    with (TABLES / "merge-outcomes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pull_request",
                "branch_merged",
                "exit_code",
                "strategy",
                "conflict_markers",
                "merge_commit_created",
                "main_tip_after",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pull_request": "#1",
                "branch_merged": "fix-loader",
                "exit_code": proc1.returncode,
                "strategy": strat1,
                "conflict_markers": 0,
                "merge_commit_created": "no",
                "main_tip_after": tip_pr1[:7],
            }
        )
        writer.writerow(
            {
                "pull_request": "#2",
                "branch_merged": "tune-model",
                "exit_code": proc2.returncode,
                "strategy": strat2,
                "conflict_markers": 0,
                "merge_commit_created": "yes",
                "main_tip_after": tip_pr2[:7],
            }
        )

    def yn(flag: bool) -> str:
        return "yes" if flag else "no"

    moments = [
        ("start (base)", base_full, count_start),
        ("after PR #1 merges", tip_pr1, count_pr1),
        ("after PR #2 merges", tip_pr2, count_pr2),
    ]
    with (TABLES / "main-history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "moment",
                "commits_reachable_from_main",
                "has_loader_change",
                "has_model_change",
            ],
        )
        writer.writeheader()
        for label, tip, count in moments:
            writer.writerow(
                {
                    "moment": label,
                    "commits_reachable_from_main": count,
                    "has_loader_change": yn(is_ancestor(box, loader_commit, tip)),
                    "has_model_change": yn(is_ancestor(box, model_commit, tip)),
                }
            )

    with (TABLES / "who-has-what.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 team-mode figures, got {figs}"
    print(
        f"{len(figs)} figures, 4 tables. Every parallel-work claim checked and passing."
    )
    print("figures:", figs)
    box.cleanup()


if __name__ == "__main__":
    main()
