"""GitHub's three PR-merge buttons, and the history each one leaves.

When you merge a pull request on GitHub you pick one of three buttons. The
difference between them is entirely the shape of history they write. This run
takes ONE feature branch with three messy commits (wip, fix typo, actually fix
it), merges it three separate ways in three sandboxes, and reads the resulting
history back out of real git with plumbing.

  Create a merge commit   git merge --no-ff feature
        keeps all three commits AND adds a merge commit with two parents.
  Squash and merge        git merge --squash feature ; git commit
        collapses the whole branch into ONE new commit on main.
  Rebase and merge        git rebase main ; git merge --ff-only feature
        replays the three commits onto main as new commits, linear, no merge.

The branch is content-identical in all three sandboxes, and main advances by the
same three commits in each (a real PR sits open while other PRs land), so the
one thing that changes is the button. Every claim the README makes is asserted
at the bottom. The strongest one: all three buttons end at the byte-identical
file tree, so the choice is about history, never about the files.
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

# The final login.py, stripped, as every button must leave it.
FINAL_LOGIN = "def login(user):\n    return True"


# ---- read-only helpers, all pulling from real git ----------------------


def out(box: Sandbox, args: str) -> str:
    return box.git("alice", args, record=False, check=False).stdout.strip()


def rev(box: Sandbox, r: str) -> str:
    return out(box, f"rev-parse {r}")


def rev_list(box: Sandbox, args: str) -> list[str]:
    s = out(box, f"rev-list {args}")
    return s.split() if s else []


def parents_of(box: Sandbox, r: str) -> list[str]:
    return out(box, f"rev-list --parents -1 {r}").split()[1:]


def count(box: Sandbox, r: str = "main") -> int:
    return int(out(box, f"rev-list --count {r}"))


def subject(box: Sandbox, r: str) -> str:
    return out(box, f"log -1 --format=%s {r}")


def tree(box: Sandbox, r: str = "main") -> str:
    return out(box, f"rev-parse {r}^{{tree}}")


def log_graph(box: Sandbox) -> list[str]:
    return [
        line for line in out(box, "log --graph --oneline main").splitlines() if line
    ]


# ---- the shared starting point -----------------------------------------


def build_feature(box: Sandbox) -> list[str]:
    """Two commits on main, then a feature branch with three messy commits.

    Returns the three feature commit ids (newest first). Content and dates are
    fixed, so these ids are byte-identical in every sandbox, which is what makes
    the three merges genuinely the same experiment with one variable changed.
    """
    box.commit("alice", "README.md", "# Project\n", "init project")
    box.commit("alice", "app.py", "def greet():\n    return 'hi'\n", "add greet")
    box.git("alice", "switch -c feature")
    box.commit("alice", "login.py", "def login(usr):\n    retrun Tru\n", "wip")
    box.commit("alice", "login.py", "def login(usr):\n    return True\n", "fix typo")
    box.commit(
        "alice", "login.py", "def login(user):\n    return True\n", "actually fix it"
    )
    box.git("alice", "switch main")
    return rev_list(box, "feature ^main")


def advance_main(box: Sandbox):
    """Main moves on by three commits while the PR is open, touching a
    different file so nothing conflicts."""
    box.commit("alice", "config.py", "DEBUG = False\n", "add config")
    box.commit("alice", "config.py", "DEBUG = False\nPORT = 8080\n", "add port")
    box.commit(
        "alice",
        "config.py",
        "DEBUG = False\nPORT = 8080\nHOST = localhost\n",
        "add host",
    )


def render(box: Sandbox, index: int, name: str):
    snaps = list(box.snapshots)
    xs, ys = layout(snaps)
    draw(snaps[index], xs, ys, FIGURES / name, mode="solo")


def main():
    clear(FIGURES, TABLES)

    # ================================================================
    # 1. Create a merge commit  ->  git merge --no-ff feature
    # ================================================================
    A = Sandbox(people=("alice",))
    orig = build_feature(A)
    assert len(orig) == 3, "the feature branch has three commits"
    advance_main(A)
    A.snap(
        "your 3 commits on feature, and main has moved on too",
        note="Three messy commits on feature. While the PR was open, three other PRs landed on main.",
    )
    before_a = count(A, "main")
    merge = A.git(
        "alice", "merge --no-ff feature -m Merge-pull-request-#7", check=False
    )
    assert merge.returncode == 0, "the merge-commit merge succeeds"
    A.snap(
        "Create a merge commit: git merge --no-ff feature",
        note="Every commit of the branch is kept, and a merge commit with two parents joins the two lines of history.",
    )
    a_tip = rev(A, "main")
    a_parents = parents_of(A, "main")
    a_merges = rev_list(A, "--min-parents=2 main")
    reach_a = set(rev_list(A, "main"))
    new_a = count(A, "main") - before_a
    tree_a = tree(A)
    login_a = out(A, "show main:login.py")
    graph_a = log_graph(A)

    # ================================================================
    # 2. Squash and merge  ->  git merge --squash feature ; git commit
    # ================================================================
    B = Sandbox(people=("alice",))
    orig_b = build_feature(B)
    assert orig_b == orig, "identical content and dates give identical commit ids"
    advance_main(B)
    B.snap("the identical starting point, rebuilt")
    before_b = count(B, "main")
    sq = B.git("alice", "merge --squash feature", check=False)
    assert sq.returncode == 0, "the squash stages the combined diff"
    commit = B.git("alice", "commit -m Add-login-feature-#7", check=False)
    assert commit.returncode == 0, "and one commit records it on main"
    B.snap(
        "Squash and merge: git merge --squash feature then git commit",
        note="The three commits collapse into ONE new commit on main. The originals stay on feature, outside main's history.",
    )
    b_tip = rev(B, "main")
    b_parents = parents_of(B, "main")
    b_merges = rev_list(B, "--min-parents=2 main")
    first_parent_b = set(rev_list(B, "--first-parent main"))
    reach_b = set(rev_list(B, "main"))
    new_b = count(B, "main") - before_b
    tree_b = tree(B)
    login_b = out(B, "show main:login.py")
    graph_b = log_graph(B)

    # ================================================================
    # 3. Rebase and merge  ->  git rebase main ; git merge --ff-only feature
    # ================================================================
    C = Sandbox(people=("alice",))
    orig_c = build_feature(C)
    assert orig_c == orig, "the same experiment a third time"
    advance_main(C)
    C.snap("the identical starting point, once more")
    before_c = count(C, "main")
    C.git("alice", "switch feature")
    rb = C.git("alice", "rebase main", check=False)
    assert rb.returncode == 0, "the three commits replay cleanly onto main"
    C.git("alice", "switch main")
    ff = C.git("alice", "merge --ff-only feature", check=False)
    assert ff.returncode == 0, "and main fast-forwards onto the replayed commits"
    C.snap(
        "Rebase and merge: git rebase main then a fast-forward",
        note="The three commits are replayed onto main as new commits. Linear history, new commit ids, no merge commit.",
    )
    c_tip = rev(C, "main")
    c_parents = parents_of(C, "main")
    c_merges = rev_list(C, "--min-parents=2 main")
    reach_c = set(rev_list(C, "main"))
    new_c = count(C, "main") - before_c
    replayed = rev_list(C, "--first-parent -3 main")  # top three = the replayed copies
    tree_c = tree(C)
    login_c = out(C, "show main:login.py")
    graph_c = log_graph(C)

    # ---- figures ----------------------------------------------------
    render(A, 0, "setup.png")
    render(A, 1, "merge-commit.png")
    render(B, 1, "squash.png")
    render(C, 1, "rebase.png")

    # ================================================================
    # THE ASSERTIONS: every claim the README makes, checked against git.
    # ================================================================

    # -- Create a merge commit ---------------------------------------
    assert len(a_parents) == 2, "the merge commit has two parents"
    assert a_merges == [a_tip], (
        "exactly one two-parent commit exists, and it is main's tip"
    )
    assert set(orig).issubset(reach_a), (
        "all three original feature commit ids are kept in main's history"
    )
    assert new_a == 4, (
        "the merge added four commits to main: the three feature commits plus the merge commit"
    )
    assert any(line.strip() == "|\\" for line in graph_a), (
        "the merge opens a visible fork in git's own graph"
    )
    assert any(line.strip() == "|/" for line in graph_a), (
        "and closes it at the commit the branch started from"
    )

    # -- Squash and merge --------------------------------------------
    assert new_b == 1, "the squash added exactly ONE new commit to main"
    assert len(b_parents) == 1, "and that commit has a single parent"
    assert b_merges == [], "main's history is linear: no two-parent commit"
    assert not (set(orig) & first_parent_b), (
        "NONE of the three original feature ids appear in main's first-parent history"
    )
    assert not (set(orig) & reach_b), (
        "in fact none of the three are reachable from main at all"
    )
    assert all(line.startswith("* ") for line in graph_b), (
        "every line of main's graph is a plain commit: no fork, no bubble"
    )

    # -- Rebase and merge --------------------------------------------
    assert c_merges == [], "the rebased history is linear: no two-parent commit"
    assert len(c_parents) == 1, "main's tip has a single parent"
    assert new_c == 3, "the rebase added three commits to main"
    assert len(replayed) == 3, "the three replayed commits sit on top of main"
    assert set(replayed).isdisjoint(set(orig)), (
        "the replayed commit ids DIFFER from the three originals: rebase rewrote them"
    )
    assert not (set(orig) & reach_c), "and not one original id survives on main"
    orig_subjects = sorted(subject(A, i) for i in orig)
    replayed_subjects = sorted(subject(C, i) for i in replayed)
    assert replayed_subjects == orig_subjects, (
        "same three messages: identical work, only the ids moved"
    )
    assert all(line.startswith("* ") for line in graph_c), "linear graph, no fork"

    # -- all three end at the identical files ------------------------
    assert login_a == login_b == login_c == FINAL_LOGIN, (
        "the final login.py is identical under all three buttons"
    )
    assert tree_a == tree_b == tree_c, (
        "the tree of main's tip is byte-identical across all three merges"
    )
    for label, bx in (("merge", A), ("squash", B), ("rebase", C)):
        assert out(bx, "show main:config.py").startswith("DEBUG = False"), (
            f"main's own config work is preserved under {label}"
        )

    # ---- the tables the README quotes from --------------------------
    styles = [
        {
            "button": "Create a merge commit",
            "git_equivalent": "git merge --no-ff feature",
            "new_commits_on_main": new_a,
            "main_tip_parents": len(a_parents),
            "merge_commit_created": "yes",
            "original_feature_ids_kept": len(set(orig) & reach_a),
            "linear_history": "no",
            "final_tree": tree_a[:7],
        },
        {
            "button": "Squash and merge",
            "git_equivalent": "git merge --squash feature; git commit",
            "new_commits_on_main": new_b,
            "main_tip_parents": len(b_parents),
            "merge_commit_created": "no",
            "original_feature_ids_kept": len(set(orig) & reach_b),
            "linear_history": "yes",
            "final_tree": tree_b[:7],
        },
        {
            "button": "Rebase and merge",
            "git_equivalent": "git rebase main; git merge --ff-only",
            "new_commits_on_main": new_c,
            "main_tip_parents": len(c_parents),
            "merge_commit_created": "no",
            "original_feature_ids_kept": len(set(orig) & reach_c),
            "linear_history": "yes",
            "final_tree": tree_c[:7],
        },
    ]
    with (TABLES / "merge-styles.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(styles[0]))
        writer.writeheader()
        writer.writerows(styles)

    # What became of each of the three commits, under each button.
    new_by_subject = {subject(C, i): i for i in replayed}
    fate = []
    for oid in reversed(orig):  # oldest first: wip, fix typo, actually fix it
        subj = subject(A, oid)
        fate.append(
            {
                "subject": subj,
                "original_id": oid[:7],
                "after_merge_commit": oid[:7],
                "after_squash": "absent",
                "after_rebase": new_by_subject[subj][:7],
            }
        )
    with (TABLES / "commit-fate.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fate[0]))
        writer.writeheader()
        writer.writerows(fate)

    # git's own drawing of the resulting history, per button.
    graphs = []
    for button, lines in (
        ("Create a merge commit", graph_a),
        ("Squash and merge", graph_b),
        ("Rebase and merge", graph_c),
    ):
        graphs += [{"button": button, "line": line} for line in lines]
    with (TABLES / "log-graph.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["button", "line"])
        writer.writeheader()
        writer.writerows(graphs)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert figs == ["merge-commit.png", "rebase.png", "setup.png", "squash.png"], (
        f"expected four figures, got {figs}"
    )

    A.cleanup()
    B.cleanup()
    C.cleanup()
    print(
        f"{len(figs)} figures, 3 tables. All three merge buttons checked, every claim passing."
    )


if __name__ == "__main__":
    main()
