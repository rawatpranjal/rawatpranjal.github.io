"""Your branch goes stale while you work, and two ways to catch it up.

Alice branches off main at C1 and writes two commits. While she works, Bob's
work lands C2 on main, so GitHub's main moves ahead and her branch is now behind:
it still hangs off C1. There are two ways to catch up, and this builds both as
real repositories in team mode:

  rebase   git fetch origin, then git rebase origin/main. Her two commits are
           replayed on top of the new main as new commits with new hashes. The
           history stays a straight line. Because the replay rewrote commits she
           had already pushed, she finishes with git push --force-with-lease.
  merge    git merge origin/main. A single two-parent merge commit joins the two
           lines, keeping her original commit ids untouched.

A third, throwaway repository shows why the safe force is --force-with-lease and
not plain --force: when someone else has pushed to the branch since you last
looked, the lease refuses and protects their commit, while plain --force would
overwrite it. Every claim the README makes is asserted at the bottom, read out
of the real repositories with plumbing. A wrong picture fails the run.
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

from lib.gitviz import Sandbox, clear, draw, layout, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# C1 is the commit everyone shares. C2 is the commit that lands on main while
# Alice works. They touch app.py; Alice's branch touches search.py, so the two
# lines are disjoint and the catch-up never hits a conflict.
APP_C1 = "def run():\n    return 'ok'\n"
APP_C2 = "def run():\n    setup()\n    return 'ok'\n"
SEARCH_1 = "def search(q):\n    return []\n"
SEARCH_2 = "def search(q):\n    return db.query(q)\n"

# The shared-branch race that the force-with-lease demo needs.
BASE = "project base\n"
SHARED_1 = "shared v1\n"
SHARED_1B = "shared v1, reworked\n"  # Alice's rewrite of S1
SHARED_2 = "shared v1\nbob added a line\n"  # Bob's honest new commit on top of S1


# ---- reading the truth back out ----------------------------------------


def _run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _full(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", rev])


def _short(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", "--short", rev])


def _is_ancestor(repo: Path, a: str, b: str) -> bool:
    """True when commit a is an ancestor of commit b."""
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", a, b], cwd=repo, capture_output=True
        ).returncode
        == 0
    )


def _merge_base(repo: Path, a: str, b: str) -> str:
    return _run(repo, ["merge-base", a, b])


def _commits_in(repo: Path, rev_range: str) -> list[tuple[str, str]]:
    """(short sha, subject) for a revision range, oldest first."""
    out = _run(repo, ["log", "--reverse", "--pretty=%h\x1f%s", rev_range])
    return [tuple(line.split("\x1f")) for line in out.splitlines() if line]


def _tree_of(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", "--short", f"{rev}^{{tree}}"])


def _parent_counts(repo: Path, rev: str) -> list[int]:
    """How many parents each commit reachable from rev has."""
    out = _run(repo, ["rev-list", "--parents", rev])
    return [len(line.split()) - 1 for line in out.splitlines() if line]


def _n_parents(repo: Path, rev: str) -> int:
    """Parents of one commit. Two means it is a merge commit."""
    return len(_run(repo, ["rev-list", "--parents", "-1", rev]).split()) - 1


# ---- the shared situation ----------------------------------------------


def build_stale_feature(box: Sandbox, snapshots: bool = True) -> dict:
    """Alice branches at C1, commits twice and pushes; Bob lands C2 on main.

    Leaves Alice on add-search, having fetched, so origin/main is C2 but her
    branch still hangs off C1. This is the stale branch every catch-up starts
    from. Returns the branch tip before and after the fetch, to prove a plain
    fetch never moves the local branch.
    """
    # C1: the commit everyone shares.
    box.commit("alice", "app.py", APP_C1, "add app entry point")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")

    # Alice branches at C1, does two commits, and pushes her branch to GitHub.
    box.git("alice", "switch -c add-search")
    box.commit("alice", "search.py", SEARCH_1, "search stub")
    box.commit("alice", "search.py", SEARCH_2, "search hits the database")
    box.git("alice", "push -u origin add-search")
    if snapshots:
        box.snap(
            "Alice branches at C1 and pushes add-search",
            note="Alice branched off main at C1 and wrote two commits on add-search, then pushed the branch. GitHub now holds main at C1 and add-search at her second commit.",
        )

    # Bob's work lands C2 on main while Alice is busy.
    box.commit("bob", "app.py", APP_C2, "app calls setup on boot")
    box.git("bob", "push")
    if snapshots:
        box.snap(
            "Bob's work lands C2 on main",
            note="Bob pushed C2 to main, so GitHub main moved ahead. Alice ran nothing, so her origin/main is still a stale cache pointing at C1, and her branch still hangs off C1.",
        )

    # Alice fetches: her caches learn about C2, her branch does not move.
    tip_before = box.read("alice").branches["add-search"]
    fetch = box.git("alice", "fetch origin", check=False)
    assert fetch.returncode == 0, f"the fetch must succeed:\n{fetch.stderr}"
    tip_after = box.read("alice").branches["add-search"]
    if snapshots:
        box.snap(
            "Alice fetches: her cache sees C2, add-search stays on C1",
            note="git fetch moved Alice's origin/main cache onto C2, but it did not touch her add-search branch. Her branch is now behind: it hangs off C1 while GitHub main is at C2.",
        )
    return {"tip_before_fetch": tip_before, "tip_after_fetch": tip_after}


def main():
    clear(FIGURES, TABLES)

    # ================= 1. the stale branch, caught up with a rebase =========
    box = Sandbox(people=("alice", "bob"), show_unreachable=True)
    info = build_stale_feature(box, snapshots=True)
    alice = box.paths["alice"]

    # The branch-is-behind facts, measured after the fetch and before the rebase.
    branch_point = _merge_base(alice, "add-search", "origin/main")
    gh_main = _full(alice, "origin/main")
    branch_tip = _full(alice, "add-search")
    base_behind_main = _is_ancestor(alice, branch_point, gh_main)
    base_equals_main = branch_point == gh_main
    main_in_branch = _is_ancestor(alice, gh_main, branch_tip)  # False: diverged

    # The two commits on the branch, before the replay.
    originals = _commits_in(alice, "origin/main..add-search")
    # C1 read as the commit two below the branch tip, while the tip is still the
    # original one (the rebase moves the tip, so this must be read now).
    c1_two_below_tip = _full(alice, "add-search~2")

    reb = box.git("alice", "rebase origin/main", check=False)
    assert reb.returncode == 0, f"the rebase must succeed:\n{reb.stderr}"
    box.snap(
        "git rebase origin/main: her two commits replay onto C2",
        note="Rebase picked up Alice's two commits and set them down again on top of C2, as new commits with new hashes. The originals sit abandoned on C1, reachable only through the reflog. GitHub's add-search still holds the old commits.",
    )

    # The two commits after the replay.
    rebased = _commits_in(alice, "origin/main..add-search")
    branch_tip_rebased = _full(alice, "add-search")
    c2_in_branch_after = _is_ancestor(alice, gh_main, branch_tip_rebased)  # True now

    # Because the replay rewrote commits she had already pushed, her local
    # add-search and GitHub's add-search now disagree. Nobody else touched the
    # branch, so the lease is satisfied and the force push goes through.
    gh_addsearch_before_force = _short(box.paths["github"], "add-search")
    fpush = box.git("alice", "push --force-with-lease", check=False)
    assert fpush.returncode == 0, (
        f"the lease is satisfied on her own branch, so it pushes:\n{fpush.stderr}"
    )
    gh_addsearch_after_force = _short(box.paths["github"], "add-search")
    box.snap(
        "git push --force-with-lease: GitHub's add-search catches up",
        note="The force-with-lease push replaced GitHub's add-search with the rebased commits. It was safe because no one else had pushed to add-search since Alice last fetched, so the lease held.",
    )

    render(box, FIGURES, TABLES, mode="team")

    # ================= 2. the same stale branch, caught up with a merge ======
    mbox = Sandbox(people=("alice", "bob"))
    build_stale_feature(mbox, snapshots=False)
    m_alice = mbox.paths["alice"]
    mbox.snap("the same stale branch")  # seed only, not drawn: fixes what is new

    m_originals = _commits_in(m_alice, "origin/main..add-search")
    mmerge = mbox.git("alice", "merge origin/main --no-edit", check=False)
    assert mmerge.returncode == 0, f"the merge must succeed:\n{mmerge.stderr}"
    msnap = mbox.snap(
        "git merge origin/main: a two-parent commit joins the two lines",
        note="Instead of replaying, a merge added one new commit with two parents, joining Alice's line and main's line. Her two original commits keep the hashes they always had.",
    )
    mxs, mys = layout(mbox.snapshots)
    draw(msnap, mxs, mys, FIGURES / "merge-route.png", mode="team")

    merge_tip_parents = _n_parents(m_alice, "add-search")
    m_kept = {sha for sha, _ in _commits_in(m_alice, "origin/main..add-search")}

    # ================= 3. why --force-with-lease, not --force ===============
    lease = Sandbox(people=("alice", "bob"))
    lease.commit("alice", "base.py", BASE, "project base")
    lease.git("alice", "push -u origin main")
    lease.git("bob", "pull")
    lease.git("alice", "switch -c shared")
    lease.commit("alice", "shared.py", SHARED_1, "shared v1")
    lease.git("alice", "push -u origin shared")
    lease.git("bob", "fetch origin")
    lease.git("bob", "switch shared")
    lease.snap("both share the branch at S1")  # seed for the highlight

    # Bob pushes an honest new commit to the shared branch.
    lease.commit("bob", "shared.py", SHARED_2, "bob extends shared")
    lease.git("bob", "push")
    l_gh = lease.paths["github"]
    s2_full = _full(l_gh, "shared")
    gh_shared_before = _short(l_gh, "shared")

    # Alice, who has not fetched, rewrites her own S1 (a rebase-like history edit).
    lease.write("alice", "shared.py", SHARED_1B)
    lease.git("alice", "add shared.py")
    lease.git("alice", "commit --amend -m shared-v1-reworked")

    # The lease refuses: GitHub's shared moved since Alice last looked at it.
    lease_push = lease.git("alice", "push --force-with-lease", check=False)
    lease_refused = lease_push.returncode != 0
    gh_shared_after_lease = _short(l_gh, "shared")
    s2_survives_lease = _is_ancestor(l_gh, s2_full, "shared")
    lease.snap(
        "git push --force-with-lease is refused",
        note="Alice rewrote S1 into S1', not knowing Bob had pushed S2. The lease compares GitHub's shared against the value Alice last saw and refuses, because they differ. Bob's S2 is protected.",
    )
    lxs, lys = layout(lease.snapshots)
    draw(lease.snapshots[-1], lxs, lys, FIGURES / "lease-refused.png", mode="team")

    # Plain --force does not check, and would overwrite Bob's commit.
    force_push = lease.git("alice", "push --force", check=False)
    force_accepted = force_push.returncode == 0
    gh_shared_after_force = _short(l_gh, "shared")
    s2_survives_force = _is_ancestor(l_gh, s2_full, "shared")

    # ================= tables the README quotes =============================
    with (TABLES / "branch-behind.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fact", "value"])
        writer.writerow(
            [
                "branch point (merge-base of add-search and origin/main)",
                _short(alice, branch_point),
            ]
        )
        writer.writerow(["GitHub main tip (origin/main)", _short(alice, gh_main)])
        writer.writerow(["add-search tip", _short(alice, branch_tip)])
        writer.writerow(
            [
                "is the branch point an ancestor of GitHub main",
                "yes" if base_behind_main else "no",
            ]
        )
        writer.writerow(
            [
                "is the branch point equal to GitHub main",
                "yes" if base_equals_main else "no",
            ]
        )
        writer.writerow(
            [
                "does add-search already contain GitHub's C2",
                "yes" if main_in_branch else "no",
            ]
        )
        writer.writerow(
            [
                "so the branch is behind and has diverged",
                "yes"
                if (base_behind_main and not base_equals_main and not main_in_branch)
                else "no",
            ]
        )

    with (TABLES / "fetch-moves-nothing.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "before git fetch", "after git fetch", "moved"])
        writer.writerow(
            [
                "add-search (Alice's branch)",
                info["tip_before_fetch"],
                info["tip_after_fetch"],
                "yes" if info["tip_before_fetch"] != info["tip_after_fetch"] else "no",
            ]
        )

    with (TABLES / "rebase-vs-merge.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question", "git rebase origin/main", "git merge origin/main"])
        writer.writerow(
            [
                "commits reachable from add-search",
                len(_parent_counts(alice, "add-search")),
                len(_parent_counts(m_alice, "add-search")),
            ]
        )
        writer.writerow(
            [
                "commits with two parents",
                sum(1 for n in _parent_counts(alice, "add-search") if n == 2),
                sum(1 for n in _parent_counts(m_alice, "add-search") if n == 2),
            ]
        )
        writer.writerow(
            [
                "your two original commit ids still on the branch",
                sum(1 for sha, _ in originals if sha in {s for s, _ in rebased}),
                sum(1 for sha, _ in m_originals if sha in m_kept),
            ]
        )
        writer.writerow(
            [
                "new commit ids created",
                len(rebased),
                1,
            ]
        )
        writer.writerow(
            [
                "history is a straight line",
                "yes" if max(_parent_counts(alice, "add-search")) == 1 else "no",
                "yes" if max(_parent_counts(m_alice, "add-search")) == 1 else "no",
            ]
        )
        writer.writerow(
            [
                "tree of the branch tip (the resulting files)",
                _tree_of(alice, "add-search"),
                _tree_of(m_alice, "add-search"),
            ]
        )

    with (TABLES / "rebase-hashes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "original commit", "rebased commit"])
        for (o_sha, subject), (n_sha, _) in zip(originals, rebased):
            writer.writerow([subject, o_sha, n_sha])

    with (TABLES / "force-with-lease.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "command",
                "outcome",
                "GitHub shared before",
                "GitHub shared after",
                "Bob's commit still on GitHub",
            ]
        )
        writer.writerow(
            [
                "git push --force-with-lease",
                "refused" if lease_refused else "accepted",
                gh_shared_before,
                gh_shared_after_lease,
                "yes" if s2_survives_lease else "no",
            ]
        )
        writer.writerow(
            [
                "git push --force",
                "accepted" if force_accepted else "refused",
                gh_shared_after_lease,
                gh_shared_after_force,
                "yes" if s2_survives_force else "no",
            ]
        )

    # ================= the oracle: every claim the README makes =============

    # (1) Alice's branch is behind: its base is an ancestor of GitHub main,
    #     strictly, and the branch has not caught up on its own.
    assert base_behind_main, (
        "the branch point is an ancestor of GitHub main, so the branch is behind"
    )
    assert not base_equals_main, (
        "the branch point is NOT GitHub main, main moved past it"
    )
    assert not main_in_branch, (
        "and GitHub's C2 is not on the branch yet, so the two lines have diverged"
    )
    assert branch_point == c1_two_below_tip, (
        "the branch point is exactly C1, the commit two commits below the original branch tip"
    )

    # (2) A plain fetch does not move the local branch.
    assert info["tip_before_fetch"] == info["tip_after_fetch"], (
        "git fetch left Alice's add-search branch exactly where it was"
    )

    # (3) After the rebase her commits sit on top of the new main, with NEW
    #     hashes, and the history is a straight line.
    assert c2_in_branch_after, (
        "after the rebase GitHub's C2 is an ancestor of add-search, her commits sit on top of it"
    )
    orig_shas = [sha for sha, _ in originals]
    new_shas = [sha for sha, _ in rebased]
    assert [s for _, s in rebased] == [s for _, s in originals], (
        "the same two changes, replayed in the same order"
    )
    assert set(new_shas).isdisjoint(orig_shas), (
        "not one original hash survived onto the rebased branch: the ids are all new"
    )
    assert max(_parent_counts(alice, "add-search")) == 1, (
        "the rebased history is linear: no commit on the branch has two parents"
    )

    # (4) The merge alternative creates exactly one two-parent commit and keeps
    #     the original ids.
    assert merge_tip_parents == 2, (
        "the merge route left a two-parent merge commit at the tip of add-search"
    )
    assert sum(1 for n in _parent_counts(m_alice, "add-search") if n == 2) == 1, (
        "the merge created exactly one commit with two parents"
    )
    assert set(sha for sha, _ in m_originals) <= m_kept, (
        "the merge kept Alice's two original commit ids on the branch, unchanged"
    )

    # (5) Both routes reach the identical files. They differ only in history.
    assert _tree_of(alice, "add-search") == _tree_of(m_alice, "add-search") != "", (
        "rebase and merge end at the identical final snapshot of the files"
    )

    # (6) --force-with-lease protects a colleague's push; plain --force does not.
    assert lease_refused, (
        "the lease refused because GitHub's shared had moved since Alice last looked"
    )
    assert s2_survives_lease, (
        "so Bob's commit S2 is still on GitHub after the refused lease push"
    )
    assert force_accepted, "plain --force does not check the lease, so it went through"
    assert not s2_survives_force, (
        "and plain --force overwrote GitHub's shared, dropping Bob's commit S2"
    )

    # ---- console summary --------------------------------------------------
    print(
        f"stale branch: base {_short(alice, branch_point)} (C1), GitHub main "
        f"{_short(alice, gh_main)} (C2). branch behind = {base_behind_main and not base_equals_main and not main_in_branch}."
    )
    print(
        f"fetch: add-search {info['tip_before_fetch']} -> {info['tip_after_fetch']} "
        f"(moved = {info['tip_before_fetch'] != info['tip_after_fetch']})."
    )
    print(
        f"rebase: {originals} -> {rebased}, "
        f"{len(_parent_counts(alice, 'add-search'))} commits, "
        f"{sum(1 for n in _parent_counts(alice, 'add-search') if n == 2)} with two parents."
    )
    print(
        f"merge:  originals kept, {len(_parent_counts(m_alice, 'add-search'))} commits, "
        f"{sum(1 for n in _parent_counts(m_alice, 'add-search') if n == 2)} with two parents."
    )
    print(
        f"both:   branch tip tree = {_tree_of(alice, 'add-search')} "
        f"({'equal' if _tree_of(alice, 'add-search') == _tree_of(m_alice, 'add-search') else 'DIFFERENT'})."
    )
    print(
        f"lease:  refused = {lease_refused}, Bob's S2 survives = {s2_survives_lease}. "
        f"force: accepted = {force_accepted}, Bob's S2 survives = {s2_survives_force}."
    )
    print(
        f"{len(list(FIGURES.glob('*.png')))} figures, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All assertions passed."
    )

    box.cleanup()
    mbox.cleanup()
    lease.cleanup()


if __name__ == "__main__":
    main()
