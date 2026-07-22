"""Tiny changes land on main constantly, through branches that live for hours.

Trunk-based development is not about avoiding branches. It is about avoiding
long-lived divergence from main (the "trunk"). Developers integrate into main
extremely often, either straight onto it or through very short-lived branches
that carry one small commit and are merged the same day. Because main must stay
releasable at all times, half-finished work is merged DARK, behind a feature
flag (a boolean that hides the new code in production until someone flips it on).

This builds one team-mode sandbox (a bare repo standing in for GitHub, plus a
clone for Alice and a clone for Bob) and watches five tiny changes reach main in
quick succession. Each change is a one-commit branch, fast-forward merged and
then deleted, so main stays a straight line of small steps and no branch is ever
more than one commit ahead of it. The fourth change ships a new checkout flow
dark behind an off switch; the fifth flips the switch on, a one-line change that
separates merging from releasing.

Every claim the README makes is asserted at the bottom, read from the real
repositories with plumbing: each branch is one commit ahead of main at merge,
each integration is a fast-forward that adds exactly one commit, every short
branch is deleted, main has no merge commits, and no branch is left dangling. A
second sandbox exhibits the thing trunk-based avoids, a long-lived branch that
drifts five commits ahead of a moving main, to make the contrast measurable. A
wrong picture fails the run.
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

# ---- the tiny changes each short branch carries ------------------------
APP_V1 = "def home():\n    return 'welcome'\n"
APP_V2 = "def home():\n    return 'welcome back'\n"
METRICS = "def log_request(path):\n    return ('request', path)\n"
LOADER = "def load():\n    return _cache()\n\n\ndef _cache():\n    return {}\n"

# The feature flag. NEW_CHECKOUT gates a whole new code path. Merged with the
# flag off, the new functions ship to production but never run, so users keep
# the old flow. Flipping the one boolean turns the feature on.
CHECKOUT_DARK = (
    "NEW_CHECKOUT = False\n"
    "\n\n"
    "def checkout(cart):\n"
    "    if NEW_CHECKOUT:\n"
    "        return new_flow(cart)\n"
    "    return old_flow(cart)\n"
    "\n\n"
    "def old_flow(cart):\n"
    "    return sum(cart)\n"
    "\n\n"
    "def new_flow(cart):\n"
    "    return sum(cart) - discount(cart)\n"
    "\n\n"
    "def discount(cart):\n"
    "    return min(5, sum(cart) // 10)\n"
)
CHECKOUT_ON = CHECKOUT_DARK.replace("NEW_CHECKOUT = False", "NEW_CHECKOUT = True", 1)


# ---- reading the truth back out ----------------------------------------


def _run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def sha(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", rev])


def rev_list(repo: Path, rng: str) -> list[str]:
    out = _run(repo, ["rev-list", rng])
    return [line for line in out.splitlines() if line]


def parents_of(repo: Path, rev: str) -> list[str]:
    return _run(repo, ["show", "-s", "--pretty=%P", rev]).split()


def show_file(repo: Path, rev: str, filename: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{filename}"], cwd=repo, capture_output=True, text=True
    )
    return proc.stdout


def commit_files(repo: Path, rev: str) -> list[str]:
    out = _run(repo, ["show", "--name-only", "--pretty=format:", rev])
    return [line for line in out.splitlines() if line]


def main():
    clear(FIGURES, TABLES)

    box = Sandbox(people=("alice", "bob"))
    A = box.paths["alice"]
    B = box.paths["bob"]
    GH = box.paths["github"]

    records: list[dict] = []

    def short_change(who, branch, filename, content, message):
        """One tiny change through a branch that lives for hours.

        Sync main, open a one-commit branch, commit, fast-forward it back onto
        main, push, delete the branch. Returns a record of what happened, read
        from the real repository, for the tables and the oracle.
        """
        repo = box.paths[who]
        box.git(who, "switch main")
        pulled = box.git(who, "pull")
        assert pulled.returncode == 0, f"{who} pull must succeed:\n{pulled.stderr}"
        box.git(who, f"switch -c {branch}")
        box.commit(who, filename, content, message)
        tip = sha(repo, branch)
        ahead = len(rev_list(repo, f"main..{branch}"))  # commits not yet on main
        box.git(who, "switch main")
        merged = box.git(who, f"merge {branch}")
        assert merged.returncode == 0, f"{branch} must merge cleanly:\n{merged.stderr}"
        was_ff = sha(repo, "main") == tip  # main now equals the branch tip
        pushed = box.git(who, "push")
        assert pushed.returncode == 0, f"{who} push must succeed:\n{pushed.stderr}"
        deleted = box.git(who, f"branch -d {branch}")
        assert deleted.returncode == 0, (
            f"{branch} must delete after merge:\n{deleted.stderr}"
        )
        rec = {
            "branch": branch,
            "who": who,
            "ahead": ahead,
            "ff": was_ff,
            "tip": tip,
            "main_count": len(rev_list(repo, "main")),
        }
        records.append(rec)
        return rec

    # ---- 1. everyone starts level on the trunk -------------------------
    box.commit("alice", "app.py", APP_V1, "project skeleton")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    box.snap(
        "Everyone starts on the trunk",
        note="Alice pushed a small skeleton and Bob pulled it. Both clones and GitHub agree on main, the shared trunk. c0 is the one commit everyone shares.",
    )
    c0 = sha(A, "main")

    # ---- 2. Alice opens a one-commit branch (shown live, before merge) --
    box.git("alice", "switch main")
    box.git("alice", "pull")
    box.git("alice", "switch -c fix-welcome-copy")
    box.commit("alice", "app.py", APP_V2, "fix welcome copy")
    b1_tip = sha(A, "fix-welcome-copy")
    b1_ahead = len(rev_list(A, "main..fix-welcome-copy"))
    box.snap(
        "A one-commit branch, opened for an hour",
        note="Alice opens a branch for a tiny copy fix. It carries one commit and is one commit ahead of main. It will be merged the same day, never left to drift.",
    )

    # ---- 3. Alice integrates the same day and deletes the branch -------
    box.git("alice", "switch main")
    m1 = box.git("alice", "merge fix-welcome-copy")
    assert m1.returncode == 0, f"b1 must merge cleanly:\n{m1.stderr}"
    b1_ff = sha(A, "main") == b1_tip
    box.git("alice", "push")
    box.git("alice", "branch -d fix-welcome-copy")
    records.append(
        {
            "branch": "fix-welcome-copy",
            "who": "alice",
            "ahead": b1_ahead,
            "ff": b1_ff,
            "tip": b1_tip,
            "main_count": len(rev_list(A, "main")),
        }
    )
    box.snap(
        "Integrated the same day, branch deleted",
        note="Alice fast-forwarded her branch back onto main and deleted it. Main moved forward by one small commit. This is the whole loop, kept short on purpose.",
    )

    # ---- 4. two more tiny changes land in quick succession -------------
    short_change(
        "bob", "add-request-metric", "metrics.py", METRICS, "add request metric"
    )
    short_change("alice", "cache-the-loader", "loader.py", LOADER, "cache the loader")
    box.snap(
        "Small changes keep landing on main",
        note="Bob adds a metric and Alice caches the loader, each a one-commit branch merged and deleted the same day. Main is a straight line of small steps, no branch drifting away from it.",
    )

    # ---- 5. Bob merges a new feature DARK, behind an off switch --------
    rec_dark = short_change(
        "bob",
        "new-checkout-behind-flag",
        "checkout.py",
        CHECKOUT_DARK,
        "add checkout behind flag",
    )
    c_dark = rec_dark["tip"]
    box.snap(
        "New code merged dark behind a feature flag",
        note="Bob merges a whole new checkout flow but ships it dark. The code is on main, the flag NEW_CHECKOUT is False, so users still get the old flow. Half-finished work can sit on a releasable main safely.",
    )

    # ---- 6. Alice flips the flag on later, a one-line change -----------
    rec_on = short_change(
        "alice",
        "enable-new-checkout",
        "checkout.py",
        CHECKOUT_ON,
        "enable new checkout",
    )
    c_on = rec_on["tip"]
    box.snap(
        "The flag is flipped on later",
        note="Later, Alice flips the flag to True in a one-line change. The same code that shipped dark is now live. Merging and releasing were separated from turning the feature on.",
    )

    render(box, FIGURES, TABLES, mode="team")

    # ==================================================================
    # The foil: a long-lived branch, the thing trunk-based avoids. One
    # feature branch drifts five commits ahead while main moves three
    # commits underneath it. Built as a separate sandbox and drawn solo.
    # ==================================================================
    cx = Sandbox(people=("alice", "bob"))
    cxA = cx.paths["alice"]
    cxB = cx.paths["bob"]
    cx.commit("alice", "app.py", APP_V1, "project skeleton")
    cx.git("alice", "push -u origin main")
    cx.git("bob", "pull")
    # Bob opens a long-lived feature branch and never integrates it.
    cx.git("bob", "switch -c big-feature")
    for i in range(1, 6):
        cx.commit("bob", "feature.py", f"# step {i}\n" * i, f"big feature step {i}")
    # Meanwhile the team keeps main moving.
    for i in range(1, 4):
        cx.commit(
            "alice", "app.py", APP_V1 + f"# main change {i}\n", f"main change {i}"
        )
        cx.git("alice", "push")
    cx.git("bob", "fetch origin")
    big_ahead = len(rev_list(cxB, "origin/main..big-feature"))  # branch not on main
    big_behind = len(rev_list(cxB, "big-feature..origin/main"))  # main not on branch
    cx.snap(
        "A long-lived branch drifts away from main",
        note="This is what trunk-based avoids. Bob's big-feature sat unmerged and drifted five commits ahead while main moved three commits underneath it. Merging it back now is a big-bang merge, not a fast-forward.",
    )
    lx, ly = layout(cx.snapshots)
    lorder: list[str] = []
    for snap in cx.snapshots:
        for state in snap.repos.values():
            for name in list(state.branches) + [
                r.split("/", 1)[1] for r in state.remotes
            ]:
                if name not in lorder:
                    lorder.append(name)
    draw(
        cx.snapshots[0],
        lx,
        ly,
        FIGURES / "long-lived-branch.png",
        mode="solo",
        repos=("bob",),
        order=lorder,
        title="The foil: a long-lived branch drifting from main",
    )
    cx.cleanup()

    # ==================================================================
    # Tables the README quotes, all read from real git above.
    # ==================================================================
    max_ahead = max(r["ahead"] for r in records)

    with (TABLES / "short-branches.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "branch",
                "author",
                "commits ahead of main at merge",
                "integration",
                "merged into main",
                "branch deleted",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r["branch"],
                    r["who"],
                    r["ahead"],
                    "fast-forward" if r["ff"] else "merge commit",
                    "yes",
                    "yes",
                ]
            )

    with (TABLES / "trunk-growth.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "change", "main tip", "commits on main", "new commits"])
        w.writerow(["0", "project skeleton", c0[:7], 1, 1])
        for i, r in enumerate(records, start=1):
            w.writerow([str(i), r["branch"], r["tip"][:7], r["main_count"], 1])

    dark_code = show_file(A, c_dark, "checkout.py")
    on_code = show_file(A, c_on, "checkout.py")
    with (TABLES / "feature-flag.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "stage",
                "flag value",
                "new checkout code on main",
                "feature visible to users",
            ]
        )
        w.writerow(["merged dark (Bob)", "NEW_CHECKOUT = False", "yes", "no"])
        w.writerow(["flag flipped (Alice)", "NEW_CHECKOUT = True", "yes", "yes"])

    with (TABLES / "divergence.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["style", "most a branch drifted ahead of main", "long-lived divergence"]
        )
        w.writerow(["trunk-based (this tutorial)", max_ahead, "no"])
        w.writerow(["a long-lived feature branch", big_ahead, "yes"])

    # ==================================================================
    # THE ORACLE: every claim the README makes, checked from real git.
    # ==================================================================

    gh = box.read("github")
    al = box.read("alice")
    bo = box.read("bob")

    # (1) Every branch used was short: one commit ahead of main at merge, and
    #     each one fast-forwarded (so integrating cost no merge commit). None is
    #     more than a couple of commits ahead, the defining trunk-based property.
    assert len(records) == 5, "five tiny changes went through five short branches"
    for r in records:
        assert r["ahead"] == 1, f"{r['branch']} was exactly one commit ahead of main"
        assert r["ff"], f"{r['branch']} integrated as a fast-forward, no merge commit"
    assert max_ahead == 1, "no branch drifted past one commit ahead of main"
    assert max_ahead <= 2, "no branch was more than a couple of commits ahead"

    # (2) Main received frequent small integrations. It grew from one commit to
    #     six, exactly one per short branch, and every step added exactly one.
    main_history = rev_list(A, "main")
    assert len(main_history) == 6, "main holds c0 plus five one-commit integrations"
    counts = [1] + [r["main_count"] for r in records]
    assert counts == [1, 2, 3, 4, 5, 6], "main grew one small commit at a time"
    deltas = [counts[i] - counts[i - 1] for i in range(1, len(counts))]
    assert deltas == [1, 1, 1, 1, 1], "each integration added exactly one commit"

    # (3) No long-lived divergence. Main is a straight line, every commit on it
    #     has at most one parent, so there is no big-bang merge commit anywhere.
    for s in main_history:
        assert len(parents_of(A, s)) <= 1, f"{s[:7]} on main has at most one parent"

    # (4) Nothing left dangling. Every short branch was deleted, so each repo
    #     ends with main and only main. GitHub never saw the short branches at
    #     all, since they were local and merged before pushing.
    assert set(gh.branches) == {"main"}, "GitHub carries only the trunk"
    assert set(al.branches) == {"main"}, "Alice has no leftover short branch"
    assert set(bo.branches) == {"main"}, "Bob has no leftover short branch"

    # (5) The feature flag: the new code shipped dark, then a one-line flip
    #     turned it on. Merging was separated from releasing.
    assert "def new_flow" in dark_code, "the new checkout code shipped when merged dark"
    assert "NEW_CHECKOUT = False" in dark_code, "but the flag was off, so it was dark"
    assert "def new_flow" in on_code, "the same code is still there after the flip"
    assert "NEW_CHECKOUT = True" in on_code, "and the flag is now on"
    assert dark_code.replace("False", "True", 1) == on_code, (
        "enabling the feature was exactly one boolean flip, nothing else changed"
    )
    assert commit_files(A, c_dark) == ["checkout.py"], (
        "the dark merge touched only checkout.py"
    )
    assert commit_files(A, c_on) == ["checkout.py"], (
        "the enable touched only checkout.py, a one-line change"
    )

    # (6) The foil is real and measured: a long-lived branch drifts far. This is
    #     precisely the state trunk-based keeps you out of.
    assert big_ahead == 5, "the long-lived branch drifted five commits ahead of main"
    assert big_behind == 3, "while main moved three commits underneath it"
    assert big_ahead > 2, "which is well past the couple-of-commits trunk-based bound"
    assert big_ahead > max_ahead, "the trunk-based branches never drifted like this"

    # Figures and tables actually landed.
    steps = sorted(p.name for p in FIGURES.glob("step-*.png"))
    assert len(steps) == 6, f"expected 6 team-mode step figures, got {steps}"
    assert (FIGURES / "long-lived-branch.png").exists(), "the foil figure was drawn"
    tables = sorted(p.name for p in TABLES.glob("*.csv"))
    assert len(tables) == 5, f"expected 5 tables, got {tables}"

    print(
        f"trunk: main grew c0 -> {main_history[0][:7]} in {len(records)} one-commit "
        f"integrations, every one a fast-forward, {len(deltas)} steps of +1 commit."
    )
    print(
        f"flag:  new checkout shipped dark at {c_dark[:7]} (NEW_CHECKOUT False), "
        f"flipped on at {c_on[:7]} in a one-line change."
    )
    print(
        f"foil:  a long-lived branch drifted {big_ahead} commits ahead of main "
        f"(main moved {big_behind}); trunk-based branches drifted at most {max_ahead}."
    )
    print(
        f"{len(list(FIGURES.glob('*.png')))} figures, {len(tables)} tables. "
        f"All assertions passed."
    )

    box.cleanup()


if __name__ == "__main__":
    main()
