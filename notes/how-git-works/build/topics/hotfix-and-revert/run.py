"""Two production scenarios that share one rule: on shared history, ADD commits, never rewrite.

Part A, the hotfix. A large feature is unfinished on a branch, but production
needs a fix now. The fix must be cut from main, not from the unfinished feature
branch, or it ships half-done feature code when it merges. Once the hotfix is on
main, every active feature branch has to incorporate the moved main or it is
still running the buggy code.

Part B, a bad change that is already on shared main. Because teammates already
have that commit and may be building on it, you do NOT rewrite history. git
revert appends a NEW commit that applies the inverse change, leaving the
original commit in place and everyone's clone intact. The antisocial
alternative, git reset --hard HEAD~1 followed by git push --force, rewrites the
public branch and strands the commit a teammate built on.

Everything below runs against real repositories in a temp dir, in team mode (a
bare repo standing in for GitHub, plus a clone for Alice and a clone for Bob).
Every claim the README makes is asserted from the true state read back with
plumbing. A wrong picture fails the run.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# ---- Part A production code. The bug: discount_pct is a percent like 20, so
# subtracting base*20 sends the price negative. The fix divides by 100.
BUG = "def price(base, discount_pct):\n    return base - base * discount_pct\n"
FIX = "def price(base, discount_pct):\n    return base - base * discount_pct / 100\n"
WIP1 = "class PricingEngine:\n    pass  # TODO everything\n"
WIP2 = "class PricingEngine:\n    def __init__(self):\n        self.tiers = {}  # half done\n"
SEARCH = "def search():\n    return []  # wip\n"

# ---- Part B production code. The bad change sets retries to zero, which
# disables every retry and takes the service down.
GOOD = "MAX_RETRIES = 3\n"
BAD = "MAX_RETRIES = 0\n"
CACHE = "def cache():\n    return {}\n"


# ---- small plumbing helpers, none of them recorded into the figure captions --


def tip(box: Sandbox, who: str, branch: str) -> str:
    return box.read(who).branches[branch]


def is_ancestor(box: Sandbox, who: str, a: str, b: str) -> bool:
    return (
        box.git(
            who, f"merge-base --is-ancestor {a} {b}", record=False, check=False
        ).returncode
        == 0
    )


def show(box: Sandbox, who: str, ref_path: str) -> str:
    return box.git(who, f"show {ref_path}", record=False, check=False).stdout


def count_main(box: Sandbox, who: str) -> int:
    out = box.git(
        who, "rev-list --count main", record=False, check=False
    ).stdout.strip()
    return int(out or 0)


def emit(tmp_figs: Path, prefix: str) -> int:
    """Copy render()'s step-NN.png/.pdf into figures/ under a descriptive name."""
    pngs = sorted(tmp_figs.glob("step-*.png"))
    for i, png in enumerate(pngs, start=1):
        shutil.copy(png, FIGURES / f"{prefix}-{i:02d}.png")
        pdf = png.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy(pdf, FIGURES / f"{prefix}-{i:02d}.pdf")
    return len(pngs)


def emit_one(tmp_figs: Path, name: str) -> None:
    png = sorted(tmp_figs.glob("step-*.png"))[0]
    shutil.copy(png, FIGURES / f"{name}.png")
    pdf = png.with_suffix(".pdf")
    if pdf.exists():
        shutil.copy(pdf, FIGURES / f"{name}.pdf")


def main() -> None:
    clear(FIGURES, TABLES)
    scratch = Path(tempfile.mkdtemp(prefix="hotfix-revert-"))

    # =================================================================
    # PART A: the hotfix
    # =================================================================
    a = Sandbox(people=("alice", "bob"))

    # Buggy production code on main. Alice pushes it, Bob pulls it.
    a.commit("alice", "pricing.py", BUG, "add pricing")
    a.git("alice", "push -u origin main")
    a.git("bob", "pull")

    # Alice is deep in a large, unfinished feature, on a local branch.
    a.git("alice", "switch -c feature/redesign")
    a.commit("alice", "redesign.py", WIP1, "wip pricing engine skeleton")
    a.commit("alice", "redesign.py", WIP2, "wip half the tiers")

    # Bob is deep in a separate unfinished feature, cut from the same old main.
    a.git("bob", "switch -c feature/search")
    a.commit("bob", "search.py", SEARCH, "wip search stub")

    a.snap(
        "Two unfinished features, one buggy main",
        note="pricing.py on main has a bug: discount_pct is a percent, so a 20 percent discount subtracts 20x the base and the price goes negative. Alice and Bob are each mid-way through a separate, unfinished feature branch.",
    )

    # Production breaks. The right move: cut the hotfix from main, not from the
    # unfinished feature branch. git switch main; git pull; git switch -c hotfix.
    a.git("alice", "switch main")
    a.git("alice", "pull")
    main_tip_at_branch = tip(a, "alice", "main")
    feat_redesign_tip = tip(a, "alice", "feature/redesign")
    a.git("alice", "switch -c hotfix/pricing")
    a.commit("alice", "pricing.py", FIX, "fix discount percent bug")
    a.snap(
        "The hotfix branches from main, not from the feature",
        note="git switch main; git pull; git switch -c hotfix/pricing. The fix sits on a branch rooted at main's tip, in its own lane. It carries none of the unfinished redesign work.",
    )
    hotfix_tip = tip(a, "alice", "hotfix/pricing")
    hotfix_parent = a.read("alice").commits[hotfix_tip].parents[0]

    # --- assert: the hotfix's parent is main's tip, NOT the feature tip -------
    assert hotfix_parent == main_tip_at_branch, (
        "the hotfix commit's parent is main's tip"
    )
    assert hotfix_parent != feat_redesign_tip, (
        "the hotfix commit's parent is NOT the feature branch's tip"
    )
    a_correct_carries = is_ancestor(a, "alice", feat_redesign_tip, hotfix_tip)
    assert not a_correct_carries, (
        "the feature's unfinished commits are NOT reachable from the hotfix"
    )

    # Ship it: push the hotfix branch (the PR), then merge it to main and push.
    a.git("alice", "push -u origin hotfix/pricing")
    a.git("alice", "switch main")
    merge_hf = a.git("alice", "merge hotfix/pricing", check=False)
    a.git("alice", "push")
    a.snap(
        "The hotfix is merged to main and shipped",
        note="main now carries the fix. Both feature branches were cut from the old main, so neither has the fix yet. Bob has not even fetched, so his origin/main is still the old tip.",
    )
    main_after_fix = tip(a, "alice", "main")

    assert merge_hf.returncode == 0, "the hotfix merges into main as a fast-forward"
    assert main_after_fix == hotfix_tip, "main fast-forwarded onto the hotfix commit"
    assert is_ancestor(a, "alice", hotfix_tip, "main"), "the fix is now on main"

    # --- assert: neither feature branch has the fix yet ----------------------
    red_has_before = is_ancestor(a, "alice", hotfix_tip, feat_redesign_tip)
    assert not red_has_before, "feature/redesign does not contain the fix yet"
    bob_feat_tip = tip(a, "bob", "feature/search")
    bob_has_before = is_ancestor(a, "bob", hotfix_tip, bob_feat_tip)
    assert not bob_has_before, "Bob's feature/search does not contain the fix yet"
    assert show(a, "alice", "feature/redesign:pricing.py") == BUG, (
        "feature/redesign still runs the buggy pricing.py"
    )
    assert show(a, "bob", "feature/search:pricing.py") == BUG, (
        "Bob's feature/search still runs the buggy pricing.py"
    )

    # Teammates incorporate the moved main so their branches carry the fix.
    a.git("alice", "switch feature/redesign")
    m_red = a.git("alice", "merge main", check=False)
    a.git("bob", "fetch origin")
    m_bob = a.git("bob", "merge origin/main", check=False)
    a.snap(
        "Both feature branches incorporate the updated main",
        note="Alice merges main into feature/redesign. Bob fetches the moved main and merges origin/main into feature/search. Now both branches carry the fix, on top of their own unfinished work.",
    )

    assert m_red.returncode == 0 and m_bob.returncode == 0, (
        "merging the moved main into each feature is a clean merge"
    )
    red_tip_after = tip(a, "alice", "feature/redesign")
    bob_tip_after = tip(a, "bob", "feature/search")
    red_has_after = is_ancestor(a, "alice", hotfix_tip, red_tip_after)
    bob_has_after = is_ancestor(a, "bob", hotfix_tip, bob_tip_after)
    assert red_has_after, "after merging main, feature/redesign contains the fix"
    assert bob_has_after, "after fetch and merge, Bob's feature contains the fix"
    assert show(a, "alice", "feature/redesign:pricing.py") == FIX, (
        "and feature/redesign's pricing.py is now the fixed version"
    )
    assert show(a, "bob", "feature/search:pricing.py") == FIX, (
        "and Bob's feature/search pricing.py is now fixed too"
    )

    tmp_a = scratch / "a-figs"
    render(a, tmp_a, scratch / "a-tabs", mode="team")
    n_a = emit(tmp_a, "hotfix")

    # The wrong way, in its own sandbox: cut the hotfix off the unfinished feature.
    w = Sandbox(people=("alice",))
    w.commit("alice", "pricing.py", BUG, "add pricing")
    w.git("alice", "push -u origin main")
    w.git("alice", "switch -c feature/redesign")
    w.commit("alice", "redesign.py", WIP1, "wip pricing engine skeleton")
    w.commit("alice", "redesign.py", WIP2, "wip half the tiers")
    w_feat_tip = tip(w, "alice", "feature/redesign")
    w.git("alice", "switch -c hotfix-from-feature")
    w.commit("alice", "pricing.py", FIX, "fix discount percent bug")
    w.snap(
        "The wrong way: hotfix cut from the unfinished feature",
        note="Branching the hotfix off feature/redesign stacks the fix on top of two unfinished commits. Merging this to main would ship the half-done redesign along with the fix.",
    )
    w_hotfix_tip = tip(w, "alice", "hotfix-from-feature")
    w_hotfix_parent = w.read("alice").commits[w_hotfix_tip].parents[0]
    assert w_hotfix_parent == w_feat_tip, "the wrong hotfix's parent IS the feature tip"
    a_wrong_carries = is_ancestor(w, "alice", w_feat_tip, w_hotfix_tip)
    assert a_wrong_carries, (
        "so the feature's unfinished commits ARE carried by the wrong hotfix"
    )
    tmp_w = scratch / "w-figs"
    render(w, tmp_w, scratch / "w-tabs", mode="team", repos=("alice",))
    emit_one(tmp_w, "hotfix-wrong")

    # =================================================================
    # PART B: revert a bad commit on shared main
    # =================================================================
    b = Sandbox(people=("alice", "bob"))
    b.commit("alice", "retries.py", GOOD, "add retry config")
    b.git("alice", "push -u origin main")
    b.git("bob", "pull")

    # A PR is squash-merged to main. It turns out to disable all retries.
    b.commit("alice", "retries.py", BAD, "merge PR perf tuning")
    b.git("alice", "push")
    bad_sha = tip(b, "alice", "main")

    # Bob pulls the (bad) main and builds his own work on top of it.
    b.git("bob", "pull")
    b.commit("bob", "cache.py", CACHE, "add cache")
    b.snap(
        "A bad change is merged to main, and Bob builds on it",
        note="The perf-tuning PR set MAX_RETRIES to 0, disabling retries. It is squash-merged to main and pushed. Bob pulls it and commits his own work on top, so his branch now depends on the bad commit.",
    )

    # The right undo on shared history: revert appends an inverse commit.
    before_revert = count_main(b, "alice")
    rev = b.git("alice", f"revert --no-edit {bad_sha}", check=False)
    b.git("alice", "push")
    b.snap(
        "git revert adds a new commit that undoes the bad one",
        note="git revert wrote a NEW commit applying the inverse change. main is now three commits long. The bad commit is still there, still reachable, still in git log. The working tree is back to MAX_RETRIES = 3.",
    )
    after_revert = count_main(b, "alice")
    revert_tip = tip(b, "alice", "main")

    # --- assert: revert ADDED a commit, and the bad commit still exists -------
    assert rev.returncode == 0, "the revert applied cleanly"
    assert after_revert == before_revert + 1, (
        "git revert ADDED a commit: the count on main went up by one"
    )
    assert is_ancestor(b, "alice", bad_sha, "main"), (
        "the bad commit is still reachable in main's history"
    )
    assert (
        b.git(
            "alice", f"cat-file -t {bad_sha}", record=False, check=False
        ).stdout.strip()
        == "commit"
    ), "the bad commit object still exists"
    # --- assert: the reverted content is undone on disk, original stays intact
    assert (b.paths["alice"] / "retries.py").read_text() == GOOD, (
        "the reverted content is undone in the working tree (retries back to 3)"
    )
    assert show(b, "alice", f"{bad_sha}:retries.py") == BAD, (
        "yet the original bad commit still holds its own content in history"
    )
    assert revert_tip != bad_sha, "the revert is a distinct new commit"

    # Bob pulls the revert. His branch and the revert both descend from the bad
    # commit, so git makes a merge commit. He keeps his work and gains the fix.
    # --no-rebase because the branches diverged, and we want the merge, not a rebase.
    b.git("bob", "pull --no-rebase")
    b.snap(
        "Bob pulls the revert and keeps his own work",
        note="Bob's branch and the revert both descend from the bad commit, so git makes a merge commit. Bob keeps his own commit AND gets the fix. Nothing was rewritten, so nothing Bob built on was invalidated.",
    )
    assert (b.paths["bob"] / "retries.py").read_text() == GOOD, "Bob now has the fix"
    assert (b.paths["bob"] / "cache.py").exists(), "and Bob still has his own work"
    assert is_ancestor(b, "bob", bad_sha, "main"), (
        "the bad commit is still in Bob's history too, nothing stranded"
    )

    tmp_b = scratch / "b-figs"
    render(b, tmp_b, scratch / "b-tabs", mode="team")
    n_b = emit(tmp_b, "revert")

    # The wrong way, in its own sandbox: reset --hard then force-push.
    c = Sandbox(people=("alice", "bob"))
    c.commit("alice", "retries.py", GOOD, "add retry config")
    c.git("alice", "push -u origin main")
    c.git("bob", "pull")
    c.commit("alice", "retries.py", BAD, "merge PR perf tuning")
    c.git("alice", "push")
    c_bad = tip(c, "alice", "main")
    c.git("bob", "pull")
    c.commit("bob", "cache.py", CACHE, "add cache")
    c_before = count_main(c, "alice")
    c.git("alice", "reset --hard HEAD~1")
    force = c.git("alice", "push --force", check=False)
    c.snap(
        "The antisocial fix: reset --hard and force-push",
        note="git reset --hard HEAD~1 threw the bad commit off main; git push --force overwrote the shared branch. GitHub's main no longer contains the bad commit. But Bob's branch was built on that commit, so his branch and the rewritten main have diverged, and the commit he depended on is gone from the server.",
    )
    c_after = count_main(c, "alice")

    assert force.returncode == 0, "the force-push overwrote shared main"
    assert c_after == c_before - 1, (
        "reset --hard REMOVED a commit from main: the count went down by one"
    )
    gh_reach = is_ancestor(c, "github", c_bad, "main")
    bob_reach = is_ancestor(c, "bob", c_bad, "main")
    assert not gh_reach, (
        "the bad commit is NO LONGER reachable from GitHub's rewritten main"
    )
    assert bob_reach, (
        "but Bob's local branch still builds on the bad commit: his branch is now invalidated"
    )
    tmp_c = scratch / "c-figs"
    render(c, tmp_c, scratch / "c-tabs", mode="team")
    emit_one(tmp_c, "revert-wrong")

    # =================================================================
    # TABLES, every cell measured above, not hand-typed
    # =================================================================
    yn = lambda flag: "yes" if flag else "no"

    with (TABLES / "hotfix-branch-point.csv").open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(
            [
                "approach",
                "hotfix branched from",
                "hotfix parent",
                "carries unfinished feature work",
            ]
        )
        wtr.writerow(
            ["from main (correct)", "main", hotfix_parent, yn(a_correct_carries)]
        )
        wtr.writerow(
            [
                "from feature (wrong)",
                "feature/redesign",
                w_hotfix_parent,
                yn(a_wrong_carries),
            ]
        )

    with (TABLES / "feature-catch-up.csv").open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(
            [
                "branch",
                "owner",
                "has the fix before merging main",
                "has the fix after merging main",
            ]
        )
        wtr.writerow(
            ["feature/redesign", "alice", yn(red_has_before), yn(red_has_after)]
        )
        wtr.writerow(["feature/search", "bob", yn(bob_has_before), yn(bob_has_after)])

    with (TABLES / "revert-vs-reset.csv").open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(
            [
                "approach",
                "command",
                "commits on main before",
                "commits on main after",
                "bad commit still in history",
                "rewrites public history",
                "teammate branch invalidated",
            ]
        )
        wtr.writerow(
            [
                "git revert (correct)",
                "git revert <sha>; git push",
                before_revert,
                after_revert,
                yn(is_ancestor(b, "alice", bad_sha, "main")),
                "no",
                "no",
            ]
        )
        wtr.writerow(
            [
                "reset --hard + force (wrong)",
                "git reset --hard HEAD~1; git push --force",
                c_before,
                c_after,
                yn(gh_reach),
                "yes",
                yn((not gh_reach) and bob_reach),
            ]
        )

    a.cleanup()
    w.cleanup()
    b.cleanup()
    c.cleanup()
    shutil.rmtree(scratch, ignore_errors=True)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == n_a + 1 + n_b + 1, f"unexpected figure set: {figs}"
    assert n_a == 4 and n_b == 3, (
        f"expected 4 hotfix + 3 revert story figures, got {n_a}, {n_b}"
    )
    print(
        f"{len(figs)} figures, 3 tables. Every hotfix and revert claim checked and passing."
    )
    print("figures:", figs)


if __name__ == "__main__":
    main()
