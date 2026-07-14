"""GitHub Flow: the everyday loop, run twice, checked against real git.

GitHub Flow is the workflow most teams should reach for by default, and it is
the right one for a data scientist shipping into a shared repository. The whole
method is one short loop, repeated:

  update main            git switch main; git pull
  branch off main        git switch -c <feature>
  commit                 edit, git add, git commit
  push the branch        git push -u origin <feature>
  open a PR              (browser) review + CI run against the branch
  squash-merge          (browser) one commit lands on main
  delete the branch     on GitHub, and locally
  pull main again        git switch main; git pull

Two rules make it work. main is always deployable, so nothing unfinished ever
sits on it. Every change reaches main only through a pull request, which is
where review and the automated tests gate it. Branches are short: they live for
hours to a few days and are deleted the moment their PR merges, so main never
diverges from anyone for long.

This run drives the loop for TWO features, add-currency then cache-rates, in
team mode with a GitHub panel over Alice's clone. A second person, Bob, reviews
each PR in the browser; the review and the CI run are prose, everything git does
is real. The squash-merge happens on GitHub's side, modelled with a throwaway
clone that is never registered as a person, so only main moving forward shows in
the figures. The assertions at the bottom check every claim: each feature went
onto a short branch off main and never onto main directly, the squash put
exactly ONE commit per feature on main, the branch's own commits are absent from
main, main stayed a single clean line with no merge commits, and every branch
was deleted after its merge.
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


# ---- read-only helpers, every one pulling from real git ----------------


def out(box: Sandbox, who: str, args: str) -> str:
    return box.git(who, args, record=False, check=False).stdout.strip()


def full(box: Sandbox, who: str, r: str) -> str:
    return out(box, who, f"rev-parse {r}")


def short(box: Sandbox, who: str, r: str) -> str:
    return out(box, who, f"rev-parse --short {r}")


def count(box: Sandbox, who: str, r: str = "main") -> int:
    return int(out(box, who, f"rev-list --count {r}"))


def reachable(box: Sandbox, who: str, r: str = "main") -> set[str]:
    s = out(box, who, f"rev-list {r}")
    return set(s.split()) if s else set()


def merge_commits(box: Sandbox, who: str, r: str = "main") -> list[str]:
    s = out(box, who, f"rev-list --min-parents=2 {r}")
    return s.split() if s else []


def first_parent(box: Sandbox, who: str, r: str = "main") -> list[str]:
    s = out(box, who, f"rev-list --first-parent {r}")
    return s.split() if s else []


def is_ancestor(box: Sandbox, who: str, a: str, b: str) -> bool:
    proc = box.git(who, f"merge-base --is-ancestor {a} {b}", record=False, check=False)
    return proc.returncode == 0


def branch_own_commits(box: Sandbox, who: str, branch: str) -> list[str]:
    """The commits that are on `branch` but not yet on main: the branch's work."""
    s = out(box, who, f"rev-list {branch} ^main")
    return s.split() if s else []


# ---- GitHub's side of the pull request ---------------------------------


def github_squash_merges_and_deletes(box: Sandbox, branch: str, message: str):
    """Model GitHub squash-merging the PR, then auto-deleting the head branch.

    Clicking "Squash and merge" runs on GitHub, not on Alice's laptop. GitHub
    already holds every commit, so it collapses the branch into a single new
    commit on main and advances main, then removes the merged branch (the common
    "automatically delete head branches" setting). A bare repo has no working
    tree, so this borrows a throwaway clone as GitHub's own server-side worktree.
    That clone is never registered with the sandbox, so it is never drawn; only
    main moving forward and the branch disappearing show up in the figures.
    """
    scratch = box.root / f"github-server-{branch}"
    box.git(
        "github", f"clone {box.paths['github']} {scratch}", cwd=box.root, record=False
    )
    box.git("github", "config user.name github", cwd=scratch, record=False)
    box.git("github", "config user.email github@example.com", cwd=scratch, record=False)
    box.git("github", "switch main", cwd=scratch, record=False)
    box.git("github", f"merge --squash origin/{branch}", cwd=scratch, record=False)
    box.git("github", f"commit -m {message}", cwd=scratch, record=False)
    box.git("github", "push origin main", cwd=scratch, record=False)
    box.git("github", f"push origin --delete {branch}", cwd=scratch, record=False)


# ---- one turn of the loop ----------------------------------------------


def run_feature(
    box: Sandbox,
    branch: str,
    edits: list[tuple[str, str, str]],
    squash_message: str,
    labels: dict[str, str],
    capture,
) -> dict:
    """Drive the full GitHub Flow loop for one feature and record what git did."""
    facts: dict = {"branch": branch}

    # update main: always start a feature from the latest deployable main.
    box.git("alice", "switch main")
    box.git("alice", "pull --prune")
    facts["branch_point"] = full(box, "alice", "main")
    main_tip_at_branch = facts["branch_point"]

    # branch off main: a short-lived branch, created straight from main's tip.
    box.git("alice", f"switch -c {branch}")
    for filename, content, message in edits:
        box.commit("alice", filename, content, message)
    facts["branch_commits"] = [
        full(box, "alice", c) for c in branch_own_commits(box, "alice", branch)
    ]

    # push the branch, then Bob reviews the PR and CI runs, in the browser.
    box.git("alice", f"push -u origin {branch}")
    box.snap(labels["open"], note=labels["open_note"])
    capture(labels["open_stage"])

    # while the branch was built, main never moved: the work stayed isolated.
    facts["main_unmoved"] = full(box, "alice", "main") == main_tip_at_branch
    facts["forked_from_main"] = out(box, "alice", f"merge-base {branch} main")
    facts["branch_isolated"] = not any(
        is_ancestor(box, "alice", c, "main") for c in facts["branch_commits"]
    )

    # squash-merge on GitHub: exactly one commit lands on main, branch removed.
    facts["main_before"] = count(box, "github", "main")
    github_squash_merges_and_deletes(box, branch, squash_message)
    facts["main_after"] = count(box, "github", "main")
    facts["squash_commit"] = full(box, "github", "main")
    facts["squash_parents"] = (
        len(out(box, "github", "rev-list --parents -1 main").split()) - 1
    )
    facts["branch_on_github_after"] = branch in box.read("github").branches
    facts["originals_reachable_after"] = any(
        is_ancestor(box, "github", c, "main") for c in facts["branch_commits"]
    )
    box.snap(labels["merge"], note=labels["merge_note"])
    capture(labels["merge_stage"])

    # delete the branch and pull the merged main back.
    box.git("alice", "switch main")
    box.git("alice", "pull --prune")
    # The safe delete refuses: squash left the branch's commits off main, so git
    # cannot see it as merged. The force delete is correct here, the work is
    # already on main as the squash commit.
    safe = box.git("alice", f"branch -d {branch}", record=False, check=False)
    facts["safe_delete_refused"] = safe.returncode != 0
    forced = box.git("alice", f"branch -D {branch}", record=False, check=False)
    facts["force_delete_ok"] = forced.returncode == 0
    facts["local_branch_gone"] = branch not in box.read("alice").branches
    facts["alice_synced"] = full(box, "alice", "main") == full(box, "github", "main")
    box.snap(labels["clean"], note=labels["clean_note"])
    capture(labels["clean_stage"])

    return facts


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice",))
    rows: list[dict] = []

    def capture(stage: str):
        g = box.read("github")
        a = box.read("alice")
        rows.append(
            {
                "stage": stage,
                "github_main": g.branches.get("main", ""),
                "github_branches": " ".join(sorted(g.branches)),
                "alice_main": a.branches.get("main", ""),
                "alice_branches": " ".join(sorted(a.branches)),
                "commits_on_main": count(box, "github", "main"),
                "merge_commits_on_main": len(merge_commits(box, "github", "main")),
            }
        )

    # ---- the starting line: a deployable main on GitHub, Alice cloned it ----
    box.commit(
        "alice", "app.py", "def price(x):\n    return x * 2\n", "initial-pricing"
    )
    box.git("alice", "push -u origin main")
    box.snap(
        "main is deployable, on GitHub",
        note="One reviewed commit on main. It is the deployable line: everything shipped goes through here, and nothing unfinished ever sits on it.",
    )
    capture("start")

    # ================================================================
    # Feature A: add-currency
    # ================================================================
    feat_a = run_feature(
        box,
        "add-currency",
        edits=[
            (
                "app.py",
                "def price(x, currency):\n    return x * 2\n",
                "add-currency-arg",
            ),
            (
                "app.py",
                'def price(x, currency="usd"):\n'
                '    rate = {"usd": 1.0, "eur": 0.9}[currency]\n'
                "    return x * 2 * rate\n",
                "apply-currency-rate",
            ),
        ],
        squash_message="Add-currency-support-(#1)",
        labels={
            "open": "add-currency: a short branch off main, pushed, PR open",
            "open_note": "Two commits on a branch that lives only until it merges. main has not moved. Bob reviews the PR in the browser and CI runs the tests against the branch.",
            "open_stage": "feature A: PR open",
            "merge": "GitHub squash-merges the PR and deletes the branch",
            "merge_note": "Squash and merge collapses the branch's two commits into ONE new commit on main, in red. GitHub then deletes the merged branch. Alice's clone has not fetched yet.",
            "merge_stage": "feature A: squash-merged",
            "clean": "git switch main; git pull --prune; git branch -D add-currency",
            "clean_note": "Alice pulls the merged main and deletes her local branch. main is now two reviewed commits, still a single line.",
            "clean_stage": "feature A: cleaned up",
        },
        capture=capture,
    )

    # ================================================================
    # Feature B: cache-rates (branches off the post-A main)
    # ================================================================
    feat_b = run_feature(
        box,
        "cache-rates",
        edits=[
            ("cache.py", "CACHE = {}\n", "add-cache-dict"),
            (
                "cache.py",
                "CACHE = {}\n\n\ndef get(key):\n    return CACHE.get(key)\n",
                "add-cache-getter",
            ),
        ],
        squash_message="Add-rate-cache-(#2)",
        labels={
            "open": "cache-rates: another short branch off the updated main",
            "open_note": "The next feature starts from the main that already contains add-currency. Same loop: branch, two commits, push, open a PR for Bob to review.",
            "open_stage": "feature B: PR open",
            "merge": "GitHub squash-merges the second PR and deletes the branch",
            "merge_note": "One more squash commit lands on main, in red. Two features, two commits: main is a clean line of squash-merged work.",
            "merge_stage": "feature B: squash-merged",
            "clean": "git switch main; git pull --prune; git branch -D cache-rates",
            "clean_note": "Alice syncs and deletes the branch. main holds exactly one commit per shipped feature, in order, with no merge bubbles.",
            "clean_stage": "feature B: cleaned up",
        },
        capture=capture,
    )

    render(box, FIGURES, TABLES, mode="team")

    # ================================================================
    # THE ASSERTIONS: every claim the README makes, checked against git.
    # ================================================================

    base = full(box, "alice", "main~2")  # the initial-pricing commit
    all_branch_commits = set(feat_a["branch_commits"]) | set(feat_b["branch_commits"])

    for f in (feat_a, feat_b):
        b = f["branch"]

        # -- each feature went onto a SHORT branch off main, not onto main ----
        assert len(f["branch_commits"]) == 2, (
            f"{b}: the feature was built as two commits on its own branch"
        )
        assert f["forked_from_main"] == f["branch_point"], (
            f"{b}: the branch forked straight from main's tip"
        )
        assert f["main_unmoved"], (
            f"{b}: main never moved while the branch was built; no commit went onto main directly"
        )
        assert f["branch_isolated"], (
            f"{b}: the branch's commits were not on main, so unfinished work stayed isolated"
        )

        # -- the squash-merge put exactly ONE commit per feature on main ------
        assert f["main_after"] - f["main_before"] == 1, (
            f"{b}: squash-merge added exactly one commit to main"
        )
        assert f["squash_parents"] == 1, (
            f"{b}: the squash commit has a single parent, so it is not a merge commit"
        )
        assert not f["originals_reachable_after"], (
            f"{b}: none of the branch's own commits are reachable from main after the squash"
        )

        # -- branches were deleted after merge -------------------------------
        assert not f["branch_on_github_after"], (
            f"{b}: the branch was deleted on GitHub when the PR merged"
        )
        assert f["safe_delete_refused"], (
            f"{b}: git branch -d refuses after a squash, because the branch is not merged into main by history"
        )
        assert f["force_delete_ok"] and f["local_branch_gone"], (
            f"{b}: git branch -D removes the finished local branch"
        )
        assert f["alice_synced"], f"{b}: after the pull, Alice's main matches GitHub's"

    # -- the branch's individual commits are NOT in main's first-parent history
    fp = set(first_parent(box, "github", "main"))
    assert all_branch_commits.isdisjoint(fp), (
        "no original branch commit appears in main's first-parent history"
    )

    # -- main stayed a clean line: linear, one commit per feature ------------
    assert merge_commits(box, "github", "main") == [], (
        "main has no merge commit anywhere: it is a single straight line"
    )
    assert count(box, "github", "main") == 3, (
        "main holds exactly three commits: the base plus one per feature"
    )
    total = reachable(box, "github", "main")
    assert total == {base, feat_a["squash_commit"], feat_b["squash_commit"]}, (
        "the only commits on main are the base and the two squash-merges"
    )
    assert len(first_parent(box, "github", "main")) == count(box, "github", "main"), (
        "first-parent history equals full history: main is perfectly linear"
    )
    assert all_branch_commits.isdisjoint(total), (
        "not one of the four original branch commits survives on main"
    )
    assert feat_b["branch_point"] == feat_a["squash_commit"], (
        "feature B branched off the main that already contained feature A's squash commit"
    )

    # -- every branch is gone at the end -------------------------------------
    assert set(box.read("github").branches) == {"main"}, (
        "GitHub has only main left: every merged branch was deleted"
    )
    assert set(box.read("alice").branches) == {"main"}, (
        "Alice has only main left: every finished branch was deleted"
    )

    # ---- the tables the README quotes from ---------------------------------
    with (TABLES / "flow-log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    line = []
    for sha in reversed(first_parent(box, "github", "main")):
        line.append(
            {
                "commit": short(box, "github", sha),
                "subject": out(box, "github", f"log -1 --format=%s {sha}"),
                "parents": len(
                    out(box, "github", f"rev-list --parents -1 {sha}").split()
                )
                - 1,
            }
        )
    with (TABLES / "main-line.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(line[0]))
        writer.writeheader()
        writer.writerows(line)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 figures, got {figs}"
    print(
        f"{len(figs)} figures, 3 tables. Two features through the GitHub Flow loop, "
        "every claim checked and passing."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
