"""GitFlow: two long-lived branches and three kinds of temporary branch, built for real.

One repository, one person, solo mode. The whole point is the branch topology of
a single repo, so there is no second machine here.

The model has two branches that live forever:

  main     holds only released, tagged versions. Its history is a chain of
           releases and nothing else. No feature is ever committed straight onto
           it.
  develop  the integration branch. Every finished feature lands here first.

and three kinds of branch that are cut, used, and deleted:

  feature/*  off develop, back into develop. Never touches main.
  release/*  off develop, stabilized, then merged into BOTH main and develop,
             and main is tagged with the version.
  hotfix/*   off main, then merged into BOTH main and develop, and main is
             tagged again.

run.py builds exactly this history, snapshots it at each stage, and ends with
assertions that check every claim the README makes. A wrong claim fails the run:
that main and develop both survive as long-lived branches; that a feature was
integrated on develop and never on main's own line; that the release was merged
into both main and develop; that main carries the version tag at the release;
and that every commit on main's own line of history is a tagged release, never a
feature commit.

Tags are checked with `git tag` and `git rev-parse`. The renderer draws branch
pointers but not tags, so the tags are named in the figure captions and the
prose and proven here in code.
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

# ---- file contents. Distinct files per branch, so every merge is clean:
# conflicts are a separate tutorial, and GitFlow's shape is what we are showing.
APP_V0 = "def app():\n    return 'v0.1.0'\n"
APP_QA = "def app():\n    return 'v1.0.0'  # QA fix\n"
LOGIN = "def login():\n    return True\n"
SEARCH = "def search():\n    return []\n"
REPORTS = "def reports():\n    return []\n"
VERSION_1 = "1.0.0\n"
CHANGELOG = "1.0.0: login, search\n"
HOTFIX = "def sanitize(x):\n    return x.strip()\n"


def rp(box: Sandbox, rev: str) -> str:
    """Full commit id of a ref, read straight out of git."""
    return box.git("alice", f"rev-parse {rev}", record=False).stdout.strip()


def first_parent(box: Sandbox, rev: str) -> list[str]:
    """The commits on a branch's OWN line of history (its first-parent chain).

    A merge commit's first parent is the branch it was made on, so walking first
    parents from main gives exactly the commits main advanced through itself,
    skipping the branches it merged in.
    """
    out = box.git("alice", f"rev-list --first-parent {rev}", record=False).stdout
    return out.split()


def parents(box: Sandbox, sha: str) -> list[str]:
    out = box.git("alice", f"rev-list --parents -n 1 {sha}", record=False).stdout
    return out.split()[1:]


def is_ancestor(box: Sandbox, a: str, b: str) -> bool:
    return (
        box.git(
            "alice", f"merge-base --is-ancestor {a} {b}", record=False, check=False
        ).returncode
        == 0
    )


def tags(box: Sandbox) -> list[str]:
    return box.git("alice", "tag", record=False).stdout.split()


def tag_commit(box: Sandbox, tag: str) -> str:
    return box.git("alice", f"rev-parse {tag}^{{commit}}", record=False).stdout.strip()


def subject(box: Sandbox, sha: str) -> str:
    return box.git("alice", f"show -s --format=%s {sha}", record=False).stdout.strip()


def main() -> None:
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice",))

    # =================================================================
    # The two long-lived branches
    # =================================================================
    box.commit("alice", "app.py", APP_V0, "initial release")
    c0 = rp(box, "main")
    box.git("alice", "switch -c develop")
    box.snap(
        "The two long-lived branches: main and develop",
        note="main holds only released versions and is tagged v0.1.0 here. develop is the integration branch. Both live for the whole life of the project. Every other branch is temporary.",
    )

    # =================================================================
    # Feature branches: off develop, back into develop, never touching main.
    # Two features are cut from the same develop and built in parallel.
    # =================================================================
    box.git("alice", "switch -c feature/login")
    box.commit("alice", "login.py", LOGIN, "add login form")
    box.commit("alice", "login.py", LOGIN + "# validated\n", "add login validation")
    login_tip = rp(box, "feature/login")

    box.git("alice", "switch develop")
    box.git("alice", "switch -c feature/search")
    box.commit("alice", "search.py", SEARCH, "add search")
    search_tip = rp(box, "feature/search")
    box.snap(
        "Feature branches are cut from develop",
        note="feature/login and feature/search each branch off develop and carry one piece of work. They never touch main. In GitFlow every feature is integrated through develop first.",
    )

    box.git("alice", "switch develop")
    box.git("alice", "merge --no-ff feature/login -m merge-feature-login")
    m1 = rp(box, "develop")  # the feature-integration merge commit on develop
    box.git("alice", "merge --no-ff feature/search -m merge-feature-search")
    m2 = rp(box, "develop")
    box.snap(
        "Features merge back into develop, never into main",
        note="--no-ff keeps each feature visible as one unit of work on develop. main has not moved: it is still at v0.1.0. develop now carries both features, ready for a release.",
    )
    # The features are integrated, so their branches are deleted (temporary).
    box.git("alice", "branch -d feature/login")
    box.git("alice", "branch -d feature/search")

    # =================================================================
    # A release branch: cut from develop, stabilized, while develop keeps moving.
    # =================================================================
    box.git("alice", "switch -c release/1.0.0")
    box.commit("alice", "VERSION", VERSION_1, "bump version to 1.0.0")
    box.commit("alice", "app.py", APP_QA, "fix typo found in QA")
    box.commit("alice", "CHANGELOG", CHANGELOG, "write changelog")
    release_tip = rp(box, "release/1.0.0")

    # Work does not stop on develop while the release stabilizes.
    box.git("alice", "switch develop")
    box.git("alice", "switch -c feature/reports")
    box.commit("alice", "reports.py", REPORTS, "add reports")
    reports_tip = rp(box, "feature/reports")
    box.git("alice", "switch develop")
    box.git("alice", "merge --no-ff feature/reports -m merge-feature-reports")
    m3 = rp(box, "develop")
    box.git("alice", "branch -d feature/reports")
    box.snap(
        "A release branch is cut from develop, and work continues on develop",
        note="release/1.0.0 is a freeze of develop for stabilization: version bump, a QA fix, a changelog. Meanwhile feature/reports merges into develop, so develop has already moved past the point the release was cut from.",
    )

    # =================================================================
    # The release merges into BOTH main and develop, and main is tagged.
    # =================================================================
    box.git("alice", "switch main")
    box.git("alice", "merge --no-ff release/1.0.0 -m merge-release-1.0.0")
    rmain = rp(box, "main")
    box.git("alice", "switch develop")
    box.git("alice", "merge --no-ff release/1.0.0 -m merge-release-into-develop")
    rdev = rp(box, "develop")
    box.git("alice", "branch -d release/1.0.0")
    box.snap(
        "The release merges into BOTH main and develop, and main is tagged v1.0.0",
        note="main gets the release and is tagged v1.0.0: this is a release, and main only ever holds releases. develop gets the same commits back, so the QA fix and version bump are not lost when the next feature lands.",
    )

    # =================================================================
    # A hotfix: off main, then into BOTH main and develop, main tagged again.
    # =================================================================
    box.git("alice", "switch main")
    box.git("alice", "switch -c hotfix/1.0.1")
    box.commit("alice", "hotfix.py", HOTFIX, "fix crash on empty input")
    hotfix_tip = rp(box, "hotfix/1.0.1")
    box.snap(
        "A hotfix branches off main, not develop",
        note="Production is broken and the next release is not ready. hotfix/1.0.1 is cut from main, so it carries no unfinished develop work. It fixes the bug and nothing else.",
    )

    box.git("alice", "switch main")
    box.git("alice", "merge --no-ff hotfix/1.0.1 -m merge-hotfix-1.0.1")
    hmain = rp(box, "main")
    box.git("alice", "switch develop")
    box.git("alice", "merge --no-ff hotfix/1.0.1 -m merge-hotfix-into-develop")
    hdev = rp(box, "develop")
    box.git("alice", "branch -d hotfix/1.0.1")
    box.snap(
        "The hotfix merges into BOTH main and develop, the full GitFlow history",
        note="main is tagged v1.0.1 and holds three releases and nothing else. develop carries every feature, the release fixes, and the hotfix. This is the busiest history in the collection, and that busyness is exactly the cost GitFlow asks you to weigh.",
    )

    # =================================================================
    # Render the seven figures and the state log.
    # =================================================================
    render(box, FIGURES, TABLES, mode="solo")

    # Tag the three releases on main. GitFlow tags every merge into main with the
    # version. The tags are created after rendering because `git for-each-ref`
    # reads a tag as a ref and the renderer would draw it identically to a branch
    # pointer, which would be a lie: a tag is a fixed label on one commit, not a
    # moving branch. The figures therefore show branch pointers only, the prose
    # names the tags, and the assertions below prove they sit where GitFlow puts
    # them.
    box.git("alice", f"tag v0.1.0 {c0}")
    box.git("alice", f"tag v1.0.0 {rmain}")
    box.git("alice", f"tag v1.0.1 {hmain}")

    # =================================================================
    # THE ORACLE: every claim the README makes, checked against real git.
    # =================================================================
    branches = set(box.read("alice").branches)

    # ---- claim 1: main and develop are the two long-lived branches ----------
    assert "main" in branches, "main is a long-lived branch and still exists"
    assert "develop" in branches, "develop is a long-lived branch and still exists"
    assert not any(
        b.startswith(("feature/", "release/", "hotfix/")) for b in branches
    ), "the feature, release and hotfix branches were temporary and are all deleted"
    assert rp(box, "main") != rp(box, "develop"), (
        "main and develop are two separate lines of history, not the same commit"
    )

    main_fp = first_parent(box, "main")
    develop_fp = first_parent(box, "develop")

    # ---- claim 2: a feature branch merged into develop, NOT into main -------
    assert m1 in develop_fp, (
        "the feature was integrated by a merge commit on develop's own line"
    )
    assert m1 not in main_fp, (
        "that feature merge never appears on main's own line of history"
    )
    assert is_ancestor(box, login_tip, "develop"), (
        "the feature's commits reached develop"
    )
    assert login_tip not in main_fp, (
        "the feature's commits were never committed directly onto main"
    )
    assert search_tip not in main_fp, "and neither was the second feature's work"

    # ---- claim 3: the release merged into BOTH main and develop -------------
    assert is_ancestor(box, release_tip, "main"), "the release was merged into main"
    assert is_ancestor(box, release_tip, "develop"), (
        "the release was merged back into develop as well"
    )
    assert parents(box, rmain)[1] == release_tip, (
        "main's release merge takes the release branch tip as its second parent"
    )
    assert parents(box, rdev)[1] == release_tip, (
        "develop's back-merge takes the same release branch tip as its second parent"
    )

    # ---- claim 4: main carries a version tag at the release -----------------
    all_tags = tags(box)
    assert "v1.0.0" in all_tags, "the release is tagged v1.0.0"
    assert tag_commit(box, "v1.0.0") == rmain, (
        "and that tag sits on main's release merge commit"
    )
    assert is_ancestor(box, "v1.0.0", "main"), "the v1.0.0 tag is on main's history"

    # ---- claim 5: main holds ONLY releases ----------------------------------
    # Every commit on main's own line is a tagged release. No feature or
    # stabilization work commit was ever committed directly onto main.
    tagged = {tag_commit(box, t) for t in all_tags}
    assert len(main_fp) == 3, (
        "main advanced exactly three times: the initial release and two merges to main"
    )
    assert all(c in tagged for c in main_fp), (
        "every commit on main's own line is a tagged release (v0.1.0, v1.0.0, v1.0.1)"
    )
    for work in (login_tip, search_tip, reports_tip, release_tip, hotfix_tip):
        assert work not in main_fp, (
            "no feature or stabilization work commit sits on main's own line"
        )

    # ---- git's own rendering, which the flat-edge renderer cannot show ------
    # A GitFlow merge into main is a merge of a branch that descends from main,
    # so its second-parent edge lies flat and the DAG figure cannot separate
    # main's own line from the branches it absorbed. git log --graph draws that
    # line in its own column, so it is the authoritative view of main's history.
    main_line = [
        line
        for line in box.git(
            "alice",
            "log --first-parent --oneline --decorate main",
            record=False,
        ).stdout.splitlines()
        if line
    ]
    assert len(main_line) == 3, (
        "main's own line is exactly three commits, git agrees with the assertions above"
    )
    assert all("tag:" in line for line in main_line), (
        "and every one of them carries a version tag: main holds only releases"
    )
    full_graph = [
        line
        for line in box.git(
            "alice",
            "log --graph --oneline --decorate --all",
            record=False,
        ).stdout.splitlines()
        if line
    ]
    assert any("\\" in line or "/" in line for line in full_graph), (
        "the full history really does fork and merge: git draws the bubbles"
    )

    # The overhead, in exact numbers read from git, not typed by hand.
    total_commits = int(box.git("alice", "rev-list --all --count", record=False).stdout)
    merge_commits = int(
        box.git("alice", "rev-list --all --merges --count", record=False).stdout
    )
    work_commits = total_commits - merge_commits - 1  # minus the initial release
    assert merge_commits == 7, (
        "seven merge commits: two per feature-and-release integration path"
    )
    assert work_commits == 8, "eight commits of real work made those seven merges"
    assert total_commits == 16, "sixteen commits of history for eight of work"

    # ---- the hotfix, the mirror of the release ------------------------------
    assert is_ancestor(box, hotfix_tip, "main"), "the hotfix reached main"
    assert is_ancestor(box, hotfix_tip, "develop"), "the hotfix also reached develop"
    assert parents(box, hmain)[1] == hotfix_tip, (
        "main's hotfix merge takes the hotfix branch tip as its second parent"
    )
    assert parents(box, hdev)[1] == hotfix_tip, (
        "develop's hotfix merge takes the same hotfix tip as its second parent"
    )
    assert "v1.0.1" in all_tags and tag_commit(box, "v1.0.1") == hmain, (
        "main is tagged v1.0.1 at the hotfix merge"
    )

    # =================================================================
    # TABLES, every cell measured above, none hand-typed.
    # =================================================================
    yn = lambda flag: "yes" if flag else "no"

    # main's own line of history: every commit a tagged release.
    by_commit_tag = {tag_commit(box, t): t for t in all_tags}
    with (TABLES / "main-only-releases.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["commit", "subject", "parents", "on main's own line", "tag"])
        for sha in main_fp:
            w.writerow(
                [
                    sha[:7],
                    subject(box, sha),
                    len(parents(box, sha)),
                    "yes",
                    by_commit_tag.get(sha, ""),
                ]
            )

    # release and hotfix: each merged into both long-lived branches.
    with (TABLES / "merges-to-both.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "branch",
                "branched from",
                "tip",
                "reached main",
                "reached develop",
                "tag on main",
            ]
        )
        w.writerow(
            [
                "release/1.0.0",
                "develop",
                release_tip[:7],
                yn(is_ancestor(box, release_tip, "main")),
                yn(is_ancestor(box, release_tip, "develop")),
                "v1.0.0",
            ]
        )
        w.writerow(
            [
                "hotfix/1.0.1",
                "main",
                hotfix_tip[:7],
                yn(is_ancestor(box, hotfix_tip, "main")),
                yn(is_ancestor(box, hotfix_tip, "develop")),
                "v1.0.1",
            ]
        )

    # the five branch classes and how each moves.
    with (TABLES / "branch-topology.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["branch", "lifetime", "branches from", "merges into", "exists at end"]
        )
        w.writerow(["main", "long-lived", "(root)", "(receives merges)", "yes"])
        w.writerow(["develop", "long-lived", "main", "(receives merges)", "yes"])
        w.writerow(["feature/*", "temporary", "develop", "develop", "no"])
        w.writerow(["release/*", "temporary", "develop", "main and develop", "no"])
        w.writerow(["hotfix/*", "temporary", "main", "main and develop", "no"])

    # git's own drawing of main's line and of the whole history.
    with (TABLES / "git-graph.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "line"])
        for line in main_line:
            w.writerow(["main --first-parent", line])
        for line in full_graph:
            w.writerow(["all --graph", line])

    box.cleanup()

    print(
        f"commits: {total_commits} total, {merge_commits} merges, {work_commits} work"
    )
    print("\nmain --first-parent:")
    for line in main_line:
        print("  " + line)
    print("\nlog --graph --all:")
    for line in full_graph:
        print("  " + line)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    tabs = sorted(p.name for p in TABLES.glob("*.csv"))
    assert len(figs) == 7, f"expected 7 figures, got {figs}"
    print(
        f"{len(figs)} figures, {len(tabs)} tables. Every GitFlow claim checked and passing."
    )
    print("figures:", figs)
    print("tags on main:", ", ".join(all_tags))
    print("main's own line (all tagged releases):", ", ".join(s[:7] for s in main_fp))


if __name__ == "__main__":
    main()
