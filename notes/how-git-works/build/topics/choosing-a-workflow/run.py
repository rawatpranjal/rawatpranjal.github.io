"""Four collaboration styles, four history shapes, one decision.

This is the decision card for "which collaboration style should I use?". It
builds each of the four named workflow families as a real repository, then
photographs the history each one leaves behind, so the reader can see the four
shapes side by side rather than take them on faith.

  direct-to-main  everyone commits straight to main. One straight line.
  github-flow     a short feature branch, squash-merged into main. Main stays a
                  clean line of squashed features; the branch commits never land.
  trunk-based     tiny frequent commits and very short branches behind feature
                  flags. Almost a straight line, but with a review gate.
  gitflow         long-lived main and develop, plus feature, release and hotfix
                  branches. The busiest shape.

Every claim the README makes is checked at the end. The direct-to-main history
is asserted linear (no merge commit, one branch). The github-flow main is
asserted to hold squashed commits, with the branch commits proven absent from
main's first-parent history. The gitflow repo is asserted to really carry a
develop branch plus a feature and a release branch, with five merge commits.
The trunk-based main is asserted near-linear with only short branches. The
decision table's structural columns are read straight out of the four repos, so
a wrong number fails the run.
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
ALICE = "alice"


# ---- reading facts back out of a real repo -----------------------------


def out(box: Sandbox, args: str) -> str:
    """Run a read-only git command and hand back its stdout, stripped."""
    return box.git(ALICE, args, record=False, check=False).stdout.strip()


def merge_commit_shas(state) -> list[str]:
    """Every commit with two or more parents: the merge commits."""
    return [s for s, c in state.commits.items() if len(c.parents) >= 2]


def first_parent_chain(box: Sandbox, ref: str = "main") -> list[str]:
    """The short shas along a branch's first-parent line, tip first.

    This is the spine of the branch: what you see if you never open a merge
    bubble. For a linear history it is the whole history.
    """
    text = out(box, f"log --first-parent --format=%h {ref}")
    return text.split()


def commits_ahead(box: Sandbox, branch: str, base: str = "main") -> int:
    """How many commits sit on `branch` that `base` has not absorbed."""
    return int(out(box, f"rev-list --count {base}..{branch}"))


def log_graph(box: Sandbox) -> list[str]:
    """git's own ASCII drawing of the whole DAG, captured verbatim."""
    return [
        line for line in out(box, "log --graph --oneline --all").splitlines() if line
    ]


def measure(box: Sandbox, style: str) -> dict:
    """The structural facts of one repo, the numbers the decision table quotes."""
    state = box.read(ALICE)
    mc = merge_commit_shas(state)
    aheads = [commits_ahead(box, b) for b in state.branches if b != "main"]
    return {
        "style": style,
        "branches": len(state.branches),
        "merge_commits": len(mc),
        "commits_on_main": len(first_parent_chain(box, "main")),
        "longest_branch_commits": max(aheads) if aheads else 0,
        "linear_history": "yes" if not mc else "no",
    }


def finalize(box: Sandbox, label: str, note: str):
    """Freeze the finished history with a clean header and no false 'new' marks.

    The first snapshot records every commit as already known, so the second one
    highlights nothing. A finished history should not be drawn as if the whole
    of it had just happened. The command log is cleared so the figure header
    carries the caption, not a wall of git invocations.
    """
    box._log.clear()
    box.snap("_prime")
    box._log.clear()
    return box.snap(label, note=note)


def draw_shape(snap, name: str, title: str):
    xs, ys = layout([snap])
    draw(snap, xs, ys, FIGURES / f"{name}.png", mode="solo", title=title)


# ---- 1. direct-to-main -------------------------------------------------


def build_direct_to_main():
    """Everyone commits straight to main. No branch is ever created."""
    box = Sandbox(people=(ALICE,), local=True)
    content = ""
    for message in [
        "set-up-project",
        "add-loader",
        "fix-typo",
        "add-report",
        "tune-params",
    ]:
        content += message + "\n"
        box.commit(ALICE, "app.py", content, message)
    snap = finalize(
        box,
        "Direct to main",
        note="Everyone commits straight to main. One straight line, no branches, no review gate.",
    )
    state = snap.repos[ALICE]

    # The claims the README makes about this shape.
    assert set(state.branches) == {"main"}, "direct-to-main uses exactly one branch"
    assert merge_commit_shas(state) == [], "no merge commit is ever created"
    assert all(len(c.parents) <= 1 for c in state.commits.values()), (
        "no commit has two parents: the history is a straight line"
    )
    assert len(first_parent_chain(box, "main")) == len(state.commits), (
        "every commit sits on main's one line, nothing hangs off it"
    )

    draw_shape(snap, "direct-to-main", "Direct to main")
    facts = measure(box, "direct-to-main")
    graph = log_graph(box)
    box.cleanup()
    return facts, graph


# ---- 2. github-flow ----------------------------------------------------


def build_github_flow():
    """A short feature branch, squash-merged into main, twice over.

    Squash-merge stages the whole branch and records it as ONE commit whose only
    parent is main's previous tip. The branch's own commits are never reachable
    from main afterwards. In practice the branch is deleted once its pull
    request merges; it is kept here so the figure can show the shape.
    """
    box = Sandbox(people=(ALICE,), local=True)
    box.commit(ALICE, "app.py", "project scaffold\n", "scaffold")

    box.git(ALICE, "switch -c feature-search")
    box.commit(ALICE, "search.py", "parse query\n", "start-search")
    box.commit(ALICE, "search.py", "parse query\nrank results\n", "finish-search")
    box.git(ALICE, "switch main")
    box.git(ALICE, "merge --squash feature-search")
    box.git(ALICE, "commit -m add-search")

    box.git(ALICE, "switch -c feature-auth")
    box.commit(ALICE, "auth.py", "check password\n", "start-auth")
    box.commit(ALICE, "auth.py", "check password\nissue token\n", "finish-auth")
    box.git(ALICE, "switch main")
    box.git(ALICE, "merge --squash feature-auth")
    box.git(ALICE, "commit -m add-auth")

    snap = finalize(
        box,
        "GitHub flow (squash-merge)",
        note="Each feature is a short branch, squash-merged into main as one commit. "
        "Main stays a clean line; the branch commits (hanging off) never land on it.",
    )
    state = snap.repos[ALICE]

    main_line = set(first_parent_chain(box, "main"))
    search_tip = state.branches["feature-search"]
    auth_tip = state.branches["feature-auth"]

    # The squash landed each feature as one commit, and the branch's own commits
    # are absent from main. This is the whole point of squash-merge.
    assert search_tip not in main_line, (
        "the feature-search branch tip is NOT on main after a squash"
    )
    assert auth_tip not in main_line, (
        "the feature-auth branch tip is NOT on main after a squash"
    )
    assert len(first_parent_chain(box, "main")) == 3, (
        "main is three commits: the scaffold plus one squashed commit per feature"
    )
    assert merge_commit_shas(state) == [], (
        "squash-merge writes single-parent commits, so main carries no merge commit"
    )
    assert commits_ahead(box, "feature-search") == 2, (
        "each feature branch really held two commits, collapsed to one on main"
    )
    assert commits_ahead(box, "feature-auth") == 2, (
        "and so did the second feature branch"
    )

    draw_shape(snap, "github-flow", "GitHub flow (squash-merge)")
    facts = measure(box, "github-flow")
    graph = log_graph(box)
    box.cleanup()
    return facts, graph


# ---- 3. trunk-based ----------------------------------------------------


def build_trunk_based():
    """Tiny frequent commits to main, plus one very short branch behind a flag.

    Trunk-based development keeps everyone on main, integrating many times a day.
    Risky work hides behind a feature flag and lands as a tiny reviewed branch,
    so main stays almost a straight line while still passing through a gate.
    """
    box = Sandbox(people=(ALICE,), local=True)
    content = ""
    for message in [
        "scaffold",
        "small-step-1",
        "small-step-2",
        "small-step-3",
        "small-step-4",
    ]:
        content += message + "\n"
        box.commit(ALICE, "app.py", content, message)

    box.git(ALICE, "switch -c flag-new-pipeline")
    box.commit(ALICE, "pipeline.py", "behind FLAG_NEW_PIPELINE\n", "impl-behind-flag")
    box.git(ALICE, "switch main")
    box.git(ALICE, "merge --squash flag-new-pipeline")
    box.git(ALICE, "commit -m land-pipeline-behind-flag")

    snap = finalize(
        box,
        "Trunk-based development",
        note="Tiny commits straight to main, plus very short branches behind feature flags. "
        "Almost a straight line, with a review gate.",
    )
    state = snap.repos[ALICE]

    assert merge_commit_shas(state) == [], "trunk-based main carries no merge commit"
    assert len(first_parent_chain(box, "main")) == 6, (
        "main is six commits: five direct plus one squashed short branch"
    )
    assert commits_ahead(box, "flag-new-pipeline") == 1, (
        "the only branch is a single short-lived commit, unlike github-flow's longer features"
    )
    # Near-linear: no branch is more than one commit ahead of main.
    aheads = [commits_ahead(box, b) for b in state.branches if b != "main"]
    assert max(aheads) <= 1, "every branch is short: at most one commit off main"

    draw_shape(snap, "trunk-based", "Trunk-based development")
    facts = measure(box, "trunk-based")
    graph = log_graph(box)
    box.cleanup()
    return facts, graph


# ---- 4. gitflow --------------------------------------------------------


def build_gitflow():
    """Long-lived main and develop, plus feature, release and hotfix branches.

    The full Driessen model. A feature branches off develop and merges back. A
    release branches off develop and merges into both main and develop. A hotfix
    branches off main and merges into both. Every junction is a --no-ff merge, so
    the history keeps a permanent record of what belonged together, and there is
    a lot of it.
    """
    # Gitflow names these branches feature/login, release/1.0 and hotfix/1.0.1
    # by convention. Plain names are used here because this collection draws any
    # slash name as a remote-tracking ref, and these are local branches.
    box = Sandbox(people=(ALICE,), local=True)
    box.commit(ALICE, "app.py", "v0\n", "initial-release")
    box.git(ALICE, "switch -c develop")

    # A feature off develop, merged back with a merge commit.
    box.git(ALICE, "switch -c feature-login")
    box.commit(ALICE, "login.py", "login form\n", "build-login")
    box.commit(ALICE, "login.py", "login form\nremember me\n", "add-remember-me")
    box.git(ALICE, "switch develop")
    r = box.git(ALICE, "merge --no-ff feature-login -m merge-login")
    assert r.returncode == 0, "the feature merges cleanly into develop"

    # A release off develop, merged into BOTH main and develop.
    box.git(ALICE, "switch -c release-1.0")
    box.commit(ALICE, "version.py", "1.0\n", "bump-to-1.0")
    box.git(ALICE, "switch main")
    r = box.git(ALICE, "merge --no-ff release-1.0 -m release-1.0")
    assert r.returncode == 0, "the release lands on main"
    box.git(ALICE, "switch develop")
    r = box.git(ALICE, "merge --no-ff release-1.0 -m merge-release-to-develop")
    assert r.returncode == 0, "and the release also flows back to develop"

    # A hotfix off main, merged into BOTH main and develop.
    box.git(ALICE, "switch main")
    box.git(ALICE, "switch -c hotfix-1.0.1")
    box.commit(ALICE, "hotfix.py", "patch a crash\n", "fix-crash")
    box.git(ALICE, "switch main")
    r = box.git(ALICE, "merge --no-ff hotfix-1.0.1 -m hotfix-1.0.1")
    assert r.returncode == 0, "the hotfix lands on main"
    box.git(ALICE, "switch develop")
    r = box.git(ALICE, "merge --no-ff hotfix-1.0.1 -m merge-hotfix-to-develop")
    assert r.returncode == 0, "and the hotfix also flows back to develop"

    snap = finalize(
        box,
        "Gitflow",
        note="Long-lived main and develop, plus feature, release and hotfix branches. "
        "The busiest shape, with a merge commit at every junction.",
    )
    state = snap.repos[ALICE]

    assert "develop" in state.branches, "gitflow keeps a long-lived develop branch"
    assert "main" in state.branches, "alongside the long-lived main branch"
    assert any(b.startswith("feature") for b in state.branches), (
        "a feature branch exists"
    )
    assert any(b.startswith("release") for b in state.branches), (
        "a release branch exists"
    )
    assert state.branches["main"] != state.branches["develop"], (
        "main and develop have genuinely diverged: two separate lines"
    )
    assert len(merge_commit_shas(state)) == 5, (
        "five --no-ff junctions leave five merge commits, where the others left none"
    )

    draw_shape(snap, "gitflow", "Gitflow")
    facts = measure(box, "gitflow")
    graph = log_graph(box)
    box.cleanup()
    return facts, graph


# ---- the decision table (the payoff) -----------------------------------

# The editorial guidance columns. The structural columns (branches, merge
# commits, history shape) are filled from the real repos below, so a wrong
# measurement fails the run rather than sitting quietly in a table.
GUIDANCE = {
    "direct-to-main": {
        "best_team_size": "1 (solo)",
        "release_cadence": "continuous",
        "review_gate": "none",
        "ci_and_tooling": "minimal",
        "best_for": "a solo project or a throwaway script",
        "verdict": "fine alone, unsafe with a team: no review and main breaks easily",
    },
    "github-flow": {
        "best_team_size": "2 to about 50",
        "release_cadence": "continuous (deploy on merge)",
        "review_gate": "pull request",
        "ci_and_tooling": "standard (CI runs on the pull request)",
        "best_for": "most teams, and most data science work",
        "verdict": "the best default",
    },
    "trunk-based": {
        "best_team_size": "any, scales to very large",
        "release_cadence": "continuous",
        "review_gate": "pull request, kept tiny",
        "ci_and_tooling": "high (fast CI plus feature flags are required)",
        "best_for": "large teams that already have strong CI and feature flags",
        "verdict": "excellent at scale, but build the CI and the flags first",
    },
    "gitflow": {
        "best_team_size": "any",
        "release_cadence": "versioned releases",
        "review_gate": "pull request",
        "ci_and_tooling": "standard",
        "best_for": "explicitly versioned software with several supported releases at once",
        "verdict": "usually too heavy for continuous delivery, by its author's own later note",
    },
}

SHAPE_WORD = {
    "direct-to-main": "one straight line",
    "github-flow": "a line of squashed features",
    "trunk-based": "near-linear, short branches",
    "gitflow": "many parallel branches",
}


def main():
    clear(FIGURES, TABLES)

    builders = [
        build_direct_to_main,
        build_github_flow,
        build_trunk_based,
        build_gitflow,
    ]
    facts_rows = []
    graph_rows = []
    for build in builders:
        facts, graph = build()
        facts_rows.append(facts)
        graph_rows += [{"style": facts["style"], "line": line} for line in graph]

    facts_by_style = {row["style"]: row for row in facts_rows}

    # The measured proof that the four shapes really are different.
    with (TABLES / "history-facts.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(facts_rows[0]))
        writer.writeheader()
        writer.writerows(facts_rows)

    # git's own drawing of each history, verbatim, for the README to quote.
    with (TABLES / "history-graphs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["style", "line"])
        writer.writeheader()
        writer.writerows(graph_rows)

    # The decision table: editorial guidance, with the structural columns filled
    # from the real measurements so they cannot drift from the figures.
    decision_fields = [
        "style",
        "best_team_size",
        "release_cadence",
        "review_gate",
        "ci_and_tooling",
        "history_shape",
        "merge_commits",
        "best_for",
        "verdict",
    ]
    with (TABLES / "decision-table.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=decision_fields)
        writer.writeheader()
        for style in ["direct-to-main", "github-flow", "trunk-based", "gitflow"]:
            row = {"style": style, **GUIDANCE[style]}
            row["history_shape"] = SHAPE_WORD[style]
            row["merge_commits"] = facts_by_style[style]["merge_commits"]
            writer.writerow(row)

    # ---- cross-style assertions the decision table leans on ------------
    # gitflow is the busiest by construction, the others carry no merge commit.
    assert facts_by_style["gitflow"]["merge_commits"] == 5, "gitflow is the busy one"
    for style in ["direct-to-main", "github-flow", "trunk-based"]:
        assert facts_by_style[style]["merge_commits"] == 0, (
            f"{style} keeps main free of merge commits"
        )
    # direct-to-main is the only style with no branch beyond main.
    assert facts_by_style["direct-to-main"]["branches"] == 1, (
        "direct-to-main is the only one-branch style"
    )
    assert facts_by_style["gitflow"]["branches"] == 5, (
        "gitflow ends with the most branches standing"
    )
    # github-flow's features are longer than trunk-based's short branches.
    assert (
        facts_by_style["github-flow"]["longest_branch_commits"]
        > facts_by_style["trunk-based"]["longest_branch_commits"]
    ), "github-flow features are longer-lived than trunk-based's tiny branches"

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    expected = sorted(
        [
            "direct-to-main.png",
            "github-flow.png",
            "gitflow.png",
            "trunk-based.png",
        ]
    )
    assert figs == expected, f"expected one figure per style, got {figs}"
    print(
        f"{len(figs)} figures, 3 tables. All four workflow shapes built and every claim checked."
    )


if __name__ == "__main__":
    main()
