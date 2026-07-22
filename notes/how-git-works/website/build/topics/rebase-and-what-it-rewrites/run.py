"""Rebase replays your commits onto a new base, and replaying makes new commits.

Builds a real stale branch, rebases it onto a main that moved, and reads the
result back out with plumbing. The same situation is then built twice more: once
handled with a merge instead, and once with a second ref still holding the
original commits, which is what a colleague's clone is.

show_unreachable=True keeps the abandoned originals in the figures, because they
really are still on disk. Every claim the README makes is asserted at the bottom
of this file.
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

APP_1 = "def main():\n    print('hello')\n"
APP_2 = "def main():\n    print('hello')\n\nTIMEOUT = 30\n"
APP_3 = "import logging\n\ndef main():\n    print('hello')\n\nTIMEOUT = 30\n"
APP_4 = "import logging\n\ndef main():\n    print('hello')\n\nTIMEOUT = 60\n"

PARSER_1 = "def parse(line):\n    return line.split()\n"
PARSER_2 = (
    "def parse(line):\n    return line.split()\n\n"
    "def test_parse():\n    assert parse('a b') == ['a', 'b']\n"
)


# ---- the shared situation ----------------------------------------------


def build_stale_branch(box: Sandbox, snapshots: bool = True):
    """Two commits on main, two on a branch off it, then main moves ahead.

    Leaves HEAD on `feature`, which is now stale: it hangs off a commit that is
    no longer the tip of main. This is the setup every rebase starts from. The
    commits on main are made here directly because this is a one-person sandbox.
    In real life they arrive in your repository when you pull.
    """
    box.commit("alice", "app.py", APP_1, "initial commit")
    box.commit("alice", "app.py", APP_2, "add timeout")
    if snapshots:
        box.snap(
            "the starting point",
            note="Two commits on main. This is the commit you are about to branch from.",
        )

    box.git("alice", "switch -c feature")
    box.commit("alice", "parser.py", PARSER_1, "parser skeleton")
    box.commit("alice", "parser.py", PARSER_2, "parser tests")
    if snapshots:
        box.snap(
            "two commits on your branch",
            note="Both of your commits were built on top of the old tip of main.",
        )

    box.git("alice", "switch main")
    box.commit("alice", "app.py", APP_3, "add logging")
    box.commit("alice", "app.py", APP_4, "raise the timeout")
    if snapshots:
        box.snap(
            "main moved on while you worked",
            note="Your branch is stale now. It hangs off a commit that is no longer where main is.",
        )

    box.git("alice", "switch feature")


# ---- reading the truth back out ----------------------------------------


def _run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _commits_in(repo: Path, rev_range: str) -> list[tuple[str, str]]:
    """(short sha, subject) for a revision range, oldest first."""
    out = _run(repo, ["log", "--reverse", "--pretty=%h\x1f%s", rev_range])
    return [tuple(line.split("\x1f")) for line in out.splitlines() if line]


def _tree_of(repo: Path, rev: str) -> str:
    """The tree object a commit points at: its snapshot of the whole project.

    One rev per call. The two-argument form, `git rev-parse a^{tree} b^{tree}`,
    does not do what it looks like it does.
    """
    return _run(repo, ["rev-parse", "--short", f"{rev}^{{tree}}"])


def _blob_of(repo: Path, rev: str, path: str) -> str:
    """The object id of one file as of one commit."""
    return _run(repo, ["rev-parse", "--short", f"{rev}:{path}"])


def _patch_id(repo: Path, rev: str) -> str:
    """A fingerprint of the DIFF a commit introduces, ignoring what it sits on.

    This is git's own answer to 'is this the same change', and it is what
    `git cherry` and rebase itself use to spot an already-applied commit.
    """
    diff = subprocess.run(
        ["git", "show", "--format=", "-p", rev],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout
    out = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=repo,
        input=diff,
        capture_output=True,
        text=True,
    ).stdout.split()
    return out[0] if out else ""


def _object_type(repo: Path, rev: str) -> str:
    """`commit` if the object is still on disk, empty if git cannot find it."""
    proc = subprocess.run(
        ["git", "cat-file", "-t", rev], cwd=repo, capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _branches_containing(repo: Path, rev: str) -> list[str]:
    out = _run(repo, ["branch", "--contains", rev, "--format=%(refname:short)"])
    return [b for b in out.splitlines() if b]


def _in_reflog(repo: Path, rev: str) -> bool:
    full = _run(repo, ["rev-parse", rev])
    return full in _run(repo, ["rev-list", "--reflog"]).split()


def _parent_counts(repo: Path, rev: str) -> list[int]:
    """How many parents each commit reachable from `rev` has."""
    out = _run(repo, ["rev-list", "--parents", rev])
    return [len(line.split()) - 1 for line in out.splitlines() if line]


def _n_parents(repo: Path, rev: str) -> int:
    """How many parents one commit has. Two means it is a merge commit."""
    return len(_run(repo, ["rev-list", "--parents", "-1", rev]).split()) - 1


def _raw_object(repo: Path, rev: str) -> list[str]:
    """The commit object exactly as git stored it, one line per field."""
    return _run(repo, ["cat-file", "-p", rev]).splitlines()


def _field(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key + " "):
            return line[len(key) + 1 :]
    return ""


def main():
    clear(FIGURES, TABLES)

    # ---- 1. the rebase ---------------------------------------------------
    box = Sandbox(people=("alice",), show_unreachable=True)
    build_stale_branch(box)
    alice = box.paths["alice"]

    originals = _commits_in(alice, "main..feature")
    orig_raw = _raw_object(alice, originals[0][0])

    rebase = box.git("alice", "rebase main")
    # `git rebase` is on gitviz's expected-failure list, so a non-zero exit
    # would not raise. Check it here or a broken rebase passes silently.
    assert rebase.returncode == 0, f"the rebase must succeed:\n{rebase.stderr}"
    final = box.snap(
        "after git rebase main",
        note="Your two commits were replayed onto the new main, as new commits with new hashes. The originals are still on disk, with nothing pointing at them.",
    )
    render(box, FIGURES, TABLES)

    rebased = _commits_in(alice, "main..feature")
    rebased_raw = _raw_object(alice, rebased[0][0])

    # ---- 2. the same situation, handled with a merge ----------------------
    merged = Sandbox(people=("alice",), show_unreachable=True)
    build_stale_branch(merged, snapshots=False)
    merged.snap("the same stale branch")  # not drawn; it seeds what counts as new
    m_alice = merged.paths["alice"]
    merging = merged.git("alice", "merge main --no-edit")
    assert merging.returncode == 0, f"the merge must succeed:\n{merging.stderr}"
    merge_snap = merged.snap(
        "git merge main, instead of rebasing",
        note="Your two commits keep the hashes they always had. One new commit, with two parents, joins the two histories.",
    )
    mxs, mys = layout(merged.snapshots)
    draw(merge_snap, mxs, mys, FIGURES / "merge-instead.png", mode="solo")
    kept = _commits_in(m_alice, "main..feature")

    # ---- 3. the golden rule, violated ------------------------------------
    # `colleague` is a ref that still names the original commits, which is
    # exactly what somebody else's clone of your branch is.
    shared = Sandbox(people=("alice",), show_unreachable=True)
    build_stale_branch(shared, snapshots=False)
    shared.git("alice", "branch colleague")
    s_alice = shared.paths["alice"]
    reb = shared.git("alice", "rebase main")
    assert reb.returncode == 0, f"the rebase must succeed:\n{reb.stderr}"
    shared.snap("you rebased, they did not")
    joined = shared.git("alice", "merge colleague --no-edit")
    assert joined.returncode == 0, f"the merge must succeed:\n{joined.stderr}"
    dup_snap = shared.snap(
        "the colleague merges the branch they still have",
        note="Every change is now in the history twice. Git had no way to know the two chains were the same two changes.",
    )
    dxs, dys = layout(shared.snapshots)
    draw(dup_snap, dxs, dys, FIGURES / "duplicated-work.png", mode="solo")
    after_merge = _commits_in(s_alice, "main..feature")

    # ---- tables ----------------------------------------------------------
    with (TABLES / "rebase-hashes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subject",
                "original commit",
                "rebased commit",
                "original parent",
                "new parent",
            ]
        )
        for (o_sha, subject), (n_sha, _) in zip(originals, rebased):
            writer.writerow(
                [
                    subject,
                    o_sha,
                    n_sha,
                    _field(_raw_object(alice, o_sha), "parent")[:7],
                    _field(_raw_object(alice, n_sha), "parent")[:7],
                ]
            )

    with (TABLES / "same-change-new-commit.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "subject",
                "patch-id before",
                "patch-id after",
                "parser.py blob before",
                "parser.py blob after",
            ]
        )
        for (o_sha, subject), (n_sha, _) in zip(originals, rebased):
            writer.writerow(
                [
                    subject,
                    _patch_id(alice, o_sha)[:7],
                    _patch_id(alice, n_sha)[:7],
                    _blob_of(alice, o_sha, "parser.py"),
                    _blob_of(alice, n_sha, "parser.py"),
                ]
            )

    with (TABLES / "what-changed-in-the-commit.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["field", "original commit", "after the rebase", "changed"])
        rows = [
            (key, _field(orig_raw, key), _field(rebased_raw, key))
            for key in ("tree", "parent", "author", "committer")
        ]
        rows.append(("message", orig_raw[-1], rebased_raw[-1]))
        rows.append(("commit id", originals[0][0], rebased[0][0]))
        for key, before, after in rows:
            # The comparison is on the full value; only the display is abbreviated.
            changed = "yes" if before != after else "no"
            short = 7 if key in ("tree", "parent") else 44
            writer.writerow([key, before[:short], after[:short], changed])

    with (TABLES / "originals-on-disk.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "original commit",
                "subject",
                "object on disk",
                "branches containing it",
                "in the reflog",
            ]
        )
        for sha, subject in originals:
            writer.writerow(
                [
                    sha,
                    subject,
                    _object_type(alice, sha) or "(gone)",
                    ", ".join(_branches_containing(alice, sha)) or "(none)",
                    "yes" if _in_reflog(alice, sha) else "no",
                ]
            )

    with (TABLES / "rebase-vs-merge.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question", "git rebase main", "git merge main"])
        writer.writerow(
            [
                "commits reachable from feature",
                len(_parent_counts(alice, "feature")),
                len(_parent_counts(m_alice, "feature")),
            ]
        )
        writer.writerow(
            [
                "commits with two parents",
                sum(1 for n in _parent_counts(alice, "feature") if n == 2),
                sum(1 for n in _parent_counts(m_alice, "feature") if n == 2),
            ]
        )
        writer.writerow(
            [
                "your two original commit ids still on the branch",
                sum(1 for sha, _ in originals if _branches_containing(alice, sha)),
                sum(1 for sha, _ in originals if _branches_containing(m_alice, sha)),
            ]
        )
        writer.writerow(
            ["new commit ids created", len(rebased), len(kept) - len(originals)]
        )
        writer.writerow(
            [
                "tree of the branch tip (the resulting files)",
                _tree_of(alice, "feature"),
                _tree_of(m_alice, "feature"),
            ]
        )

    with (TABLES / "duplicated-work.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["commit", "subject", "parents"])
        for sha, subject in after_merge:
            writer.writerow([sha, subject, _n_parents(s_alice, sha)])

    # ---- the oracle: every claim the README makes, checked ----------------
    orig_shas = [sha for sha, _ in originals]
    new_shas = [sha for sha, _ in rebased]

    # The rebase replayed the same two changes, in the same order.
    assert len(rebased) == 2, "the branch still carries exactly two commits"
    assert [s for _, s in rebased] == [s for _, s in originals], (
        "the same two changes, in the same order"
    )

    # ... as NEW commits. Not one hash survived.
    assert set(new_shas).isdisjoint(orig_shas), (
        "no rebased commit reuses an original commit id"
    )

    # The change each commit carries is byte-identical. Same patch-id, and the
    # file the branch touched is literally the same blob object.
    for (o_sha, subject), (n_sha, _) in zip(originals, rebased):
        assert _patch_id(alice, o_sha) == _patch_id(alice, n_sha) != "", (
            f"the rebased '{subject}' introduces the identical diff"
        )
        assert _blob_of(alice, o_sha, "parser.py") == _blob_of(
            alice, n_sha, "parser.py"
        ), f"the rebased '{subject}' stores the identical parser.py blob"

    # The snapshot, however, is NOT identical, and it cannot be: the replayed
    # commit sits on top of main's new work, so its tree contains that too.
    # This is a second, independent reason the hash has to change.
    for (o_sha, _), (n_sha, _) in zip(originals, rebased):
        assert _tree_of(alice, o_sha) != _tree_of(alice, n_sha), (
            "the replayed commit snapshots the whole project, which now includes main's commits"
        )
    assert _blob_of(alice, rebased[0][0], "app.py") == _blob_of(
        alice, "main", "app.py"
    ), (
        "the app.py inside the rebased commit is main's version, not the one your branch had"
    )
    assert _blob_of(alice, originals[0][0], "app.py") != _blob_of(
        alice, "main", "app.py"
    ), "the original commit predates main's changes to app.py"

    # What did NOT change: who wrote it, when, and what they called it. The new
    # commit id comes from the tree and the parent, nothing else.
    for key in ("author", "committer"):
        assert _field(orig_raw, key) == _field(rebased_raw, key) != "", (
            f"the rebase left the {key} line, timestamp and all, untouched"
        )
    assert orig_raw[-1] == rebased_raw[-1], "the commit message is unchanged"
    assert _field(orig_raw, "parent") != _field(rebased_raw, "parent"), (
        "the replayed commit has a different parent"
    )

    # The originals are still on disk, just unreferenced. This is what the
    # figure draws, and it is the reason show_unreachable exists.
    for sha, _ in originals:
        assert _object_type(alice, sha) == "commit", (
            f"{sha} is still a commit object on disk after the rebase"
        )
        assert _branches_containing(alice, sha) == [], (
            f"no branch points at {sha} any more"
        )
        assert _in_reflog(alice, sha), f"{sha} is still findable through the reflog"
        assert sha in final.repos["alice"].commits, (
            f"{sha} is in the figure, drawn as the orphan it now is"
        )

    # The rebased history is linear. The merged alternative is not.
    assert max(_parent_counts(alice, "feature")) == 1, (
        "after the rebase, no commit on the branch has two parents"
    )
    assert sum(1 for n in _parent_counts(m_alice, "feature") if n == 2) == 1, (
        "the merge created exactly one commit with two parents"
    )
    assert _n_parents(m_alice, "feature") == 2, "and it is the tip of the branch"

    # Merge keeps your commits. Rebase does not.
    assert set(orig_shas) <= {sha for sha, _ in kept}, (
        "after a merge, your original commit ids are still on the branch, unchanged"
    )
    assert len(kept) == len(originals) + 1, (
        "a merge adds exactly one commit to the branch: the merge commit"
    )

    # And yet both routes end at the same files. Rebase and merge differ in the
    # history they leave behind, not in the result.
    assert _tree_of(alice, "feature") == _tree_of(m_alice, "feature") != "", (
        "rebase and merge produce the identical final snapshot"
    )

    # The golden rule, in hashes. A colleague still holding the originals merges,
    # and every change is now in the history twice.
    subjects = [s for _, s in after_merge]
    for _, subject in originals:
        assert subjects.count(subject) == 2, (
            f"'{subject}' is now in the history twice: once as the original, "
            "once as the rebased copy"
        )
    assert set(orig_shas) <= {sha for sha, _ in after_merge}, (
        "the originals came back, because the colleague's ref still named them"
    )
    assert sum(1 for _, s in after_merge for _ in [1] if s.startswith("Merge")) == 1, (
        "one merge commit, joining a history to a rewritten copy of itself"
    )

    print(
        f"rebase: {len(originals)} commits replayed as {len(new_shas)} new ids, "
        f"{len(_parent_counts(alice, 'feature'))} commits on the branch, 0 with two parents."
    )
    print(
        f"merge:  originals kept ({', '.join(orig_shas)}), "
        f"{len(_parent_counts(m_alice, 'feature'))} commits on the branch, 1 with two parents."
    )
    print(
        f"both:   the branch tip snapshots the identical tree {_tree_of(alice, 'feature')}."
    )
    print(
        f"golden rule violated: {len(after_merge)} commits on the branch, every change present twice."
    )
    print(
        f"{len(list(FIGURES.glob('*.png')))} figures, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All assertions passed."
    )

    box.cleanup()
    merged.cleanup()
    shared.cleanup()


if __name__ == "__main__":
    main()
