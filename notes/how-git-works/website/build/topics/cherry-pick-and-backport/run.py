"""Backport one fix to an old release line with git cherry-pick.

main has moved on to version 2, but customers still run release/v1, a long-lived
branch cut from an older commit. A bug is fixed on main with one commit, F. That
same fix has to land on release/v1 as well. `git cherry-pick F`, run from
release/v1, applies just that one commit's change onto the old line.

The backported commit F' carries the identical change but a DIFFERENT hash,
because a commit's hash covers its parent and its whole snapshot, and both differ
on the old line. This is the same mechanism rebase leans on: same change, new
parent, new id. Everything the README claims is asserted at the bottom of this
file, so a wrong picture fails the run.
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

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# app.py holds the buggy function and, after the fix, the corrected one. The fix
# is the only change to this file anywhere in the history, so cherry-pick applies
# it cleanly onto the old line: the surrounding lines match byte for byte.
APP_BUGGY = "def parse_amount(text):\n    return int(text)\n"
APP_FIXED = 'def parse_amount(text):\n    return int(text.strip().lstrip("$"))\n'

CLI = 'from app import parse_amount\n\nprint(parse_amount("42"))\n'
CURRENCIES = 'CURRENCIES = ["USD", "EUR", "GBP"]\n'
VERSION_1 = "1.0\n"
VERSION_2 = "2.0\n"


# ---- reading the truth back out ----------------------------------------


def _run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _short(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", "--short", rev])


def _subject(repo: Path, rev: str) -> str:
    return _run(repo, ["log", "-1", "--pretty=%s", rev])


def _tree_of(repo: Path, rev: str) -> str:
    """The tree object a commit points at: its snapshot of the whole project."""
    return _run(repo, ["rev-parse", "--short", f"{rev}^{{tree}}"])


def _blob_of(repo: Path, rev: str, path: str) -> str:
    """The object id of one file as of one commit."""
    return _run(repo, ["rev-parse", "--short", f"{rev}:{path}"])


def _patch_id(repo: Path, rev: str) -> str:
    """A fingerprint of the DIFF a commit introduces, ignoring what it sits on.

    This is git's own answer to 'is this the same change', and it is what
    `git cherry` and cherry-pick itself use to spot an already-applied commit.
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


def _raw_object(repo: Path, rev: str) -> list[str]:
    """The commit object exactly as git stored it, one line per field."""
    return _run(repo, ["cat-file", "-p", rev]).splitlines()


def _field(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key + " "):
            return line[len(key) + 1 :]
    return ""


def _branches_containing(repo: Path, rev: str) -> list[str]:
    out = _run(repo, ["branch", "--contains", rev, "--format=%(refname:short)"])
    return [b for b in out.splitlines() if b]


def _n_parents(repo: Path, rev: str) -> int:
    """How many parents one commit has. Two means it is a merge commit."""
    return len(_run(repo, ["rev-list", "--parents", "-1", rev]).split()) - 1


def _count_commits(repo: Path, rev: str) -> int:
    return len([s for s in _run(repo, ["rev-list", rev]).splitlines() if s])


def _n_merges(repo: Path, rev: str) -> int:
    return len([s for s in _run(repo, ["rev-list", "--merges", rev]).splitlines() if s])


def main():
    clear(FIGURES, TABLES)

    box = Sandbox(people=("alice",))
    alice = box.paths["alice"]

    # ---- 1. the shared history, and the v1 release ----------------------
    box.commit("alice", "app.py", APP_BUGGY, "add-parser")
    box.commit("alice", "cli.py", CLI, "add-cli")
    box.commit("alice", "VERSION", VERSION_1, "release-1.0")
    # Cut the long-lived release line here, at the 1.0 release commit.
    box.git("alice", "branch release/v1")
    box.snap(
        "version 1 is released",
        note="release/v1 is cut from main at the 1.0 release. Both branches point at the same commit for now.",
    )

    # ---- 2. main moves on to version 2 ----------------------------------
    box.commit("alice", "currencies.py", CURRENCIES, "v2-add-currencies")
    box.commit("alice", "VERSION", VERSION_2, "v2-bump-version")
    box.snap(
        "main moves on to version 2",
        note="main has advanced two commits. release/v1 is untouched: customers still run it.",
    )

    # ---- 3. the bug is fixed on main, with commit F ---------------------
    box.commit("alice", "app.py", APP_FIXED, "fix-amount-parsing")
    box.snap(
        "the bug is fixed on main",
        note="F fixes parse_amount on main. The same bug is still live on release/v1.",
    )
    f_full = _run(alice, ["rev-parse", "HEAD"])
    f_short = _short(alice, "HEAD")

    # ---- 4. backport: cherry-pick F onto release/v1 ---------------------
    box.git("alice", "switch release/v1")
    # cherry-pick is on gitviz's expected-failure list, so a non-zero exit would
    # not raise. Check it here or a broken pick passes silently.
    pick = box.git("alice", f"cherry-pick {f_short}", check=False)
    assert pick.returncode == 0, f"the cherry-pick must apply cleanly:\n{pick.stderr}"
    box.snap(
        "cherry-pick the fix onto release/v1",
        note="F' carries the identical fix but a new id. main keeps F, release/v1 gets F', and the branches stay distinct.",
    )
    render(box, FIGURES, TABLES, mode="solo")

    fp_full = _run(alice, ["rev-parse", "release/v1"])
    fp_short = _short(alice, "release/v1")

    f_raw = _raw_object(alice, f_full)
    fp_raw = _raw_object(alice, fp_full)

    # ---- tables ---------------------------------------------------------
    with (TABLES / "backport-hashes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "subject",
                "commit on main",
                "backported to v1",
                "parent on main",
                "parent on v1",
            ]
        )
        w.writerow(
            [
                _subject(alice, f_full),
                f_short,
                fp_short,
                _field(f_raw, "parent")[:7],
                _field(fp_raw, "parent")[:7],
            ]
        )

    with (TABLES / "same-change.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fingerprint", "on main (F)", "on release/v1 (F')"])
        w.writerow(
            [
                "patch-id of the diff",
                _patch_id(alice, f_full)[:7],
                _patch_id(alice, fp_full)[:7],
            ]
        )
        w.writerow(
            [
                "app.py blob it produces",
                _blob_of(alice, f_full, "app.py"),
                _blob_of(alice, fp_full, "app.py"),
            ]
        )

    with (TABLES / "what-changed-in-the-commit.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "commit F (main)", "backported F' (v1)", "changed"])
        rows = [
            ("tree", _tree_of(alice, f_full), _tree_of(alice, fp_full)),
            ("parent", _field(f_raw, "parent")[:7], _field(fp_raw, "parent")[:7]),
            ("author", _field(f_raw, "author"), _field(fp_raw, "author")),
            ("message", f_raw[-1], fp_raw[-1]),
            ("commit id", f_short, fp_short),
        ]
        for field, before, after in rows:
            w.writerow([field, before, after, "yes" if before != after else "no"])

    with (TABLES / "two-branches.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "main", "release/v1"])
        w.writerow(["branch tip", f_short, fp_short])
        w.writerow(
            ["tip's parent", _field(f_raw, "parent")[:7], _field(fp_raw, "parent")[:7]]
        )
        w.writerow(
            [
                "app.py at the tip",
                _blob_of(alice, "main", "app.py"),
                _blob_of(alice, "release/v1", "app.py"),
            ]
        )
        w.writerow(
            [
                "whole-tree at the tip",
                _tree_of(alice, "main"),
                _tree_of(alice, "release/v1"),
            ]
        )
        w.writerow(
            [
                "commits on the branch",
                _count_commits(alice, "main"),
                _count_commits(alice, "release/v1"),
            ]
        )
        w.writerow(
            [
                "merge commits on the branch",
                _n_merges(alice, "main"),
                _n_merges(alice, "release/v1"),
            ]
        )

    # ---- the oracle: every claim the README makes, checked --------------

    # (1) The pick created a DIFFERENT commit id from F.
    assert fp_full != f_full, "the backported commit is a new commit, not F itself"
    assert fp_short != f_short, "and its short id differs from F's too"

    # (2) The change is the same. Two independent proofs: the diff each commit
    # introduces has the identical patch-id, and the app.py the fix produces is
    # literally the same blob object on both lines.
    assert _patch_id(alice, f_full) == _patch_id(alice, fp_full) != "", (
        "F and F' introduce the identical diff (same patch-id)"
    )
    assert _blob_of(alice, f_full, "app.py") == _blob_of(alice, fp_full, "app.py"), (
        "the fixed app.py is the same blob on main and on release/v1"
    )
    # And the fix really is present on release/v1: the buggy text is gone, the
    # fixed text is there.
    v1_app = (alice / "app.py").read_text()
    assert "text.strip().lstrip" in v1_app, "the fix is present in release/v1's app.py"
    assert v1_app == APP_FIXED, "release/v1's app.py is exactly the fixed version"

    # (3) F' hangs off release/v1's own tip, not main's. The parent is the 1.0
    # release commit, not the version-2 commit F was built on.
    assert _field(fp_raw, "parent") != _field(f_raw, "parent"), (
        "F' has a different parent from F"
    )
    b_release = _run(
        alice, ["rev-parse", "release/v1~1"]
    )  # the release/v1 tip's parent
    assert _field(fp_raw, "parent") == b_release, (
        "F's parent on the backport is release/v1's previous tip (the 1.0 release)"
    )
    d_main = _run(alice, ["rev-parse", f"{f_full}~1"])
    assert _field(f_raw, "parent") == d_main, (
        "F's parent on main is main's version-2 commit"
    )
    # And that snapshot differs, which is the second reason the id had to change:
    # release/v1 has no version-2 files, so F' snapshots a different project.
    assert _tree_of(alice, f_full) != _tree_of(alice, fp_full), (
        "F and F' snapshot different projects (v1 has no version-2 files)"
    )
    assert not (alice / "currencies.py").exists(), (
        "the version-2 file never reached release/v1: only the one fix was picked"
    )

    # (4) main and release/v1 stayed distinct. The pick copied one commit, it did
    # not merge the lines: no branch contains the other's tip, no merge commit
    # appeared, and the two tips are different commits.
    assert _branches_containing(alice, f_full) == ["main"], "F is on main only"
    assert _branches_containing(alice, fp_full) == ["release/v1"], (
        "F' is on release/v1 only"
    )
    assert _n_parents(alice, fp_full) == 1, "F' has one parent: a copy, not a merge"
    assert _n_merges(alice, "release/v1") == 0, (
        "no merge commit was created on release/v1"
    )
    assert _run(alice, ["rev-parse", "main"]) != _run(
        alice, ["rev-parse", "release/v1"]
    ), "main and release/v1 point at different commits"

    # What cherry-pick preserved and what it changed. Author (the original one)
    # and the message survive; the parent, the tree and therefore the id do not.
    assert _field(f_raw, "author") == _field(fp_raw, "author") != "", (
        "cherry-pick kept the original author, timestamp and all"
    )
    assert f_raw[-1] == fp_raw[-1], "the commit message is unchanged"

    print(
        f"backport: fix {f_short} on main copied to {fp_short} on release/v1, "
        f"different id, same patch-id {_patch_id(alice, f_full)[:7]}."
    )
    print(
        f"parents:  F sits on main's {_field(f_raw, 'parent')[:7]}, "
        f"F' sits on release/v1's {_field(fp_raw, 'parent')[:7]}."
    )
    print(
        f"branches: main tip {_short(alice, 'main')} (contains F), "
        f"release/v1 tip {fp_short} (contains F'), 0 merges, tips differ."
    )
    print(
        f"{len(list(FIGURES.glob('*.png')))} figures, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All assertions passed."
    )

    box.cleanup()


if __name__ == "__main__":
    main()
