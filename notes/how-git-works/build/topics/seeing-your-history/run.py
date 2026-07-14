"""Seeing your history: log, show and diff, run for real.

A first-steps tutorial, so it shows what a beginner actually sees. A tiny repo
is built (three commits growing one shopping list), then each history command is
run in it, its real output captured, the temp path swapped for a friendly one,
and the result written to tables/ for the README to quote verbatim. The DAG
figures come from the same repository. Nothing is typed by hand, and the run
ends with assertions that check every claim the README makes.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, render_split  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# What the reader's own machine would show instead of a temp path.
FRIENDLY = "/home/you/shopping"


def clean(text: str, real_root: Path) -> str:
    """Swap the throwaway temp path for a friendly one, so the output reads like
    something off the reader's own laptop. Both the raw and the resolved form are
    swapped, because macOS reports temp paths through /private."""
    for form in (str(real_root.resolve()), str(real_root)):
        text = text.replace(form, FRIENDLY)
    return text.rstrip("\n")


def _env(box: Sandbox):
    """The same fixed-date env the sandbox uses, so commit hashes are stable."""
    return {
        "GIT_AUTHOR_DATE": "2026-01-01T12:00:00",
        "GIT_COMMITTER_DATE": "2026-01-01T12:00:00",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(box.root),
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }


def main():
    clear(FIGURES, TABLES)
    outputs: dict[str, str] = {}

    # A plain local repo, no remote: the reader has not met GitHub yet.
    box = Sandbox(people=("alice",), local=True)
    repo = box.paths["alice"]

    def show(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    def make_commit(content: str, message: str) -> None:
        """Edit the one file, stage it, and commit with a real spaced message.
        box.commit would hyphenate the message, so commit via subprocess and
        record the step by hand so the figure caption stays honest."""
        box.write("alice", "shopping.txt", content, record=False)
        box.git("alice", "add shopping.txt")
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            capture_output=True,
            text=True,
            env=_env(box),
        )
        box._log.append(("alice", f"git commit -m '{message}'"))

    # 1. Three commits, growing one shopping list, so history has a shape.
    make_commit("milk\neggs\n", "Start the shopping list")
    make_commit("milk\neggs\nbread\nbutter\n", "Add bread and butter")
    make_commit("milk\neggs\nbread\nbutter\ncoffee\n", "Add coffee")

    box._log.clear()  # a short, uncluttered caption; the note does the teaching
    box.snap(
        "your history, three commits",
        note="Three saved versions of one file. git log reads this chain back, newest first.",
    )

    # 2. The two views of history.
    outputs["log_full"] = show("log")
    outputs["log_oneline"] = show("log --oneline")

    # 3. What a single commit changed. Grab the second commit's hash from git
    #    itself, then show it, so the diff in the README is a real one.
    hashes = subprocess.run(
        ["git", "log", "--reverse", "--pretty=%H"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.split()
    second_full = hashes[1]
    second_short = subprocess.run(
        ["git", "rev-parse", "--short", second_full],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    outputs["show_second"] = clean(
        subprocess.run(
            ["git", "show", second_full], cwd=repo, capture_output=True, text=True
        ).stdout,
        repo,
    )

    # 4. Nothing is staged and the tree is clean, so both diffs are empty.
    outputs["diff_clean"] = show("diff")
    outputs["diff_staged_empty"] = show("diff --staged")

    # 5. An edit that is not committed. Change one existing line, so the diff
    #    shows both a removed line and an added line.
    box.write(
        "alice",
        "shopping.txt",
        "milk (2 litres)\neggs\nbread\nbutter\ncoffee\n",
        record=False,
    )
    box._log.clear()
    box.snap(
        "a change you have not committed",
        note="The file on disk now has an edit that is in no commit. git diff shows exactly this.",
    )
    outputs["diff_unstaged"] = show("diff")

    # 6. Stage that edit. Now git diff is empty and git diff --staged shows it.
    box.git("alice", "add shopping.txt")
    outputs["diff_after_add"] = show("diff")
    outputs["diff_staged"] = show("diff --staged")

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    with (TABLES / "session.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output"])
        for key, value in outputs.items():
            writer.writerow([key, value])

    # The oracle: every fact the README states, checked against real git.
    state = box.read("alice")
    assert len(state.commits) == 3, "three commits were made, so three exist"
    assert state.remotes == {}, "a local repo has no remote, so no origin appears"

    assert len(outputs["log_oneline"].splitlines()) == 3, (
        "git log --oneline prints exactly one line per commit, so three lines"
    )
    for message in ("Start the shopping list", "Add bread and butter", "Add coffee"):
        assert message in outputs["log_full"], f"git log shows the message: {message}"
    assert "Author: alice" in outputs["log_full"], "git log names the author"
    assert "Date:" in outputs["log_full"], "git log dates each commit"

    assert second_short in outputs["show_second"], (
        "git show names the commit by its hash"
    )
    assert "Add bread and butter" in outputs["show_second"], (
        "git show prints the commit's message"
    )
    assert "+bread" in outputs["show_second"] and "+butter" in outputs["show_second"], (
        "git show marks the two added lines with a plus"
    )

    assert outputs["diff_clean"] == "", "with nothing changed, git diff prints nothing"
    assert outputs["diff_staged_empty"] == "", (
        "with nothing staged, git diff --staged prints nothing"
    )

    assert "-milk" in outputs["diff_unstaged"], (
        "git diff marks the removed line with a minus"
    )
    assert "+milk (2 litres)" in outputs["diff_unstaged"], (
        "git diff marks the added line with a plus"
    )

    assert outputs["diff_after_add"] == "", (
        "once the edit is staged, git diff shows nothing: the change moved to --staged"
    )
    assert (
        "-milk" in outputs["diff_staged"]
        and "+milk (2 litres)" in outputs["diff_staged"]
    ), "git diff --staged now shows the very change git diff had shown before the add"

    print(
        f"3 commits, {len(list(FIGURES.glob('*.png')))} figures, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All checks passed."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
