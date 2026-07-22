"""Your first commit: init, status, add, commit, run for real.

This is a first-steps tutorial, so it shows what a beginner actually sees. Each
git command is run in a real repository and its real output is captured, path
sanitized to a friendly name, and written to tables/ for the README to quote.
The figures are two views of the same moment: the working directory on the left
(the files and what git thinks of each) and the commit graph on the right.
Nothing is typed by hand.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw_system, render_split  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# What the reader's own machine would show instead of a temp path.
FRIENDLY = "/home/you/sales-report"


def clean(text: str, real_root: Path) -> str:
    """Swap the throwaway temp path for a friendly one, so the output reads
    like something off the reader's own laptop. Both the raw and the resolved
    form are swapped, because macOS reports temp paths through /private."""
    for form in (str(real_root.resolve()), str(real_root)):
        text = text.replace(form, FRIENDLY)
    return text.rstrip("\n")


def main():
    clear(FIGURES, TABLES)
    outputs: dict[str, str] = {}

    # 1. git init, captured from a real init so the reader sees the real line.
    init_dir = Path(tempfile.mkdtemp(prefix="gitviz-init-")) / "sales-report"
    init = subprocess.run(
        ["git", "init", "--initial-branch=main", str(init_dir)],
        capture_output=True,
        text=True,
    )
    outputs["init"] = clean(init.stdout, init_dir)
    subprocess.run(["rm", "-rf", str(init_dir.parent)])

    # The real work happens in a local sandbox: a plain repo, no remote, because
    # the reader has not met GitHub yet.
    box = Sandbox(people=("alice",), local=True)
    repo = box.paths["alice"]

    def show(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    # 2. A new file: git can see it but is not tracking it yet.
    box.write("alice", "analysis.py", "import pandas as pd\n", record=False)
    outputs["status_untracked"] = show("status")
    box._log.append(("alice", "vim analysis.py"))
    box.snap(
        "a new file, not yet tracked",
        note="analysis.py sits on your disk. Git can see it but is not saving it. "
        "No commit exists yet.",
    )

    # 3. git add stages it.
    box.git("alice", "add analysis.py")
    outputs["status_staged"] = show("status")
    box.snap(
        "git add lines it up",
        note="git add moves analysis.py into the staging area, chosen for the next "
        "commit. It is still not saved.",
    )

    # 4. git commit saves the staged snapshot. Run via subprocess so the real,
    # spaced commit message survives (box.git would split it on spaces), then
    # record it by hand so the figure caption matches what actually happened.
    commit = subprocess.run(
        ["git", "commit", "-m", "Add the data loader"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(box),
    )
    outputs["commit"] = clean(commit.stdout, repo)
    box._log.append(("alice", "git commit -m 'Add the data loader'"))
    box.snap(
        "your first commit",
        note="The staged file is saved as one commit. History has one entry, and "
        "the working tree is clean.",
    )
    outputs["log_one"] = show("log")

    # 5. Change the file: now it is modified, not untracked.
    box.write(
        "alice",
        "analysis.py",
        "import pandas as pd\n\ndf = pd.read_csv('sales.csv')\n",
        record=False,
    )
    outputs["status_modified"] = show("status")
    box._log.append(("alice", "vim analysis.py"))
    box.snap(
        "you change the file",
        note="Editing a tracked file makes it modified. The change is on disk but "
        "not yet in a commit.",
    )

    # 6. A second commit, so history has a shape.
    box.git("alice", "add analysis.py")
    subprocess.run(
        ["git", "commit", "-m", "Load the sales data"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(box),
    )
    box._log.append(("alice", "git commit -m 'Load the sales data'"))
    box.snap(
        "two commits",
        note="A second commit points back at the first. This chain is your history.",
    )
    outputs["log_two"] = show("log --oneline")
    outputs["status_clean"] = show("status")

    # Two-column figures (working directory plus commit graph), one per stage.
    render_split(box, FIGURES, TABLES, "alice", folder="sales-report")

    # One systemic view: where a change travels on your own machine.
    draw_system(
        FIGURES / "step-06.png",
        stages=["working tree", "staging area", "history (.git)"],
        arrows=["git add", "git commit"],
        title="Where a change travels, on your own machine",
        frame_label="your laptop, no server yet",
        note="A change starts as an edit on disk, is lined up with git add, and is "
        "saved into history with git commit. Nothing leaves your laptop.",
    )

    with (TABLES / "session.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output"])
        for key, value in outputs.items():
            writer.writerow([key, value])

    # The oracle: the everyday facts the README states, checked against real git.
    state = box.read("alice")
    assert len(state.commits) == 2, "two commits were made, so two exist"
    assert state.remotes == {}, (
        "this repo has no remote yet, so no origin appears anywhere"
    )
    assert "Untracked files" in outputs["status_untracked"], (
        "a brand new file shows up as untracked, not as a change"
    )
    assert "Changes to be committed" in outputs["status_staged"], (
        "after git add, the file is staged"
    )
    assert "root-commit" in outputs["commit"], (
        "the very first commit in a repo is labelled the root commit"
    )
    assert "Add the data loader" in outputs["commit"], (
        "and it carries the message we gave it"
    )
    assert "nothing to commit, working tree clean" in outputs["status_clean"], (
        "once everything is committed, git status is clean"
    )
    assert outputs["log_two"].count("\n") == 1, (
        "git log --oneline shows the two commits, one per line"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures (5 stages + 1 system), got {figs}"

    print(f"2 commits, {len(figs)} figures, 1 table. All checks passed.")
    box.cleanup()


def _env(box: Sandbox):
    """The same fixed-date env the sandbox uses, so commit hashes are stable."""
    return {
        "GIT_AUTHOR_DATE": "2026-01-01T12:00:00",
        "GIT_COMMITTER_DATE": "2026-01-01T12:00:00",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(box.root),
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }


if __name__ == "__main__":
    main()
