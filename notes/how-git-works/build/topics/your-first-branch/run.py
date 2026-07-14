"""Your first branch: create, commit, switch, merge, delete, for real.

A first-steps tutorial, so it shows what a beginner actually sees. Every git
command is run in a real local repository (a plain `git init`, no remote, so no
phantom origin appears) and its real output is captured, path sanitized to a
friendly name, and written to tables/ for the README to quote. The DAG figures
come from the same repository. Nothing is typed by hand.

The story: you have working code on main. You want to try something risky, so
you make a branch, do the work there, watch the files change as you switch back
and forth, then merge the work into main and delete the finished branch.
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
FRIENDLY = "/home/you/my-app"

# The file the experiment grows, one version per commit.
APP_V1 = 'print("Welcome to the app")\n'
APP_V2 = 'print("Welcome to the app")\nprint("Here is a tip for today")\n'
APP_V3 = (
    'print("Welcome to the app")\n'
    'print("Here is a tip for today")\n'
    'print("See you tomorrow")\n'
)
# A small, unrelated bit of work you do on main while the experiment waits. It
# lives in its own file, so bringing the experiment back is a clean merge.
NOTES = "Remember to write tests\n"


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

    # A plain local repo, no remote, because the reader has not met GitHub yet.
    box = Sandbox(people=("alice",), local=True)
    repo = box.paths["alice"]

    def do_commit(filename: str, content: str, message: str):
        """Edit, stage, commit with a real spaced message and a fixed date."""
        box.write("alice", filename, content, record=False)
        box.git("alice", f"add {filename}")
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            capture_output=True,
            text=True,
            env=_env(box),
        )
        box._log.append(("alice", f"git commit -m '{message}'"))

    def cap(proc) -> str:
        """The real output of a command. git prints 'Switched to ...' to stderr
        and the merge summary to stdout, so we keep both."""
        return clean((proc.stdout or "") + (proc.stderr or ""), repo)

    def git_out(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    def cat(name: str) -> str:
        return clean(
            subprocess.run(
                ["cat", name], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    def obj_type(sha: str) -> str:
        return subprocess.run(
            ["git", "cat-file", "-t", sha], cwd=repo, capture_output=True, text=True
        ).stdout.strip()

    # 1. The starting point: one commit on main. This is where your first-commit
    #    tutorial left off.
    do_commit("app.py", APP_V1, "Add the welcome message")
    sha_A = box.read("alice").head_sha
    box.snap(
        "your project, one commit on main",
        note="One commit on main, which is the branch git made for you. This is your safe, working code.",
    )

    # 2. Make a branch and move onto it. -c means create, switch moves you there.
    outputs["switch_create"] = cap(box.git("alice", "switch -c experiment"))
    st_branch = box.read("alice")
    sha_main_at_branch = st_branch.branches["main"]
    sha_exp_at_branch = st_branch.branches["experiment"]
    box.snap(
        "a new branch called experiment",
        note="The branch is a new label at the same commit. You are now standing on experiment, not main.",
    )

    # 3. Two commits of experimental work, all on the branch.
    do_commit("app.py", APP_V2, "Add the experimental greeting")
    do_commit("app.py", APP_V3, "Add a sign-off line")
    outputs["log_experiment"] = git_out("log --oneline")
    st_ahead = box.read("alice")
    sha_main_after_work = st_ahead.branches["main"]
    sha_C = st_ahead.branches["experiment"]
    box.snap(
        "two commits on the experiment branch",
        note="main has not moved. Your new work sits on experiment, two commits ahead of main.",
    )

    # 4. The aha moment: switch away, and the file on disk changes back.
    outputs["switch_main"] = cap(box.git("alice", "switch main"))
    outputs["cat_main"] = cat("app.py")
    outputs["switch_back"] = cap(box.git("alice", "switch experiment"))
    outputs["cat_experiment"] = cat("app.py")

    # 5. Back on main, you do a small unrelated thing while the experiment waits.
    box._log.clear()  # keep the next figure's command line to the point
    box.git("alice", "switch main")
    do_commit("notes.txt", NOTES, "Add a note to self")
    box.snap(
        "main moved on while experiment waited",
        note="You switched to main and made a separate change there. The two branches have each gone their own way.",
    )

    # 6. Merge the experiment back into main. main has both lines of work now.
    outputs["merge"] = cap(box.git("alice", "merge --no-edit experiment"))
    outputs["cat_merged"] = cat("app.py")
    outputs["log_main"] = git_out("log --oneline")
    st_merged = box.read("alice")
    sha_merge = st_merged.head_sha
    box.snap(
        "merge brings experiment back into main",
        note="A new commit ties the two lines together. main now has both the experiment and the note.",
    )

    # 7. Delete the finished branch. -d is the safe one: it refuses if unmerged.
    outputs["delete"] = cap(box.git("alice", "branch -d experiment"))
    st_deleted = box.read("alice")
    box.snap(
        "the finished branch, deleted",
        note="The experiment label is gone. Every commit it pointed at is still here, now part of main's history.",
    )

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    with (TABLES / "session.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output"])
        for key, value in outputs.items():
            writer.writerow([key, value])

    # ---- The oracle: every claim the README makes, checked against real git ----

    # This is a local repo, so nothing about a server appears anywhere.
    assert box.bares == set(), "local mode creates no server"
    assert st_deleted.remotes == {}, "and the working repo has no origin"

    # git switch -c put you onto the new branch, which starts at main's commit.
    assert st_branch.head == "experiment", (
        "after git switch -c, HEAD is on the new branch"
    )
    assert sha_exp_at_branch == sha_A == sha_main_at_branch, (
        "a fresh branch points at the very commit you were on"
    )
    assert "Switched to a new branch 'experiment'" in outputs["switch_create"], (
        "git confirms the branch was created and switched to"
    )

    # A commit on the branch moves experiment, not main.
    assert sha_main_after_work == sha_A, "committing on experiment does not move main"
    assert sha_C != sha_A, "the experiment branch has advanced past where it started"

    # Switching changes the files on disk: main's app.py is not the branch's.
    assert "Switched to branch 'main'" in outputs["switch_main"], (
        "git confirms the switch back to main"
    )
    assert "Here is a tip for today" not in outputs["cat_main"], (
        "on main, the experiment's work is not in the file"
    )
    assert 'print("Welcome to the app")' in outputs["cat_main"], (
        "on main, the file is back to its committed version"
    )
    assert outputs["cat_main"] != outputs["cat_experiment"], (
        "the same file reads differently on the two branches"
    )
    assert "Switched to branch 'experiment'" in outputs["switch_back"], (
        "git confirms the switch back to experiment"
    )
    assert (
        "Here is a tip for today" in outputs["cat_experiment"]
        and "See you tomorrow" in outputs["cat_experiment"]
    ), "on experiment, the work returns to the file exactly as it was"

    # The merge brought the branch's commit into main.
    assert "Merge made by" in outputs["merge"], "the merge created a merge commit"
    assert len(st_merged.commits[sha_merge].parents) == 2, (
        "a merge commit joins two lines, so it has two parents"
    )
    assert sha_C in st_merged.commits, (
        "the experiment's last commit is now part of main's history"
    )
    assert "Here is a tip for today" in outputs["cat_merged"], (
        "after the merge, main's app.py carries the experiment's work"
    )

    # git branch -d removed the label; the commit object is untouched.
    assert "Deleted branch experiment" in outputs["delete"], (
        "git confirms the branch label was removed"
    )
    assert "experiment" not in st_deleted.branches, "the experiment branch ref is gone"
    assert obj_type(sha_C) == "commit", (
        "the commit itself still exists as an object on disk"
    )
    assert sha_C in st_deleted.commits, "and it is still reachable, now through main"

    # The quoted history blocks have the shape the README shows.
    assert outputs["log_experiment"].count("\n") == 2, (
        "git log --oneline on experiment lists three commits"
    )
    assert outputs["log_main"].count("\n") == 4, (
        "git log --oneline on main lists five commits after the merge"
    )

    print(
        f"5 commits, {len(list(FIGURES.glob('*.png')))} figures, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All checks passed."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
