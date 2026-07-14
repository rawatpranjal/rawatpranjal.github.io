"""Ignoring files: keep junk out of git, for real, with captured output.

This is a first-steps tutorial, so it shows what a beginner actually sees. A
real repository is filled with the clutter a data science project makes (data
files, caches, a model, a secret, OS junk), git status is captured before and
after a .gitignore, and the one trap everyone hits (a file that was already
committed keeps being tracked even after you ignore it) is demonstrated for
real. Every quoted block below is captured from a live git command and asserted
to say what the README claims. The DAG figures come from the same repository.
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

# What the reader's own machine would show instead of a throwaway temp path.
FRIENDLY = "/home/you/sales-report"

# The clutter a normal afternoon of data science leaves in a project folder.
# None of it belongs in git. Some sit in subfolders, which git creates on write.
JUNK = {
    "data/sales.csv": "date,units\n2026-01-01,42\n",
    "data/big.parquet": "PAR1... (imagine two gigabytes of columns) ...PAR1\n",
    "export.csv": "id,score\n1,0.9\n",
    "features.parquet": "PAR1... a feature matrix ...PAR1\n",
    "__pycache__/analysis.cpython-311.pyc": "compiled bytecode, not source\n",
    ".ipynb_checkpoints/analysis-checkpoint.ipynb": '{"cells": []}\n',
    "model.pkl": "a pickled model, a big binary blob\n",
    ".env": "API_KEY=sk-do-not-commit-this\n",
    ".DS_Store": "macOS folder metadata\n",
}

# The .gitignore we write. Each block is a pattern kind the README explains.
GITIGNORE = """\
# Data files: big, and they change all the time
data/
*.csv
*.parquet

# Python leftovers: regenerated every run
__pycache__/
.ipynb_checkpoints/

# Model files: large binaries
model.pkl

# Secrets: never put these in git
.env

# macOS clutter
.DS_Store

# A settings file we committed months ago, before we knew better
config.py
"""

# One representative file per pattern, and the pattern that should catch it.
# Verified for real with `git check-ignore`, so the README's per-line
# explanation is checked, not asserted by hand.
PATTERN_CHECKS = [
    ("data/sales.csv", "data/"),
    ("export.csv", "*.csv"),
    ("features.parquet", "*.parquet"),
    ("__pycache__/analysis.cpython-311.pyc", "__pycache__/"),
    (".ipynb_checkpoints/analysis-checkpoint.ipynb", ".ipynb_checkpoints/"),
    ("model.pkl", "model.pkl"),
    (".env", ".env"),
    (".DS_Store", ".DS_Store"),
]


def _env(box: Sandbox):
    """The fixed-date env the sandbox uses, so commit hashes stay stable."""
    return {
        "GIT_AUTHOR_DATE": "2026-01-01T12:00:00",
        "GIT_COMMITTER_DATE": "2026-01-01T12:00:00",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(box.root),
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }


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

    # A plain repo, no remote, because the reader has not met GitHub yet.
    box = Sandbox(people=("alice",), local=True)
    repo = box.paths["alice"]

    def show(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    def commit(message: str):
        """A commit with a real, spaced message (box.git would split on spaces)."""
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            capture_output=True,
            text=True,
            env=_env(box),
        )
        box._log.append(("alice", f"git commit -m '{message}'"))

    # 1. An existing project: two files, already committed, so both are tracked.
    box.write("alice", "analysis.py", "import pandas as pd\n", record=False)
    box.write(
        "alice",
        "config.py",
        'DB_HOST = "localhost"\nDB_PASSWORD = "hunter2"\n',
        record=False,
    )
    box.git("alice", "add analysis.py config.py")
    commit("Start the project")
    box._log.clear()  # step-01's caption should start fresh at the .gitignore work

    # 2. A normal afternoon of work leaves clutter. git status drowns in it.
    for name, content in JUNK.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    outputs["status_mess"] = show("status")

    # 3. Write the .gitignore, and the noise disappears from status.
    (repo / ".gitignore").write_text(GITIGNORE)
    outputs["gitignore_file"] = clean((repo / ".gitignore").read_text(), repo)
    outputs["status_after_ignore"] = show("status")

    # 4. Prove each pattern really catches its file, with git's own checker.
    pattern_rows = []
    for target, expected in PATTERN_CHECKS:
        proc = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", target],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{target} should be ignored but was not"
        source, _line, matched = proc.stdout.split("\t")[0].split(":", 2)
        assert matched == expected, f"{target} matched {matched}, expected {expected}"
        pattern_rows.append((expected, target, source))

    # 5. The .gitignore is part of the project, so commit it like any file.
    box.git("alice", "add .gitignore")
    commit("Add a .gitignore")
    outputs["status_clean"] = show("status")
    box.snap(
        "the .gitignore is committed",
        note="A .gitignore is a normal file. Commit it, and everyone who clones "
        "the project gets the same rules.",
    )

    # 6. The trap. config.py was committed on day one and is in .gitignore now,
    #    yet git still tracks it: an ignore rule only ever affects UNtracked files.
    outputs["ls_files_tracked"] = show("ls-files")

    # The symptom you actually feel: change the ignored-but-tracked file, and
    # git still reports the change, as if the ignore rule were not there.
    box.write(
        "alice",
        "config.py",
        'DB_HOST = "localhost"\nDB_PASSWORD = "changed-my-password"\n',
        record=False,
    )
    outputs["status_trap"] = show("status")
    box.git("alice", "restore config.py")  # put it back, clean, before the fix
    box._log.clear()  # step-02's caption is just the fix and its commit

    # 7. The fix: git rm --cached removes it from git's tracking, not from disk.
    rm = subprocess.run(
        ["git", "rm", "--cached", "config.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(box),
    )
    outputs["rm_cached"] = clean(rm.stdout, repo)
    box._log.append(("alice", "git rm --cached config.py"))
    outputs["status_after_rm"] = show("status")
    config_still_on_disk = (repo / "config.py").exists()
    commit("Stop tracking config.py")
    outputs["ls_files_after"] = show("ls-files")
    box.snap(
        "config.py is no longer tracked",
        note="git rm --cached made this commit. config.py is off git's radar now, "
        "but the file is still on your disk.",
    )

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    with (TABLES / "session.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output"])
        for key, value in outputs.items():
            writer.writerow([key, value])

    with (TABLES / "patterns.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pattern", "example_file_ignored", "matched_in"])
        for pattern, target, source in pattern_rows:
            writer.writerow([pattern, target, source])

    # The oracle: every fact the README states, checked against real git.
    state = box.read("alice")
    assert state.remotes == {}, "local repo has no remote, so no origin appears"
    assert len(state.commits) == 3, "three commits were made, so three exist"

    assert "Untracked files" in outputs["status_mess"], (
        "before .gitignore, the junk shows up as untracked noise"
    )
    for token in (
        ".DS_Store",
        ".env",
        ".ipynb_checkpoints/",
        "__pycache__/",
        "data/",
        "export.csv",
        "features.parquet",
        "model.pkl",
    ):
        assert token in outputs["status_mess"], (
            f"{token} is part of the mess git status lists"
        )

    assert ".gitignore" in outputs["status_after_ignore"], (
        "after writing it, the only new untracked file is .gitignore itself"
    )
    for token in ("data/", "model.pkl", ".env", "__pycache__/", "export.csv"):
        assert token not in outputs["status_after_ignore"], (
            f"once ignored, git stops mentioning {token}"
        )

    for line in (
        "data/",
        "*.csv",
        "*.parquet",
        "__pycache__/",
        ".ipynb_checkpoints/",
        "model.pkl",
        ".env",
        ".DS_Store",
        "config.py",
    ):
        assert line in outputs["gitignore_file"], (
            f"the committed .gitignore contains the {line} pattern the README quotes"
        )

    assert "nothing to commit, working tree clean" in outputs["status_clean"], (
        "after committing the .gitignore, git status is clean"
    )

    assert "config.py" in outputs["ls_files_tracked"], (
        "the trap: config.py is still tracked even though it is in .gitignore"
    )
    assert (
        "modified:" in outputs["status_trap"] and "config.py" in outputs["status_trap"]
    ), "git still reports changes to config.py, ignore rule notwithstanding"

    assert "config.py" in outputs["rm_cached"], (
        "git rm --cached reports the file it removed from tracking"
    )
    assert "deleted:" in outputs["status_after_rm"], (
        "after git rm --cached, the removal is staged as a deletion"
    )
    assert config_still_on_disk, (
        "git rm --cached leaves the file on disk, it only stops tracking it"
    )
    assert "config.py" not in outputs["ls_files_after"], (
        "after the commit, config.py is no longer tracked"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 2, "two DAG snapshots, two figures"

    print(
        f"3 commits, {len(figs)} figures, 2 tables. "
        f"{len(pattern_rows)} patterns verified with git check-ignore. All checks passed."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
