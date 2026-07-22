"""What git is: the motivation, shown with a real repository.

This is the very first tutorial a beginner reads, so it is light on commands
and heavy on why git exists. It still obeys the collection's hard rule: every
line of output the README quotes comes from a real git repository built during
this run, path sanitized to a friendly name, never typed by hand. The figure is
a photograph of that repository, and the run ends by asserting the facts the
README states.
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

from lib.gitviz import Sandbox, clear, render_split  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# What the reader's own machine would show instead of a throwaway temp path.
FRIENDLY = "/home/you/project"


def clean(text: str, real_root: Path) -> str:
    """Swap the throwaway temp path for a friendly one, so the output reads
    like something off the reader's own laptop. Both the raw and the resolved
    form are swapped, because macOS reports temp paths through /private."""
    for form in (str(real_root.resolve()), str(real_root)):
        text = text.replace(form, FRIENDLY)
    return text.rstrip("\n")


def _env(box: Sandbox):
    """The fixed-date env the sandbox uses, so commit hashes are stable."""
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

    # 1. A real git init, captured so the reader sees the true first line.
    init_dir = Path(tempfile.mkdtemp(prefix="gitviz-init-")) / "project"
    init = subprocess.run(
        ["git", "init", "--initial-branch=main", str(init_dir)],
        capture_output=True,
        text=True,
    )
    outputs["init"] = clean(init.stdout, init_dir)
    subprocess.run(["rm", "-rf", str(init_dir.parent)])

    # The teaser runs in a local sandbox: a plain repo with no remote, because
    # the reader has not met GitHub yet and no phantom origin should appear.
    box = Sandbox(people=("alice",), local=True)
    repo = box.paths["alice"]

    def show(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    def save(content: str, message: str):
        """Edit analysis.py and save the new version as one commit. The commit
        goes through subprocess so the spaced message survives (box.commit would
        hyphenate it), then it is logged by hand."""
        box.write("alice", "analysis.py", content, record=False)
        box.git("alice", "add analysis.py")
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo,
            capture_output=True,
            text=True,
            env=_env(box),
        )
        box._log.append(("alice", f"git commit -m '{message}'"))

    # 2. Two saved versions of one file. Git keeps both, so the reader never has
    #    to keep analysis_v1.py and analysis_v2_final.py side by side.
    save("import pandas as pd\n", "Start the analysis")
    save(
        "import pandas as pd\n\ndf = pd.read_csv('sales.csv')\nfit_model(df)\n",
        "Add the model",
    )

    # 3. The history, one line per saved version.
    outputs["log"] = show("log --oneline")

    # 4. Proof it is all local: git remote lists the servers this repo talks to,
    #    and there are none.
    outputs["remote"] = show("remote -v")

    # 5. Everything saved, nothing pending.
    outputs["status"] = show("status")

    # Keep the figure's command subtitle empty. It is a picture of the history,
    # not of a single command, so the caption carries the meaning instead.
    box._log.clear()
    box.snap(
        "your project, two saved versions",
        note="Two versions of one file, saved on your own laptop. Each points back at the one before it.",
    )

    render_split(box, FIGURES, TABLES, "alice", folder="project")

    with (TABLES / "session.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "output"])
        for key, value in outputs.items():
            writer.writerow([key, value])

    # The oracle: every fact the README states, checked against real git.
    state = box.read("alice")
    assert len(state.commits) == 2, "two versions were saved, so two commits exist"
    assert state.remotes == {}, "this repo has no remote, so nothing left the laptop"
    assert box.bares == set(), "local mode built no server, so no origin can appear"
    assert (
        "Initialized empty Git repository in /home/you/project/.git/" in outputs["init"]
    ), "git init reports the repository it made, at the friendly path"
    assert outputs["log"].count("\n") == 1, (
        "git log --oneline prints the two commits, one per line"
    )
    assert (
        "Start the analysis" in outputs["log"] and "Add the model" in outputs["log"]
    ), "both saved versions carry the messages we gave them"
    assert outputs["remote"].strip() == "", (
        "git remote -v prints nothing, because there is no server"
    )
    assert "nothing to commit, working tree clean" in outputs["status"], (
        "with both versions saved, git status is clean"
    )

    print(
        f"2 commits, {len(list(FIGURES.glob('*.png')))} figure, "
        f"{len(list(TABLES.glob('*.csv')))} tables. All checks passed."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
