"""Two agents, one repository, two git worktrees. Photographs of the real thing.

A worktree is a second working directory attached to the same repository. It has
its own HEAD and its own files, but shares the one object store and the one set
of branches. That is exactly what you want when two coding agents run at once:
each edits in its own directory, on its own branch, and neither can touch the
other's files.

Everything below is built by running real git in a temporary directory and
reading the state back out with plumbing, then asserted at the end.
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

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# Panels are roles, not people, so name them for the reader.
PANELS = ("dev", "agent-a", "agent-b")
LABELS = {
    "dev": "Main checkout (branch: main)",
    "agent-a": "Agent A worktree (branch: agent-a)",
    "agent-b": "Agent B worktree (branch: agent-b)",
}

BASE = "def run():\n    return 1\n"
FEATURE_A = "def run():\n    return 1\n\ndef search(q):\n    return []\n"
FEATURE_B = "def run():\n    return 1\n\ndef cache():\n    return {}\n"


def add_worktree(box: Sandbox, key: str, branch: str) -> subprocess.CompletedProcess:
    """Attach a new worktree at <root>/<key> on a new branch, and register it.

    git worktree add lives inside the main clone, so we drive it there, then
    point the sandbox at the new directory so snap() and render() treat it as a
    panel of its own.
    """
    path = box.root / key
    # A relative path (../key) keeps the throwaway temp root out of the figure
    # header and the state log; it resolves to <root>/<key> from the dev clone.
    proc = box.git("dev", f"worktree add ../{key} -b {branch}")
    box.paths[key] = path
    return proc


def main() -> None:
    clear(FIGURES, TABLES)
    box = Sandbox(people=("dev",))

    # One repository, one commit, pushed so there is a shared remote to branch from.
    box.commit("dev", "app.py", BASE, "start the app")
    box.git("dev", "push -u origin main")

    # Give each agent its own worktree on its own branch.
    add_worktree(box, "agent-a", "agent-a")
    add_worktree(box, "agent-b", "agent-b")
    box.people = PANELS
    box.snap(
        "one repository, two worktrees",
        note="Each worktree has its own HEAD and its own files. They share one "
        "object store and one set of branches.",
    )

    # Agent A works in its worktree. Only its files change.
    box.commit("agent-a", "app.py", FEATURE_A, "add search")
    box.snap(
        "Agent A commits on agent-a",
        note="The commit lands on branch agent-a. The main checkout and Agent B's "
        "worktree are untouched.",
    )

    # Agent B works in its worktree, at the same time, with no coordination.
    box.commit("agent-b", "app.py", FEATURE_B, "add cache")
    box.snap(
        "Agent B commits on agent-b",
        note="Two branches moved forward independently from the same start, with "
        "no conflict and no shared working directory.",
    )

    # git worktree list: the ground truth of who is where.
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=box.paths["dev"],
        capture_output=True,
        text=True,
    ).stdout

    # git refuses to check out a branch that is already checked out in a worktree.
    dup = box.git("dev", "worktree add ../dup agent-a", check=False)

    render(box, FIGURES, TABLES, mode="team", repos=PANELS, panel_labels=LABELS)

    # ---- read the truth back out for the tables and the assertions ----
    dev = box.read("dev")
    a = box.read("agent-a")
    b = box.read("agent-b")

    # tables/worktrees.csv: path, branch, head sha, per worktree
    wt_rows = []
    cur = {}
    for line in listing.splitlines() + [""]:
        if not line:
            if cur:
                wt_rows.append(cur)
                cur = {}
            continue
        parts = line.split(" ", 1)
        cur[parts[0]] = parts[1] if len(parts) > 1 else ""
    with (TABLES / "worktrees.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "branch", "head"])
        # git worktree list reports resolved paths (/private/var on macOS), so
        # resolve both sides before matching or every role degrades to "?".
        roles = {
            str(box.paths["dev"].resolve()): "main checkout",
            str(box.paths["agent-a"].resolve()): "Agent A",
            str(box.paths["agent-b"].resolve()): "Agent B",
        }
        for r in wt_rows:
            path = str(Path(r.get("worktree", ".")).resolve())
            branch = r.get("branch", "").replace("refs/heads/", "")
            w.writerow([roles.get(path, "?"), branch, r.get("HEAD", "")[:7]])

    # tables/isolation.csv: the one-line proof that files are isolated
    with (TABLES / "isolation.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "answer"])
        w.writerow(["agent-a HEAD is on branch", a.head])
        w.writerow(["agent-b HEAD is on branch", b.head])
        w.writerow(
            [
                "agent A's search() present in agent B's files",
                "search" in (box.paths["agent-b"] / "app.py").read_text(),
            ]
        )
        w.writerow(
            [
                "agent-a commit visible in shared history everywhere",
                a.branches["agent-a"] in b.commits,
            ]
        )
        w.writerow(["second checkout of agent-a refused", dup.returncode != 0])

    # ---- ASSERTIONS: every claim the README makes ----
    # (A) each worktree stands on its own branch
    assert dev.head == "main", dev.head
    assert a.head == "agent-a", a.head
    assert b.head == "agent-b", b.head

    # (B) the branches (refs) are shared: all three see the same branch set
    assert set(a.branches) == set(b.branches) == set(dev.branches), (
        "worktrees share one set of branches"
    )
    assert {"main", "agent-a", "agent-b"} <= set(dev.branches)

    # (C) each agent's commit landed on its own branch, and moved only that branch
    assert a.branches["agent-a"] != dev.branches["main"], "agent A moved agent-a"
    assert b.branches["agent-b"] != dev.branches["main"], "agent B moved agent-b"
    assert a.branches["main"] == dev.branches["main"], "nobody moved main"

    # (D) files are isolated: A's new function is not in B's working tree
    assert "search" in (box.paths["agent-a"] / "app.py").read_text()
    assert "search" not in (box.paths["agent-b"] / "app.py").read_text(), (
        "agent B's working tree never saw agent A's edit"
    )
    assert "cache" not in (box.paths["agent-a"] / "app.py").read_text()

    # (E) but the history is shared: A's commit exists in B's object store
    assert a.branches["agent-a"] in b.commits, "one repository, one shared history"

    # (F) git refuses to check out an already-checked-out branch (default safeguard)
    assert dup.returncode != 0, "git refuses a second checkout of agent-a"
    assert "already" in (dup.stderr + dup.stdout).lower()

    # (G) three worktrees exist, the roles resolved (not "?"), and figures exist
    assert len(wt_rows) == 3, f"expected 3 worktrees, got {len(wt_rows)}"
    csv_roles = [
        row["role"]
        for row in csv.DictReader((TABLES / "worktrees.csv").read_text().splitlines())
    ]
    assert set(csv_roles) == {"main checkout", "Agent A", "Agent B"}, (
        f"role labels must resolve, got {csv_roles}"
    )
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {[f.name for f in figs]}"

    box.cleanup()
    print("worktrees-for-parallel-agents: ok, 3 figures, all assertions passed")


if __name__ == "__main__":
    main()
