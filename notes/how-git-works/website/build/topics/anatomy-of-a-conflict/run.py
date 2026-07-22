"""Anatomy of a conflict: what git will merge for you, and what it refuses to guess.

Builds two real repositories. In the first, two branches change different lines
of the same file and git merges them with no help. In the second, two branches
change the SAME line, git stops, writes the markers, and hands the decision to a
human. Everything the README claims about exit codes, markers, status codes and
parents is read back out of git and asserted here.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():  # walk up to the collection root
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw, draw_conflict, layout  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

FILE = "pipeline.py"
MARKERS = ("<<<<<<<", "=======", ">>>>>>>")

BASE = """TIMEOUT = 30
RETRIES = 3

def fetch(job):
    return http_get(job.url, timeout=TIMEOUT)

def clean(rows):
    return [row for row in rows if row]

def report(rows):
    print(len(rows), "rows")
"""

FETCH_LINE = "    return http_get(job.url, timeout=TIMEOUT)"

# Story 1: two branches, two different lines of the one file.
TUNING = BASE.replace("TIMEOUT = 30", "TIMEOUT = 90")
LOGGING = BASE.replace('print(len(rows), "rows")', 'print(len(rows), "rows kept")')

# Story 2: two branches, the SAME line, for two different reasons.
RETRY_SIDE = BASE.replace(
    FETCH_LINE, "    return http_get(job.url, timeout=TIMEOUT, retries=RETRIES)"
)
TLS_SIDE = BASE.replace(
    FETCH_LINE, "    return http_get(job.url, timeout=TIMEOUT, verify=True)"
)
# Neither side is right on its own. The retry and the certificate check are both
# wanted, so the resolution takes something from each.
RESOLVED = BASE.replace(
    FETCH_LINE,
    "    return http_get(job.url, timeout=TIMEOUT, retries=RETRIES, verify=True)",
)


def status(box: Sandbox) -> str:
    """The porcelain status line for the one file, or (clean) when git is silent."""
    out = box.git("alice", f"status --porcelain {FILE}", record=False).stdout
    return out.rstrip("\n") or "(clean)"


def contents(box: Sandbox) -> str:
    return (box.paths["alice"] / FILE).read_text()


def markers_in(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(MARKERS))


def unmerged(box: Sandbox) -> int:
    """Files with more than one version staged. That is what a conflict IS to git."""
    out = box.git("alice", "ls-files -u", record=False).stdout
    return len({line.split("\t")[-1] for line in out.splitlines() if line})


def index_versions(box: Sandbox) -> int:
    """How many versions of the file the index holds. One normally, three mid-conflict."""
    out = box.git("alice", f"ls-files -u {FILE}", record=False).stdout
    return len(out.splitlines()) or 1


def merging(box: Sandbox) -> bool:
    return (box.paths["alice"] / ".git" / "MERGE_HEAD").exists()


def render_dag(box: Sandbox, prefix: str) -> int:
    xs, ys = layout(box.snapshots)
    for i, snap in enumerate(box.snapshots, start=1):
        draw(snap, xs, ys, FIGURES / f"{prefix}-{i:02d}.png", mode="solo")
    return len(box.snapshots)


def log_rows(box: Sandbox, story: str) -> list[dict]:
    rows = []
    for i, snap in enumerate(box.snapshots, start=1):
        state = snap.repos["alice"]
        rows.append(
            {
                "story": story,
                "step": i,
                "label": snap.label,
                "command": snap.command,
                "head": state.head or "(detached)",
                "head_sha": state.head_sha or "",
                "branches": " ".join(
                    f"{b}={s}" for b, s in sorted(state.branches.items())
                ),
                "unmerged_files": len([d for d in state.dirty if d.startswith("UU")]),
            }
        )
    return rows


def new_sandbox() -> Sandbox:
    box = Sandbox(people=("alice",))
    # Pin git's own default marker style, so the figures cannot drift with a
    # differently configured machine. Not recorded: it is a determinism guard,
    # not a step of the tutorial.
    box.git("alice", "config merge.conflictstyle merge", record=False)
    return box


def clean_merge() -> tuple[Sandbox, dict]:
    """Two branches change different lines of one file. Git merges them alone."""
    box = new_sandbox()

    box.commit("alice", FILE, BASE, "add the pipeline")
    box.snap(
        "one file on main",
        note="Eleven lines. Two branches are about to edit two different parts of it.",
    )

    box.git("alice", "switch -c tuning")
    box.commit("alice", FILE, TUNING, "raise the timeout")
    box.snap(
        "tuning changes line 1",
        note="TIMEOUT goes from 30 to 90. Nothing else in the file is touched.",
    )

    box.git("alice", "switch main")
    box.commit("alice", FILE, LOGGING, "say how many rows were kept")
    before = box.read("alice")
    box.snap(
        "main changes the last line",
        note="Both branches have now edited pipeline.py. The history has forked.",
    )

    merge = box.git("alice", f"merge {'tuning'}", check=False)
    after = box.snap(
        "git merge tuning",
        note="Exit code 0. Git took line 1 from tuning and the last line from main by itself.",
    )

    merged = contents(box)
    state = after.repos["alice"]
    tip = state.branches["main"]

    # ---- the oracle for story 1 ----------------------------------------
    assert merge.returncode == 0, (
        f"a merge of disjoint lines must succeed, git returned {merge.returncode}"
    )
    assert markers_in(merged) == 0, (
        "git wrote no conflict markers, because it never had to ask anything"
    )
    assert "<<<<<<<" not in merged and ">>>>>>>" not in merged, (
        "the merged file is ordinary source, not a conflicted file"
    )
    assert "TIMEOUT = 90" in merged, "the merge kept the tuning branch's edit"
    assert '"rows kept"' in merged, "the merge kept main's edit"
    assert "def clean(rows):" in merged, "the lines neither side touched are untouched"
    assert len(state.commits) == len(before.commits) + 1, (
        "a true merge of two diverged branches creates exactly one new commit"
    )
    assert len(state.commits[tip].parents) == 2, (
        "and that commit has two parents, one per side of the merge"
    )
    assert set(state.commits[tip].parents) == {
        before.branches["main"],
        before.branches["tuning"],
    }, "the two parents are exactly the two tips that were merged"
    assert status(box) == "(clean)", "nothing is left for a human to do"

    assert index_versions(box) == 1, (
        "the index holds one version of the file, as always"
    )

    facts = {
        "exit_code": merge.returncode,
        "markers": markers_in(merged),
        "status": status(box),
        "index_versions": index_versions(box),
        "new_commits": len(state.commits) - len(before.commits),
        "parents": len(state.commits[tip].parents),
        "merge_sha": tip,
    }
    draw_conflict(
        box,
        "alice",
        FILE,
        FIGURES / "clean-file.png",
        "The same file after the merge, with no help from anyone",
        "git merge tuning",
        "Both edits are in. git status is empty, so it prints nothing after the colon.",
    )
    return box, facts


def conflicted_merge() -> tuple[Sandbox, dict, dict, dict]:
    """Two branches change the SAME line. Git stops and asks."""
    box = new_sandbox()

    box.commit("alice", FILE, BASE, "add the pipeline")
    box.snap(
        "the same starting point",
        note="This time both branches will edit line 5, the fetch call.",
    )

    box.git("alice", "switch -c retries")
    box.commit("alice", FILE, RETRY_SIDE, "retry failed fetches")
    box.snap(
        "retries rewrites line 5",
        note="The fetch call gains retries=RETRIES.",
    )

    box.git("alice", "switch main")
    box.commit("alice", FILE, TLS_SIDE, "verify tls certificates")
    before = box.read("alice")
    box.snap(
        "main rewrites the same line 5",
        note="The same line, a different edit, for a different reason.",
    )

    merge = box.git("alice", "merge retries", check=False)
    conflicted = contents(box)
    during = box.snap(
        "git merge retries",
        note="Exit code 1. No commit was made. The file on disk now holds both versions.",
    )

    # ---- the oracle for the conflict ------------------------------------
    assert merge.returncode != 0, (
        f"a same-line merge must stop, git returned {merge.returncode}"
    )
    assert "<<<<<<< HEAD" in conflicted, "HEAD is what you have: main's version"
    assert "=======" in conflicted, "the fence between the two versions"
    assert ">>>>>>> retries" in conflicted, (
        "the other side is named by the branch you are merging in"
    )
    assert markers_in(conflicted) == 3, "exactly three marker lines, one conflict"
    assert "verify=True" in conflicted and "retries=RETRIES" in conflicted, (
        "git threw nothing away, both intents sit in the file"
    )
    assert status(box) == f"UU {FILE}", (
        f"git status says both sides modified it, got {status(box)!r}"
    )
    assert unmerged(box) == 1, "one file is unmerged in the index"
    assert (
        len(box.git("alice", "ls-files -u", record=False).stdout.splitlines()) == 3
    ), "the index holds three versions of it: the ancestor, ours, theirs"
    assert merging(box), ".git/MERGE_HEAD exists, so git knows a merge is in progress"
    assert during.repos["alice"].head_sha == before.head_sha, (
        "the conflicted merge made no commit, HEAD has not moved"
    )
    assert len(during.repos["alice"].commits) == len(before.commits), (
        "and it created no commit object either"
    )

    conflict_facts = {
        "exit_code": merge.returncode,
        "markers": markers_in(conflicted),
        "status": status(box),
        "index_versions": index_versions(box),
        "new_commits": 0,
        "unmerged": unmerged(box),
    }
    draw_conflict(
        box,
        "alice",
        FILE,
        FIGURES / "conflict-file.png",
        "What git actually wrote into the file",
        "git merge retries",
        "HEAD is what you already had. The lower half is what you asked git to merge in.",
    )

    # ---- git merge --abort: the merge never happened ---------------------
    abort_before = {
        "head": before.head_sha,
        "merge_head": merging(box),
        "unmerged": unmerged(box),
        "commits": len(before.commits),
        "file_is_mains": contents(box) == TLS_SIDE,
    }
    abort = box.git("alice", "merge --abort", check=False)
    after_abort = box.read("alice")
    box.snap(
        "git merge --abort",
        note="The same three commits, the same HEAD, a clean working tree. The merge never happened.",
    )
    assert abort.returncode == 0, "the abort itself succeeds"
    assert not merging(box), "MERGE_HEAD is gone, git is no longer mid-merge"
    assert unmerged(box) == 0, "no file is unmerged any more"
    assert status(box) == "(clean)", "the working tree is clean again"
    assert contents(box) == TLS_SIDE, (
        "the file is back to exactly what main had before the merge"
    )
    assert after_abort.head_sha == before.head_sha, "HEAD never moved"
    assert len(after_abort.commits) == len(before.commits), "no commit was left behind"

    abort_facts = {
        "before": {
            "head": before.head_sha,
            "merge_head": "no",
            "unmerged": 0,
            "commits": len(before.commits),
            "file": "main's version",
        },
        "during": {
            "head": abort_before["head"],
            "merge_head": "yes",
            "unmerged": abort_before["unmerged"],
            "commits": abort_before["commits"],
            "file": "both versions, with markers",
        },
        "after": {
            "head": after_abort.head_sha,
            "merge_head": "no",
            "unmerged": unmerged(box),
            "commits": len(after_abort.commits),
            "file": "main's version",
        },
    }

    # ---- do it again, and this time resolve it ---------------------------
    again = box.git("alice", "merge retries", check=False)
    assert again.returncode != 0, "the same merge conflicts again, nothing was cured"
    assert markers_in(contents(box)) == 3, "the same three markers come back"

    box.write("alice", FILE, RESOLVED)
    box.git("alice", f"add {FILE}")
    resolved = contents(box)
    staged_status = status(box)
    assert staged_status == f"M  {FILE}", (
        f"git add is how you tell git the conflict is settled, got {staged_status!r}"
    )
    assert unmerged(box) == 0, "the index now holds one version of the file, not three"
    draw_conflict(
        box,
        "alice",
        FILE,
        FIGURES / "resolved-file.png",
        "The resolution: one line taken from neither side",
        "edit pipeline.py; git add pipeline.py",
        "UU has become M. git add is the act of saying: this is the answer.",
    )

    box.git("alice", "commit --no-edit")
    final = box.snap(
        "git commit",
        note="The merge commit records both parents, so the history remembers both edits.",
    )

    state = final.repos["alice"]
    tip = state.branches["main"]
    ours = "    return http_get(job.url, timeout=TIMEOUT, verify=True)"
    theirs = "    return http_get(job.url, timeout=TIMEOUT, retries=RETRIES)"
    combined = (
        "    return http_get(job.url, timeout=TIMEOUT, retries=RETRIES, verify=True)"
    )

    # ---- the oracle for the resolution -----------------------------------
    assert markers_in(resolved) == 0, "a resolved file has no markers left in it"
    assert combined in resolved, "the resolved line is the one the README quotes"
    assert "retries=RETRIES" in combined and "verify=True" in combined, (
        "the resolution takes something from each side"
    )
    assert ours not in resolved and theirs not in resolved, (
        "and it is neither side's line verbatim, which is the whole point"
    )
    assert len(state.commits[tip].parents) == 2, (
        "the resolved merge commit has exactly two parents"
    )
    assert set(state.commits[tip].parents) == {
        before.branches["main"],
        before.branches["retries"],
    }, "and they are the two tips that were merged"
    assert len(state.commits) == len(before.commits) + 1, (
        "resolving added exactly one commit, the merge commit"
    )
    assert status(box) == "(clean)", (
        "and the working tree is clean once it is committed"
    )
    assert not merging(box), "the merge is over, MERGE_HEAD is gone"

    resolve_facts = {
        "exit_code": 0,
        "markers": markers_in(resolved),
        "status": status(box),
        "index_versions": index_versions(box),
        "new_commits": len(state.commits) - len(before.commits),
        "parents": len(state.commits[tip].parents),
        "merge_sha": tip,
        "ours": ours.strip(),
        "theirs": theirs.strip(),
        "resolved": combined.strip(),
        "conflicted_text": conflicted,
        "staged_status": staged_status,
    }
    return box, conflict_facts, abort_facts, resolve_facts


def write_tables(clean_facts, conflict_facts, abort_facts, resolve_facts, log):
    with (TABLES / "merge-outcomes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "what the two branches changed",
                "git exit code",
                "conflict marker lines written",
                "git status for the file",
                "versions of the file in the index",
                "commits created",
                "parents of the new commit",
            ]
        )
        writer.writerow(
            [
                "different lines of pipeline.py",
                clean_facts["exit_code"],
                clean_facts["markers"],
                clean_facts["status"],
                clean_facts["index_versions"],
                clean_facts["new_commits"],
                clean_facts["parents"],
            ]
        )
        writer.writerow(
            [
                "the same line of pipeline.py",
                conflict_facts["exit_code"],
                conflict_facts["markers"],
                conflict_facts["status"],
                conflict_facts["index_versions"],
                conflict_facts["new_commits"],
                "(no commit)",
            ]
        )
        writer.writerow(
            [
                "the same line, after a human resolved it",
                resolve_facts["exit_code"],
                resolve_facts["markers"],
                resolve_facts["status"],
                resolve_facts["index_versions"],
                resolve_facts["new_commits"],
                resolve_facts["parents"],
            ]
        )

    with (TABLES / "the-conflicted-file.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["line", "text", "belongs to"])
        side = "both sides agree"
        for i, text in enumerate(
            resolve_facts["conflicted_text"].splitlines(), start=1
        ):
            if text.startswith("<<<<<<<"):
                side, tag = "ours", "marker"
            elif text.startswith("======="):
                side, tag = "theirs", "marker"
            elif text.startswith(">>>>>>>"):
                side, tag = "both sides agree", "marker"
            else:
                tag = {
                    "ours": "HEAD (main, what you have)",
                    "theirs": "retries (what you are merging in)",
                }.get(side, "both sides agree")
            writer.writerow([i, text, tag])

    with (TABLES / "the-resolution.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["version of line 5", "the line", "kept?"])
        writer.writerow(
            ["HEAD (main, what you have)", resolve_facts["ours"], "not on its own"]
        )
        writer.writerow(
            [
                "retries (what you are merging in)",
                resolve_facts["theirs"],
                "not on its own",
            ]
        )
        writer.writerow(
            ["what the human wrote", resolve_facts["resolved"], "yes, both intentions"]
        )

    with (TABLES / "merge-abort.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "what git knows",
                "before the merge",
                "during the conflict",
                "after git merge --abort",
            ]
        )
        keys = [
            ("head", "HEAD"),
            ("merge_head", ".git/MERGE_HEAD exists"),
            ("unmerged", "unmerged files"),
            ("commits", "commits in the repository"),
            ("file", "pipeline.py holds"),
        ]
        for key, label in keys:
            writer.writerow(
                [
                    label,
                    abort_facts["before"][key],
                    abort_facts["during"][key],
                    abort_facts["after"][key],
                ]
            )

    fields = [
        "story",
        "step",
        "label",
        "command",
        "head",
        "head_sha",
        "branches",
        "unmerged_files",
    ]
    with (TABLES / "state-log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log)


def main():
    clear(FIGURES, TABLES)

    # The README describes the file it is about. Check that description too.
    lines = BASE.splitlines()
    assert len(lines) == 11, "the README calls pipeline.py an eleven line file"
    assert lines[0] == "TIMEOUT = 30", "tuning rewrites line 1"
    assert lines[4] == FETCH_LINE, "both branches of the second story rewrite line 5"
    assert lines[10] == '    print(len(rows), "rows")', "main rewrites line 11"

    clean_box, clean_facts = clean_merge()
    n_clean = render_dag(clean_box, "clean")
    log = log_rows(clean_box, "different lines")

    conflict_box, conflict_facts, abort_facts, resolve_facts = conflicted_merge()
    n_conflict = render_dag(conflict_box, "conflict")
    log += log_rows(conflict_box, "the same line")

    write_tables(clean_facts, conflict_facts, abort_facts, resolve_facts, log)

    # The figures the README embeds all exist, and the two stories really are
    # the same commands with a different outcome.
    figures = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figures) == n_clean + n_conflict + 3, (
        f"every snapshot plus the three file figures, got {figures}"
    )
    for name in ("clean-file.png", "conflict-file.png", "resolved-file.png"):
        assert name in figures, f"{name} was not drawn"
    assert clean_facts["exit_code"] == 0 and conflict_facts["exit_code"] != 0, (
        "the only difference between the two stories is WHICH lines were edited"
    )

    print(
        f"{len(figures)} figures, {len(list(TABLES.glob('*.csv')))} tables. "
        f"Clean merge exit {clean_facts['exit_code']}, conflicted merge exit "
        f"{conflict_facts['exit_code']}, status {conflict_facts['status']}. "
        "All assertions passed."
    )
    clean_box.cleanup()
    conflict_box.cleanup()


if __name__ == "__main__":
    main()
