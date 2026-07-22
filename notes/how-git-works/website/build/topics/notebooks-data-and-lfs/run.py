"""The data scientist's coda: notebooks, data, and big binaries in git.

Git is built for source and small text, and it shows its limits the moment a
data scientist hands it the three things that fill a real project: notebooks,
datasets, and model files. This tutorial demonstrates each problem for real,
never by assertion.

  1. A notebook is JSON, and git merges by line. Two harmless-looking edits to
     one .ipynb (one person just re-ran it, the other edited a cell) collide in
     the machine-generated execution_count and outputs, and git writes conflict
     markers straight into the JSON. The file is then no longer valid JSON, so
     Jupyter cannot open it. This is the centerpiece, shown from a real merge.
  2. A big binary committed once lives in history forever. Adding a model
     checkpoint and then removing it in a later commit does not shrink the
     repository, because the old commit still holds the blob. Measured on disk.
  3. A .gitignore keeps data/ and *.parquet out of git status. Checked with
     git's own check-ignore, mirroring 00-first-steps/ignoring-files/.

Every claim the README makes is checked against real git at the bottom. A wrong
picture fails the run.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():  # walk up to the collection root
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw_conflict, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

NB = "analysis.ipynb"
MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
FRIENDLY = "/home/you/sales-analysis"


# ---- the tiny notebook, built as real nbformat JSON --------------------


def notebook(execution_count: int, out_text: str, source: list[str]) -> str:
    """One code cell, written the way Jupyter writes it: pretty JSON, one token
    per line. That multi-line layout is exactly why git tries to merge a
    notebook line by line, and why it fails."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {},
                "outputs": [
                    {
                        "data": {"text/plain": [out_text]},
                        "execution_count": execution_count,
                        "metadata": {},
                        "output_type": "execute_result",
                    }
                ],
                "source": source,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1) + "\n"


# The base notebook, committed to main. One cell computes a mean, run once.
BASE = notebook(1, "42.0", ["df['units'].mean()"])

# Alice does nothing but click Run All after the monthly data landed. She did
# not touch a line of code. Only the machine-written parts move: the cell ran a
# third time this session, and the number changed because the data changed.
ALICE = notebook(3, "44.0", ["df['units'].mean()"])

# Bob, on his own branch, adds a median to the same cell and runs it. His code
# edit is real, and it too rewrites execution_count and the output.
BOB = notebook(
    2,
    "(42.0, 40.0)",
    ["mean = df['units'].mean()\n", "median = df['units'].median()\n", "mean, median"],
)


def clean(text: str, real_root: Path) -> str:
    """Swap the throwaway temp path for a friendly one, both raw and resolved,
    because macOS reports temp paths through /private."""
    for form in (str(real_root.resolve()), str(real_root)):
        text = text.replace(form, FRIENDLY)
    return text.rstrip("\n")


def objects_size(repo: Path) -> int:
    """Total bytes in the object store. This is the history git will carry in
    every clone, forever, whether or not any branch still points at a file."""
    base = repo / ".git" / "objects"
    return sum(p.stat().st_size for p in base.rglob("*") if p.is_file())


def markers_in(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(MARKERS))


def main():
    clear(FIGURES, TABLES)
    outputs: dict[str, str] = {}

    # ==================================================================
    # PART 1: a real notebook merge conflict
    # ==================================================================
    box = Sandbox(people=("alice",), local=True)
    box.git("alice", "config merge.conflictstyle merge", record=False)
    repo = box.paths["alice"]

    def show(args: str) -> str:
        return clean(
            subprocess.run(
                ["git", *args.split()], cwd=repo, capture_output=True, text=True
            ).stdout,
            repo,
        )

    # main carries the base notebook.
    box.commit("alice", NB, BASE, "add-the-analysis-notebook")
    box.snap(
        "The notebook lives on main",
        note="One committed notebook. It is a JSON file. Its execution counts "
        "and cell outputs are written by Jupyter, not by you.",
    )

    # Two branches from that one base. Alice re-runs, Bob edits a cell.
    box.git("alice", "switch -c refreshed-data")
    box.commit("alice", NB, ALICE, "rerun-after-the-monthly-data-landed")
    box.git("alice", "switch main")
    box.git("alice", "switch -c add-a-median")
    box.commit("alice", NB, BOB, "add-a-median-to-the-cell")
    box.snap(
        "Two branches, two edits to the same notebook",
        note="refreshed-data only re-ran the notebook. add-a-median edited the "
        "cell. Both changed the same JSON lines: execution_count and outputs.",
    )

    # Merge the two. Git tries a line merge of the JSON and cannot.
    box.git("alice", "switch refreshed-data")
    merge = box.git("alice", "merge add-a-median", check=False)
    outputs["merge_stdout"] = clean(merge.stdout, repo)
    conflicted = (repo / NB).read_text()
    outputs["status"] = show("status")
    outputs["status_porcelain"] = show(f"status --porcelain {NB}")

    # An excerpt of the wreckage, for the README: the first conflict hunk with a
    # little context, taken straight from the file git wrote.
    lines = conflicted.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("<<<<<<<"))
    stop = next(i for i, ln in enumerate(lines) if ln.startswith(">>>>>>>"))
    lo = max(0, start - 2)
    hi = min(len(lines), stop + 2)
    outputs["conflict_excerpt"] = "\n".join(lines[lo:hi])

    # Is the file still a notebook? Try to parse it as JSON the way Jupyter would.
    try:
        json.loads(conflicted)
        nb_still_valid = True
    except json.JSONDecodeError:
        nb_still_valid = False

    draw_conflict(
        box,
        "alice",
        NB,
        FIGURES / "notebook-conflict.png",
        "A notebook after a merge conflict, exactly as git wrote it",
        "git merge add-a-median",
        "The markers sit inside the JSON. The file no longer parses, so Jupyter "
        "cannot open it. Never resolve this by blindly keeping one side.",
    )

    # ---- the oracle for the notebook conflict -------------------------
    assert merge.returncode != 0, (
        f"merging two notebook edits must conflict, git returned {merge.returncode}"
    )
    assert "CONFLICT" in merge.stdout, "git reported a merge conflict"
    assert NB in merge.stdout, "and named the notebook as the conflicted file"
    assert markers_in(conflicted) >= 3, (
        "git wrote conflict markers into the notebook JSON"
    )
    assert all(m in conflicted for m in MARKERS), (
        "all three marker kinds are present in the file"
    )
    assert not nb_still_valid, (
        "the conflicted .ipynb is no longer valid JSON, so Jupyter cannot open it"
    )
    assert json.loads(BASE) and json.loads(ALICE) and json.loads(BOB), (
        "each individual version was valid JSON before the merge; only the "
        "conflicted merge product is broken"
    )
    assert outputs["status_porcelain"].startswith("UU"), (
        f"git status marks the notebook unmerged, got {outputs['status_porcelain']!r}"
    )
    assert "both modified:" in outputs["status"], (
        "git status spells out that both sides changed the notebook"
    )
    assert "<<<<<<<" in outputs["conflict_excerpt"], (
        "the README excerpt really contains the conflict markers"
    )
    # The number Alice saw and the number Bob saw are both trapped in the file.
    assert "44.0" in conflicted and "40.0" in conflicted, (
        "both people's outputs are stranded inside the conflicted JSON"
    )

    render(box, FIGURES, TABLES)  # the DAG snapshots + state-log.csv
    box.cleanup()

    # ==================================================================
    # PART 2: a big binary bloats history forever
    # ==================================================================
    binbox = Sandbox(people=("alice",), local=True)
    brepo = binbox.paths["alice"]

    # A deterministic megabyte of pseudo-random bytes. Random data does not
    # compress, so git stores very nearly the whole megabyte, the way a real
    # model checkpoint would land.
    blob = random.Random(0).randbytes(1024 * 1024)

    binbox.commit("alice", "train.py", "# fits the model\n", "add-the-training-script")
    size_before = objects_size(brepo)

    # The mistake: someone commits a model checkpoint before they know better.
    (brepo / "model.ckpt").write_bytes(blob)
    binbox.git("alice", "add model.ckpt")
    binbox.git("alice", "commit -m commit-a-model-checkpoint")
    size_after_add = objects_size(brepo)

    # Realising the mistake, they delete it in a new commit.
    binbox.git("alice", "rm model.ckpt")
    binbox.git("alice", "commit -m remove-the-checkpoint")
    size_after_rm = objects_size(brepo)

    # The file is gone from the working tree and from the latest commit ...
    working_tree_has_it = (brepo / "model.ckpt").exists()
    head_has_it = (
        binbox.git("alice", "cat-file -e HEAD:model.ckpt", check=False).returncode == 0
    )
    # ... but the blob is still reachable from the earlier commit, at full size.
    blob_sha = binbox.git(
        "alice", "rev-parse HEAD~1:model.ckpt", record=False
    ).stdout.strip()
    blob_exists = (
        binbox.git("alice", f"cat-file -e {blob_sha}", check=False).returncode == 0
    )
    stored_size = int(
        binbox.git("alice", f"cat-file -s {blob_sha}", record=False).stdout.strip()
    )

    # ---- the oracle for the bloat -------------------------------------
    assert size_after_add - size_before > 900_000, (
        "committing the checkpoint added roughly its megabyte to the object store"
    )
    assert not working_tree_has_it, "git rm deleted the file from the working tree"
    assert not head_has_it, "and the latest commit no longer contains it"
    assert blob_exists, (
        "yet the blob is still in the object database, reachable from the old commit"
    )
    assert stored_size == len(blob), (
        f"and it is stored at full size ({stored_size} bytes), not trimmed"
    )
    assert size_after_rm >= 0.9 * size_after_add, (
        "removing the file in a new commit did NOT shrink the repository: "
        f"{size_after_rm} bytes after rm vs {size_after_add} after add"
    )
    binbox.cleanup()

    # ==================================================================
    # PART 3: a .gitignore keeps data and parquet out of git status
    # ==================================================================
    igbox = Sandbox(people=("alice",), local=True)
    irepo = igbox.paths["alice"]

    igbox.commit("alice", "train.py", "# fits the model\n", "start-the-project")
    for name, content in {
        "data/sales.parquet": "PAR1... imagine two gigabytes ...PAR1\n",
        "data/raw.csv": "id,units\n1,42\n",
        "features.parquet": "PAR1... a feature matrix ...PAR1\n",
    }.items():
        path = irepo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    status_before_ignore = clean(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=irepo, capture_output=True, text=True
        ).stdout,
        irepo,
    )
    (irepo / ".gitignore").write_text("data/\n*.parquet\n")
    status_after_ignore = clean(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=irepo, capture_output=True, text=True
        ).stdout,
        irepo,
    )

    ignore_rows = []
    for target, expected in [
        ("data/sales.parquet", "data/"),
        ("data/raw.csv", "data/"),
        ("features.parquet", "*.parquet"),
    ]:
        proc = subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", target],
            cwd=irepo,
            capture_output=True,
            text=True,
        )
        matched = proc.stdout.split("\t")[0].split(":", 2)[-1] if proc.stdout else ""
        ignore_rows.append((target, expected, matched, proc.returncode == 0))

    # ---- the oracle for the .gitignore --------------------------------
    for token in ("data/", "features.parquet"):
        assert token in status_before_ignore, (
            f"before the .gitignore, {token} shows up as untracked noise"
        )
    for token in ("data/", ".parquet"):
        assert token not in status_after_ignore, (
            f"after the .gitignore, git stops mentioning {token}"
        )
    assert ".gitignore" in status_after_ignore, (
        "the only thing git still sees is the .gitignore itself"
    )
    for target, expected, matched, ok in ignore_rows:
        assert ok and matched == expected, (
            f"{target} should be ignored by {expected}, git said {matched!r}"
        )
    igbox.cleanup()

    # ==================================================================
    # TABLES
    # ==================================================================
    with (TABLES / "notebook-conflict.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fact", "value"])
        w.writerow(["git merge exit code", merge.returncode])
        w.writerow(["conflict marker lines in the notebook", markers_in(conflicted)])
        w.writerow(["git status for the notebook", outputs["status_porcelain"]])
        w.writerow(["valid JSON before the merge", "yes (each branch)"])
        w.writerow(
            ["valid JSON after the merge", "no" if not nb_still_valid else "yes"]
        )

    with (TABLES / "binary-bloat.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "object store bytes", "note"])
        w.writerow(
            ["after the training script", size_before, "a few small text objects"]
        )
        w.writerow(
            [
                "after committing model.ckpt",
                size_after_add,
                "the megabyte is now stored",
            ]
        )
        w.writerow(
            [
                "after git rm + commit",
                size_after_rm,
                "still there: the old commit holds the blob",
            ]
        )
        w.writerow(
            [
                "blob still readable from HEAD~1",
                stored_size,
                f"full size, sha {blob_sha}",
            ]
        )

    with (TABLES / "gitignore-check.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "expected pattern", "matched pattern", "ignored"])
        for target, expected, matched, ok in ignore_rows:
            w.writerow([target, expected, matched, "yes" if ok else "no"])

    # A reference table for the reader: what belongs in git and what does not.
    where = [
        (
            "Source code",
            ".py, .sql, .R, Dockerfile",
            "commit to git",
            "small, text, the line diff is meaningful",
        ),
        (
            "Notebooks",
            ".ipynb",
            "commit, but clear outputs first",
            "JSON merges badly, outputs and counts are churn",
        ),
        (
            "Small reference data",
            "a 5 KB lookup table, fixtures",
            "commit to git",
            "tiny and stable, versioning it helps",
        ),
        (
            "Large or churning data",
            ".parquet, big .csv, a db dump",
            "gitignore, then S3, DVC, or LFS",
            "git keeps every version forever, the repo balloons",
        ),
        (
            "Models and checkpoints",
            ".pkl, .pt, .ckpt, .onnx",
            "gitignore, then artifact storage or LFS",
            "large binaries, regenerated by training",
        ),
        (
            "Generated outputs",
            "figures, .html reports, build files",
            "gitignore, regenerate from source",
            "reproducible from code, nothing to store",
        ),
        (
            "Secrets",
            ".env, keys, tokens",
            "never commit, use a secret manager",
            "a committed secret lives in history forever",
        ),
        (
            "Caches",
            "__pycache__/, .ipynb_checkpoints/",
            "gitignore",
            "tools regenerate them on their own",
        ),
    ]
    with (TABLES / "what-goes-where.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["what", "examples", "where it goes", "why"])
        w.writerows(where)

    # Save the captured console output for the README to quote verbatim.
    with (TABLES / "session.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "output"])
        for key, value in outputs.items():
            w.writerow([key, value])

    # ==================================================================
    # figure and table counts
    # ==================================================================
    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert "notebook-conflict.png" in figs, "the conflict figure was drawn"
    assert len([f for f in figs if f.startswith("step-")]) == 2, (
        "two DAG snapshots: base notebook, and the two diverged branches"
    )

    tabs = sorted(p.name for p in TABLES.glob("*.csv"))
    for needed in (
        "notebook-conflict.csv",
        "binary-bloat.csv",
        "gitignore-check.csv",
        "what-goes-where.csv",
    ):
        assert needed in tabs, f"{needed} was written"

    print(
        f"notebook merge exit {merge.returncode}, "
        f"{markers_in(conflicted)} marker lines, JSON valid after merge: "
        f"{nb_still_valid}. Bloat: {size_before} -> {size_after_add} -> "
        f"{size_after_rm} object bytes, blob {blob_sha} still {stored_size} B. "
        f"gitignore hid {sum(1 for r in ignore_rows if r[3])}/{len(ignore_rows)} files."
    )
    print("figures:", figs)
    print("tables:", tabs)
    print("All assertions passed.")


if __name__ == "__main__":
    main()
