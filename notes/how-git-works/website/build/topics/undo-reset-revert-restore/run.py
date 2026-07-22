"""Undo: restore, reset, revert, read as moves on the three trees.

Every undo command in git is a move on one, two or three of the three trees
(working tree, index, HEAD). This script runs each of them for real against an
identical starting history, reads all three trees back out afterwards, and
derives the command-to-tree mapping table from what it observed. Nothing in the
table is typed by hand, so a wrong claim in the README fails the run.

The recoverability claims are checked the same way. An edit is "gone" only if
the blob git would need to give it back is genuinely absent from the object
store, which is checked with `git hash-object --stdin` plus `git cat-file -e`,
never assumed.
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

from lib.gitviz import (  # noqa: E402
    Sandbox,
    clear,
    draw,
    draw_trees,
    layout,
    read_trees,
)

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

PIPE = "pipeline.py"
NOTES = "notes.md"

# Three versions of the script. V2 is the last commit, the one every undo below
# is aimed at. V3 is an edit that only ever gets staged, never committed.
PIPE_V1 = "load()\n"
PIPE_V2 = "load()\nclean()\n"
PIPE_V3 = "load()\nclean()\nmodel()\n"

# Two versions of the notes. V2 is never staged and never committed, so git has
# no copy of it anywhere. That is what makes it the one thing git cannot restore.
NOTES_V1 = "TODO: tune the model\n"
NOTES_V2 = "TODO: tune the model\nTODO: add a smoke test\n"


# ---- reading facts back out of the real repository ----------------------


def run(box: Sandbox, *args: str, who: str = "alice", stdin: str | None = None):
    return subprocess.run(
        ["git", *args],
        cwd=box.paths[who],
        input=stdin,
        capture_output=True,
        text=True,
    )


def head_sha(box: Sandbox) -> str:
    return run(box, "rev-parse", "--short", "HEAD").stdout.strip()


def commit_count(box: Sandbox) -> int:
    return int(run(box, "rev-list", "--count", "HEAD").stdout.strip())


def status(box: Sandbox, filename: str) -> str:
    # The two porcelain columns are index-vs-HEAD and working-tree-vs-index, and
    # the first one can be a space, so only the trailing newline comes off.
    return run(box, "status", "--porcelain", filename).stdout.rstrip("\n") or "(clean)"


def reachable(box: Sandbox, sha: str) -> bool:
    """Can you still get to this commit by walking back from HEAD?

    This is the whole question behind 'safe on shared history'. If a command
    leaves the old commit reachable it added to history. If not, it rewrote it.
    """
    return run(box, "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0


def blob_in_store(box: Sandbox, content: str) -> bool:
    """Does git's object store hold a blob with exactly these bytes?

    hash-object without -w computes the hash the content would have and writes
    nothing, so asking the question cannot change the answer. cat-file -e then
    says whether that object actually exists. If it does not, git cannot give
    this content back, by any command, ever.
    """
    sha = run(box, "hash-object", "--stdin", stdin=content).stdout.strip()
    return run(box, "cat-file", "-e", sha).returncode == 0


def object_type(box: Sandbox, sha: str) -> str:
    return run(box, "cat-file", "-t", sha).stdout.strip()


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


# ---- the identical starting point every undo is measured against --------


def quiet_commit(box: Sandbox, filename: str, content: str, message: str):
    """A commit that does not show up in the figure's command line."""
    box.write("alice", filename, content, record=False)
    box.git("alice", f"add {filename}", record=False)
    box.git("alice", f"commit -m {message}", record=False)


def two_commits() -> Sandbox:
    box = Sandbox(people=("alice",), show_unreachable=True)
    quiet_commit(box, NOTES, NOTES_V1, "add-notes")
    quiet_commit(box, PIPE, PIPE_V1, "add-loader")
    return box


def base() -> Sandbox:
    """Three commits on main. The last one, add-cleaning, is what we undo."""
    box = two_commits()
    quiet_commit(box, PIPE, PIPE_V2, "add-cleaning")
    return box


# ---- the observation harness -------------------------------------------


def event(box: Sandbox, command: str, filename: str, run_it) -> dict:
    """Snapshot the three trees, run one command, snapshot them again.

    Which trees the command moved is then a fact about the two snapshots, not
    a claim. Every row of the mapping table is built this way.
    """
    before_trees = read_trees(box, "alice", filename)
    before = (head_sha(box), before_trees["index"], before_trees["working tree"])
    count_before = commit_count(box)

    run_it()

    after_trees = read_trees(box, "alice", filename)
    after = (head_sha(box), after_trees["index"], after_trees["working tree"])
    count_after = commit_count(box)

    moved_pointer = before[0] != after[0]
    old_head_still_reachable = reachable(box, before[0])
    return {
        "command": command,
        "file_observed": filename,
        "moves_working_tree": yn(before[2] != after[2]),
        "moves_index": yn(before[1] != after[1]),
        "moves_branch_pointer": yn(moved_pointer),
        "commits_before": count_before,
        "commits_after": count_after,
        "previous_head_still_reachable": yn(old_head_still_reachable),
        "rewrites_history": yn(moved_pointer and not old_head_still_reachable),
    }


def main():
    clear(FIGURES, TABLES)
    rows: dict[str, dict] = {}

    # ================================================================
    # 1. restore: the two undos that never touch history at all.
    # ================================================================
    r = base()
    c3 = head_sha(r)

    # Stage an edit. The index and the working tree now agree with each other
    # and disagree with HEAD.
    r.write("alice", PIPE, PIPE_V3, record=False)
    r.git("alice", f"add {PIPE}", record=False)
    assert status(r, PIPE) == "M  " + PIPE, "the edit is staged"
    draw_trees(
        read_trees(r, "alice", PIPE),
        FIGURES / "trees-01-staged.png",
        "An edit, staged",
        f"edit {PIPE}; git add {PIPE}",
        "The index and the working tree hold the new version. HEAD still holds the old one.",
    )

    rows["restore --staged"] = event(
        r,
        f"git restore --staged {PIPE}",
        PIPE,
        lambda: r.git("alice", f"restore --staged {PIPE}"),
    )
    trees = read_trees(r, "alice", PIPE)
    assert trees["index"] == trees["HEAD"] == PIPE_V2, (
        "unstage pulled the index back to HEAD"
    )
    assert trees["working tree"] == PIPE_V3, (
        "unstage left the working tree alone: the edit is still there"
    )
    assert status(r, PIPE) == " M " + PIPE, "the edit is now modified but unstaged"
    assert rows["restore --staged"]["moves_index"] == "yes"
    assert rows["restore --staged"]["moves_working_tree"] == "no"
    assert rows["restore --staged"]["moves_branch_pointer"] == "no"
    assert blob_in_store(r, PIPE_V3), (
        "git add had already written this blob, and unstaging does not delete it"
    )
    draw_trees(
        trees,
        FIGURES / "trees-02-restore-staged.png",
        "git restore --staged pipeline.py",
        f"git restore --staged {PIPE}",
        "The index moved back to HEAD. The working tree did not move. The edit is safe.",
    )

    # Now an edit that is never staged. Git has no copy of it anywhere.
    r.write("alice", NOTES, NOTES_V2, record=False)
    assert status(r, NOTES) == " M " + NOTES, "modified, not staged"
    assert not blob_in_store(r, NOTES_V2), (
        "an unstaged edit was never written into the object store"
    )
    draw_trees(
        read_trees(r, "alice", NOTES),
        FIGURES / "trees-03-dirty.png",
        "An edit that was never staged",
        f"edit {NOTES}",
        "This version lives only on disk. Git has never been given a copy of it.",
    )

    rows["restore"] = event(
        r, f"git restore {NOTES}", NOTES, lambda: r.git("alice", f"restore {NOTES}")
    )
    trees = read_trees(r, "alice", NOTES)
    assert trees["working tree"] == trees["index"] == trees["HEAD"] == NOTES_V1, (
        "restore overwrote the working tree from the index"
    )
    assert status(r, NOTES) == "(clean)"
    assert rows["restore"]["moves_working_tree"] == "yes"
    assert rows["restore"]["moves_index"] == "no"
    assert rows["restore"]["moves_branch_pointer"] == "no"
    assert not blob_in_store(r, NOTES_V2), (
        "the edit is unrecoverable: git never stored it, so no git command can return it"
    )
    draw_trees(
        trees,
        FIGURES / "trees-04-restore.png",
        "git restore notes.md",
        f"git restore {NOTES}",
        "The working tree was overwritten from the index. The edit is gone, and git never had a copy.",
    )
    restore_lost_forever = not blob_in_store(r, NOTES_V2)

    # ================================================================
    # 2. reset --hard: the graph move, drawn, plus the orphan it leaves.
    # ================================================================
    h = two_commits()
    c2 = head_sha(h)
    h.snap("two commits on main")  # seeds the highlight set; not drawn
    h.commit("alice", PIPE, PIPE_V2, "add cleaning")
    assert head_sha(h) == c3, "same content, same parent, same author: the same commit"
    h.write("alice", NOTES, NOTES_V2, record=False)
    h.snap(
        "Three commits on main",
        note="notes.md also carries an uncommitted, unstaged edit. Watch what happens to it.",
    )

    rows["reset --hard"] = event(
        h,
        "git reset --hard HEAD~1",
        PIPE,
        lambda: h.git("alice", "reset --hard HEAD~1"),
    )
    h.snap(
        "After git reset --hard HEAD~1",
        note="main moved back one commit. The commit it left behind is unreferenced, not deleted.",
    )

    state = h.read("alice")
    assert state.branches["main"] == c2, "the branch pointer moved back one commit"
    assert c3 in state.commits, "the orphaned commit is still in the repository"
    assert object_type(h, c3) == "commit", (
        "the commit object itself is still on disk, byte for byte"
    )
    assert not reachable(h, c3), "but nothing points at it any more"
    trees_hard = read_trees(h, "alice", PIPE)
    assert (
        trees_hard["working tree"]
        == trees_hard["index"]
        == trees_hard["HEAD"]
        == PIPE_V1
    ), "--hard moved all three trees"
    assert status(h, PIPE) == "(clean)", "no gaps left between the three trees"
    assert (h.paths["alice"] / NOTES).read_text() == NOTES_V1, (
        "--hard also wiped the uncommitted edit to notes.md"
    )
    assert not blob_in_store(h, NOTES_V2), "and that edit is unrecoverable too"
    hard_orphan_survives = object_type(h, c3) == "commit"
    hard_lost_forever = not blob_in_store(h, NOTES_V2)

    xs, ys = layout(h.snapshots)
    draw(h.snapshots[1], xs, ys, FIGURES / "graph-01-before-reset.png", mode="solo")
    draw(h.snapshots[2], xs, ys, FIGURES / "graph-02-after-reset.png", mode="solo")
    draw_trees(
        trees_hard,
        FIGURES / "trees-07-reset-hard.png",
        "git reset --hard HEAD~1",
        "git reset --hard HEAD~1",
        "All three trees moved. The working tree is wiped clean of the undone work.",
    )

    # ================================================================
    # 3. reset --soft and reset --mixed: the same graph move, different trees.
    # ================================================================
    s = base()
    assert head_sha(s) == c3
    rows["reset --soft"] = event(
        s,
        "git reset --soft HEAD~1",
        PIPE,
        lambda: s.git("alice", "reset --soft HEAD~1"),
    )
    trees_soft = read_trees(s, "alice", PIPE)
    assert trees_soft["HEAD"] == PIPE_V1, "the branch pointer moved back"
    assert trees_soft["index"] == PIPE_V2, (
        "--soft did NOT touch the index: the undone commit's content is still staged"
    )
    assert trees_soft["working tree"] == PIPE_V2, "and the working tree is untouched"
    assert status(s, PIPE) == "M  " + PIPE, "staged, ready to be committed again"
    assert rows["reset --soft"]["moves_branch_pointer"] == "yes"
    assert rows["reset --soft"]["moves_index"] == "no"
    assert rows["reset --soft"]["moves_working_tree"] == "no"
    draw_trees(
        trees_soft,
        FIGURES / "trees-05-reset-soft.png",
        "git reset --soft HEAD~1",
        "git reset --soft HEAD~1",
        "Only the branch pointer moved. The undone work is still staged, ready to recommit.",
    )

    m = base()
    rows["reset --mixed"] = event(
        m,
        "git reset --mixed HEAD~1",
        PIPE,
        lambda: m.git("alice", "reset --mixed HEAD~1"),
    )
    trees_mixed = read_trees(m, "alice", PIPE)
    assert trees_mixed["HEAD"] == trees_mixed["index"] == PIPE_V1, (
        "--mixed moved the branch pointer and the index"
    )
    assert trees_mixed["working tree"] == PIPE_V2, (
        "the work survives in the working tree, unstaged"
    )
    assert status(m, PIPE) == " M " + PIPE, "modified, not staged"
    assert rows["reset --mixed"]["moves_branch_pointer"] == "yes"
    assert rows["reset --mixed"]["moves_index"] == "yes"
    assert rows["reset --mixed"]["moves_working_tree"] == "no"
    draw_trees(
        trees_mixed,
        FIGURES / "trees-06-reset-mixed.png",
        "git reset --mixed HEAD~1 (the default)",
        "git reset --mixed HEAD~1",
        "The pointer and the index moved. The work survives on disk, unstaged.",
    )

    # The three modes are the SAME move on the graph. They differ only below the
    # pointer, which is exactly why one figure of the graph serves all three.
    hard_graph = (h.read("alice").branches["main"], set(h.read("alice").commits))
    for name, other in (("--soft", s), ("--mixed", m)):
        state = other.read("alice")
        assert (state.branches["main"], set(state.commits)) == hard_graph, (
            f"reset {name} makes the identical move on the commit graph as --hard"
        )
        assert c3 in state.commits, (
            f"reset {name} orphans the commit too, it does not delete it"
        )

    # ================================================================
    # 4. revert: the only one that adds instead of rewriting.
    # ================================================================
    v = base()
    v.snap("three commits on main")  # seeds the highlight set; not drawn
    rows["revert"] = event(
        v, f"git revert {c3}", PIPE, lambda: v.git("alice", f"revert --no-edit {c3}")
    )
    v.snap(
        "After git revert",
        note="A brand new commit that applies the inverse change. Nothing was rewritten.",
    )

    assert rows["revert"]["commits_after"] == rows["revert"]["commits_before"] + 1, (
        "revert made the commit count go UP"
    )
    assert reachable(v, c3), (
        "the reverted commit is still reachable from HEAD: history was added to, not rewritten"
    )
    assert rows["revert"]["rewrites_history"] == "no"
    state = v.read("alice")
    tip = state.branches["main"]
    assert tip not in (c2, c3), "HEAD is a brand new commit, not an old one"
    assert state.commits[tip].parents == [c3], (
        "the revert commit sits directly on top of the commit it undoes"
    )
    trees_revert = read_trees(v, "alice", PIPE)
    assert (
        trees_revert["working tree"]
        == trees_revert["index"]
        == trees_revert["HEAD"]
        == PIPE_V1
    ), "the content is back to the pre-commit version"
    assert status(v, PIPE) == "(clean)"

    xs, ys = layout(v.snapshots)
    draw(v.snapshots[1], xs, ys, FIGURES / "graph-03-after-revert.png", mode="solo")
    draw_trees(
        trees_revert,
        FIGURES / "trees-08-revert.png",
        f"git revert {c3}",
        f"git revert {c3}",
        "The content is back, but HEAD is a new commit. The old one is still in the history.",
    )

    # ================================================================
    # The tables, built from the observations above and nothing else.
    # ================================================================
    order = [
        "restore",
        "restore --staged",
        "reset --soft",
        "reset --mixed",
        "reset --hard",
        "revert",
    ]
    fields = [
        "command",
        "file_observed",
        "moves_working_tree",
        "moves_index",
        "moves_branch_pointer",
        "commits_before",
        "commits_after",
        "previous_head_still_reachable",
        "rewrites_history",
    ]
    with (TABLES / "which-tree-moves.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in order:
            writer.writerow(rows[key])

    with (TABLES / "reset-modes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "mode",
                "branch_tip",
                "head_holds",
                "index_holds",
                "working_tree_holds",
                "git_status",
            ]
        )
        for mode, box, trees in (
            ("--soft", s, trees_soft),
            ("--mixed (the default)", m, trees_mixed),
            ("--hard", h, trees_hard),
        ):
            flat = [
                (trees[k] or "").strip().replace("\n", " / ")
                for k in ("HEAD", "index", "working tree")
            ]
            writer.writerow(
                [
                    mode,
                    box.read("alice").branches["main"],
                    flat[0],
                    flat[1],
                    flat[2],
                    status(box, PIPE),
                ]
            )

    with (TABLES / "what-survives.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "event",
                "what it overwrote",
                "object git would need",
                "still in the object store",
                "recoverable with",
            ]
        )
        for event_name, what, kind, survived in (
            (
                f"git restore {NOTES}",
                "an uncommitted, unstaged edit",
                "a blob",
                not restore_lost_forever,
            ),
            (
                "git reset --hard HEAD~1",
                f"the commit add-cleaning ({c3})",
                "a commit",
                hard_orphan_survives,
            ),
            (
                "git reset --hard HEAD~1",
                "an uncommitted, unstaged edit to notes.md",
                "a blob",
                not hard_lost_forever,
            ),
        ):
            writer.writerow(
                [
                    event_name,
                    what,
                    kind,
                    yn(survived),
                    "the reflog" if survived else "nothing. git never stored it",
                ]
            )

    figures = len(list(FIGURES.glob("*.png")))
    print(f"{figures} figures, 3 tables. All checks passed.")
    print(f"  the undone commit is {c3}, its parent is {c2}")
    print(
        f"  reset --hard orphaned {c3} but the object survives: {hard_orphan_survives}"
    )
    print(f"  git restore destroyed an edit git never stored: {restore_lost_forever}")
    for key in order:
        row = rows[key]
        print(
            f"  {row['command']:<32} "
            f"working tree {row['moves_working_tree']:<3} "
            f"index {row['moves_index']:<3} "
            f"pointer {row['moves_branch_pointer']:<3} "
            f"rewrites history {row['rewrites_history']}"
        )

    for box in (r, s, m, h, v):
        box.cleanup()


if __name__ == "__main__":
    main()
