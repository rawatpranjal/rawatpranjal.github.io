"""The reflog: how to get back work that no branch names any more.

Stages three real disasters in a real repository, recovers from each with the
reflog, and checks every claim the README makes. A hard reset, a deleted branch
and a regretted rebase all move a pointer and leave the commit objects exactly
where they were, so the recovered tip comes back byte-identical rather than as a
copy. The limits are checked too: a fresh clone carries no reflog, and an edit
that was never committed is in no object and cannot come back.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw_trees, read_trees, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

PIPELINE = [
    "def load(): ...\n",
    "def clean(): ...\n",
    "def features(): ...\n",
    "def model(): ...\n",
    "def evaluate(): ...\n",
]

NEVER_COMMITTED = "# the eval split must be time-ordered\n"
STAGED_ONLY = "# cache the feature matrix on disk\n"


# ---- reading the truth back out ----------------------------------------


def sha(box: Sandbox, rev: str, who: str = "alice") -> str:
    """The full 40-character hash a name resolves to, right now."""
    return box.git(who, f"rev-parse {rev}", record=False).stdout.strip()


def exists(box: Sandbox, obj: str, who: str = "alice") -> bool:
    """git cat-file -e: does this object exist in this repository at all?"""
    return box.git(who, f"cat-file -e {obj}", record=False, check=False).returncode == 0


def obj_type(box: Sandbox, obj: str, who: str = "alice") -> str:
    return box.git(who, f"cat-file -t {obj}", record=False).stdout.strip()


def reachable(box: Sandbox, who: str = "alice") -> set[str]:
    """Every commit any branch or tag can walk to. The reflog is not consulted."""
    return set(box.git(who, "rev-list --all", record=False).stdout.split())


def reflog(box: Sandbox, who: str = "alice") -> list[tuple[str, str, str]]:
    """The real reflog, as (selector, full sha, action) triples."""
    raw = box.git(who, "reflog --format=%gd%x1f%H%x1f%gs", record=False).stdout
    out = []
    for line in raw.splitlines():
        selector, full, action = line.split("\x1f")
        out.append((selector, full, action))
    return out


def write_reflog(rows: list[tuple[str, str, str]], path: Path):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entry", "commit", "what moved HEAD there"])
        for selector, full, action in rows:
            writer.writerow([selector, full[:7], action])


def objects_holding(box: Sandbox, needle: str, who: str = "alice") -> list[str]:
    """Every object in the repository whose contents contain this text.

    Zero means git never saw it, and no command can bring it back.
    """
    listing = box.git(
        who, "cat-file --batch-all-objects --batch-check", record=False
    ).stdout
    hits = []
    for line in listing.splitlines():
        obj = line.split()[0]
        body = box.git(who, f"cat-file -p {obj}", record=False, check=False).stdout
        if needle in body:
            hits.append(obj)
    return hits


def main():
    clear(FIGURES, TABLES)
    # show_unreachable walks the reflog as well as the branches, so a commit that
    # nothing points at any more is still drawn. That is the whole subject here:
    # these commits are not gone, they are merely unnamed.
    box = Sandbox(people=("alice",), show_unreachable=True)

    # ---- a history worth losing -----------------------------------------
    for i, message in enumerate(
        ["add loader", "add cleaning", "add features", "add model", "add eval"], start=1
    ):
        box.commit("alice", "pipeline.py", "".join(PIPELINE[:i]), message)

    good_tip = sha(box, "HEAD")
    doomed = [sha(box, "HEAD"), sha(box, "HEAD~1"), sha(box, "HEAD~2")]
    good_file = (box.paths["alice"] / "pipeline.py").read_text()
    box.snap(
        "five commits on main",
        note="Five commits. main points at the newest one, and HEAD is attached to main.",
    )

    # ---- disaster 1: git reset --hard HEAD~3 ----------------------------
    box.git("alice", "reset --hard HEAD~3")
    after_reset = box.snap(
        "git reset --hard HEAD~3",
        note="main moved back three commits. The three commits it left behind are still on disk, named by nothing.",
    )

    live = reachable(box)
    for lost in doomed:
        assert lost not in live, "no branch can reach the commit any more"
        assert exists(box, lost), "but the commit object is still on disk"
        assert obj_type(box, lost) == "commit", "and it is still a commit"
    assert len(after_reset.repos["alice"].commits) == 5, (
        "the reflog view still sees all five: reset destroyed no object"
    )
    assert len(live) == 2, "only two commits are reachable from a branch now"
    assert (box.paths["alice"] / "pipeline.py").read_text() != good_file, (
        "--hard threw the working tree away too"
    )

    log_after_reset = reflog(box)
    assert log_after_reset[0][1] == sha(box, "HEAD"), "HEAD@{0} is where HEAD is now"
    assert log_after_reset[1][1] == good_tip, (
        "HEAD@{1} is where HEAD was before the reset"
    )
    assert sha(box, "HEAD@{1}") == good_tip, "and git resolves that selector to it"
    assert len(log_after_reset) == 6, "five commits plus the reset itself"
    write_reflog(log_after_reset, TABLES / "reflog-after-the-reset.csv")

    # ---- recovery 1 -----------------------------------------------------
    box.git("alice", "reset --hard HEAD@{1}")
    box.snap(
        "git reset --hard HEAD@{1}",
        note="HEAD@{1} is the position HEAD held one move ago. main is back on the same commit, not a copy of it.",
    )

    assert sha(box, "HEAD") == good_tip, (
        "the recovered tip is the identical hash, so it is the identical commit"
    )
    recovered_tip = sha(box, "main")
    assert recovered_tip == good_tip, "main points at it again"
    assert set(doomed) <= reachable(box), "all three commits are reachable again"
    assert (box.paths["alice"] / "pipeline.py").read_text() == good_file, (
        "the working tree came back with them"
    )

    # ---- disaster 2: a deleted branch -----------------------------------
    box.git("alice", "switch -c feature")
    box.commit("alice", "plots.py", "def roc(): ...\n", "add roc plot")
    box.commit("alice", "plots.py", "def roc(): ...\ndef pr(): ...\n", "add pr plot")
    feature_tip = sha(box, "HEAD")
    feature_first = sha(box, "HEAD~1")
    box.snap(
        "two commits on feature",
        note="feature is two commits ahead of main. Nothing is pushed, so these two commits exist on this laptop only.",
    )

    box.git("alice", "switch main")
    box.commit(
        "alice", "pipeline.py", "".join(PIPELINE) + "def report(): ...\n", "add report"
    )
    box.commit(
        "alice",
        "pipeline.py",
        "".join(PIPELINE) + "def report(): ...\ndef publish(): ...\n",
        "add publish",
    )
    main_tip = sha(box, "HEAD")
    box.snap(
        "main moves on",
        note="main gains two commits of its own. The history has forked.",
    )

    box.git("alice", "branch -D feature")
    box.snap(
        "git branch -D feature",
        note="The name is gone. Both of its commits are still here, reachable from the reflog and from nothing else.",
    )

    assert "feature" not in box.read("alice").branches, "the pointer is gone"
    assert feature_tip not in reachable(box), "no branch reaches the work any more"
    assert exists(box, feature_tip) and exists(box, feature_first), (
        "both commits are still objects on disk"
    )

    log_after_delete = reflog(box)
    delete_idx = next(
        i for i, (_, full, _) in enumerate(log_after_delete) if full == feature_tip
    )
    assert delete_idx == 3, "the README quotes HEAD@{3} as the last commit on feature"
    assert log_after_delete[delete_idx][2] == "commit: add-pr-plot", (
        "and that entry is the commit that made it"
    )
    assert sha(box, "HEAD@{3}") == feature_tip
    write_reflog(log_after_delete, TABLES / "reflog-after-the-branch-delete.csv")

    # ---- recovery 2 -----------------------------------------------------
    box.git("alice", "branch feature HEAD@{3}")
    box.snap(
        "git branch feature HEAD@{3}",
        note="A new pointer at an old commit. The branch is not rebuilt, it is renamed back onto work that never left.",
    )

    assert sha(box, "feature") == feature_tip, "the same tip, hash for hash"
    assert set([feature_tip, feature_first]) <= reachable(box), "both commits are back"

    # ---- disaster 3: a rebase you regret --------------------------------
    box.git("alice", "switch feature")
    box.git("alice", "rebase main")
    rebased_tip = sha(box, "HEAD")
    rebased_first = sha(box, "HEAD~1")
    box.snap(
        "git rebase main",
        note="feature now points at two brand new commits (red outline). The two originals are untouched, and nothing names them.",
    )

    assert rebased_tip != feature_tip, "a rebase copies, it never moves a commit"
    assert sha(box, f"{rebased_first}^") == main_tip, "the copies hang off main now"
    assert obj_type(box, f"{rebased_tip}^{{tree}}") == "tree"
    assert sha(box, f"{rebased_tip}^{{tree}}") != sha(box, f"{feature_tip}^{{tree}}"), (
        "the copy carries main's file as well, so even its tree differs"
    )
    assert exists(box, feature_tip), "the original commit survives the rebase"
    assert feature_tip not in reachable(box), "but nothing points at it"

    log_after_rebase = reflog(box)
    start_idx = next(
        i
        for i, (_, _, action) in enumerate(log_after_rebase)
        if action.startswith("rebase (start)")
    )
    pre_rebase_idx = start_idx + 1  # the entry one older is where HEAD was before
    assert pre_rebase_idx == 4, "the README quotes HEAD@{4} as the pre-rebase position"
    assert sha(box, "HEAD@{4}") == feature_tip, (
        "and that selector resolves to the original feature tip"
    )
    write_reflog(log_after_rebase, TABLES / "reflog-after-rebase.csv")

    # ---- recovery 3 -----------------------------------------------------
    box.git("alice", "reset --hard HEAD@{4}")
    box.snap(
        "git reset --hard HEAD@{4}",
        note="feature is back on the original commits, with their original hashes. The rebased copies are the orphans now.",
    )

    assert sha(box, "feature") == feature_tip, (
        "the original commits are back, not re-created: same hashes"
    )
    assert set([feature_tip, feature_first]) <= reachable(box)
    assert rebased_tip not in reachable(box), "the copies are the unreferenced ones now"
    assert exists(box, rebased_tip), "and they too are still on disk, for now"

    render(box, FIGURES, TABLES)

    # ---- limit 1: the reflog is local ------------------------------------
    box.git("alice", "switch main", record=False)
    box.git("alice", "push -u origin main", record=False)
    fresh = box.root / "fresh-clone"
    box.git("alice", f"clone {box.remote} {fresh}", cwd=box.root, record=False)
    box.paths["fresh"] = fresh

    clone_log = reflog(box, who="fresh")
    assert len(clone_log) == 1, "a fresh clone's reflog has exactly one entry"
    assert clone_log[0][2].startswith("clone:"), "and that entry is the clone itself"
    assert not any(
        action.startswith(("reset:", "rebase")) for _, _, action in clone_log
    ), "none of the local moves travelled with the push"
    assert len(reflog(box)) > 10, "while the local reflog holds every move made here"
    assert not exists(box, rebased_tip, who="fresh"), (
        "an unreferenced commit is not even sent to a clone"
    )
    assert not exists(box, feature_tip, who="fresh"), (
        "and neither is a branch that was never pushed"
    )
    write_reflog(clone_log, TABLES / "reflog-in-a-fresh-clone.csv")

    # ---- limit 2: it cannot recover what was never committed --------------
    box.write("alice", "pipeline.py", good_file + NEVER_COMMITTED, record=False)
    trees = read_trees(box, "alice", "pipeline.py")
    assert NEVER_COMMITTED in trees["working tree"]
    assert NEVER_COMMITTED not in trees["index"], "it was never staged"
    assert NEVER_COMMITTED not in trees["HEAD"], "and never committed"
    assert objects_holding(box, NEVER_COMMITTED.strip()) == [], (
        "the edit is in no object: git has never seen it"
    )
    draw_trees(
        trees,
        FIGURES / "uncommitted-edit.png",
        "An edit git has never seen",
        "edit pipeline.py",
        "The new line exists on disk and in no git object. Nothing, including the reflog, can bring it back once it is overwritten.",
    )

    box.git("alice", "restore pipeline.py")
    assert NEVER_COMMITTED not in (box.paths["alice"] / "pipeline.py").read_text(), (
        "git restore overwrote the file from the index"
    )
    assert objects_holding(box, NEVER_COMMITTED.strip()) == [], (
        "and there is no object to recover it from. It is gone for good."
    )
    assert not any(NEVER_COMMITTED.strip() in action for _, _, action in reflog(box)), (
        "the reflog logs pointer moves, and this was never a pointer move"
    )

    # The one nuance: git add writes a blob. That blob outlives a restore, so a
    # staged edit is recoverable, though not by the reflog.
    box.write("alice", "pipeline.py", good_file + STAGED_ONLY, record=False)
    box.git("alice", "add pipeline.py")
    staged_blob = sha(box, ":pipeline.py")
    box.git("alice", "restore --source=HEAD --staged --worktree pipeline.py")

    assert STAGED_ONLY not in (box.paths["alice"] / "pipeline.py").read_text()
    assert exists(box, staged_blob), "git add wrote a blob, and the blob is still there"
    assert objects_holding(box, STAGED_ONLY.strip()) == [staged_blob], (
        "exactly one object holds it"
    )
    assert not any(staged_blob in full for _, full, _ in reflog(box)), (
        "no reflog entry names it: a blob is not a pointer position"
    )
    dangling = box.git("alice", "fsck --unreachable", record=False, check=False).stdout
    assert f"unreachable blob {staged_blob}" in dangling, (
        "git fsck finds it, which is the tool for this case, not the reflog"
    )

    # ---- the claims table -------------------------------------------------
    with (TABLES / "what-survives.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["claim checked in run.py", "measured"])
        rows = [
            ("commits reachable from a branch before the reset", 5),
            (
                "commits reachable from a branch after git reset --hard HEAD~3",
                len(live),
            ),
            ("of the 3 lost commits, how many still exist as objects", len(doomed)),
            ("reflog entries after the reset", len(log_after_reset)),
            ("HEAD@{1} resolves to the pre-reset tip", "yes"),
            ("recovered main tip equals the pre-disaster tip, hash for hash", "yes"),
            ("commits of the deleted branch still on disk after git branch -D", 2),
            ("commits the rebase copied", 2),
            ("original pre-rebase commits still on disk after the rebase", 2),
            ("HEAD@{4} resolves to the pre-rebase tip", "yes"),
            ("recovered feature tip equals the pre-rebase tip, hash for hash", "yes"),
            ("reflog entries in this repository", len(reflog(box))),
            ("reflog entries in a fresh clone of it", len(clone_log)),
            ("objects in the clone holding the unpushed commits", 0),
            ("objects holding an edit that was never staged", 0),
            ("objects holding an edit that was staged then restored", 1),
        ]
        writer.writerows(rows)

    figures = sorted(p.name for p in FIGURES.glob("*.png"))
    tables = sorted(p.name for p in TABLES.glob("*.csv"))
    print(f"figures: {', '.join(figures)}")
    print(f"tables:  {', '.join(tables)}")
    print(f"pre-reset main tip:     {good_tip}")
    print(f"recovered main tip:     {recovered_tip}")
    print(f"pre-rebase feature tip: {feature_tip}")
    print(f"recovered feature tip:  {sha(box, 'feature')}")
    print("all assertions passed")
    box.cleanup()


if __name__ == "__main__":
    main()
