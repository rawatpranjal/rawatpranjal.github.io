"""Contributing to a repo you do not own: the fork model, built for real.

There are two servers here, not one. `upstream` is the original project, which
the contributor has no permission to push to. `github` is the contributor's own
server-side copy of it, a fork, which the contributor owns and can push to. The
contributor clones the fork (so `origin` is the fork), adds `upstream` as a
second remote to stay current, does the work on a branch, and pushes that branch
to the fork. Upstream never moves under the contributor, because the contributor
cannot move it.

The original history on `upstream` and the later commit that advances it both
belong to the project maintainer, so they are made in a throwaway maintainer
clone that is never registered as a person or a server and therefore never drawn.
Only the two servers and the contributor Alice appear in the figures.

Every claim the README makes is checked at the bottom of main(). Two servers
exist. After Alice pushes her branch to her fork, the fork has her commit and
upstream has none of it. Her clone knows both remotes. After she fetches
upstream, upstream's newer commit shows up as `upstream/main` in her repo. A
wrong claim fails the run.
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

from lib.gitviz import Sandbox, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

# The four servers-and-people the figures show. The maintainer clone is not
# here, so it never becomes a panel.
PANELS = ("upstream", "github", "alice")


def maintainer_commit(box: Sandbox, filename: str, content: str, message: str):
    """A commit made in the maintainer's own clone, off the drawn stage.

    Records nothing, so it never leaks into a contributor figure's caption.
    """
    (box.paths["maintainer"] / filename).write_text(content)
    box.git("maintainer", f"add {filename}", record=False)
    proc = box.git("maintainer", f"commit -m {message.replace(' ', '-')}", record=False)
    assert proc.returncode == 0, proc.stderr


def contains(state, sha: str) -> bool:
    return sha in state.commits


def main():
    from lib.gitviz import clear

    clear(FIGURES, TABLES)

    # ---- the two servers ------------------------------------------------
    # Sandbox gives us one bare server (github) plus Alice's clone of it. The
    # github bare will play the role of Alice's fork: it is the server she owns.
    box = Sandbox(people=("alice",))
    box.add_bare("upstream")  # the second server: the original project

    # ---- seed the original project, entirely as the maintainer ----------
    # A maintainer clone that is never registered, so it is never drawn. It is
    # the only thing that ever writes to upstream.
    box.paths["maintainer"] = box.root / "maintainer"
    box.git(
        "maintainer",
        f"clone {box.paths['upstream']} {box.paths['maintainer']}",
        cwd=box.root,
        record=False,
    )
    box.git("maintainer", "config user.name maintainer", record=False)
    box.git("maintainer", "config user.email maintainer@example.com", record=False)

    maintainer_commit(box, "README.md", "# widgets\n", "add readme")
    maintainer_commit(box, "core.py", "def run():\n    pass\n", "add core module")
    pushed = box.git("maintainer", "push -u origin main", record=False)
    assert pushed.returncode == 0, pushed.stderr

    # GitHub's "Fork" button copies upstream's history into Alice's own server.
    # Pushing the same objects into the github bare reproduces exactly that copy,
    # so the fork starts life byte-identical to upstream.
    forked = box.git("maintainer", f"push {box.paths['github']} main", record=False)
    assert forked.returncode == 0, forked.stderr

    base_tip = box.read("upstream").branches["main"]
    base_commits = set(box.read("upstream").commits)
    assert box.read("github").branches["main"] == base_tip, (
        "a fork is a copy: the fork's main starts at the exact commit upstream's main is on"
    )

    # ---- Alice wires up her clone: origin is her fork, upstream is added -
    box.add_remote("alice", "upstream", "upstream")
    box.git("alice", "fetch origin", record=False)
    box.git("alice", "switch main", record=False)  # local main tracking origin/main
    box.git("alice", "fetch upstream", record=True)  # this is the caption of step 1
    step1 = box.snap(
        "The setup: two servers, and two remotes in your clone",
        note="origin is your fork, the copy you own and push to. upstream is the original, which you fetch from and cannot push to.",
    )

    # ---- Alice does the work on a branch --------------------------------
    box.git("alice", "switch -c fix-typo", record=True)
    box.commit(
        "alice",
        "README.md",
        "# widgets\n\nA tiny widget toolkit.\n",
        "document the toolkit",
    )
    step2 = box.snap(
        "Your fix goes on a branch in your own clone",
        note="One new commit, in your clone only. Neither server knows about it yet.",
    )
    fix_before_rebase = step2.repos["alice"].branches["fix-typo"]

    # ---- meanwhile upstream moves on, and Alice fetches it ---------------
    # The maintainer merges someone else's pull request. Alice cannot see it
    # until she asks upstream for it.
    maintainer_commit(
        box, "core.py", "def run():\n    return 0\n", "merge another contributors fix"
    )
    moved = box.git("maintainer", "push origin main", record=False)
    assert moved.returncode == 0, moved.stderr
    upstream_new = box.read("upstream").branches["main"]

    alice_before_fetch = set(box.read("alice").commits)
    box.git("alice", "fetch upstream", record=True)
    step3 = box.snap(
        "Upstream moved on: git fetch upstream brings it in",
        note="upstream/main jumps to the maintainer's newer commit. Your own branch has not moved.",
    )

    # ---- Alice rebases onto upstream/main to stay current ---------------
    box.git("alice", "rebase upstream/main", record=True)
    step4 = box.snap(
        "Rebase your branch onto upstream/main to stay current",
        note="Your fix is replayed on top of upstream's newest commit, so it will apply cleanly when the maintainer merges it.",
    )
    fix_after_rebase = step4.repos["alice"].branches["fix-typo"]

    # ---- Alice pushes her branch to her fork, not to upstream -----------
    box.git("alice", "push -u origin fix-typo", record=True)
    step5 = box.snap(
        "Push your branch to your fork (origin), never to upstream",
        note="Your fork gains the branch. Upstream is unchanged: you have no push access there, so your commit cannot land on it.",
    )

    render(box, FIGURES, TABLES, mode="team", repos=PANELS)

    # ---- the tables the README quotes from ------------------------------
    with (TABLES / "two-remotes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["remote", "points at", "you can push"])
        writer.writerow(["origin", "your fork (github), which you own", "yes"])
        writer.writerow(["upstream", "the original project", "no"])

    steps = [
        ("1 wired up", step1),
        ("2 fix on a branch", step2),
        ("3 fetched upstream", step3),
        ("4 rebased", step4),
        ("5 pushed to fork", step5),
    ]
    with (TABLES / "where-is-the-work.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "step",
                "upstream commits",
                "fork commits",
                "your clone commits",
                "your fix on upstream",
                "your fix on the fork",
            ]
        )
        for label, snap in steps:
            up = snap.repos["upstream"]
            gh = snap.repos["github"]
            al = snap.repos["alice"]
            fix = al.branches.get("fix-typo")
            writer.writerow(
                [
                    label,
                    len(up.commits),
                    len(gh.commits),
                    len(al.commits),
                    "yes" if fix and contains(up, fix) else "no",
                    "yes" if fix and contains(gh, fix) else "no",
                ]
            )

    # ================= the claims, checked =================

    # 1. There really are two servers.
    assert "upstream" in box.bares and "github" in box.bares, (
        "two server-side repositories exist: the original and the fork"
    )
    assert len(box.bares) == 2, "exactly two servers, no more"

    # 2. Alice's clone knows both remotes.
    remotes = box.git("alice", "remote", record=False).stdout.split()
    assert "origin" in remotes and "upstream" in remotes, (
        "the contributor's clone has both origin (the fork) and upstream (the original)"
    )
    assert step1.repos["alice"].remotes["origin/main"] == base_tip, (
        "origin/main points at the fork's main"
    )

    # 3. The first fetch established upstream/main from the second server.
    assert step1.repos["alice"].remotes["upstream/main"] == base_tip, (
        "after git fetch upstream, upstream's main appears as upstream/main in the clone"
    )

    # 4. Upstream moving on is only visible to Alice after she fetches it.
    assert upstream_new not in alice_before_fetch, (
        "the maintainer's new commit did not exist in Alice's clone before she fetched"
    )
    assert step3.repos["alice"].remotes["upstream/main"] == upstream_new, (
        "after git fetch upstream, upstream/main advances to the maintainer's newer commit"
    )
    assert step3.repos["alice"].branches["fix-typo"] == fix_before_rebase, (
        "the fetch updated a remote-tracking ref only: her own branch did not move"
    )

    # 5. The rebase replayed her fix on top of upstream's newest commit.
    assert fix_after_rebase != fix_before_rebase, (
        "rebasing rewrote the fix commit onto a new base, so it has a new hash"
    )
    assert step4.repos["alice"].commits[fix_after_rebase].parents == [upstream_new], (
        "the rebased fix now sits directly on top of upstream's newest commit"
    )

    # 6. The push went to the fork, and upstream did not move.
    fork_final = step5.repos["github"]
    up_final = step5.repos["upstream"]
    assert contains(fork_final, fix_after_rebase), (
        "after the push, the fork holds the contribution commit"
    )
    assert fork_final.branches.get("fix-typo") == fix_after_rebase, (
        "and the fork has the fix-typo branch pointing at it"
    )
    assert not contains(up_final, fix_after_rebase), (
        "upstream has none of the contribution: the contributor cannot push there"
    )
    assert not contains(up_final, fix_before_rebase), (
        "not the pre-rebase version of it either"
    )
    assert "fix-typo" not in up_final.branches, (
        "upstream never gained the contributor's branch"
    )
    assert up_final.branches["main"] == upstream_new, (
        "upstream's main is exactly where the maintainer last left it, no further"
    )
    # Everything upstream holds is the maintainer's, not Alice's: the shared
    # base the fork was copied from, plus the maintainer's own later commit.
    assert set(up_final.commits) == base_commits | {upstream_new}, (
        "upstream holds only the shared base plus the maintainer's own later commit"
    )

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 5, f"expected 5 figures, got {figs}"
    print(
        f"{len(figs)} figures, 2 tables plus the state log. All fork claims checked and passing."
    )
    box.cleanup()


if __name__ == "__main__":
    main()
