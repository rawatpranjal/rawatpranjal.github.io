"""Four commands move code between your machine and GitHub, and three refs get confused.

Your local `main`, the remote-tracking cache `origin/main`, and GitHub's own
`main` are three distinct references. Almost all remote confusion is not knowing
which of the three you are looking at. This builds one team-mode sandbox (a bare
repo standing in for GitHub, plus a clone for Alice and a clone for Bob) and walks
the load-bearing sequence:

  in sync   Alice's first commit is pushed and Bob has pulled it. All three agree.
  push      Alice commits again and pushes. GitHub's main moves. Bob's origin/main
            is now a STALE cache: it still points where GitHub used to be.
  fetch     Bob fetches. His origin/main cache catches up to GitHub, but his own
            main and his working files do not move at all.
  merge     Bob merges origin/main. Now his main advances and his files update.

Every claim the README makes is asserted below, read from the real repositories:
the stale cache after the push, the untouched main and files after the fetch, the
caught-up cache after the fetch, and the commit GitHub gained from the push. A
second sandbox proves `git pull` lands Bob in exactly the same place as a fetch
followed by a merge. A wrong diagram fails the run.
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

from lib.gitviz import Sandbox, clear, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

V1 = "line one\n"
V2 = "line one\nline two\n"


def moved(before: str, after: str) -> str:
    return "yes" if before != after else "no"


def main():
    clear(FIGURES, TABLES)

    box = Sandbox(people=("alice", "bob"))

    # ---- 1. everyone in sync -------------------------------------------
    box.commit("alice", "readme.md", V1, "first commit")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    s1 = box.snap(
        "Everyone in sync after clone, push and pull",
        note="clone gave Alice and Bob each a full copy plus an origin remote. Alice pushed the first commit, Bob pulled it. GitHub main, both local mains and both origin/main caches all point at the same commit.",
    )
    bob_file_1 = (box.paths["bob"] / "readme.md").read_text()

    # ---- 2. alice pushes, bob's cache goes stale ------------------------
    box.commit("alice", "readme.md", V2, "second commit")
    box.git("alice", "push")
    s2 = box.snap(
        "Alice pushes: GitHub moves, Bob's cache goes stale",
        note="Alice's push moved GitHub main onto the second commit and updated her own origin/main. Bob ran nothing, so Bob's main and Bob's origin/main both still point at the first commit. Bob's origin/main is now a stale cache of where GitHub used to be.",
    )
    bob_file_2 = (box.paths["bob"] / "readme.md").read_text()

    # ---- 3. bob fetches: the cache catches up, main does not ------------
    box.git("bob", "fetch origin")
    s3 = box.snap(
        "Bob fetches: the cache catches up, main does not move",
        note="git fetch moved Bob's origin/main onto GitHub's second commit. It did not touch Bob's own main and did not change one byte of his working files.",
    )
    bob_file_3 = (box.paths["bob"] / "readme.md").read_text()

    # ---- 4. bob merges origin/main: now caught up ----------------------
    merged = box.git("bob", "merge origin/main", check=False)
    s4 = box.snap(
        "Bob merges origin/main: now Bob is caught up",
        note="git merge origin/main fast-forwarded Bob's main onto the fetched commit and updated his files. Bob is level with GitHub. git pull would have run this fetch and this merge in one command.",
    )
    bob_file_4 = (box.paths["bob"] / "readme.md").read_text()

    # ---- the snapshots as plain repo states ----------------------------
    g1, a1, b1 = s1.repos["github"], s1.repos["alice"], s1.repos["bob"]
    g2, a2, b2 = s2.repos["github"], s2.repos["alice"], s2.repos["bob"]
    g3, b3 = s3.repos["github"], s3.repos["bob"]
    g4, b4 = s4.repos["github"], s4.repos["bob"]

    # ================= ASSERTIONS: every README claim ===================

    # (1) At the start, all three references agree.
    assert g1.branches["main"] == a1.branches["main"] == b1.branches["main"], (
        "at the start GitHub main, Alice main and Bob main are the same commit"
    )
    assert b1.remotes["origin/main"] == g1.branches["main"], (
        "and Bob's origin/main cache matches GitHub's main"
    )
    assert len(g1.commits) == 1, "GitHub holds exactly the first commit"

    # (2) The push moves GitHub up by one commit and lands Alice's work there.
    assert len(g2.commits) == len(g1.commits) + 1, (
        "the push added exactly one commit to GitHub"
    )
    assert g2.branches["main"] != g1.branches["main"], (
        "GitHub main moved to a new commit"
    )
    assert a2.branches["main"] == g2.branches["main"], (
        "Alice's push landed: her local main is now GitHub's main"
    )

    # (3) THE load-bearing fact: after the push, Bob's cache is stale.
    assert g2.branches["main"] != b2.remotes["origin/main"], (
        "GitHub's real main is ahead of Bob's cached origin/main"
    )
    assert b2.remotes["origin/main"] == b1.remotes["origin/main"], (
        "Bob's origin/main did not move: he never fetched"
    )
    assert b2.branches["main"] == b1.branches["main"], (
        "and Bob's own main did not move either"
    )
    assert len(b2.commits) == 1, (
        "before fetching, Bob's repo does not even contain the second commit"
    )
    assert bob_file_2 == V1, (
        "Bob's working file is still the first version: the push did not reach his tree"
    )

    # (4) fetch moves the cache only: not local main, not the files.
    assert b3.remotes["origin/main"] == g3.branches["main"], (
        "after fetch, Bob's origin/main equals GitHub's main"
    )
    assert b3.remotes["origin/main"] != b2.remotes["origin/main"], (
        "the fetch moved Bob's cache forward"
    )
    assert b3.branches["main"] == b2.branches["main"], (
        "the fetch did NOT move Bob's own main"
    )
    assert bob_file_3 == bob_file_2 == V1, (
        "the fetch changed no file in Bob's working tree"
    )
    assert b3.dirty == [], "and left Bob's working tree clean"
    assert len(b3.commits) == 2, (
        "the fetch downloaded the second commit object into Bob's repo"
    )

    # (5) merge advances local main and updates the files, creating no commit.
    assert merged.returncode == 0, "the merge is a clean fast-forward"
    assert b4.branches["main"] == g4.branches["main"], (
        "after merge, Bob's main equals GitHub's main"
    )
    assert b4.remotes["origin/main"] == g4.branches["main"], (
        "and his cache still matches GitHub too"
    )
    assert bob_file_4 == V2, (
        "the merge brought the second version into Bob's working files"
    )
    assert set(b4.commits) == set(b3.commits), (
        "the merge was a fast-forward: it created no new commit"
    )
    assert len(b4.commits) == len(g4.commits), "Bob now holds every commit GitHub holds"

    # (6) git pull equals fetch then merge, proven in a second sandbox.
    pullbox = Sandbox(people=("alice", "bob"))
    pullbox.commit("alice", "readme.md", V1, "first commit")
    pullbox.git("alice", "push -u origin main")
    pullbox.git("bob", "pull")
    pullbox.commit("alice", "readme.md", V2, "second commit")
    pullbox.git("alice", "push")
    pb_before = pullbox.read("bob")
    pb_before_file = (pullbox.paths["bob"] / "readme.md").read_text()
    pb_g_before = pullbox.read("github").branches["main"]
    pullbox.git("bob", "pull")
    pb_after = pullbox.read("bob")
    pb_after_file = (pullbox.paths["bob"] / "readme.md").read_text()
    pb_g_after = pullbox.read("github").branches["main"]
    assert pb_after.branches["main"] == g4.branches["main"], (
        "git pull lands Bob at the same commit as fetch then merge"
    )
    assert pb_after_file == V2, "and with the same file contents on disk"
    pull_cols = {
        "github_main": moved(pb_g_before, pb_g_after),
        "bob_origin_main_cache": moved(
            pb_before.remotes["origin/main"], pb_after.remotes["origin/main"]
        ),
        "bob_local_main": moved(pb_before.branches["main"], pb_after.branches["main"]),
        "bob_working_files": moved(pb_before_file, pb_after_file),
    }
    pullbox.cleanup()

    # ================= FIGURES and TABLES the README quotes =============

    render(box, FIGURES, TABLES, mode="team")

    def in_sync(g, b):
        return "yes" if g.branches["main"] == b.branches["main"] else "no"

    refs = [
        {
            "step": 1,
            "event": "everyone in sync",
            "github_main": g1.branches["main"],
            "bob_local_main": b1.branches["main"],
            "bob_origin_main_cache": b1.remotes["origin/main"],
            "bob_in_sync_with_github": in_sync(g1, b1),
        },
        {
            "step": 2,
            "event": "alice pushes",
            "github_main": g2.branches["main"],
            "bob_local_main": b2.branches["main"],
            "bob_origin_main_cache": b2.remotes["origin/main"],
            "bob_in_sync_with_github": in_sync(g2, b2),
        },
        {
            "step": 3,
            "event": "bob fetches",
            "github_main": g3.branches["main"],
            "bob_local_main": b3.branches["main"],
            "bob_origin_main_cache": b3.remotes["origin/main"],
            "bob_in_sync_with_github": in_sync(g3, b3),
        },
        {
            "step": 4,
            "event": "bob merges origin/main",
            "github_main": g4.branches["main"],
            "bob_local_main": b4.branches["main"],
            "bob_origin_main_cache": b4.remotes["origin/main"],
            "bob_in_sync_with_github": in_sync(g4, b4),
        },
    ]
    with (TABLES / "three-references.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(refs[0]))
        writer.writeheader()
        writer.writerows(refs)

    commands = [
        {
            "command": "git push (alice)",
            "github_main": moved(g1.branches["main"], g2.branches["main"]),
            "bob_origin_main_cache": moved(
                b1.remotes["origin/main"], b2.remotes["origin/main"]
            ),
            "bob_local_main": moved(b1.branches["main"], b2.branches["main"]),
            "bob_working_files": moved(bob_file_1, bob_file_2),
        },
        {
            "command": "git fetch (bob)",
            "github_main": moved(g2.branches["main"], g3.branches["main"]),
            "bob_origin_main_cache": moved(
                b2.remotes["origin/main"], b3.remotes["origin/main"]
            ),
            "bob_local_main": moved(b2.branches["main"], b3.branches["main"]),
            "bob_working_files": moved(bob_file_2, bob_file_3),
        },
        {
            "command": "git merge origin/main (bob)",
            "github_main": moved(g3.branches["main"], g4.branches["main"]),
            "bob_origin_main_cache": moved(
                b3.remotes["origin/main"], b4.remotes["origin/main"]
            ),
            "bob_local_main": moved(b3.branches["main"], b4.branches["main"]),
            "bob_working_files": moved(bob_file_3, bob_file_4),
        },
        {"command": "git pull (bob)", **pull_cols},
    ]
    with (TABLES / "what-each-command-moves.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(commands[0]))
        writer.writeheader()
        writer.writerows(commands)

    box.cleanup()

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 team-mode figures, got {figs}"
    print(f"{len(figs)} figures, 3 tables. Every remote claim checked and passing.")
    print("figures:", figs)


if __name__ == "__main__":
    main()
