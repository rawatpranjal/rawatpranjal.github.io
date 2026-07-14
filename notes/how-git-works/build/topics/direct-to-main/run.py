"""Direct to main: everyone commits straight to main, no branches, no review.

The simplest way two people can share a repository is to both commit onto main
and push. There is no feature branch and no pull request. History is one straight
line that everybody appends to. This is a perfectly good workflow for a solo
researcher, a notebook, a tutorial, or a very small trusted team, and it is also
where the everyday sync friction of git shows up in its plainest form.

The story runs in team mode, one panel per repository (GitHub on top, then Alice,
then Bob). A commit sits at the same horizontal position in every panel, so who
holds what is readable at a glance.

  1. everyone in sync: Alice makes the first commit on main, pushes, Bob pulls.
  2. Alice edits and commits straight to main, then pushes. GitHub main moves,
     Bob has heard nothing. No branch was ever created.
  3. Bob pulls Alice's work. His main fast-forwards onto her commit.
  4. Bob commits on top of it and pushes. GitHub main is now one shared line
     that both people have appended to: base, Alice, Bob.
  5. Alice syncs, commits again straight to main, and pushes. Bob did not sync,
     so Bob is now behind GitHub by one commit.
  6. Bob commits locally without pulling and tries to push. Git REJECTS it,
     because the remote holds a commit Bob does not have. This is the everyday
     rule: you cannot push when you are behind.
  7. Bob runs git pull --rebase, which replays his commit on top of Alice's,
     then pushes. It succeeds. History is a single straight line again.

Every claim the README makes is asserted from the real repositories: that no
branch other than main ever exists in any repo, that the final history has no
merge commit (every commit has at most one parent), that Bob's push while behind
returns a non-zero exit code and the same push succeeds after the rebase, and
that the final main on GitHub carries both people's commits in one straight line.
A wrong diagram fails the run.
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

# One tiny shared project. Each person owns different files, so nothing collides
# and the rebase in step 7 replays cleanly. The point here is the workflow, not
# a conflict (conflicts have their own tutorials in section 04).
BASE_NOTES = "# Weekly report\n\nnotes go here\n"
ALICE_LOADER = "def load(path):\n    return open(path).read()\n"
BOB_REPORT = "## Findings\n\nInitial results look good.\n"
ALICE_METRICS = "def rmse(y, yhat):\n    return ((y - yhat) ** 2).mean() ** 0.5\n"
BOB_PLOTS = "def plot(series):\n    series.plot()\n"


def short(box: Sandbox, who: str, ref: str) -> str:
    return box.git(
        who, f"rev-parse --short {ref}", record=False, check=False
    ).stdout.strip()


def count(box: Sandbox, who: str, ref: str = "main") -> int:
    out = box.git(who, f"rev-list --count {ref}", record=False, check=False).stdout
    return int(out.strip() or 0)


def is_ancestor(box: Sandbox, who: str, older: str, newer: str) -> bool:
    return (
        box.git(
            who,
            f"merge-base --is-ancestor {older} {newer}",
            record=False,
            check=False,
        ).returncode
        == 0
    )


def main_log(box: Sandbox, who: str) -> list[tuple[str, str, str, int]]:
    """The commits reachable from main, oldest first: short sha, author, subject, #parents."""
    # %x1f is git's own unit-separator placeholder. It stays printable inside the
    # command string (so Sandbox.git's args.split() does not shatter the format on
    # it, the way it would on a literal 0x1f) and git emits the real byte in output.
    raw = box.git(
        who,
        "log main --reverse --pretty=%h%x1f%an%x1f%s%x1f%p",
        record=False,
        check=False,
    ).stdout
    rows = []
    for line in raw.splitlines():
        h, an, s, p = line.split("\x1f")
        rows.append((h, an, s, len(p.split()) if p else 0))
    return rows


def main():
    clear(FIGURES, TABLES)
    box = Sandbox(people=("alice", "bob"))

    # ---- 1. everyone in sync on main -----------------------------------
    box.commit("alice", "notes.md", BASE_NOTES, "start the report")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    s1 = box.snap(
        "Everyone in sync on main",
        note="Alice made the first commit on main and pushed it. Bob cloned and pulled. GitHub main, both local mains and both origin/main caches all point at the same commit. No branch exists anywhere but main.",
    )
    base_sha = short(box, "github", "main")

    # ---- 2. Alice commits straight to main and pushes ------------------
    box.commit("alice", "loader.py", ALICE_LOADER, "add the data loader")
    box.git("alice", "push")
    s2 = box.snap(
        "Alice commits straight to main and pushes",
        note="Alice did not make a branch. She committed onto main and pushed. GitHub main moved onto her new commit and so did her own origin/main cache. Bob ran nothing, so Bob still sits one commit back and does not even hold Alice's commit yet.",
    )
    a1_sha = short(box, "github", "main")

    # ---- 3. Bob pulls Alice's work -------------------------------------
    box.git("bob", "pull")
    s3 = box.snap(
        "Bob pulls Alice's work",
        note="git pull fast-forwarded Bob's main onto Alice's commit and brought her file into his working tree. Bob wrote no commit of his own. All three references agree again.",
    )

    # ---- 4. Bob commits on top and pushes: one shared line -------------
    box.commit("bob", "report.md", BOB_REPORT, "write up findings")
    bob_push_1 = box.git("bob", "push", check=False)
    s4 = box.snap(
        "Bob builds on it and pushes: one shared line",
        note="Bob committed straight onto the same main, on top of Alice's work, and pushed. GitHub main is now a single straight line that both people have appended to: the base, then Alice, then Bob.",
    )
    b1_sha = short(box, "github", "main")

    # ---- 5. Alice syncs, commits again, pushes: Bob is now behind ------
    box.git("alice", "pull")
    box.commit("alice", "metrics.py", ALICE_METRICS, "add a metric")
    box.git("alice", "push")
    s5 = box.snap(
        "Alice pushes again; now Bob is behind",
        note="Alice pulled Bob's commit, committed her own onto main, and pushed. GitHub main advanced again. Bob has not pulled since his own push, so Bob's main is now one commit behind GitHub and his origin/main cache is stale.",
    )
    a2_sha = short(box, "github", "main")

    # ---- 6. Bob commits while behind and the push is rejected ---------
    box.commit("bob", "plots.py", BOB_PLOTS, "add plots")
    b2_local_sha = short(box, "bob", "main")
    bob_rejected = box.git("bob", "push", check=False)
    s6 = box.snap(
        "Bob pushes while behind: git refuses",
        note="Bob committed onto main without pulling first, then tried to push. Git rejected the push. GitHub holds Alice's latest commit, which Bob's history does not contain, so accepting Bob's push would silently drop Alice's work. Git will not do that.",
    )
    g6_main = short(box, "github", "main")
    b6_main = short(box, "bob", "main")

    # ---- 7. Bob rebases onto main and pushes: the line is straight again
    bob_rebase = box.git("bob", "pull --rebase", check=False)
    bob_push_2 = box.git("bob", "push", check=False)
    s7 = box.snap(
        "Bob rebases onto main and pushes: the line is straight again",
        note="git pull --rebase fetched Alice's commit and replayed Bob's commit on top of it, so main stays one straight line with no merge commit. The second push succeeded. GitHub main now holds all five commits in a single line.",
    )
    b2_final_sha = short(box, "github", "main")

    render(box, FIGURES, TABLES, mode="team")

    # ================= ASSERTIONS: every README claim ===================

    g1, a1r, b1r = s1.repos["github"], s1.repos["alice"], s1.repos["bob"]
    g2, b2r = s2.repos["github"], s2.repos["bob"]
    g, a, b = box.read("github"), box.read("alice"), box.read("bob")

    # (A) DIRECT TO MAIN: no branch other than main ever exists, in any repo,
    #     at any moment, and everyone always stays on main. That is the workflow.
    for snap in box.snapshots:
        for name, st in snap.repos.items():
            assert set(st.branches) <= {"main"}, (
                f"no branch other than main is ever created ({name} at '{snap.label}')"
            )
            if name not in snap.bares:
                assert st.head == "main", (
                    f"everyone commits with main checked out ({name} at '{snap.label}')"
                )

    # (B) Everyone starts in sync on one commit.
    assert g1.branches["main"] == a1r.branches["main"] == b1r.branches["main"], (
        "at the start GitHub main, Alice main and Bob main are the same commit"
    )
    assert len(g1.commits) == 1, "GitHub holds exactly the first commit"

    # (C) Alice commits straight to main: GitHub moves by one, Bob learns nothing.
    assert len(g2.commits) == len(g1.commits) + 1, (
        "Alice's direct commit added exactly one commit to GitHub"
    )
    assert g2.branches["main"] == a1_sha and a1_sha != base_sha, (
        "GitHub main advanced onto Alice's new commit"
    )
    assert b2r.branches["main"] == base_sha, "Bob's main did not move: he ran nothing"
    assert len(b2r.commits) == 1, "Bob does not even hold Alice's commit yet"

    # (D) Bob's first push (after pulling) is accepted; it builds the shared line.
    assert bob_push_1.returncode == 0, "Bob's push after pulling was accepted"
    assert is_ancestor(box, "github", a1_sha, b1_sha), (
        "Bob committed on top of Alice's work, so hers is in the history of his"
    )

    # (E) THE REJECTION: a push while behind is refused, and the very same push
    #     succeeds once Bob has integrated the remote work.
    assert bob_rejected.returncode != 0, (
        "git refuses Bob's push while he is behind GitHub"
    )
    assert "rejected" in bob_rejected.stderr.lower(), (
        "git says the push was rejected, in its own words"
    )
    assert g6_main == a2_sha, (
        "at the rejection, GitHub main is on Alice's latest commit"
    )
    assert not is_ancestor(box, "bob", a2_sha, b6_main), (
        "Bob's history does not contain GitHub's latest commit: he is behind"
    )
    assert bob_rebase.returncode == 0, "the rebase-pull replayed Bob's commit cleanly"
    assert bob_push_2.returncode == 0, (
        "the same push succeeds after Bob integrates the remote work"
    )

    # (F) The rejected commit never landed; its rebased twin did instead.
    assert not is_ancestor(box, "github", b2_local_sha, "main"), (
        "the commit git refused never reached GitHub"
    )
    assert b2_local_sha not in g.commits, (
        "GitHub does not hold the pre-rebase commit at all"
    )

    # (G) SINGLE STRAIGHT LINE: only main, and not one merge commit anywhere.
    assert set(g.branches) == {"main"}, "GitHub carries exactly one branch, main"
    assert all(len(c.parents) <= 1 for c in g.commits.values()), (
        "no commit on GitHub has two parents: the history has no merge commit"
    )
    assert count(box, "github", "main") == 5, "main is exactly five commits long"
    assert len(g.commits) == 5, (
        "and GitHub holds exactly those five, nothing off to the side"
    )

    # (H) The final main holds BOTH people's commits, in that one line.
    for label, sha in [
        ("Alice's first (loader)", a1_sha),
        ("Bob's first (report)", b1_sha),
        ("Alice's second (metrics)", a2_sha),
        ("Bob's second (plots, rebased)", b2_final_sha),
    ]:
        assert is_ancestor(box, "github", sha, "main"), (
            f"{label} is in the history of the final main"
        )
    authors = {row[1] for row in main_log(box, "github")}
    assert authors == {"alice", "bob"}, (
        "the one line of history is authored by both people"
    )

    # (I) After the last push Bob is level with GitHub, and Alice, who has not
    #     pulled since, is honestly one commit behind: the same rule that caught
    #     Bob will catch her until she pulls.
    assert b.branches["main"] == g.branches["main"] == b2_final_sha, (
        "Bob just pushed, so Bob and GitHub sit on the same final commit"
    )
    assert a.branches["main"] == a2_sha and a2_sha != g.branches["main"], (
        "Alice has not pulled Bob's last commit, so she is one behind GitHub"
    )

    # ================= TABLES the README quotes =========================

    # The final history of main, one row per commit, proving one straight line
    # authored by both people with no merge commit.
    line = main_log(box, "github")
    with (TABLES / "history-line.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["position", "commit", "author", "subject", "parents"]
        )
        writer.writeheader()
        for i, (h, an, s, np) in enumerate(line, start=1):
            writer.writerow(
                {
                    "position": i,
                    "commit": h,
                    "author": an,
                    "subject": s,
                    "parents": np,
                }
            )

    # The sync loop: every push and pull Bob and Alice ran, its exit code, and
    # what git did. The two rows that matter are Bob's rejected push and the same
    # push succeeding after the rebase.
    def outcome(rc: int, accepted_msg: str, refused_msg: str) -> str:
        return accepted_msg if rc == 0 else refused_msg

    events = [
        ("alice", "git push", 0, "accepted, GitHub main moves"),
        ("bob", "git push", bob_push_1.returncode, "accepted, on top of Alice"),
        ("alice", "git push", 0, "accepted, GitHub main moves"),
        (
            "bob",
            "git push",
            bob_rejected.returncode,
            outcome(bob_rejected.returncode, "accepted", "REJECTED: Bob is behind"),
        ),
        (
            "bob",
            "git pull --rebase",
            bob_rebase.returncode,
            "replays Bob's commit onto Alice's",
        ),
        (
            "bob",
            "git push",
            bob_push_2.returncode,
            outcome(bob_push_2.returncode, "accepted after the rebase", "rejected"),
        ),
    ]
    with (TABLES / "sync-loop.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["who", "command", "exit_code", "what_git_did"]
        )
        writer.writeheader()
        for who, cmd, rc, what in events:
            writer.writerow(
                {"who": who, "command": cmd, "exit_code": rc, "what_git_did": what}
            )

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    assert len(figs) == 7, f"expected 7 team-mode figures, got {figs}"
    print(
        f"{len(figs)} figures, 2 tables. Every direct-to-main claim checked and passing."
    )
    print("figures:", figs)
    print("history line:", " -> ".join(h for h, *_ in line))
    print("rejected push exit code:", bob_rejected.returncode)
    print("push-after-rebase exit code:", bob_push_2.returncode)
    box.cleanup()


if __name__ == "__main__":
    main()
