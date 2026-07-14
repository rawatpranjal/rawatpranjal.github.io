"""Two people change the same line, and the second one to land does the merging.

This is the collaboration version of a merge conflict. In the solo tutorial one
person merges two of their own branches. Here the two branches belong to two
different people. Alice and Bob both branch from main and rewrite the SAME line
of the same file, for two different reasons. Alice's pull request is reviewed and
lands first, so GitHub's main now carries her line. Bob then has to update his
branch on top of the new main before his own pull request can merge, and git
stops: it cannot pick between Alice's line and Bob's.

The whole point is drawn out here. Whoever pushes second inherits the job of
resolving, and the correct answer is usually neither side alone but a combination
of both intentions. Blindly taking one side silently deletes a colleague's work,
or, in a rebase, your own. Every claim the README makes about exit codes, the
markers, the UU status and the resolution is read back out of the real
repositories and asserted at the bottom. A wrong picture fails the run.

Bob updates with `git rebase origin/main`, which replays his commit on top of
Alice's landed work. During that rebase the conflict markers are backwards from
the intuition: HEAD is Alice's line (the base being replayed onto) and the lower
half is Bob's own commit.
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

from lib.gitviz import Sandbox, clear, draw_conflict, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

FILE = "client.py"
MARKERS = ("<<<<<<<", "=======", ">>>>>>>")

BASE = """BASE_URL = "https://api.example.com"
TIMEOUT = 30

def make_session():
    return Session(base_url=BASE_URL, timeout=TIMEOUT)

def fetch(session, path):
    return session.get(path)
"""

LINE = "    return Session(base_url=BASE_URL, timeout=TIMEOUT)"

# Alice makes the client retry failed requests. Bob gives it a connection pool.
# Both edits are wanted, and both land on the one line that builds the session.
ALICE_LINE = "    return Session(base_url=BASE_URL, timeout=TIMEOUT, retries=5)"
BOB_LINE = "    return Session(base_url=BASE_URL, timeout=TIMEOUT, pool_size=20)"
RESOLVED_LINE = (
    "    return Session(base_url=BASE_URL, timeout=TIMEOUT, retries=5, pool_size=20)"
)

ALICE = BASE.replace(LINE, ALICE_LINE)
BOB = BASE.replace(LINE, BOB_LINE)
RESOLVED = BASE.replace(LINE, RESOLVED_LINE)


# ---- small readers over the real repositories --------------------------


def status_of(box: Sandbox, who: str) -> str:
    """The porcelain status line for the one file, or (clean) when git is silent."""
    out = box.git(who, f"status --porcelain {FILE}", record=False).stdout
    return out.rstrip("\n") or "(clean)"


def contents(box: Sandbox, who: str) -> str:
    return (box.paths[who] / FILE).read_text()


def markers_in(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith(MARKERS))


def index_versions(box: Sandbox, who: str) -> int:
    """How many versions of the file the index holds. One normally, three in a conflict."""
    out = box.git(who, f"ls-files -u {FILE}", record=False).stdout
    return len(out.splitlines()) or 1


def rebasing(box: Sandbox, who: str) -> bool:
    return (box.paths[who] / ".git" / "rebase-merge").exists()


def is_ancestor(box: Sandbox, who: str, older: str, newer: str) -> bool:
    proc = box.git(
        who, f"merge-base --is-ancestor {older} {newer}", record=False, check=False
    )
    return proc.returncode == 0


def github_lands_pr(box: Sandbox, feature_ref: str, message: str):
    """Model GitHub merging an approved pull request into main, server-side.

    Clicking Merge on GitHub does not run on anyone's laptop. GitHub already has
    every object and merges the branch itself, then advances main. A bare repo
    has no working tree to merge in, so this uses a throwaway clone standing in
    for GitHub's own server-side worktree, merges the feature branch into main
    with a merge commit (what the default Merge button writes) and pushes main
    back. The clone is never registered with the sandbox, so it is never drawn.
    Only GitHub's main moving forward shows up in the figures.
    """
    scratch = box.root / "github-server-worktree"
    box.git(
        "github", f"clone {box.paths['github']} {scratch}", cwd=box.root, record=False
    )
    box.git("github", "config user.name github", cwd=scratch, record=False)
    box.git("github", "config user.email github@example.com", cwd=scratch, record=False)
    box.git("github", "switch main", cwd=scratch, record=False)
    merge = box.git(
        "github",
        f"merge --no-ff {feature_ref} -m {message}",
        cwd=scratch,
        record=False,
        check=False,
    )
    box.git("github", "push origin main", cwd=scratch, record=False)
    return merge


def new_sandbox() -> Sandbox:
    box = Sandbox(people=("alice", "bob"))
    for who in ("alice", "bob"):
        # Pin the default marker style so the figures cannot drift with a
        # differently configured machine. Not recorded: a determinism guard.
        box.git(who, "config merge.conflictstyle merge", record=False)
    # A non-interactive editor, so `git rebase --continue` reuses Bob's commit
    # message instead of trying to open an editor under a subprocess.
    box.git("bob", "config core.editor true", record=False)
    return box


def main():
    clear(FIGURES, TABLES)

    # The README describes the file it is about. Check that description too.
    base_lines = BASE.splitlines()
    assert len(base_lines) == 8, "the README calls client.py an eight line file"
    assert base_lines[4] == LINE, "both people rewrite line 5, the session builder"
    assert ALICE_LINE != BOB_LINE, "the two edits differ"
    assert "retries=5" in ALICE_LINE and "pool_size" not in ALICE_LINE, (
        "Alice adds retries and nothing else"
    )
    assert "pool_size=20" in BOB_LINE and "retries" not in BOB_LINE, (
        "Bob adds a pool and nothing else"
    )

    box = new_sandbox()
    captures: list[dict] = []

    def capture(event: str):
        g, b = box.read("github"), box.read("bob")
        real_main = g.branches.get("main", "")
        # Does Bob's branch contain GitHub's REAL current main (Alice's landed
        # work), not his possibly stale cache of it? Checked against the true tip,
        # so the answer flips to no the instant Alice lands, which is the point.
        on_top = "no"
        if "bob-pooling" in b.branches and real_main:
            on_top = (
                "yes" if is_ancestor(box, "bob", real_main, "bob-pooling") else "no"
            )
        captures.append(
            {
                "event": event,
                "github_main": real_main,
                "bob_branch_tip": b.branches.get("bob-pooling", "(none)"),
                "bob_origin_main_cache": b.remotes.get("origin/main", "(none)"),
                "bob_branch_has_current_main": on_top,
            }
        )

    # ---- 1. a shared file on main, both people have it ------------------
    box.commit("alice", FILE, BASE, "add the api client")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    box.snap(
        "A shared file on main. Alice and Bob both have it.",
        note="One commit on main, on GitHub and in both clones. make_session builds line 5, the line both people are about to edit.",
    )
    capture("shared base on main")

    # ---- 2. both branch from main and edit the SAME line ---------------
    box.git("alice", "switch -c alice-retries")
    box.commit("alice", FILE, ALICE, "add retry support")
    box.git("bob", "switch -c bob-pooling")
    box.commit("bob", FILE, BOB, "add connection pooling")
    box.snap(
        "Alice and Bob each branch from main and rewrite line 5",
        note="Two people, two branches, the same line. Alice adds retries, Bob adds a connection pool. Neither branch is on GitHub yet.",
    )
    capture("both branched, same line edited")

    # ---- 3. Alice's pull request lands first ---------------------------
    box.git("alice", "push -u origin alice-retries")
    server_merge = github_lands_pr(
        box, "origin/alice-retries", "Merge-pull-request-alice-retries"
    )
    box.snap(
        "Alice's pull request lands first, on GitHub main",
        note="Her branch was reviewed and merged into main on GitHub. Bob ran nothing, so his branch and his origin/main cache still point at the old base.",
    )
    capture("alice PR landed on main")

    github_after_alice = box.read("github")
    bob_before_fetch = box.read("bob")

    # ---- 4. Bob updates his branch: fetch + rebase, and git stops ------
    box.git("bob", "fetch origin")
    rebase = box.git("bob", "rebase origin/main", check=False)
    conflicted = contents(box, "bob")
    box.snap(
        "Bob rebases onto the new main, and git stops on line 5",
        note="git replayed Bob's commit onto Alice's landed main and hit the same line. HEAD (detached) sits on Alice's commit while it replays. The file now holds both versions with markers.",
    )
    capture("bob rebase conflicts")

    bob_during = box.read("bob")

    # ---- the oracle for Alice landing and Bob conflicting ---------------
    assert server_merge.returncode == 0, "Alice's pull request merged cleanly on GitHub"
    assert "retries=5" in box.git("github", "show main:" + FILE, record=False).stdout, (
        "GitHub's main now contains Alice's retries line"
    )
    assert is_ancestor(box, "github", "alice-retries", "main"), (
        "Alice's commit is an ancestor of GitHub main: her merge to main succeeded"
    )

    assert rebase.returncode != 0, (
        f"Bob's update must stop on the same-line conflict, git returned {rebase.returncode}"
    )
    assert "CONFLICT" in rebase.stdout, "git reported a content conflict"
    assert markers_in(conflicted) == 3, "exactly three marker lines, one conflict"
    assert "<<<<<<< HEAD" in conflicted, (
        "during a rebase HEAD is the base you are replaying onto, which is Alice's line"
    )
    assert ALICE_LINE in conflicted, (
        "Alice's landed line is the HEAD side of the markers"
    )
    assert ">>>>>>>" in conflicted, (
        "the lower marker names Bob's own commit being replayed"
    )
    assert BOB_LINE in conflicted, "Bob's own line is the lower side of the markers"
    assert status_of(box, "bob") == f"UU {FILE}", (
        f"git status shows the file unmerged, got {status_of(box, 'bob')!r}"
    )
    assert index_versions(box, "bob") == 3, (
        "the index holds three versions: the ancestor, Alice's, and Bob's"
    )
    assert rebasing(box, "bob"), (
        ".git/rebase-merge exists, so git knows a rebase is paused"
    )
    assert (
        bob_during.branches["bob-pooling"] == bob_before_fetch.branches["bob-pooling"]
    ), "the paused rebase has not moved bob-pooling yet: no resolved commit exists"

    draw_conflict(
        box,
        "bob",
        FILE,
        FIGURES / "conflict-file.png",
        "The same line, as git wrote it during Bob's rebase",
        "git rebase origin/main",
        "HEAD is Alice's landed line, not Bob's. In a rebase the base is what you replay onto, so it is the upper half.",
    )

    # ---- picking one side alone silently drops the other's work ---------
    assert "retries=5" in ALICE_LINE and "pool_size" not in ALICE_LINE, (
        "taking Alice's line alone keeps retries and drops Bob's pool"
    )
    assert "pool_size=20" in BOB_LINE and "retries" not in BOB_LINE, (
        "taking Bob's line alone keeps the pool and drops Alice's retries"
    )

    # ---- 5. Bob resolves by combining both intents, then continues -----
    box.write("bob", FILE, RESOLVED)
    box.git("bob", f"add {FILE}")
    resolved = contents(box, "bob")
    staged_status = status_of(box, "bob")
    draw_conflict(
        box,
        "bob",
        FILE,
        FIGURES / "resolved-file.png",
        "Bob's resolution: one line that carries both intentions",
        "edit client.py; git add client.py",
        "Neither side alone. The retry count and the pool size both survive, on a line that existed on no branch.",
    )
    box.git("bob", "rebase --continue", check=False)
    box.snap(
        "Bob combines both changes and continues the rebase",
        note="git add told git the conflict was settled, and rebase --continue replanted the resolved commit on the new main. bob-pooling now sits on top of Alice's landed work.",
    )
    capture("bob resolved and continued")

    bob_after = box.read("bob")
    resolved_tip = bob_after.branches["bob-pooling"]

    # ---- the oracle for the resolution ---------------------------------
    assert staged_status == f"M  {FILE}", (
        f"git add is how you mark a rebase conflict settled, got {staged_status!r}"
    )
    assert markers_in(resolved) == 0, "the resolved file has no markers left in it"
    assert RESOLVED_LINE in resolved, "the resolved line is the one the README quotes"
    assert "retries=5" in RESOLVED_LINE and "pool_size=20" in RESOLVED_LINE, (
        "the resolution takes something from each side"
    )
    assert RESOLVED_LINE != ALICE_LINE and RESOLVED_LINE != BOB_LINE, (
        "and it is neither person's line verbatim, which is the whole point"
    )
    assert bob_after.dirty == [], "after continuing, Bob's working tree is clean"
    assert status_of(box, "bob") == "(clean)", "git status is silent again"
    assert not rebasing(box, "bob"), "the rebase is over, .git/rebase-merge is gone"
    assert bob_after.head == "bob-pooling", (
        "HEAD is back on Bob's branch, no longer detached"
    )
    assert is_ancestor(box, "bob", "origin/main", "bob-pooling"), (
        "bob-pooling now sits on top of Alice's landed main"
    )
    assert "retries=5" in contents(box, "bob"), (
        "Bob's branch now carries Alice's intent"
    )
    assert "pool_size=20" in contents(box, "bob"), "and still carries his own"
    assert len(bob_after.commits[resolved_tip].parents) == 1, (
        "a rebase keeps history linear: the replayed commit has one parent"
    )
    assert resolved_tip != bob_before_fetch.branches["bob-pooling"], (
        "the rebase rewrote Bob's commit, so its hash changed"
    )

    # ---- 6. Bob pushes his conflict-free branch ------------------------
    push = box.git("bob", "push -u origin bob-pooling", check=False)
    box.snap(
        "Bob pushes his rebased, conflict-free branch",
        note="Because Bob resolved on top of Alice's work, GitHub can merge his branch with no further conflict. The second person did the merging.",
    )
    capture("bob pushed")

    github_final = box.read("github")
    assert push.returncode == 0, "Bob's push of the fresh branch succeeds"
    gh_bob = box.git("github", "show bob-pooling:" + FILE, record=False).stdout
    assert "retries=5" in gh_bob and "pool_size=20" in gh_bob, (
        "GitHub's copy of Bob's branch carries both intents"
    )
    assert is_ancestor(box, "github", "alice-retries", "bob-pooling"), (
        "and it sits on top of Alice's merged work, so it will merge with no conflict"
    )

    # ================= FIGURES and TABLES =============================
    render(box, FIGURES, TABLES, mode="team")

    write_tables(captures, rebase, conflicted, resolved, staged_status, resolved_tip)

    figs = sorted(p.name for p in FIGURES.glob("*.png"))
    n_snaps = len(box.snapshots)
    assert len(figs) == n_snaps + 2, (
        f"every snapshot plus the two file figures, got {figs}"
    )
    for name in ("conflict-file.png", "resolved-file.png"):
        assert name in figs, f"{name} was not drawn"

    tabs = sorted(p.name for p in TABLES.glob("*.csv"))
    print(
        f"{len(figs)} figures, {len(tabs)} tables. "
        f"Alice PR merge exit {server_merge.returncode}, Bob rebase exit "
        f"{rebase.returncode}, status {status_of(box, 'bob') or 'clean'}. "
        f"Resolved line carries both intents. All assertions passed."
    )
    print("figures:", figs)
    print("tables:", tabs)

    # unused var guard for linters, and a final truth check on the whole story
    assert github_after_alice.branches["main"] == github_final.branches["main"], (
        "GitHub main did not move again after Alice landed: Bob rebased, he did not merge to main"
    )
    box.cleanup()


def write_tables(captures, rebase, conflicted, resolved, staged_status, resolved_tip):
    # 1. what Bob's rebase produced, at a glance
    with (TABLES / "conflict-outcome.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "stage",
                "git exit code",
                "conflict marker lines",
                "git status for client.py",
                "versions of the file in the index",
            ]
        )
        writer.writerow(
            [
                "git rebase origin/main (Bob)",
                rebase.returncode,
                markers_in(conflicted),
                f"UU {FILE}",
                3,
            ]
        )
        writer.writerow(
            [
                "after Bob resolved and staged",
                0,
                markers_in(resolved),
                staged_status,
                1,
            ]
        )

    # 2. the two changes and the resolution
    with (TABLES / "the-two-changes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["version of line 5", "whose change", "the line", "kept on its own?"]
        )
        writer.writerow(
            [
                "HEAD (the rebase base)",
                "Alice, already on main",
                ALICE_LINE.strip(),
                "no, drops the pool",
            ]
        )
        writer.writerow(
            [
                "the replayed commit",
                "Bob, being rebased",
                BOB_LINE.strip(),
                "no, drops the retries",
            ]
        )
        writer.writerow(
            [
                "what Bob wrote",
                "both intentions",
                RESOLVED_LINE.strip(),
                "yes, both survive",
            ]
        )

    # 3. the conflicted file, line by line, and which side each line is
    with (TABLES / "the-conflicted-file.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["line", "text", "belongs to"])
        side = "both sides agree"
        for i, text in enumerate(conflicted.splitlines(), start=1):
            if text.startswith("<<<<<<<"):
                side, tag = "ours", "marker"
            elif text.startswith("======="):
                side, tag = "theirs", "marker"
            elif text.startswith(">>>>>>>"):
                side, tag = "both sides agree", "marker"
            else:
                tag = {
                    "ours": "HEAD (Alice's landed line)",
                    "theirs": "Bob's replayed commit",
                }.get(side, "both sides agree")
            writer.writerow([i, text, tag])

    # 4. how Bob's branch caught up to the moved main
    with (TABLES / "bob-catches-up.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(captures[0]))
        writer.writeheader()
        writer.writerows(captures)


if __name__ == "__main__":
    main()
