"""Bob's branch is stacked on Alice's unmerged branch, and restacked with --onto.

Alice builds an API on `alice-api`, unmerged. Bob needs it to build his model, so
Bob branches `bob-model` from `alice-api`, not from `main`. Two stacked pull
requests are open at once: alice-api into main, and bob-model into alice-api.

When Alice's pull request is squash-merged, `main` gains one new commit that is
not either of Alice's originals. Bob's branch still hangs off Alice's old
commits. If Bob only repointed his pull request to main, the pull request would
show Alice's commits alongside his own, as if Bob had authored them. The fix is
one command:

    git rebase --onto main alice-api bob-model

which replays only the commits in `alice-api..bob-model`, that is only Bob's own
work, onto the new `main`, dropping Alice's now-merged commits from underneath
him.

Every claim the README makes is asserted at the bottom, read from the real
repositories with plumbing. A wrong picture fails the run.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():  # walk up to the collection root
    root = root.parent
sys.path.insert(0, str(root))

from lib.gitviz import Sandbox, clear, draw, layout, render  # noqa: E402

FIGURES = HERE / "figures"
TABLES = HERE / "tables"

README_TXT = "myproject\n"
API_1 = "def serve():\n    pass\n"
API_2 = "def serve():\n    return {'ok': True}\n\ndef health():\n    return 'up'\n"
MODEL_1 = "def model():\n    pass\n"
MODEL_2 = "def model():\n    return predict()\n\ndef predict():\n    return 0.9\n"


# ---- reading the truth back out ----------------------------------------


def _run(repo: Path, args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def _sha(repo: Path, rev: str) -> str:
    """The full object id of a revision."""
    return _run(repo, ["rev-parse", rev])


def _short(repo: Path, rev: str) -> str:
    return _run(repo, ["rev-parse", "--short", rev])


def _subjects(repo: Path, rev_range: str) -> list[str]:
    """Commit subjects in a range, oldest first."""
    out = _run(repo, ["log", "--reverse", "--pretty=%s", rev_range])
    return [line for line in out.splitlines() if line]


def _rev_list(repo: Path, rev_range: str) -> list[str]:
    """Full object ids in a range, git's default order (newest first)."""
    out = _run(repo, ["rev-list", rev_range])
    return [line for line in out.splitlines() if line]


def _touching(repo: Path, ref: str, path: str) -> list[str]:
    """Commits reachable from `ref` that changed `path`. The duplication meter."""
    out = _run(repo, ["rev-list", ref, "--", path])
    return [line for line in out.splitlines() if line]


def _is_ancestor(repo: Path, a: str, b: str) -> bool:
    """True when commit `a` is an ancestor of commit `b`."""
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", a, b],
            cwd=repo,
            capture_output=True,
        ).returncode
        == 0
    )


def _order(snapshots) -> list[str]:
    """One branch order across the run, so a branch keeps its color."""
    order: list[str] = []
    for snap in snapshots:
        for state in snap.repos.values():
            for name in list(state.branches) + [
                r.split("/", 1)[1] for r in state.remotes
            ]:
                if name not in order:
                    order.append(name)
    return order


def main():
    clear(FIGURES, TABLES)

    # ==================================================================
    # The collaboration story, in team mode: GitHub, Alice and Bob.
    # ==================================================================
    box = Sandbox(people=("alice", "bob"))
    github = box.paths["github"]
    bob = box.paths["bob"]

    # ---- 1. everyone starts on main ----------------------------------
    box.commit("alice", "readme.md", README_TXT, "project-start")
    box.git("alice", "push -u origin main")
    box.git("bob", "pull")
    box.snap(
        "Everyone starts on main",
        note="Alice pushed the first commit and Bob pulled it. Both clones and GitHub agree on main.",
    )

    # ---- 2. Alice builds the API on a branch, unmerged ---------------
    box.git("alice", "switch -c alice-api")
    box.commit("alice", "api.py", API_1, "api-skeleton")
    box.commit("alice", "api.py", API_2, "api-endpoints")
    box.git("alice", "push -u origin alice-api")
    box.snap(
        "Alice pushes alice-api and opens a PR into main",
        note="Alice's API lives only on the alice-api branch. It is not merged. The PR alice-api into main is open and unapproved.",
    )

    # ---- 3. Bob stacks his branch on alice-api, not on main ----------
    box.git("bob", "fetch origin")
    box.git("bob", "switch -c alice-api origin/alice-api")
    box.git("bob", "switch -c bob-model")
    box.commit("bob", "model.py", MODEL_1, "model-skeleton")
    box.commit("bob", "model.py", MODEL_2, "train-model")
    box.git("bob", "push -u origin bob-model")
    box.snap(
        "Bob branches from alice-api, not main",
        note="Bob needs Alice's API to build his model, so bob-model is stacked on alice-api. Two PRs are open: alice-api into main, bob-model into alice-api.",
    )

    # Facts at the moment of stacking, read from Bob's repository.
    a2 = _sha(bob, "alice-api")
    a1 = _sha(bob, "alice-api~1")
    old_b2 = _sha(bob, "bob-model")
    old_b1 = _sha(bob, "bob-model~1")
    bob_subjects_before = _subjects(bob, "alice-api..bob-model")
    ancestor_a1_before = _is_ancestor(bob, a1, "bob-model")
    ancestor_a2_before = _is_ancestor(bob, a2, "bob-model")
    touching_before = _touching(bob, "bob-model", "api.py")
    stack_history = set(_rev_list(bob, "bob-model"))

    # ---- 4. Alice's PR is squash-merged into main -------------------
    box.git("alice", "switch main")
    squashed = box.git("alice", "merge --squash alice-api")
    assert squashed.returncode == 0, (
        f"the squash must stage cleanly:\n{squashed.stderr}"
    )
    box.git("alice", "commit -m add-api-squashed")
    box.git("alice", "push")
    deleted = box.git("alice", "push origin --delete alice-api")
    assert deleted.returncode == 0, (
        f"GitHub deletes the merged branch:\n{deleted.stderr}"
    )
    box.git("alice", "branch -D alice-api")
    box.snap(
        "Alice's PR is squash-merged into main",
        note="GitHub squashed Alice's two commits into one new commit on main, then deleted the alice-api branch, as it does by default. Alice drops her local copy too.",
    )
    s = _sha(github, "main")

    # ---- 5. Bob fetches: origin/main is ahead, bob-model is not ------
    # Bob only fetches here. His local main and his bob-model are untouched,
    # so bob-model still hangs off Alice's old commits. This is the moment the
    # stacked branch has to be dealt with.
    box.git("bob", "fetch --prune origin")
    box.snap(
        "Bob fetches: main has moved, bob-model has not",
        note="origin/main is now Alice's squashed commit. Bob's bob-model still sits on Alice's old commits. If he repointed his PR to main now, it would show Alice's commits as his own.",
    )

    # The hazard, measured: what Bob's PR into main (origin/main) would show.
    leak = _rev_list(bob, "origin/main..bob-model")
    bob_originmain_before = _sha(bob, "origin/main")
    base_before = _sha(bob, "bob-model~2")

    # ---- 6. Bob syncs main, restacks with --onto, drops the old base -
    box.git("bob", "switch main")
    ff = box.git("bob", "merge origin/main")
    assert ff.returncode == 0, f"Bob's main must fast-forward:\n{ff.stderr}"
    box.git("bob", "switch bob-model")
    reb = box.git("bob", "rebase --onto main alice-api bob-model")
    assert reb.returncode == 0, f"the restack must succeed:\n{reb.stderr}"

    new_b2 = _sha(bob, "bob-model")
    new_b1 = _sha(bob, "bob-model~1")
    base_after = _sha(bob, "bob-model~2")
    bob_main_after = _sha(bob, "main")
    bob_subjects_after = _subjects(bob, "main..bob-model")
    pr_after = _rev_list(bob, "main..bob-model")
    touching_after = _touching(bob, "bob-model", "api.py")
    ancestor_a1_after = _is_ancestor(bob, a1, "bob-model")
    ancestor_a2_after = _is_ancestor(bob, a2, "bob-model")
    ancestor_main_after = _is_ancestor(bob, "main", "bob-model")
    model_file = (bob / "model.py").read_text()
    api_file = (bob / "api.py").read_text()

    box.git("bob", "branch -D alice-api")
    pf = box.git("bob", "push --force-with-lease origin bob-model")
    assert pf.returncode == 0, f"the force-push must succeed:\n{pf.stderr}"
    box.snap(
        "Bob restacks with rebase --onto, then force-pushes",
        note="git rebase --onto main alice-api bob-model replayed only Bob's two commits onto the new main. He drops the stale alice-api ref and force-pushes. His PR, repointed to main, now shows his work and nothing of Alice's.",
    )
    gh_bob_model = _sha(github, "bob-model")
    gh_main = _sha(github, "main")
    gh_pr = _rev_list(github, "main..bob-model")

    render(box, FIGURES, TABLES, mode="team")

    # ==================================================================
    # A zoom into the --onto surgery, solo, just Bob's DAG. Drawn from the
    # SAME scenario above (not a separate sandbox) so every hash in these two
    # figures matches the ones the README and tables quote. The "before" is
    # the moment Bob has fetched and main has moved but his branch has not;
    # the "after" is the restack. Bob's panel is rendered alone with the same
    # layout the team figures use.
    # ==================================================================
    bx, by = layout(box.snapshots)
    border = _order(box.snapshots)
    draw(
        box.snapshots[4],  # "Bob fetches: main has moved, bob-model has not"
        bx,
        by,
        FIGURES / "restack-before.png",
        mode="solo",
        repos=("bob",),
        order=border,
        title="Before the restack: bob-model still sits on alice-api",
    )
    draw(
        box.snapshots[5],  # "Bob restacks with rebase --onto, then force-pushes"
        bx,
        by,
        FIGURES / "restack-after.png",
        mode="solo",
        repos=("bob",),
        order=border,
        title="After the restack: bob-model sits directly on main",
    )

    # ==================================================================
    # Why --onto and not plain `git rebase main`. The plain form replays
    # the whole range main..bob-model, which here includes Alice's commits,
    # so it conflicts trying to re-add api.py that main already has.
    # ==================================================================
    contrast = Sandbox(people=("bob",), local=True)
    contrast.commit("bob", "readme.md", README_TXT, "project-start")
    contrast.git("bob", "switch -c alice-api")
    contrast.commit("bob", "api.py", API_1, "api-skeleton")
    contrast.commit("bob", "api.py", API_2, "api-endpoints")
    contrast.git("bob", "switch -c bob-model")
    contrast.commit("bob", "model.py", MODEL_1, "model-skeleton")
    contrast.commit("bob", "model.py", MODEL_2, "train-model")
    contrast.git("bob", "switch main")
    contrast.git("bob", "merge --squash alice-api")
    contrast.git("bob", "commit -m add-api-squashed")
    contrast.git("bob", "switch bob-model")
    c_repo = contrast.paths["bob"]
    c_a2 = _sha(c_repo, "alice-api")
    c_a1 = _sha(c_repo, "alice-api~1")
    plain_range = _rev_list(c_repo, "main..bob-model")  # what plain rebase replays
    onto_range = _rev_list(c_repo, "alice-api..bob-model")  # what --onto replays
    plain = contrast.git("bob", "rebase main", check=False)
    plain_conflict = (c_repo / "api.py").read_text()
    plain_status = _run(c_repo, ["status", "--porcelain", "api.py"])
    contrast.git("bob", "rebase --abort")
    contrast.cleanup()

    with (TABLES / "plain-vs-onto.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["command", "commits it replays", "result"])
        w.writerow(
            ["git rebase main bob-model", len(plain_range), "conflict on api.py"]
        )
        w.writerow(
            [
                "git rebase --onto main alice-api bob-model",
                len(onto_range),
                "clean, only Bob's commits",
            ]
        )

    # ==================================================================
    # Tables the README quotes, all read from real git above.
    # ==================================================================
    alice_in_leak = sum(1 for x in leak if x in (a1, a2))
    bob_in_leak = len(leak) - alice_in_leak
    alice_in_after = sum(1 for x in pr_after if x in (a1, a2))

    with (TABLES / "bob-pr-contents.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "stage",
                "commits in Bob's PR into main",
                "Alice's commits shown",
                "Bob's commits shown",
            ]
        )
        w.writerow(
            ["repoint to main, no restack", len(leak), alice_in_leak, bob_in_leak]
        )
        w.writerow(
            ["after git rebase --onto", len(pr_after), alice_in_after, len(pr_after)]
        )

    with (TABLES / "onto-surgery.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "your commit",
                "before restack",
                "base before",
                "after restack",
                "base after",
            ]
        )
        w.writerow(
            [
                "model-skeleton",
                _short(bob, old_b1),
                _short(bob, a2),
                _short(bob, new_b1),
                _short(bob, s),
            ]
        )
        w.writerow(
            [
                "train-model",
                _short(bob, old_b2),
                _short(bob, old_b1),
                _short(bob, new_b2),
                _short(bob, new_b1),
            ]
        )

    with (TABLES / "alice-work-under-bob.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "stage",
                "commits reachable from bob-model touching api.py",
                "which",
            ]
        )
        w.writerow(
            [
                "before restack",
                len(touching_before),
                ", ".join(_short(bob, x) for x in touching_before),
            ]
        )
        w.writerow(
            [
                "after restack",
                len(touching_after),
                ", ".join(_short(bob, x) for x in touching_after),
            ]
        )

    # ==================================================================
    # THE ORACLE: every claim the README makes, checked.
    # ==================================================================

    # (a) Bob's branch initially carries Alice's API commits as ancestors.
    assert ancestor_a1_before, "before the restack, A1 is an ancestor of bob-model"
    assert ancestor_a2_before, "before the restack, A2 is an ancestor of bob-model"
    assert a1 != a2, "Alice made two distinct commits"

    # The stack is real: at stacking time bob-model reached Alice's two commits
    # and Bob's original two, five commits including the base.
    assert {a1, a2, old_b1, old_b2}.issubset(stack_history), (
        "at stacking time bob-model's history included Alice's two commits and Bob's two"
    )
    assert len(stack_history) == 5, "init plus Alice's two plus Bob's two"

    # The squash gave main a NEW commit, not one of Alice's originals.
    assert s not in (a1, a2), (
        "main's tip after the squash is a new commit, not A1 or A2"
    )
    assert bob_originmain_before == s, (
        "after fetching, Bob's origin/main is the squashed commit"
    )
    assert bob_main_after == s, (
        "Bob fast-forwarded his local main onto the squashed commit before restacking"
    )

    # The hazard, in commits. Without a restack, Bob's PR into main would show
    # FOUR commits: Alice's two and Bob's two.
    assert len(leak) == 4, (
        "before the restack, origin/main..bob-model holds four commits"
    )
    assert {a1, a2}.issubset(set(leak)), (
        "and two of them are Alice's, leaking into what would be Bob's PR"
    )
    assert {old_b1, old_b2}.issubset(set(leak)), "the other two are Bob's own"
    assert base_before == a2, (
        "before the restack, Bob's lowest commit sits on Alice's A2"
    )

    # (b) After the restack, Bob's commits sit on main's tip, not on alice-api.
    assert base_after == s, "after the restack, Bob's lowest commit sits on main's tip"
    assert base_after != a2, "no longer on Alice's old A2"
    assert ancestor_main_after, "main's tip is now an ancestor of bob-model"
    assert not ancestor_a1_after, "A1 is no longer an ancestor of bob-model"
    assert not ancestor_a2_after, "A2 is no longer an ancestor of bob-model"

    # (c) Bob's own commits are preserved: same subjects, in order, new hashes.
    assert bob_subjects_before == ["model-skeleton", "train-model"], (
        "Bob authored exactly these two commits"
    )
    assert bob_subjects_after == bob_subjects_before, (
        "the restack preserved Bob's commit subjects, in order"
    )
    assert {new_b1, new_b2}.isdisjoint({old_b1, old_b2}), (
        "the replayed commits have new hashes, because their base changed"
    )

    # And the files Bob ends with are intact: Alice's API plus his model.
    assert model_file == MODEL_2, "Bob's model.py survived the restack unchanged"
    assert api_file == API_2, "and Alice's API is present at its endpoints version"

    # (d) Alice's commits are no longer duplicated under Bob. His PR into main
    # now shows only his two commits, and Alice's API change appears once.
    assert set(pr_after) == {new_b1, new_b2}, (
        "Bob's PR into main now shows only his two commits"
    )
    assert alice_in_after == 0, "no Alice commit remains in Bob's PR range"
    assert len(touching_before) == 2, (
        "before the restack, two commits under Bob touched api.py: Alice's A1 and A2"
    )
    assert len(touching_after) == 1, (
        "after the restack, exactly one commit reachable from bob-model touches api.py"
    )
    assert touching_after == [s], (
        "and it is main's own squashed commit, not a copy of Alice's work under Bob"
    )

    # The force-push landed the restacked branch on GitHub, and GitHub's view of
    # Bob's PR agrees: only Bob's two commits.
    assert gh_bob_model == new_b2, (
        "the force-push updated GitHub's bob-model to the restacked tip"
    )
    assert gh_main == s, "GitHub's main is the squashed commit"
    assert set(gh_pr) == {new_b1, new_b2}, (
        "on GitHub too, Bob's PR into main shows only his two commits"
    )

    # Why --onto is the right tool. Plain `git rebase main` replays the whole
    # main..bob-model range, which here is four commits including Alice's, and it
    # conflicts trying to re-add api.py that main already carries. The --onto form
    # replays only the alice-api..bob-model range, which is Bob's two commits.
    assert len(plain_range) == 4, (
        "plain git rebase main would replay four commits: Alice's two and Bob's two"
    )
    assert {c_a1, c_a2}.issubset(set(plain_range)), (
        "and two of them are Alice's, which is the problem"
    )
    assert len(onto_range) == 2, "the --onto range is only Bob's two commits"
    assert {c_a1, c_a2}.isdisjoint(set(onto_range)), (
        "the --onto range excludes Alice's commits entirely"
    )
    assert plain.returncode != 0, "plain git rebase main stops on a conflict"
    assert "<<<<<<<" in plain_conflict, "git wrote conflict markers into api.py"
    assert plain_status.startswith("AA"), (
        "the conflict is add/add on api.py: both sides add the file"
    )

    # Figures and tables actually landed.
    steps = sorted(p.name for p in FIGURES.glob("step-*.png"))
    assert len(steps) == 6, f"expected 6 team-mode step figures, got {steps}"
    assert (FIGURES / "restack-before.png").exists()
    assert (FIGURES / "restack-after.png").exists()
    tables = sorted(p.name for p in TABLES.glob("*.csv"))
    assert len(tables) == 5, f"expected 5 tables, got {tables}"

    print(
        f"stacked: bob-model carried Alice's {a1[:7]}, {a2[:7]} as ancestors; "
        f"PR into main would have shown {len(leak)} commits."
    )
    print(
        f"restack: git rebase --onto replayed Bob's 2 commits onto main's {s[:7]}; "
        f"new ids {new_b1[:7]}, {new_b2[:7]}; base moved {a2[:7]} -> {s[:7]}."
    )
    print(
        f"clean:   Bob's PR into main now shows {len(pr_after)} commits, "
        f"api.py touched by {len(touching_after)} commit under bob-model (was {len(touching_before)})."
    )
    print(
        f"{len(list(FIGURES.glob('*.png')))} figures, {len(tables)} tables. All assertions passed."
    )

    box.cleanup()


if __name__ == "__main__":
    main()
