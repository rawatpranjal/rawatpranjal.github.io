"""Run real git in a sandbox, snapshot the true state, draw it.

Every figure in this collection is a photograph of a repository that actually
exists on disk for the duration of the run, never a hand-drawn picture. The
sandbox holds a bare repo standing in for GitHub plus one clone per person.
After each command the full commit graph and every ref are read back out with
plumbing, so a diagram cannot drift from what git really did.

Two layout modes:

  solo  a single commit graph, Pro Git style, for one person on one machine.
  team  the same graph drawn once per repository, stacked, so a commit sits at
        the same x in every box and you can see at a glance who has what.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Dark palette, tuned to match the deck's design tokens (bg #16171a, accent
# #7cb0de). PAPER is the canvas/background; NODE_FILL gives commits a subtle
# body a shade lighter than the canvas; NODE_EDGE (accent) outlines them.
INK = "#dbe3ee"  # node hash text, HEAD, default light ink
MUTED = "#7d8894"  # parent-pointer arrows, secondary text
PAPER = "#16171a"  # figure + axes background; hollow (remote) box fill
PANEL = "#1c1f26"  # team-mode per-repo panel, a shade above the canvas
NODE_FILL = "#1d2733"  # commit circle body
NODE_EDGE = "#7cb0de"  # commit circle outline (deck accent)
LANE_COLORS = ["#7cb0de", "#e8834a", "#b19cf5", "#6bbf8a", "#e06b9c", "#7aa2f0"]
HEAD_COLOR = "#dbe3ee"
NEW_COLOR = "#e8a13c"  # the commit the last command just created (amber)

COMMIT_R = 0.24
X_STEP = 1.9
Y_STEP = 1.4


@dataclass
class Commit:
    sha: str
    parents: list[str]
    subject: str


@dataclass
class RepoState:
    """What one repository knows, at one moment."""

    name: str
    commits: dict[str, Commit]
    branches: dict[str, str]  # local branch -> sha
    remotes: dict[str, str]  # origin/main -> sha
    head: str | None  # branch name, or None when detached
    head_sha: str | None
    dirty: list[str] = field(default_factory=list)  # porcelain lines
    files: list[tuple[str, str]] = field(
        default_factory=list
    )  # (filename, status) for the working-directory view


@dataclass
class Snapshot:
    """The whole sandbox, at one moment, plus the command that got us here."""

    label: str
    command: str
    note: str
    repos: dict[str, RepoState]
    highlight: set[str] = field(default_factory=set)  # shas to draw as new
    bares: set[str] = field(default_factory=set)  # server-side repos, no working tree


class Sandbox:
    """A throwaway git universe: a bare 'github' plus one clone per person."""

    def __init__(
        self,
        people: tuple[str, ...] = ("alice",),
        initial_branch: str = "main",
        show_unreachable: bool = False,
        local: bool = False,
    ):
        # show_unreachable walks the reflog too, so commits no branch points at
        # any more still appear. Rebase, reset and reflog want that (the old
        # commits really are still on disk). Everywhere else it is clutter.
        #
        # local=True skips the bare server and clone entirely, so each person is
        # a plain `git init` with no remote. The first-steps tutorials use this,
        # because a phantom origin/main would confuse a reader who has not met
        # GitHub yet.
        self.root = Path(tempfile.mkdtemp(prefix="gitviz-"))
        self.people = people
        self.initial_branch = initial_branch
        self.show_unreachable = show_unreachable
        self.snapshots: list[Snapshot] = []
        self._log: list[tuple[str, str]] = []

        self.paths: dict[str, Path] = {}
        self.bares: set[str] = set()  # repos with no working tree, drawn as servers
        self.remote_names: set[str] = {"origin"}

        if not local:
            self.remote = self.add_bare("github")

        for person in people:
            path = self.root / person
            if local:
                self.git(
                    person,
                    f"init --initial-branch={initial_branch} {path}",
                    cwd=self.root,
                    record=False,
                )
            else:
                self.git(
                    person, f"clone {self.remote} {path}", cwd=self.root, record=False
                )
            self.paths[person] = path
            self.git(person, f"config user.name {person}", record=False)
            self.git(person, f"config user.email {person}@example.com", record=False)
            # Deterministic hashes across runs: fixed author and committer dates.
            self.git(person, "config commit.gpgsign false", record=False)

    def add_bare(self, name: str) -> Path:
        """Another server-side repository. A fork needs two of these."""
        path = self.root / f"{name}.git"
        self.git(
            name,
            f"init --bare --initial-branch={self.initial_branch} {path}",
            cwd=self.root,
            record=False,
        )
        self.paths[name] = path
        self.bares.add(name)
        return path

    def add_remote(self, who: str, remote: str, bare: str):
        """Point someone's clone at another server, e.g. `upstream` alongside `origin`."""
        self.git(who, f"remote add {remote} {self.paths[bare]}", record=False)
        self.remote_names.add(remote)

    # ---- driving git --------------------------------------------------

    def git(
        self,
        who: str,
        args: str,
        cwd: Path | None = None,
        record: bool = True,
        check: bool = True,
    ):
        """Run one git command as `who`. Returns the CompletedProcess."""
        where = cwd if cwd is not None else self.paths[who]
        env = {
            "GIT_AUTHOR_DATE": "2026-01-01T12:00:00",
            "GIT_COMMITTER_DATE": "2026-01-01T12:00:00",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(self.root),
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
        proc = subprocess.run(
            ["git", *args.split()],
            cwd=where,
            env=env,
            capture_output=True,
            text=True,
        )
        if check and proc.returncode != 0 and not _expected_failure(args):
            raise RuntimeError(f"git {args} (as {who}) failed:\n{proc.stderr}")
        if record:
            self._log.append((who, f"git {args}"))
        return proc

    def write(self, who: str, filename: str, content: str, record: bool = True):
        """Edit a file in someone's working tree."""
        (self.paths[who] / filename).write_text(content)
        if record:
            self._log.append((who, f"edit {filename}"))

    def commit(self, who: str, filename: str, content: str, message: str):
        """The common case: edit, stage, commit. Records the three steps."""
        self.write(who, filename, content, record=False)
        self.git(who, f"add {filename}")
        self.git(who, f"commit -m {message.replace(' ', '-')}")

    # ---- reading the truth back out ------------------------------------

    def read(self, name: str) -> RepoState:
        path = self.paths[name]
        walk = ["git", "log", "--all", "--pretty=%h\x1f%p\x1f%s"]
        if self.show_unreachable:
            walk.insert(3, "--reflog")
        raw = subprocess.run(walk, cwd=path, capture_output=True, text=True).stdout
        commits: dict[str, Commit] = {}
        for line in raw.splitlines():
            sha, parents, subject = line.split("\x1f")
            commits[sha] = Commit(sha, parents.split() if parents else [], subject)

        refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname:short)\x1f%(objectname:short)"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout
        branches, remotes = {}, {}
        for line in refs.splitlines():
            ref, sha = line.split("\x1f")
            if "/" in ref and ref.split("/")[0] in self.remote_names:
                remotes[ref] = sha
            else:
                branches[ref] = sha

        head = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head_sha = subprocess.run(
            ["git", "rev-parse", "--short", "-q", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = [
            line
            for line in subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        ]
        # Every file in the working directory with the status git reports for it,
        # for the working-directory view. A porcelain code (index, worktree) maps
        # to a label; a tracked file with no porcelain line is committed and clean.
        dirty_map = {line[3:]: line[:2] for line in dirty}
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=path, capture_output=True, text=True
        ).stdout.split()
        files = [
            (n, _file_status(dirty_map[n]) if n in dirty_map else "committed")
            for n in sorted(set(tracked) | set(dirty_map))
        ]
        return RepoState(
            name=name,
            commits=commits,
            branches=branches,
            remotes=remotes,
            head=head or None,
            head_sha=head_sha or None,
            dirty=dirty,
            files=files,
        )

    def snap(self, label: str, note: str = "") -> Snapshot:
        """Freeze the current state of every repo and remember what caused it."""
        command = "; ".join(cmd for _, cmd in self._log) if self._log else ""
        self._log.clear()
        repos = {name: self.read(name) for name in (*sorted(self.bares), *self.people)}
        known = set()
        for prev in self.snapshots:
            for state in prev.repos.values():
                known |= set(state.commits)
        now = set()
        for state in repos.values():
            now |= set(state.commits)
        snap = Snapshot(
            label=label,
            command=command,
            note=note,
            repos=repos,
            highlight=now - known,
            bares=set(self.bares),
        )
        self.snapshots.append(snap)
        return snap

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _expected_failure(args: str) -> bool:
    """Commands a tutorial runs on purpose to show git refusing or conflicting."""
    return any(
        token in args
        for token in ("merge", "rebase", "cherry-pick", "pull", "push", "apply", "am")
    )


# The working-directory view colors each file by what git thinks of it.
FILE_STATUS_COLOR = {
    "untracked": MUTED,
    "modified": LANE_COLORS[1],  # orange, edited but not staged
    "staged": LANE_COLORS[3],  # green, ready for the next commit
    "committed": INK,  # clean, saved in history
}


def _file_status(code: str) -> str:
    """A git porcelain status code (index char, worktree char) to a plain label."""
    return {
        "??": "untracked",
        " M": "modified",
        " D": "modified",
        "M ": "staged",
        "A ": "staged",
        "MM": "staged",
        "AM": "staged",
        "MD": "staged",
    }.get(code, "modified")


# ---- layout ------------------------------------------------------------


def layout(snapshots: list[Snapshot]) -> tuple[dict[str, float], dict[str, float]]:
    """One global (x, y) per commit, shared by every repo box and every frame.

    x is the commit's depth in the graph, so a child always sits right of its
    parents. y is a lane, chosen so the trunk runs flat along the bottom and
    each side branch gets its own row above it.
    """
    commits: dict[str, Commit] = {}
    branch_order: list[str] = []
    for snap in snapshots:
        for state in snap.repos.values():
            commits.update(state.commits)
            for name in list(state.branches) + [
                r.split("/", 1)[1] for r in state.remotes
            ]:
                if name not in branch_order:
                    branch_order.append(name)

    trunk = [b for b in branch_order if b in ("main", "master")]
    branch_order = trunk + [b for b in branch_order if b not in trunk]

    depth: dict[str, float] = {}

    def get_depth(sha: str) -> int:
        if sha in depth:
            return int(depth[sha])
        commit = commits.get(sha)
        if commit is None or not commit.parents:
            depth[sha] = 0
            return 0
        depth[sha] = 1 + max(get_depth(p) for p in commit.parents if p in commits)
        return int(depth[sha])

    for sha in commits:
        get_depth(sha)

    # Which branch tips can reach each commit, across every repo and frame.
    tips: dict[str, set[str]] = {b: set() for b in branch_order}
    for snap in snapshots:
        for state in snap.repos.values():
            for ref, sha in {
                **state.branches,
                **{k.split("/", 1)[1]: v for k, v in state.remotes.items()},
            }.items():
                if ref in tips:
                    tips[ref].add(sha)

    reach: dict[str, set[str]] = {}
    for branch, heads in tips.items():
        seen: set[str] = set()
        stack = list(heads)
        while stack:
            sha = stack.pop()
            if sha in seen or sha not in commits:
                continue
            seen.add(sha)
            stack.extend(commits[sha].parents)
        reach[branch] = seen

    lane: dict[str, float] = {}
    for sha in commits:
        owners = [i for i, b in enumerate(branch_order) if sha in reach.get(b, ())]
        lane[sha] = float(min(owners)) if owners else float(len(branch_order))

    # Two commits must never land on the same dot.
    taken: set[tuple[int, float]] = set()
    for sha in sorted(commits, key=lambda s: (depth[s], lane[s])):
        while (int(depth[sha]), lane[sha]) in taken:
            lane[sha] += 1
        taken.add((int(depth[sha]), lane[sha]))

    xs = {sha: depth[sha] * X_STEP for sha in commits}
    ys = {sha: lane[sha] * Y_STEP for sha in commits}
    return xs, ys


def _wrap(text: str, width_in: float, fontsize: float = 7.5) -> list[str]:
    """Break a caption into lines that fit the figure width.

    A proportional font at `fontsize` averages about 0.52 * fontsize points per
    character; convert that to a character budget for the given width in inches.
    """
    import textwrap

    chars = max(20, int((width_in - 0.2) * 72 / (0.52 * fontsize)))
    return textwrap.wrap(text, width=chars) or [text]


def _branch_color(name: str, order: list[str]) -> str:
    if name in ("main", "master"):
        return LANE_COLORS[0]
    idx = order.index(name) if name in order else 0
    return LANE_COLORS[(idx + 1) % len(LANE_COLORS)]


def _draw_repo(
    ax,
    state: RepoState,
    snap: Snapshot,
    xs,
    ys,
    order,
    show_remotes: bool,
    show_head: bool = True,
):
    """Draw one repository's commits, branch pointers and HEAD onto one axes."""
    for sha, commit in state.commits.items():
        for parent in commit.parents:
            if parent not in state.commits:
                continue
            # The arrow runs child -> parent, the way the pointer really runs:
            # a commit records its parent, never its children.
            ax.add_patch(
                FancyArrowPatch(
                    (xs[sha] - COMMIT_R, ys[sha]),
                    (xs[parent] + COMMIT_R, ys[parent]),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    shrinkA=0,
                    shrinkB=0,
                    color=MUTED,
                    linewidth=1.1,
                    connectionstyle="arc3,rad=0.0"
                    if ys[parent] == ys[sha]
                    else "arc3,rad=-0.14",
                    zorder=1,
                )
            )

    for sha in state.commits:
        is_new = sha in snap.highlight
        ax.add_patch(
            Circle(
                (xs[sha], ys[sha]),
                COMMIT_R,
                facecolor=NODE_FILL,
                edgecolor=NEW_COLOR if is_new else NODE_EDGE,
                linewidth=2.0 if is_new else 1.3,
                zorder=2,
            )
        )
        ax.text(
            xs[sha],
            ys[sha],
            sha[:5],
            ha="center",
            va="center",
            fontsize=6.5,
            family="monospace",
            color=NEW_COLOR if is_new else INK,
            zorder=3,
        )

    refs = dict(state.branches)
    if show_remotes:
        refs.update(state.remotes)

    stack: dict[str, int] = {}
    for name, sha in sorted(refs.items()):
        if sha not in xs:
            continue
        level = stack.get(sha, 0)
        stack[sha] = level + 1
        y = ys[sha] + COMMIT_R + 0.30 + level * 0.42
        remote = "/" in name
        color = _branch_color(name.split("/")[-1], order)
        width = max(
            0.84, 0.115 * len(name) + 0.22
        )  # the label has to fit inside the box
        ax.add_patch(
            FancyBboxPatch(
                (xs[sha] - width / 2, y - 0.14),
                width,
                0.28,
                boxstyle="round,pad=0.04",
                facecolor=PAPER if remote else color,
                edgecolor=color,
                linewidth=1.2,
                linestyle="--" if remote else "-",
                zorder=4,
                mutation_aspect=1.0,
            )
        )
        ax.text(
            xs[sha],
            y,
            name,
            ha="center",
            va="center",
            fontsize=6.0,
            family="monospace",
            color=color if remote else PAPER,
            zorder=5,
        )
        ax.plot(
            [xs[sha], xs[sha]],
            [ys[sha] + COMMIT_R, y - 0.14],
            color=color,
            linewidth=0.9,
            linestyle=":",
            zorder=1,
        )

        if show_head and state.head == name:
            ax.text(
                xs[sha] + width / 2 + 0.12,
                y,
                "HEAD",
                ha="left",
                va="center",
                fontsize=6.0,
                family="monospace",
                fontweight="bold",
                color=HEAD_COLOR,
                zorder=5,
            )

    if state.head is None and state.head_sha in xs:  # detached
        sha = state.head_sha
        ax.text(
            xs[sha],
            ys[sha] - COMMIT_R - 0.34,
            "HEAD (detached)",
            ha="center",
            va="center",
            fontsize=6.0,
            family="monospace",
            fontweight="bold",
            color=NEW_COLOR,
            zorder=5,
        )

    if state.dirty:
        ax.text(
            0.995,
            0.04,
            f"working tree: {len(state.dirty)} changed",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.0,
            family="monospace",
            color=MUTED,
        )


def draw(
    snap: Snapshot,
    xs,
    ys,
    path: Path,
    mode: str = "solo",
    title: str | None = None,
    repos: tuple[str, ...] | None = None,
    order: list[str] | None = None,
    panel_labels: dict[str, str] | None = None,
):
    """Render one snapshot. `solo` draws one repo, `team` stacks them.

    order fixes the branch->color mapping. render() passes one order computed
    across every snapshot, so a branch keeps its color through the whole
    tutorial (and the deck flipbook). Left None, it is derived per-figure.
    """
    if order is None:
        order = []
        for state in snap.repos.values():
            for name in list(state.branches) + [
                r.split("/", 1)[1] for r in state.remotes
            ]:
                if name not in order:
                    order.append(name)

    names = repos or (("alice",) if mode == "solo" else tuple(snap.repos))

    # Headroom above the top commit has to clear the branch pointers stacked on
    # it, otherwise a busy commit pushes its own labels out of the axes.
    busiest = 1
    for name in names:
        state = snap.repos[name]
        refs = list(state.branches.values()) + (
            list(state.remotes.values()) if name not in snap.bares else []
        )
        for sha in set(refs):
            busiest = max(busiest, refs.count(sha))

    x_lo = min(xs.values(), default=0) - 0.9
    x_hi = max(xs.values(), default=0) + 1.5
    y_lo = min(ys.values(), default=0) - 0.7
    y_hi = max(ys.values(), default=0) + 0.75 + 0.42 * busiest

    # Fixed inches-per-data-unit, so a commit is the same size in every figure
    # of the collection and the axes carry no dead space.
    scale = 0.60
    panel_h = (y_hi - y_lo) * scale
    width = max(5.6, (x_hi - x_lo) * scale)
    # A narrow history would letterbox inside a min-width figure, so widen the
    # data range to match instead. Then the axes box already has the data's
    # aspect and an equal-aspect axes fills it exactly.
    slack = width / scale - (x_hi - x_lo)
    x_lo -= slack / 2
    x_hi += slack / 2

    # Wrap the note and the command line to the figure width now, so the header
    # and footer can be sized to hold however many lines each takes. Unwrapped,
    # a long command or note runs off the right edge (monospace is a touch wider
    # per character, so it gets a smaller budget).
    note_lines = _wrap(snap.note, width) if snap.note else []
    cmd_lines = _wrap(snap.command, width, fontsize=8.4) if snap.command else []
    header = 0.42 + 0.16 * len(cmd_lines)
    footer = 0.12 + 0.16 * len(note_lines)
    height = panel_h * len(names) + header + footer

    fig, axes = plt.subplots(
        len(names), 1, figsize=(width, height), squeeze=False, facecolor=PAPER
    )
    axes = [a[0] for a in axes]
    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1 - header / height,
        bottom=footer / height,
        hspace=0.06,
    )

    for ax, name in zip(axes, names):
        state = snap.repos[name]
        ax.set_facecolor(PANEL if mode == "team" else PAPER)
        # The bare repo on GitHub has no working tree, so it has no meaningful
        # HEAD to show. Drawing one there invites the reader to think someone
        # is standing in it.
        _draw_repo(
            ax,
            state,
            snap,
            xs,
            ys,
            order,
            show_remotes=(name not in snap.bares),
            show_head=(name not in snap.bares),
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(MUTED if mode == "team" else PAPER)
            spine.set_linewidth(0.8)
        if mode == "team":
            # When an upstream server is present, "github" is the reader's own
            # fork, so name it as such rather than the ambiguous "GitHub".
            has_upstream = "upstream" in snap.bares
            labels = {
                "github": "Your fork (GitHub)" if has_upstream else "GitHub",
                "upstream": "Upstream (GitHub)",
            }
            # A tutorial that names its own panels (e.g. worktrees, each a role
            # rather than a person) passes panel_labels; otherwise fall back to
            # the built-in names and a plain capitalize.
            label = (panel_labels or {}).get(name) or labels.get(
                name, name.capitalize()
            )
            ax.text(
                0.008,
                0.94,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color=INK,
            )

    heading = title or snap.label
    fig.text(
        0.015,
        1 - 0.20 / height,
        heading,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
    )
    for i, line in enumerate(cmd_lines):
        fig.text(
            0.015,
            1 - (0.45 + 0.16 * i) / height,
            line,
            fontsize=7.0,
            family="monospace",
            color=MUTED,
            ha="left",
            va="center",
        )
    # Stack the wrapped note lines up from the bottom, so the last line sits at
    # the same baseline a single-line note would.
    for i, line in enumerate(reversed(note_lines)):
        fig.text(
            0.015,
            (0.12 + 0.16 * i) / height,
            line,
            fontsize=7.5,
            color=INK,
            ha="left",
            va="center",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER)
    fig.savefig(
        path.with_suffix(".pdf"), facecolor=PAPER
    )  # the LaTeX note reuses these
    plt.close(fig)


def render(
    sandbox: Sandbox,
    figures: Path,
    tables: Path,
    mode: str = "solo",
    repos=None,
    panel_labels: dict[str, str] | None = None,
):
    """Draw every snapshot and write the state log. The two artifacts of a run."""
    xs, ys = layout(sandbox.snapshots)

    # One branch order across the whole run, so a branch keeps its color from
    # the first figure to the last (and through the deck flipbook).
    order: list[str] = []
    for snap in sandbox.snapshots:
        for state in snap.repos.values():
            for name in list(state.branches) + [
                r.split("/", 1)[1] for r in state.remotes
            ]:
                if name not in order:
                    order.append(name)

    for i, snap in enumerate(sandbox.snapshots, start=1):
        draw(
            snap,
            xs,
            ys,
            figures / f"step-{i:02d}.png",
            mode=mode,
            repos=repos,
            order=order,
            panel_labels=panel_labels,
        )

    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "state-log.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "step",
                "label",
                "command",
                "repo",
                "head",
                "head_sha",
                "branches",
                "dirty",
            ]
        )
        for i, snap in enumerate(sandbox.snapshots, start=1):
            for name, state in snap.repos.items():
                writer.writerow(
                    [
                        i,
                        snap.label,
                        snap.command,
                        name,
                        state.head or "(detached)",
                        state.head_sha or "",
                        " ".join(f"{b}={s}" for b, s in sorted(state.branches.items())),
                        len(state.dirty),
                    ]
                )


def draw_files(ax, state: RepoState, folder: str | None = None):
    """The working directory as a card: one row per file, colored by its status.

    The counterpart to the commit graph. The graph is the history; this is what
    is actually on disk right now, and what git thinks of each file.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.01",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=0.8,
        )
    )
    ax.text(
        0.06, 0.92, "Working directory", fontsize=10.5, fontweight="bold", color=INK
    )
    ax.text(
        0.06,
        0.85,
        f"{folder or state.name}/",
        fontsize=8.5,
        family="monospace",
        color=MUTED,
    )
    rows = state.files or [("(empty)", "committed")]
    y = 0.74
    for name, label in rows:
        color = FILE_STATUS_COLOR.get(label, INK)
        ax.text(0.10, y, "●", fontsize=10, color=color, va="center")
        ax.text(0.145, y, name, fontsize=9, family="monospace", color=INK, va="center")
        ax.add_patch(
            FancyBboxPatch(
                (0.60, y - 0.032),
                0.34,
                0.064,
                boxstyle="round,pad=0.006",
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                alpha=0.16,
            )
        )
        ax.text(
            0.77,
            y,
            label,
            fontsize=7.5,
            family="monospace",
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
        )
        y -= 0.105
    ax.text(
        0.06,
        0.06,
        "each dot is a file, colored by its status",
        fontsize=7,
        color=MUTED,
        style="italic",
    )


def draw_system(
    path: Path,
    stages: list[str],
    arrows: list[str],
    title: str,
    frame_label: str = "",
    note: str = "",
):
    """A higher-level schematic: named stages in a row, labeled arrows between.

    Not a photograph of the graph, a systemic view of where a change travels.
    Use it at the conceptual hinges, not on every step.
    """
    n = len(stages)
    fig, ax = plt.subplots(figsize=(2.6 * n + 0.6, 3.0), facecolor=PAPER)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if frame_label:
        ax.add_patch(
            FancyBboxPatch(
                (0.04, 0.10),
                n - 0.08,
                0.66,
                boxstyle="round,pad=0.01",
                facecolor=PANEL,
                edgecolor=MUTED,
                linewidth=1.0,
                linestyle="--",
            )
        )
        ax.text(0.12, 0.70, frame_label, fontsize=8.5, color=MUTED, fontweight="bold")

    for i, label in enumerate(stages):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        ax.add_patch(
            FancyBboxPatch(
                (i + 0.14, 0.34),
                0.72,
                0.20,
                boxstyle="round,pad=0.01",
                facecolor=PAPER,
                edgecolor=color,
                linewidth=1.6,
            )
        )
        ax.text(
            i + 0.5,
            0.44,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            family="monospace",
            color=INK,
        )
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (i + 0.88, 0.44),
                    (i + 1.12, 0.44),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color=INK,
                    linewidth=1.3,
                )
            )
            if i < len(arrows):
                ax.text(
                    i + 1.0,
                    0.55,
                    arrows[i],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    family="monospace",
                    color=INK,
                )

    fig.text(
        0.015, 0.955, title, fontsize=10.5, fontweight="bold", color=INK, ha="left"
    )
    if note:
        fig.text(0.015, 0.04, note, fontsize=7.5, color=INK, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def draw_split(
    snap: Snapshot,
    xs,
    ys,
    path: Path,
    who: str,
    order: list[str],
    folder: str | None = None,
):
    """One snapshot as two columns: the working directory and the commit graph."""
    state = snap.repos[who]
    note_lines = _wrap(snap.note, 6.0) if snap.note else []
    cmd_lines = _wrap(snap.command, 6.5, fontsize=8.4) if snap.command else []
    header = 0.55 + 0.17 * len(cmd_lines)
    footer = 0.12 + 0.17 * len(note_lines)
    height = 3.8 + header + footer

    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(11.0, height),
        facecolor=PAPER,
        gridspec_kw={"width_ratios": [1.0, 1.28], "wspace": 0.06},
    )
    fig.subplots_adjust(
        left=0.02, right=0.985, top=1 - header / height, bottom=footer / height
    )

    draw_files(axL, state, folder=folder or who)

    axR.set_facecolor(PAPER)
    _draw_repo(axR, state, snap, xs, ys, order, show_remotes=(who not in snap.bares))
    axR.set_xlim(min(xs.values(), default=0) - 0.9, max(xs.values(), default=1) + 1.7)
    axR.set_ylim(min(ys.values(), default=0) - 0.8, max(ys.values(), default=0) + 1.6)
    axR.set_aspect("equal")
    axR.axis("off")
    axR.text(
        0.02,
        0.97,
        "History (the commit graph)",
        transform=axR.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="top",
    )

    fig.text(
        0.02, 1 - 0.22 / height, snap.label, fontsize=12.0, fontweight="bold", color=INK
    )
    for i, line in enumerate(cmd_lines):
        fig.text(
            0.02,
            1 - (0.46 + 0.17 * i) / height,
            line,
            fontsize=8.5,
            family="monospace",
            color=LANE_COLORS[0],
        )
    for i, line in enumerate(reversed(note_lines)):
        fig.text(0.02, (0.12 + 0.17 * i) / height, line, fontsize=8.0, color=INK)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)


def render_split(
    sandbox: Sandbox, figures: Path, tables: Path, who: str, folder: str | None = None
):
    """Like render(), but each figure is the two-column working-dir plus graph."""
    xs, ys = layout(sandbox.snapshots)
    order: list[str] = []
    for snap in sandbox.snapshots:
        for state in snap.repos.values():
            for name in list(state.branches):
                if name not in order:
                    order.append(name)
    for i, snap in enumerate(sandbox.snapshots, start=1):
        draw_split(
            snap, xs, ys, figures / f"step-{i:02d}.png", who, order, folder=folder
        )

    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "state-log.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["step", "label", "command", "repo", "head", "head_sha", "files"]
        )
        for i, snap in enumerate(sandbox.snapshots, start=1):
            state = snap.repos[who]
            writer.writerow(
                [
                    i,
                    snap.label,
                    snap.command,
                    who,
                    state.head or "(detached)",
                    state.head_sha or "",
                    " ".join(f"{n}:{s}" for n, s in state.files),
                ]
            )


def read_trees(sandbox: "Sandbox", who: str, filename: str) -> dict[str, str]:
    """The same file as the three trees see it, read out of real git.

    Working tree is the bytes on disk. Index is `git show :file`, the staged
    blob. HEAD is `git show HEAD:file`, the committed blob. When they differ,
    that difference IS what `git status` is reporting.
    """
    path = sandbox.paths[who]
    disk = (path / filename).read_text() if (path / filename).exists() else None

    def show(rev: str) -> str | None:
        proc = subprocess.run(
            ["git", "show", f"{rev}:{filename}"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        return proc.stdout if proc.returncode == 0 else None

    return {"working tree": disk, "index": show(""), "HEAD": show("HEAD")}


def draw_trees(
    trees: dict[str, str], path: Path, title: str, command: str = "", note: str = ""
):
    """Three columns: what the working tree, the index and HEAD each hold."""
    labels = ["working tree", "index", "HEAD"]
    fig, ax = plt.subplots(figsize=(7.4, 2.9), facecolor=PAPER)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, label in enumerate(labels):
        content = trees.get(label)
        missing = content is None
        ax.add_patch(
            FancyBboxPatch(
                (i + 0.04, 0.10),
                0.74,
                0.66,
                boxstyle="round,pad=0.02",
                facecolor=PANEL if not missing else PAPER,
                edgecolor=MUTED if missing else LANE_COLORS[i],
                linewidth=1.4,
                linestyle=":" if missing else "-",
            )
        )
        ax.text(
            i + 0.41,
            0.83,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=MUTED if missing else LANE_COLORS[i],
        )
        ax.text(
            i + 0.41,
            0.43,
            "(does not exist here)" if missing else content.strip() or "(empty)",
            ha="center",
            va="center",
            fontsize=8,
            family="monospace",
            color=MUTED if missing else INK,
        )

    # The arrows between the columns are the two commands that promote a file.
    for i, verb in enumerate(["git add", "git commit"]):
        ax.add_patch(
            FancyArrowPatch(
                (i + 0.82, 0.43),
                (i + 1.00, 0.43),
                arrowstyle="-|>",
                mutation_scale=13,
                color=INK,
                linewidth=1.2,
            )
        )
        ax.text(
            i + 0.91,
            0.53,
            verb,
            ha="center",
            va="center",
            fontsize=7,
            family="monospace",
            color=INK,
        )

    fig.text(
        0.015, 0.955, title, fontsize=10.5, fontweight="bold", color=INK, ha="left"
    )
    if command:
        fig.text(
            0.015,
            0.90,
            command,
            fontsize=7.0,
            family="monospace",
            color=MUTED,
            ha="left",
        )
    if note:
        fig.text(0.015, 0.03, note, fontsize=7.5, color=INK, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)


def draw_conflict(
    sandbox: "Sandbox",
    who: str,
    filename: str,
    path: Path,
    title: str,
    command: str = "",
    note: str = "",
):
    """The conflicted file exactly as git wrote it, markers and all.

    Reads the bytes off disk rather than composing them, so the markers in the
    figure are the ones the reader will actually see in their editor.
    """
    text = (sandbox.paths[who] / filename).read_text()
    lines = text.splitlines() or ["(empty)"]

    status = subprocess.run(
        ["git", "status", "--porcelain", filename],
        cwd=sandbox.paths[who],
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")

    row_h = 0.30
    fig, ax = plt.subplots(
        figsize=(7.6, 1.5 + row_h * len(lines) * 0.42), facecolor=PAPER
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(lines) - 0.5, 2.2)
    ax.axis("off")

    # Which side of the conflict each line belongs to, walking the markers the
    # way a human reads them.
    side = "context"
    for i, line in enumerate(lines):
        if line.startswith("<<<<<<<"):
            side = "ours"
        elif line.startswith("======="):
            side = "theirs"
        elif line.startswith(">>>>>>>"):
            side = "end"

        marker = line.startswith(("<<<<<<<", "=======", ">>>>>>>"))
        color = {
            "ours": LANE_COLORS[0],
            "theirs": LANE_COLORS[1],
            "context": INK,
            "end": LANE_COLORS[1],
        }[side]
        if marker:
            color = NEW_COLOR
        if side == "end" and not marker:
            side, color = "context", INK

        ax.text(
            0.25,
            -i,
            line or " ",
            ha="left",
            va="center",
            fontsize=8.5,
            family="monospace",
            fontweight="bold" if marker else "normal",
            color=color,
        )

    ax.text(
        0.25,
        1.5,
        f"{filename}   git status: {status}",
        ha="left",
        va="center",
        fontsize=7.5,
        family="monospace",
        color=MUTED,
    )

    fig.text(
        0.015, 0.965, title, fontsize=10.5, fontweight="bold", color=INK, ha="left"
    )
    if command:
        fig.text(
            0.015,
            0.90,
            command,
            fontsize=7.0,
            family="monospace",
            color=MUTED,
            ha="left",
        )
    if note:
        fig.text(0.015, 0.03, note, fontsize=7.5, color=INK, ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def clear(*dirs: Path):
    """run.py regenerates figures/ and tables/ from scratch, as the schema requires."""
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


if __name__ == "__main__":
    # Self-check: five real commands, five figures, and the DAG facts they claim.
    box = Sandbox(people=("alice",))
    box.commit("alice", "readme.md", "one", "first commit")
    box.snap("one commit")
    box.commit("alice", "readme.md", "two", "second commit")
    box.snap("two commits")
    box.git("alice", "switch -c feature")
    box.snap("a branch is a pointer")
    box.commit("alice", "feature.md", "work", "work on feature")
    box.snap("commit on the branch")
    box.git("alice", "switch main")
    box.git("alice", "merge feature")
    final = box.snap("fast-forward merge")

    alice = final.repos["alice"]
    assert len(alice.commits) == 3, "fast-forward must not create a commit"
    assert alice.branches["main"] == alice.branches["feature"], (
        "both refs point at the same commit"
    )
    assert all(len(c.parents) <= 1 for c in alice.commits.values()), (
        "no merge commit here"
    )

    out = Path(tempfile.mkdtemp(prefix="gitviz-check-"))
    render(box, out / "figures", out / "tables")
    assert len(list((out / "figures").glob("*.png"))) == 5
    box.cleanup()

    # Local mode: a plain git init with no remote, for the first-steps section.
    solo = Sandbox(people=("alice",), local=True)
    solo.commit("alice", "notes.md", "first line", "first commit")
    first = solo.snap("a first repo, no remote")
    assert solo.bares == set(), "local mode creates no server"
    assert first.repos["alice"].remotes == {}, "and the working repo has no origin"
    assert len(first.repos["alice"].commits) == 1, "one commit made, one exists"
    solo.cleanup()

    # Team mode: two people, one remote, and the fact that a push moves GitHub
    # while a clone that has not fetched still knows nothing about it.
    team = Sandbox(people=("alice", "bob"))
    team.commit("alice", "readme.md", "start", "first commit")
    team.git("alice", "push -u origin main")
    team.git("bob", "pull")
    team.snap("both cloned")
    team.commit("alice", "readme.md", "alice was here", "alice edits")
    team.git("alice", "push")
    pushed = team.snap("alice pushes")

    assert len(pushed.repos["github"].commits) == 2, "the push moved GitHub"
    assert len(pushed.repos["bob"].commits) == 1, (
        "Bob has not fetched, so Bob cannot know"
    )
    assert (
        pushed.repos["bob"].remotes["origin/main"]
        != pushed.repos["github"].branches["main"]
    ), "Bob's origin/main is a stale cache, not GitHub's live main"

    render(team, out / "team-figures", out / "team-tables", mode="team")
    assert len(list((out / "team-figures").glob("*.png"))) == 2
    team.cleanup()

    # A real conflict: two people change the same line, and git refuses to guess.
    fight = Sandbox(people=("alice", "bob"))
    fight.commit("alice", "config.py", "TIMEOUT = 30\n", "set timeout")
    fight.git("alice", "push -u origin main")
    fight.git("bob", "pull")
    fight.commit("alice", "config.py", "TIMEOUT = 60\n", "alice doubles it")
    fight.git("alice", "push")
    fight.commit("bob", "config.py", "TIMEOUT = 10\n", "bob cuts it")
    rejected = fight.git("bob", "push", check=False)
    assert rejected.returncode != 0, "git refuses a push that would drop Alice's commit"

    fight.git("bob", "fetch origin")
    merging = fight.git("bob", "merge origin/main", check=False)
    assert merging.returncode != 0, "the merge stops on a conflict"

    conflicted = (fight.paths["bob"] / "config.py").read_text()
    assert "<<<<<<<" in conflicted and ">>>>>>>" in conflicted, (
        "git wrote conflict markers into the file"
    )
    assert "TIMEOUT = 60" in conflicted and "TIMEOUT = 10" in conflicted, (
        "both intents are preserved in the file for a human to choose between"
    )
    draw_conflict(
        fight,
        "bob",
        "config.py",
        out / "conflict.png",
        "A conflict, as git actually wrote it",
        "git merge origin/main",
        "Git kept both versions and stopped. It will not guess which timeout is right.",
    )
    assert (out / "conflict.png").exists()
    fight.cleanup()

    # A fork: two servers, and a clone that knows about both.
    fork = Sandbox(people=("alice",))
    fork.add_bare("upstream")
    fork.add_remote("alice", "upstream", "upstream")
    fork.commit("alice", "readme.md", "hello", "first commit")
    fork.git("alice", "push -u origin main")
    forked = fork.snap("alice pushes to her fork, not to upstream")
    assert "upstream" in forked.bares and "github" in forked.bares, "two servers exist"
    assert len(forked.repos["upstream"].commits) == 0, (
        "upstream has nothing: a contributor cannot push there"
    )
    assert len(forked.repos["github"].commits) == 1, "her own fork has the commit"
    fork.cleanup()

    print(f"ok: solo, team, conflict and fork modes. Artifacts in {out}")
