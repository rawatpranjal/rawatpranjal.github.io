"""Draw tinyharness's true state: every figure is a photograph of a real run.

Design tokens are ported verbatim from the RAG deck's ragviz.py (which ported
them from the git deck's gitviz.py), so the three decks read as one system.

Flipbook figures use a FIXED canvas (no bbox_inches="tight"): reveal.js
auto-animate tweens between consecutive same-title slides, and a constant
canvas is what makes the context column read as growing rather than jumping.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from harness import Message, Snapshot

# Dark palette, matching the deck's design tokens (bg #16171a, accent #7cb0de).
INK = "#dbe3ee"
MUTED = "#7d8894"
PAPER = "#16171a"
PANEL = "#1c1f26"
NODE_FILL = "#1d2733"
NODE_EDGE = "#7cb0de"
LANE_COLORS = ["#7cb0de", "#e8834a", "#b19cf5", "#6bbf8a", "#e06b9c", "#7aa2f0"]
NEW_COLOR = "#e8a13c"
GREEN = "#6bbf8a"
PINK = "#e06b9c"
VIOLET = "#b19cf5"
BLUE = "#7cb0de"
ORANGE = "#e8834a"

ROLE_COLORS = {
    ("system", "text"): MUTED,
    ("system", "skill_index"): VIOLET,
    ("system", "skill_body"): VIOLET,
    ("system", "hook_inject"): NEW_COLOR,
    ("user", "hook_inject"): NEW_COLOR,
    ("system", "summary"): PINK,
    ("system", "memory"): "#7aa2f0",
    ("user", "text"): BLUE,
    ("assistant", "text"): INK,
    ("assistant", "tool_use"): ORANGE,
    ("tool", "tool_result"): GREEN,
}

LEGEND = [
    ("system", MUTED),
    ("user", BLUE),
    ("assistant", INK),
    ("tool call", ORANGE),
    ("tool result", GREEN),
    ("hook", NEW_COLOR),
    ("skill", VIOLET),
    ("summary", PINK),
]


def msg_color(m: Message) -> str:
    if m.kind == "tool_result" and (
        m.content.startswith("ERROR") or m.content.startswith("BLOCKED")
    ):
        return PINK
    return ROLE_COLORS.get((m.role, m.kind), INK)


def _short(s: str, n: int = 36) -> str:
    s = s.splitlines()[0] if s else ""
    return s if len(s) <= n else s[: n - 1] + "…"


def clear(*dirs: Path):
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


def _new_fig(title: str, note: str = "", figsize=(11.4, 6.4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PAPER)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    ax.text(2, 57.3, title, fontsize=13, fontweight="bold", color=INK)
    if note:
        ax.text(2, 1.0, note, fontsize=8, color=MUTED)
    return fig, ax


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER)
    fig.savefig(path.with_suffix(".pdf"), facecolor=PAPER)
    plt.close(fig)


# ---- the context column (the deck's signature element) ----------------------


def _draw_context_column(
    ax, snap: Snapshot, x0=4.0, x1=38.0, y0=8.0, y1=52.0, show_legend=True
):
    """Stacked message blocks, height proportional to real token counts,
    scaled to the budget so a filling column reads as a filling column."""
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 0.8, y0 - 0.8),
            (x1 - x0) + 1.6,
            (y1 - y0) + 1.6,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=1.0,
        )
    )
    ax.text(x0, y1 + 1.6, "context window", fontsize=9, color=MUTED, fontweight="bold")

    total = snap.total_tokens
    budget = snap.budget or max(total, 1)
    height = y1 - y0
    # Heights are token-proportional with a readable floor; the gauge on the
    # flank carries the exact ratio. If the floors overflow the column
    # (a crowded pre-compaction context), everything squeezes to fit.
    scale = height / max(budget, total, 1)
    heights = [max(m.tokens * scale, 2.4) for m in snap.context]
    if sum(heights) > height:
        squeeze = height / sum(heights)
        heights = [h * squeeze for h in heights]
    y = y1
    for i, m in enumerate(snap.context):
        h = heights[i]
        y -= h
        color = msg_color(m)
        new = i in snap.new_msgs
        ax.add_patch(
            FancyBboxPatch(
                (x0, y + 0.12),
                (x1 - x0),
                max(h - 0.24, 0.5),
                boxstyle="round,pad=0.06",
                facecolor=NODE_FILL,
                edgecolor=NEW_COLOR if new else color,
                linewidth=2.0 if new else 1.1,
            )
        )
        if h >= 1.7:
            if m.kind == "tool_use" and m.tool_name:
                label = f"tool_use {m.tool_name}"
            elif m.kind == "tool_result" and m.tool_name:
                label = f"{m.tool_name} result"
            else:
                label = m.kind if m.kind != "text" else m.role
            ax.text(
                x0 + 0.9,
                y + h / 2,
                _short(
                    f"{label}: {m.content.splitlines()[0] if m.content else ''}", 46
                ),
                fontsize=6.8,
                family="monospace",
                color=color,
                va="center",
            )
    # token gauge
    gx = x1 + 2.4
    ax.add_patch(
        FancyBboxPatch(
            (gx, y0),
            1.6,
            height,
            boxstyle="round,pad=0.05",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=0.8,
        )
    )
    frac = min(total / budget, 1.0)
    gcolor = GREEN if frac < 0.7 else (NEW_COLOR if frac < 0.95 else PINK)
    ax.add_patch(
        FancyBboxPatch(
            (gx, y0),
            1.6,
            height * frac,
            boxstyle="round,pad=0.05",
            facecolor=gcolor,
            edgecolor="none",
            alpha=0.75,
        )
    )
    ax.text(
        gx + 0.8,
        y0 - 2.2,
        f"{total}/{budget} tok",
        fontsize=7,
        family="monospace",
        color=gcolor,
        ha="center",
    )
    if show_legend:
        lx = x0
        for name, color in LEGEND:
            ax.add_patch(
                FancyBboxPatch(
                    (lx, 3.1),
                    0.9,
                    0.9,
                    boxstyle="round,pad=0.03",
                    facecolor=color,
                    edgecolor="none",
                )
            )
            ax.text(lx + 1.3, 3.5, name, fontsize=5.8, color=MUTED, va="center")
            lx += 1.7 + len(name) * 0.62


# ---- right panels ------------------------------------------------------------


def _panel_frame(ax, label: str, x0=47.0, x1=98.0, y0=4.5, y1=54.0):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.3",
            facecolor=PAPER,
            edgecolor="#2a2c31",
            linewidth=1.2,
        )
    )
    ax.text(x0 + 1.2, y1 - 2.0, label, fontsize=9, color=MUTED, fontweight="bold")
    return x0, x1, y0, y1


def _draw_loop_panel(ax, active: str, labels: dict | None = None):
    """The agent loop as a ring; the active node is lit amber."""
    labels = labels or {}
    x0, x1, y0, y1 = _panel_frame(ax, "the loop")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 1
    nodes = {
        "model": (cx, cy + 15, labels.get("model", "MODEL")),
        "hooks": (cx + 19, cy, labels.get("hooks", "HOOK GATE")),
        "tool": (cx, cy - 15, labels.get("tool", "TOOL")),
        "context": (cx - 19, cy, labels.get("context", "CONTEXT")),
    }
    arrows = [
        ("model", "hooks", "tool call"),
        ("hooks", "tool", ""),
        ("tool", "context", "result"),
        ("context", "model", ""),
    ]
    for a, b, lab in arrows:
        (xa, ya, _), (xb, yb, _) = nodes[a], nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                (xa, ya),
                (xb, yb),
                connectionstyle="arc3,rad=-0.35",
                arrowstyle="-|>",
                mutation_scale=13,
                color=MUTED,
                linewidth=1.4,
                shrinkA=24,
                shrinkB=24,
            )
        )
    for name, (x, y, lab) in nodes.items():
        hot = name == active
        ax.add_patch(
            FancyBboxPatch(
                (x - 7.5, y - 2.6),
                15,
                5.2,
                boxstyle="round,pad=0.25",
                facecolor="#3a2f14" if hot else NODE_FILL,
                edgecolor=NEW_COLOR if hot else NODE_EDGE,
                linewidth=2.2 if hot else 1.4,
            )
        )
        ax.text(
            x,
            y,
            lab,
            fontsize=8.5,
            family="monospace",
            color=NEW_COLOR if hot else INK,
            ha="center",
            va="center",
            fontweight="bold" if hot else "normal",
        )
    if active == "done":
        ax.add_patch(
            FancyArrowPatch(
                (nodes["model"][0] + 8, nodes["model"][1]),
                (nodes["model"][0] + 20, nodes["model"][1]),
                arrowstyle="-|>",
                mutation_scale=13,
                color=GREEN,
                linewidth=2,
            )
        )
        ax.text(
            nodes["model"][0] + 21,
            nodes["model"][1],
            "end turn",
            fontsize=8,
            color=GREEN,
            va="center",
            fontweight="bold",
        )


def _feature_boxes(ax, features, x, y, w=2.5, per_row=6, gap=0.6):
    """The feature_list.json burn-down checkboxes -- truth read from disk."""
    for i, f in enumerate(features):
        col, row = i % per_row, i // per_row
        bx, by = x + col * (w + gap), y - row * (w + gap)
        ok = f.get("passes")
        ax.add_patch(
            FancyBboxPatch(
                (bx, by),
                w,
                w,
                boxstyle="round,pad=0.08",
                facecolor="#1e3327" if ok else "#331e24",
                edgecolor=GREEN if ok else PINK,
                linewidth=1.4,
            )
        )
        mark = "✓" if ok else "✗"
        ax.text(
            bx + w / 2,
            by + w / 2,
            mark,
            fontsize=9,
            color=GREEN if ok else PINK,
            ha="center",
            va="center",
            fontweight="bold",
        )


def _commit_dots(ax, commits, x, y, max_dots=16):
    ax.text(x, y + 2.0, "git log", fontsize=7.5, color=MUTED, family="monospace")
    shown = commits[-max_dots:]
    for i, (sha, _msg) in enumerate(shown):
        ax.add_patch(
            Circle(
                (x + 1.2 + i * 2.6, y),
                0.9,
                facecolor=NODE_FILL,
                edgecolor=NODE_EDGE,
                linewidth=1.4,
            )
        )
    if shown:
        sha, msg = shown[-1]
        ax.text(
            x + 1.2 + (len(shown) - 1) * 2.6,
            y - 2.3,
            sha,
            fontsize=6,
            family="monospace",
            color=MUTED,
            ha="center",
        )


def _draw_files_panel(ax, snap: Snapshot, show_content: tuple = ()):
    """The disk panel: the real workspace read back from disk."""
    x0, x1, y0, y1 = _panel_frame(ax, "the disk (real files)")
    y = y1 - 5.0
    listed = [
        f
        for f in snap.files
        if f != "feature_list.json" and not f.startswith(".checkpoint")
    ]
    for f in listed[:11]:
        changed = f in snap.changed_files
        ax.text(
            x0 + 2.5,
            y,
            f,
            fontsize=7.8,
            family="monospace",
            color=NEW_COLOR if changed else INK,
            fontweight="bold" if changed else "normal",
        )
        if f in show_content:
            first = _short(snap.files[f], 44)
            ax.text(
                x0 + 4.5, y - 1.7, first, fontsize=6.2, family="monospace", color=MUTED
            )
            y -= 1.9
        y -= 2.5
    if snap.features:
        n_pass = sum(1 for f in snap.features if f.get("passes"))
        ax.text(
            x0 + 2.5,
            y - 0.5,
            f"feature_list.json  {n_pass}/{len(snap.features)} passing",
            fontsize=7.8,
            family="monospace",
            color=NEW_COLOR if "feature_list.json" in snap.changed_files else INK,
        )
        _feature_boxes(ax, snap.features, x0 + 2.5, y - 4.6)
        y -= 10.5
    if snap.commits:
        _commit_dots(ax, snap.commits, x0 + 2.5, y0 + 3.2)


RAIL_EVENTS = [
    ("session_start", "SessionStart"),
    ("user_prompt_submit", "UserPromptSubmit"),
    ("pre_tool_use", "PreToolUse"),
    ("post_tool_use", "PostToolUse"),
    ("stop", "Stop"),
    ("pre_compact", "PreCompact"),
]

ACTION_COLORS = {
    "allow": MUTED,
    "deny": PINK,
    "inject": NEW_COLOR,
    "continue_": NEW_COLOR,
}


def _draw_rail_panel(ax, events: list[dict], registered: dict | None = None):
    """The hook rail: the six lifecycle stations, with a flag for every real
    firing so far. Red flag = a hook really blocked something."""
    x0, x1, y0, y1 = _panel_frame(ax, "the hook rail")
    rail_y = y1 - 9.0
    xs = {}
    for i, (key, label) in enumerate(RAIL_EVENTS):
        x = x0 + 4.5 + i * (x1 - x0 - 9) / 5
        xs[key] = x
        has_hook = registered and key in registered
        ax.add_patch(
            Circle(
                (x, rail_y),
                1.1,
                facecolor=NODE_FILL,
                edgecolor=NODE_EDGE if has_hook else "#2a2c31",
                linewidth=2 if has_hook else 1.2,
            )
        )
        ax.text(
            x,
            rail_y + 2.6,
            label,
            fontsize=6.4,
            family="monospace",
            color=INK if has_hook else MUTED,
            ha="center",
            rotation=18,
        )
    ax.plot([x0 + 3, x1 - 3], [rail_y, rail_y], color="#2a2c31", linewidth=2, zorder=0)
    # flags cascade in global firing order, one row per firing, so adjacent
    # stations can never collide horizontally
    for d, e in enumerate(events[:11]):
        key = e["event"]
        fy = rail_y - 3.8 - d * 3.1
        color = ACTION_COLORS.get(e["action"], MUTED)
        ax.plot(
            [xs[key], xs[key]],
            [rail_y - 1.1, fy + 1.1],
            color=color,
            linewidth=1.0,
            alpha=0.7,
        )
        tag = {
            "deny": "BLOCKED",
            "inject": "inject",
            "continue_": "not done!",
            "allow": "ok",
        }[e["action"]]
        label = f"{e['hook']}: {tag}"
        w = max(9.0, len(label) * 0.78 + 1.6)
        fx = min(max(xs[key], x0 + 1.5 + w / 2), x1 - 1.5 - w / 2)
        ax.add_patch(
            FancyBboxPatch(
                (fx - w / 2, fy - 1.1),
                w,
                2.3,
                boxstyle="round,pad=0.12",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.5 if e["action"] != "allow" else 0.9,
            )
        )
        ax.text(
            fx,
            fy,
            label,
            fontsize=6.0,
            family="monospace",
            color=color,
            ha="center",
            va="center",
        )


def draw_frame(
    snap: Snapshot,
    path: Path,
    title: str,
    note: str = "",
    right: str = "loop",
    loop_labels: dict | None = None,
    registered_hooks: dict | None = None,
    show_content: tuple = (),
    show_legend: bool = True,
):
    """The workhorse: context column left, one live panel right."""
    fig, ax = _new_fig(title, note)
    _draw_context_column(ax, snap, show_legend=show_legend)
    if right == "loop":
        _draw_loop_panel(ax, snap.loop_node, loop_labels)
    elif right == "files":
        _draw_files_panel(ax, snap, show_content=show_content)
    elif right == "rail":
        _draw_rail_panel(ax, snap.events, registered_hooks)
    elif right == "none":
        pass
    _save(fig, path)


# ---- the MCP wire ------------------------------------------------------------


def _msg_summary(direction: str, m: dict) -> str:
    if "method" in m:
        p = m.get("params", {})
        detail = ""
        if m["method"] == "tools/call":
            detail = f" {p['name']}({', '.join(f'{k}={v!r}' for k, v in p.get('arguments', {}).items())})"
        return f'{{"jsonrpc":"2.0","id":{m["id"]},"method":"{m["method"]}"{detail}}}'
    r = m.get("result", {})
    if "serverInfo" in r:
        return f"id:{m['id']} result: serverInfo {r['serverInfo']['name']}, capabilities: tools"
    if "tools" in r:
        names = ", ".join(t["name"] for t in r["tools"])
        return f"id:{m['id']} result: tools [{names}]"
    if "content" in r:
        return f'id:{m["id"]} result: "{_short(r["content"][0]["text"], 40)}"'
    return f"id:{m.get('id')} result"


def draw_wire(
    wire: list[tuple[str, dict]],
    path: Path,
    title: str,
    note: str = "",
    client_tools: list[str] | None = None,
):
    """The literal JSON-RPC traffic between harness and MCP server."""
    fig, ax = _new_fig(title, note)
    # endpoint boxes
    for x, label in ((22, "tinyharness (client)"), (78, "checker server")):
        ax.add_patch(
            FancyBboxPatch(
                (x - 15, 50),
                30,
                5,
                boxstyle="round,pad=0.3",
                facecolor=NODE_FILL,
                edgecolor=NODE_EDGE,
                linewidth=1.8,
            )
        )
        ax.text(
            x,
            52.5,
            label,
            fontsize=10,
            family="monospace",
            color=INK,
            ha="center",
            va="center",
        )
    ax.plot([22, 22], [8, 50], color="#2a2c31", linewidth=1.6)
    ax.plot([78, 78], [8, 50], color="#2a2c31", linewidth=1.6)
    ax.text(50, 47.5, "stdio pipe", fontsize=7.5, color=MUTED, ha="center")

    y = 43.0
    for direction, m in wire[-9:]:
        is_req = direction == "->"
        color = BLUE if is_req else GREEN
        x_from, x_to = (22, 78) if is_req else (78, 22)
        ax.add_patch(
            FancyArrowPatch(
                (x_from, y),
                (x_to, y),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=1.3,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (28, y + 0.35),
                44,
                2.6,
                boxstyle="round,pad=0.15",
                facecolor=PANEL,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(
            50,
            y + 1.65,
            _short(_msg_summary(direction, m), 62),
            fontsize=6.6,
            family="monospace",
            color=color,
            ha="center",
            va="center",
        )
        y -= 4.4
    if client_tools:
        ax.text(4, 44, "tool belt", fontsize=8, color=MUTED, fontweight="bold")
        ty = 41.5
        for t in client_tools:
            mcp = t.startswith("mcp__")
            ax.text(
                4,
                ty,
                t,
                fontsize=6.6,
                family="monospace",
                color=NEW_COLOR if mcp else INK,
                fontweight="bold" if mcp else "normal",
            )
            ty -= 2.2
    _save(fig, path)


# ---- session lanes (the long-running signature) -----------------------------


def draw_session_lanes(
    sessions: list[dict],
    path: Path,
    title: str,
    note: str = "",
    features: list[dict] | None = None,
    commits: list | None = None,
    disk_files: list[str] | None = None,
):
    """One lane per session; lanes touch ONLY through the disk strip below.
    sessions: {label, state: done|active|dead|future, blocks, note}."""
    fig, ax = _new_fig(title, note)
    state_style = {
        "done": (GREEN, "-"),
        "active": (NEW_COLOR, "-"),
        "dead": (PINK, "-"),
        "future": ("#3a3d44", "--"),
    }
    n = len(sessions)
    lane_h = min(7.5, 30 / max(n, 1))
    y = 52.0
    for s in sessions:
        color, ls = state_style[s["state"]]
        y -= lane_h + 1.2
        ax.add_patch(
            FancyBboxPatch(
                (14, y),
                74,
                lane_h,
                boxstyle="round,pad=0.25",
                facecolor=PANEL if s["state"] != "future" else PAPER,
                edgecolor=color,
                linewidth=1.8,
                linestyle=ls,
            )
        )
        ax.text(
            4,
            y + lane_h / 2,
            s["label"],
            fontsize=9.5,
            family="monospace",
            color=color,
            va="center",
            fontweight="bold",
        )
        # context blocks inside the lane: a fresh, separate column each session
        for b in range(min(s.get("blocks", 0), 20)):
            ax.add_patch(
                FancyBboxPatch(
                    (16 + b * 2.6, y + lane_h / 2 - 1.0),
                    2.1,
                    2.0,
                    boxstyle="round,pad=0.05",
                    facecolor=NODE_FILL,
                    edgecolor=color,
                    linewidth=0.8,
                )
            )
        if s.get("note"):
            ax.text(
                87,
                y + lane_h / 2,
                s["note"],
                fontsize=7.2,
                color=color,
                va="center",
                ha="right",
                family="monospace",
            )
        if s["state"] == "dead":
            ax.text(
                52,
                y + lane_h / 2,
                "✗ KILLED",
                fontsize=11,
                color=PINK,
                ha="center",
                va="center",
                fontweight="bold",
            )
    # the disk strip: the only thing the lanes share
    ax.add_patch(
        FancyBboxPatch(
            (14, 4.5),
            74,
            y - 7.0,
            boxstyle="round,pad=0.25",
            facecolor="#14161a",
            edgecolor=MUTED,
            linewidth=1.2,
        )
    )
    ax.text(
        16, y - 4.6, "the disk (shared)", fontsize=8, color=MUTED, fontweight="bold"
    )
    # one row inside the strip for each artifact family: checkboxes left,
    # commit dots right, filenames along the bottom, no overlap
    if features:
        _feature_boxes(ax, features, 17, y - 8.4, w=1.9, per_row=12, gap=0.45)
    if commits:
        _commit_dots(ax, commits, 52, y - 7.5)
    if disk_files:
        ax.text(
            17,
            5.7,
            "  ".join(disk_files[:4]),
            fontsize=6.4,
            family="monospace",
            color=INK,
        )
    _save(fig, path)


# ---- diff + verdict (the verifier firewall) ---------------------------------


def draw_diff_verdict(
    diff_text: str,
    path: Path,
    title: str,
    verdict: str = "",
    evidence: str = "",
    note: str = "",
):
    fig, ax = _new_fig(title, note)
    lines = diff_text.splitlines()[:22]
    ax.add_patch(
        FancyBboxPatch(
            (4, 6),
            58,
            47,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=1.0,
        )
    )
    y = 50.5
    for line in lines:
        color = (
            GREEN if line.startswith("+") else PINK if line.startswith("-") else MUTED
        )
        ax.text(6, y, _short(line, 56), fontsize=6.6, family="monospace", color=color)
        y -= 2.0
    if verdict:
        vcolor = GREEN if verdict.upper().startswith("PASS") else PINK
        ax.add_patch(
            FancyBboxPatch(
                (68, 38),
                26,
                10,
                boxstyle="round,pad=0.4",
                facecolor=PAPER,
                edgecolor=vcolor,
                linewidth=3.0,
            )
        )
        ax.text(
            81,
            43,
            verdict.upper(),
            fontsize=17,
            color=vcolor,
            ha="center",
            va="center",
            fontweight="bold",
            family="monospace",
        )
    if evidence:
        y = 32
        ax.text(68, y, "evidence (rerun, not read):", fontsize=7.5, color=MUTED)
        for ln in textwrap.wrap(evidence, 34)[:8]:
            y -= 2.4
            ax.text(68, y, ln, fontsize=7.0, family="monospace", color=INK)
    _save(fig, path)


# ---- token ledger + burn-down ------------------------------------------------


def draw_token_ledger(
    series: list[tuple[str, int]],
    budget: int,
    path: Path,
    title: str,
    note: str = "",
    mark: dict | None = None,
):
    """Bars of real context size per event; the compaction cliff is the shot."""
    fig, ax = _new_fig(title, note)
    n = len(series)
    if not n:
        _save(fig, path)
        return
    w = min(6.0, 84 / n)
    top = max(max(t for _, t in series), budget) * 1.12
    ax.plot(
        [6, 6 + n * w + 2],
        [8 + 40 * budget / top] * 2,
        color=PINK,
        linewidth=1.4,
        linestyle="--",
    )
    ax.text(
        6 + n * w + 3,
        8 + 40 * budget / top,
        "budget",
        fontsize=7.5,
        color=PINK,
        va="center",
    )
    for i, (label, tok) in enumerate(series):
        x = 6 + i * w
        h = 40 * tok / top
        hot = mark and mark.get(i)
        ax.add_patch(
            FancyBboxPatch(
                (x, 8),
                w * 0.75,
                max(h, 0.4),
                boxstyle="round,pad=0.05",
                facecolor="#3a2f14" if hot else NODE_FILL,
                edgecolor=NEW_COLOR if hot else NODE_EDGE,
                linewidth=1.6 if hot else 1.0,
            )
        )
        if hot:
            ax.text(
                x + w * 0.38,
                8 + h + 1.8,
                mark[i],
                fontsize=7.2,
                color=NEW_COLOR,
                ha="center",
                fontweight="bold",
            )
        if n <= 20:
            ax.text(
                x + w * 0.38,
                6.4,
                label,
                fontsize=5.8,
                family="monospace",
                color=MUTED,
                ha="center",
                rotation=30,
            )
    _save(fig, path)


def draw_burndown(
    history: list[tuple[str, int]],
    total: int,
    path: Path,
    title: str,
    note: str = "",
):
    """Features passing after each session -- real counts from feature_list.json."""
    fig, ax = _new_fig(title, note)
    n = len(history)
    for i, (label, done) in enumerate(history):
        x = 10 + i * (80 / max(n, 1))
        h = 38 * done / total
        ax.add_patch(
            FancyBboxPatch(
                (x, 10),
                9,
                max(h, 0.4),
                boxstyle="round,pad=0.1",
                facecolor="#1e3327",
                edgecolor=GREEN,
                linewidth=1.6,
            )
        )
        ax.text(
            x + 4.5,
            10 + h + 1.6,
            f"{done}/{total}",
            fontsize=8.5,
            color=GREEN,
            ha="center",
            family="monospace",
        )
        ax.text(
            x + 4.5, 7.6, label, fontsize=8, color=INK, ha="center", family="monospace"
        )
    ax.plot([8, 94], [48, 48], color=MUTED, linewidth=1, linestyle="--")
    ax.text(94, 49.2, f"all {total}", fontsize=7.5, color=MUTED, ha="right")
    _save(fig, path)


# ---- skill tiers -------------------------------------------------------------


def draw_skill_tiers(
    measured: list[tuple[str, str, int]],
    path: Path,
    title: str,
    note: str = "",
    hot: int = -1,
):
    """The three-tier context-cost pyramid with REAL measured token counts:
    measured = [(tier, example, tokens)] top-down."""
    fig, ax = _new_fig(title, note)
    widths = [30, 54, 78]
    y = 44
    for i, (tier, example, tokens) in enumerate(measured):
        w = widths[min(i, 2)]
        x = 50 - w / 2
        is_hot = i == hot
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 9),
                w,
                9,
                boxstyle="round,pad=0.3",
                facecolor="#3a2f14" if is_hot else PANEL,
                edgecolor=NEW_COLOR if is_hot else VIOLET,
                linewidth=2.2 if is_hot else 1.4,
            )
        )
        ax.text(
            50, y - 3.1, tier, fontsize=10, color=INK, ha="center", fontweight="bold"
        )
        ax.text(
            50,
            -6.1 + y,
            example,
            fontsize=7.2,
            family="monospace",
            color=MUTED,
            ha="center",
        )
        ax.text(
            50 + w / 2 + 2,
            y - 4.5,
            f"{tokens} tok" if tokens >= 0 else "unbounded",
            fontsize=8.5,
            family="monospace",
            color=NEW_COLOR if is_hot else VIOLET,
            va="center",
        )
        y -= 12.5
    _save(fig, path)


# ---- self-check --------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    from fixtures import build_workspace
    from harness import Agent, Harness, ScriptedModel, Turn, builtin_tools

    out = Path(tempfile.mkdtemp(prefix="harnessviz-check-"))
    ws = build_workspace(stage=5)
    tools = builtin_tools(ws)
    model = ScriptedModel(
        [
            Turn(
                text="Reading the plan.", tool="read_file", args={"path": "CLAUDE.md"}
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="Stage 5 is green."),
        ]
    )
    agent = Agent(model, tools, ws, budget=1500)
    h = Harness(ws)
    agent.run("Check the project state.", on_event=lambda e, m: h.snap(e, agent))
    assert len(h.snapshots) >= 5

    draw_frame(h.snapshots[-1], out / "frame_loop.png", "Frame: loop", right="loop")
    draw_frame(h.snapshots[-1], out / "frame_files.png", "Frame: files", right="files")
    draw_frame(
        h.snapshots[-1],
        out / "frame_rail.png",
        "Frame: rail",
        right="rail",
        registered_hooks=agent.hooks.registered(),
    )
    draw_wire(
        [
            ("->", {"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            (
                "<-",
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"serverInfo": {"name": "checker"}},
                },
            ),
        ],
        out / "wire.png",
        "Wire",
        client_tools=tools.names(),
    )
    draw_session_lanes(
        [
            {"label": "S1", "state": "done", "blocks": 8, "note": "F01-F05"},
            {"label": "S2", "state": "active", "blocks": 4, "note": "F06"},
            {"label": "S3", "state": "future", "blocks": 0},
        ],
        out / "lanes.png",
        "Lanes",
        features=h.snapshots[-1].features,
        commits=ws.git_log(),
    )
    draw_diff_verdict(
        "+ def cmd_search():\n-  pass",
        out / "diff.png",
        "Diff",
        verdict="PASS",
        evidence="test_f06_search: ok",
    )
    draw_token_ledger(
        [(s.label[:8], s.total_tokens) for s in h.snapshots],
        1500,
        out / "ledger.png",
        "Ledger",
    )
    draw_burndown([("S1", 5), ("S2", 6)], 12, out / "burn.png", "Burn-down")
    draw_skill_tiers(
        [
            ("index", "one line per skill", 22),
            ("body", "SKILL.md", 380),
            ("spokes", "references/*", -1),
        ],
        out / "tiers.png",
        "Tiers",
    )
    import os

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 9, pngs
    sizes = {p.name: os.path.getsize(p) for p in out.glob("*.png")}
    assert all(s > 20_000 for s in sizes.values()), sizes
    print(f"ok: 9 primitives rendered to {out}")
    ws.cleanup()
