"""Draw the langchain/langgraph deck's true state: every figure is a
photograph of a real run at Beanline Coffee.

Design tokens are ported verbatim from the harness deck's harnessviz.py
(itself ported from ragviz.py / gitviz.py), so all four decks read as one
system.

Flipbook figures use a FIXED canvas (no bbox_inches="tight"): reveal.js
auto-animate tweens between consecutive same-title slides, and a constant
canvas is what makes a growing message column or a lighting-up graph node
read as evolving rather than jumping.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Deck text is never math: real payloads carry "$4.50" and JSON-schema "$defs",
# which must render literally, not as TeX.
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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

ROLE_COLOR = {
    "system": MUTED,
    "human": BLUE,
    "user": BLUE,
    "ai": NODE_EDGE,
    "assistant": NODE_EDGE,
    "tool": GREEN,
}


def _short(s: str, n: int = 36) -> str:
    s = s.splitlines()[0] if s else ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _wrap(s: str, width: int) -> list[str]:
    return textwrap.wrap(s, width) or [""]


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


# ---- shared chrome ------------------------------------------------------------


def _panel_frame(ax, label: str, x0: float, x1: float, y0: float, y1: float):
    """A titled rounded frame -- ported from harnessviz's right-panel border."""
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
    if label:
        ax.text(x0 + 1.2, y1 - 2.0, label, fontsize=9, color=MUTED, fontweight="bold")
    return x0, x1, y0, y1


def _panel_bg(ax, x0: float, x1: float, y0: float, y1: float, label: str = ""):
    """A titled rounded panel -- ported from harnessviz's context-column border."""
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
    if label:
        ax.text(x0, y1 + 1.6, label, fontsize=9, color=MUTED, fontweight="bold")


def _colored_line(ax, x: float, y: float, segments, fontsize: float = 7.2):
    """Draw text pieces in different colors on one baseline (monospace pitch
    is estimated, not measured -- fine for short chip-style highlights)."""
    char_w = fontsize * 0.075
    for text, color in segments:
        if not text:
            continue
        ax.text(
            x, y, text, fontsize=fontsize, family="monospace", color=color, va="center"
        )
        x += len(text) * char_w
    return x


def _strike(s: str) -> str:
    """Cheap strikethrough via a combining overlay -- no extra dependency."""
    return "".join(ch + "̶" for ch in s)


# ---- pipeline (the LCEL chain) ------------------------------------------------


_PIPE_STYLE = {
    "pending": (MUTED, 1.0, "--", 0.55),
    "active": (NEW_COLOR, 2.4, "-", 1.0),
    "done": (NODE_EDGE, 1.6, "-", 1.0),
    "error": (PINK, 2.2, "-", 1.0),
}


def draw_pipeline(stages: list[dict], path: Path, title: str, note: str = ""):
    """Horizontal boxes joined by big '|' glyphs, the LCEL pipe."""
    fig, ax = _new_fig(title, note)
    n = len(stages)
    if not n:
        _save(fig, path)
        return
    x0, x1 = 4.0, 96.0
    slot_w = (x1 - x0) / n
    box_w = slot_w * 0.72
    box_h = 9.0
    y_mid = 30.0
    centers = []
    for i, s in enumerate(stages):
        cx = x0 + slot_w * i + slot_w / 2
        centers.append(cx)
        color, lw, ls, alpha = _PIPE_STYLE.get(
            s.get("state", "pending"), _PIPE_STYLE["pending"]
        )
        ax.add_patch(
            FancyBboxPatch(
                (cx - box_w / 2, y_mid - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.25",
                facecolor=NODE_FILL,
                edgecolor=color,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
            )
        )
        ax.text(
            cx,
            y_mid,
            _short(s["label"], 16),
            fontsize=9,
            family="monospace",
            color=color,
            ha="center",
            va="center",
            fontweight="bold" if s.get("state") == "active" else "normal",
            alpha=alpha,
        )
        if s.get("type_label"):
            ax.text(
                cx,
                y_mid + box_h / 2 + 2.6,
                _short(s["type_label"], 20),
                fontsize=7,
                family="monospace",
                color=MUTED,
                ha="center",
            )
        if s.get("payload"):
            py = y_mid - box_h / 2 - 2.6
            for ln in _wrap(s["payload"], 18)[:3]:
                ax.text(
                    cx,
                    py,
                    ln,
                    fontsize=6.4,
                    family="monospace",
                    color=INK,
                    ha="center",
                    alpha=alpha,
                )
                py -= 1.8
    for i in range(n - 1):
        mx = (centers[i] + centers[i + 1]) / 2
        ax.text(
            mx,
            y_mid,
            "|",
            fontsize=24,
            color=INK,
            ha="center",
            va="center",
            fontweight="bold",
        )
    _save(fig, path)


# ---- message columns -----------------------------------------------------------


def _msg_parts(m):
    """Duck-type a langchain BaseMessage or accept a plain {'role','text'} dict.
    Returns (role, text, tool_call_str_or_None)."""
    if hasattr(m, "type"):
        role = m.type
        content = m.content if isinstance(m.content, str) else str(m.content)
        tool_calls = getattr(m, "tool_calls", None) or []
        if role == "ai" and tool_calls:
            tc = tool_calls[0]
            args = ", ".join(f"{k}={v!r}" for k, v in (tc.get("args") or {}).items())
            return role, content, f"tool_call: {tc.get('name')}({args})"
        return role, content, None
    return m.get("role", "text"), m.get("text", ""), None


def _msg_color(role: str, has_call: bool) -> str:
    if has_call:
        return ORANGE
    return ROLE_COLOR.get(role, INK)


_LEGEND = [
    ("system", MUTED),
    ("human", BLUE),
    ("ai", NODE_EDGE),
    ("tool call", ORANGE),
    ("tool", GREEN),
]


def _draw_message_column(
    ax, msgs, x0, x1, y0, y1, new=(), show_legend=False, header=None, verdict=None
):
    """Vertical column of role-colored blocks, height proportional to text
    length with a readable floor and squeeze-to-fit (port of harnessviz's
    _draw_context_column algorithm, using char count in place of tokens)."""
    if header is not None:
        hcolor, mark = INK, ""
        if verdict == "ok":
            hcolor, mark = GREEN, "  ✓"
        elif verdict == "bad":
            hcolor, mark = PINK, "  ✗"
        ax.text(
            x0, y1 + 1.8, header + mark, fontsize=9, color=hcolor, fontweight="bold"
        )

    height = y1 - y0
    parts = [_msg_parts(m) for m in msgs]
    if not parts:
        return
    lens = [max(len(text) + (len(call) if call else 0), 10) for _, text, call in parts]
    scale = height / max(sum(lens), 1)
    floor = 4.8
    heights = [max(l * scale, floor) for l in lens]
    if sum(heights) > height:
        squeeze = height / sum(heights)
        heights = [h * squeeze for h in heights]

    y = y1
    for i, (role, text, call) in enumerate(parts):
        color = _msg_color(role, call is not None)
        h = heights[i]
        y -= h
        is_new = i in new
        ax.add_patch(
            FancyBboxPatch(
                (x0, y + 0.15),
                x1 - x0,
                max(h - 0.3, 0.5),
                boxstyle="round,pad=0.06",
                facecolor=NODE_FILL,
                edgecolor=NEW_COLOR if is_new else color,
                linewidth=2.0 if is_new else 1.1,
            )
        )
        box_top, box_bottom = y + h - 0.15, y + 0.15
        label_h, line_h, pad_top = 1.3, 1.5, 0.7
        if h >= pad_top + label_h:
            ax.text(
                x0 + 0.8,
                box_top - pad_top,
                role,
                fontsize=6.6,
                family="monospace",
                color=color,
                fontweight="bold",
                va="top",
            )
            body = (call if call else text).replace("\n", " ").strip()
            content_top = box_top - pad_top - label_h
            max_lines = max(0, int((content_top - box_bottom) / line_h))
            if body and max_lines:
                width_chars = max(int((x1 - x0) * 2.3), 12)
                wrapped = _wrap(body, width_chars)
                shown = wrapped[:max_lines]
                if len(wrapped) > max_lines and shown:
                    shown[-1] = _short(" ".join(wrapped[max_lines - 1 :]), width_chars)
                ty = content_top
                for ln in shown:
                    ax.text(
                        x0 + 0.8,
                        ty,
                        ln,
                        fontsize=6.0,
                        family="monospace",
                        color=color,
                        va="top",
                    )
                    ty -= line_h

    if show_legend:
        lx = x0
        ly = max(y0 - 3.0, 1.5)
        for name, color in _LEGEND:
            ax.add_patch(
                FancyBboxPatch(
                    (lx, ly),
                    0.9,
                    0.9,
                    boxstyle="round,pad=0.03",
                    facecolor=color,
                    edgecolor="none",
                )
            )
            ax.text(lx + 1.3, ly + 0.45, name, fontsize=5.8, color=MUTED, va="center")
            lx += 1.7 + len(name) * 0.62


def _text_card(ax, text: str, x0, x1, y0, y1, title: str = ""):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=NODE_EDGE,
            linewidth=1.2,
        )
    )
    if title:
        ax.text(x0 + 1.5, y1 - 2.4, title, fontsize=9, color=MUTED, fontweight="bold")
    y = y1 - (5.0 if title else 2.5)
    width_chars = max(int((x1 - x0) * 1.55), 10)
    for line in text.splitlines():
        for ln in _wrap(line, width_chars):
            if y < y0 + 1.5:
                return
            ax.text(x0 + 1.5, y, ln, fontsize=7.6, family="monospace", color=INK)
            y -= 2.1


def draw_messages(
    msgs,
    path: Path,
    title: str,
    note: str = "",
    new=(),
    right_text: str | None = None,
    right_title: str = "",
):
    fig, ax = _new_fig(title, note)
    if right_text is not None:
        lx0, lx1, y0, y1 = 4.0, 46.0, 10.0, 52.0
        _panel_bg(ax, lx0, lx1, y0, y1)
        _draw_message_column(
            ax, msgs, lx0 + 1.0, lx1 - 1.0, y0 + 1.0, y1 - 1.0, new=new
        )
        rx0, rx1 = 52.0, 97.0
        _text_card(ax, right_text, rx0, rx1, y0, y1, right_title)
    else:
        lx0, lx1, y0, y1 = 10.0, 70.0, 12.0, 51.0
        _panel_bg(ax, lx0, lx1, y0, y1)
        _draw_message_column(
            ax,
            msgs,
            lx0 + 1.0,
            lx1 - 1.0,
            y0 + 1.0,
            y1 - 1.0,
            new=new,
            show_legend=True,
        )
    _save(fig, path)


def draw_dual_messages(
    left_msgs,
    right_msgs,
    path: Path,
    title: str,
    left_label: str,
    right_label: str,
    note: str = "",
    left_verdict: str | None = None,
    right_verdict: str | None = None,
):
    fig, ax = _new_fig(title, note)
    lx0, lx1 = 4.0, 47.0
    rx0, rx1 = 53.0, 96.0
    y0, y1 = 10.0, 49.0
    _panel_bg(ax, lx0, lx1, y0, y1)
    _panel_bg(ax, rx0, rx1, y0, y1)
    _draw_message_column(
        ax,
        left_msgs,
        lx0 + 1.0,
        lx1 - 1.0,
        y0 + 1.0,
        y1 - 1.0,
        header=left_label,
        verdict=left_verdict,
    )
    _draw_message_column(
        ax,
        right_msgs,
        rx0 + 1.0,
        rx1 - 1.0,
        y0 + 1.0,
        y1 - 1.0,
        header=right_label,
        verdict=right_verdict,
    )
    _save(fig, path)


# ---- the graph (langgraph's true shape) ----------------------------------------


def _state_value_segments(row: dict):
    """See langviz.py module notes / build report: with only `value` and
    `delta` on a row (no separate prior-value field), the "old" slot in
    "old (+) delta -> value" is dropped and, for overwrite, `delta` stands
    in for the discarded old value (struck through)."""
    value = _short(str(row.get("value", "")), 22)
    delta = row.get("delta")
    reducer = row.get("reducer")
    if not delta:
        return [(value, INK)]
    delta_s = _short(str(delta), 18)
    if reducer in ("add", "add_messages"):
        return [("⊕ ", VIOLET), (delta_s, INK), ("  ->  ", MUTED), (value, INK)]
    if reducer == "overwrite":
        return [(_strike(delta_s), MUTED), ("  ←  ", MUTED), (value, INK)]
    return [(value, INK)]


def _draw_state_rows(ax, rows: list[dict], x0, x1, y0, y1, header: str = "state"):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=MUTED,
            linewidth=1.0,
        )
    )
    ax.text(x0 + 1.2, y1 - 2.0, header, fontsize=9, color=MUTED, fontweight="bold")
    n = len(rows)
    if not n:
        return
    row_h = min(9.0, (y1 - y0 - 4.0) / n)
    y = y1 - 4.5
    for r in rows:
        changed = r.get("changed")
        reducer = r.get("reducer")
        if changed:
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + 1.0, y - row_h + 1.2),
                    x1 - x0 - 2.0,
                    row_h - 1.4,
                    boxstyle="round,pad=0.1",
                    facecolor="none",
                    edgecolor=NEW_COLOR,
                    linewidth=1.6,
                )
            )
        tag = f"[{reducer}]" if reducer else ""
        ax.text(
            x0 + 2.0,
            y - 1.7,
            f"{r['channel']}  {tag}",
            fontsize=7.4,
            family="monospace",
            color=INK,
            fontweight="bold" if changed else "normal",
        )
        _colored_line(ax, x0 + 2.0, y - 3.8, _state_value_segments(r), fontsize=6.8)
        y -= row_h


def draw_state(rows: list[dict], path: Path, title: str, note: str = ""):
    fig, ax = _new_fig(title, note)
    _draw_state_rows(ax, rows, 4.0, 96.0, 6.0, 52.0)
    _save(fig, path)


def draw_graph(
    drawable,
    positions: dict,
    path: Path,
    title: str,
    active=None,
    visited=(),
    taken_edges=(),
    edge_label=None,
    state_rows=None,
    note: str = "",
):
    """drawable is the real object from compiled_graph.get_graph(); positions
    is caller-owned (0..100 x 0..60), we draw exactly the nodes it names."""
    fig, ax = _new_fig(title, note)
    if state_rows is not None:
        _draw_state_rows(ax, state_rows, 66.0, 98.0, 6.0, 52.0)

    nodes = list(drawable.nodes)
    edges = [(e.source, e.target, e.conditional) for e in drawable.edges]
    taken = set(taken_edges)

    for src, dst, cond in edges:
        if src not in positions or dst not in positions:
            continue
        p1, p2 = positions[src], positions[dst]
        is_taken = (src, dst) in taken
        color = GREEN if is_taken else MUTED
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                connectionstyle="arc3,rad=0.15",
                arrowstyle="-|>",
                mutation_scale=13,
                color=color,
                linewidth=2.6 if is_taken else 1.2,
                linestyle="--" if cond else "-",
                shrinkA=16,
                shrinkB=16,
                zorder=1,
            )
        )
        if edge_label and edge_label[0] == src and edge_label[1] == dst:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 1.8
            lbl = _short(str(edge_label[2]), 14)
            w = len(lbl) * 0.9 + 1.6
            ax.add_patch(
                FancyBboxPatch(
                    (mx - w / 2, my - 1.1),
                    w,
                    2.2,
                    boxstyle="round,pad=0.12",
                    facecolor="#3a2f14",
                    edgecolor=NEW_COLOR,
                    linewidth=1.4,
                    zorder=2,
                )
            )
            ax.text(
                mx,
                my,
                lbl,
                fontsize=6.6,
                family="monospace",
                color=NEW_COLOR,
                ha="center",
                va="center",
                zorder=3,
            )

    for n in nodes:
        if n not in positions:
            continue
        x, y = positions[n]
        if n in ("__start__", "__end__"):
            ax.add_patch(
                Circle(
                    (x, y),
                    1.0,
                    facecolor=NODE_FILL,
                    edgecolor=MUTED,
                    linewidth=1.4,
                    zorder=2,
                )
            )
            ax.text(
                x,
                y - 2.6,
                "START" if n == "__start__" else "END",
                fontsize=6.2,
                family="monospace",
                color=MUTED,
                ha="center",
                zorder=2,
            )
            continue
        is_active = n == active
        is_visited = n in visited
        w = max(10.0, len(n) * 0.78 + 3.4)
        h = 5.4
        if is_active:
            ax.add_patch(
                FancyBboxPatch(
                    (x - w / 2 - 0.9, y - h / 2 - 0.9),
                    w + 1.8,
                    h + 1.8,
                    boxstyle="round,pad=0.3",
                    facecolor=NEW_COLOR,
                    edgecolor="none",
                    alpha=0.18,
                    zorder=1,
                )
            )
        edgecolor = NEW_COLOR if is_active else (GREEN if is_visited else NODE_EDGE)
        lw = 2.6 if is_active else (2.0 if is_visited else 1.4)
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.25",
                facecolor=NODE_FILL,
                edgecolor=edgecolor,
                linewidth=lw,
                zorder=2,
            )
        )
        ax.text(
            x,
            y,
            n,
            fontsize=8.2,
            family="monospace",
            color=edgecolor,
            ha="center",
            va="center",
            fontweight="bold" if is_active else "normal",
            zorder=3,
        )
    _save(fig, path)


# ---- streaming ------------------------------------------------------------------


def draw_stream(chunks: list[str], upto: int, path: Path, title: str, note: str = ""):
    fig, ax = _new_fig(title, note)
    n = len(chunks)
    tx0, tx1, ty = 6.0, 94.0, 46.0
    ax.plot([tx0, tx1], [ty, ty], color="#2a2c31", linewidth=2, zorder=0)
    if n:
        w = (tx1 - tx0) / n
        for i in range(n):
            x = tx0 + i * w + w / 2
            arrived = i < upto
            ax.add_patch(
                FancyBboxPatch(
                    (x - w * 0.35, ty - 1.3),
                    w * 0.7,
                    2.6,
                    boxstyle="round,pad=0.05",
                    facecolor=NODE_FILL if arrived else PAPER,
                    edgecolor=NODE_EDGE if arrived else MUTED,
                    linewidth=1.2 if arrived else 0.8,
                    linestyle="-" if arrived else "--",
                    alpha=1.0 if arrived else 0.4,
                )
            )
        if upto >= 1:
            x0c = tx0 + w / 2
            ax.annotate(
                "first token",
                xy=(x0c, ty + 1.4),
                xytext=(x0c, ty + 6.5),
                fontsize=7.5,
                color=NEW_COLOR,
                ha="center",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=NEW_COLOR, lw=1.4),
            )

    ax.add_patch(
        FancyBboxPatch(
            (6, 10),
            88,
            26,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=NODE_EDGE,
            linewidth=1.2,
        )
    )
    ax.text(8, 33.4, "assembled so far", fontsize=8, color=MUTED, fontweight="bold")
    assembled = "".join(chunks[:upto])
    cursor = "▌" if upto < n else ""
    y = 30.4
    for ln in _wrap(assembled + cursor, 74)[:9]:
        ax.text(8, y, ln, fontsize=9.5, family="monospace", color=INK)
        y -= 2.7
    ax.text(
        6,
        6.0,
        f"{upto}/{n} chunks arrived",
        fontsize=7.5,
        color=MUTED,
        family="monospace",
    )
    _save(fig, path)


# ---- thread lanes (langgraph's checkpointer) ------------------------------------


_LANE_STYLE = {
    "active": (NEW_COLOR, "-"),
    "done": (GREEN, "-"),
    "dimmed": (MUTED, "--"),
}


def draw_thread_lanes(
    threads: list[dict],
    path: Path,
    title: str,
    note: str = "",
    store_label: str = "one checkpointer -- one lane per thread_id",
):
    fig, ax = _new_fig(title, note)
    n = max(len(threads), 1)
    lane_x0, lane_w = 16.0, 76.0
    band = 52.0 - 15.0  # reserve the bottom band for the shared store strip
    lane_h = min(8.5, band / n * 0.62)
    gap = min(4.5, band / n * 0.38)
    y = 52.0
    for t in threads:
        color, ls = _LANE_STYLE[t["state"]]
        y -= lane_h + gap
        ax.add_patch(
            FancyBboxPatch(
                (lane_x0, y),
                lane_w,
                lane_h,
                boxstyle="round,pad=0.25",
                facecolor=PANEL if t["state"] != "dimmed" else PAPER,
                edgecolor=color,
                linewidth=1.8,
                linestyle=ls,
            )
        )
        ax.text(
            4,
            y + lane_h / 2,
            t["label"],
            fontsize=9.5,
            family="monospace",
            color=color,
            va="center",
            fontweight="bold",
        )

        cps = t.get("checkpoints", [])
        m = len(cps)
        cy = y + lane_h * 0.66
        inner0, inner1 = lane_x0 + 5, lane_x0 + lane_w - 5
        dot_xs = []
        for i, cp in enumerate(cps):
            cx = inner0 + (i + 0.5) * (inner1 - inner0) / max(m, 1)
            dot_xs.append(cx)
            ax.add_patch(
                Circle(
                    (cx, cy),
                    0.8,
                    facecolor=NODE_FILL,
                    edgecolor=color,
                    linewidth=1.6,
                    zorder=2,
                )
            )
            ax.text(
                cx,
                cy - 2.0,
                cp.get("step", ""),
                fontsize=6.2,
                family="monospace",
                color=INK,
                ha="center",
                va="top",
            )
            if cp.get("hint") and lane_h >= 6.5:
                ax.text(
                    cx,
                    cy - 3.5,
                    _short(cp["hint"], 16),
                    fontsize=5.4,
                    color=MUTED,
                    ha="center",
                    va="top",
                )

        resume_from = t.get("resume_from")
        if resume_from is not None and 0 <= resume_from < len(dot_xs) and gap >= 2.5:
            rx = dot_xs[resume_from]
            ax.add_patch(
                FancyArrowPatch(
                    (rx, y - gap + 0.4),
                    (rx, y - 0.3),
                    connectionstyle="arc3,rad=0.3",
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=NEW_COLOR,
                    linewidth=1.8,
                    zorder=1,
                )
            )
            ax.text(
                rx,
                y - gap + 0.2,
                "resume",
                fontsize=6.2,
                color=NEW_COLOR,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    store_y1 = y - 2.0
    ax.add_patch(
        FancyBboxPatch(
            (lane_x0, 4.5),
            lane_w,
            max(store_y1 - 4.5, 2.0),
            boxstyle="round,pad=0.25",
            facecolor="#14161a",
            edgecolor=MUTED,
            linewidth=1.2,
        )
    )
    ax.text(
        lane_x0 + 2,
        store_y1 - 2.2,
        store_label,
        fontsize=7.5,
        color=MUTED,
        family="monospace",
    )
    _save(fig, path)


# ---- the tool wire (model <-> python fn) ----------------------------------------


def draw_tool_wire(
    phase: int,
    path: Path,
    title: str,
    call: dict | None = None,
    result: dict | None = None,
    world: list | None = None,
    world_title: str = "stock room",
    note: str = "",
    schema: dict | None = None,
):
    fig, ax = _new_fig(title, note)
    have_world = world is not None and phase >= 3
    left_x = 20.0
    right_x = 58.0 if have_world else 76.0
    top_y, bottom_y = 50.0, 10.0
    fn_active = phase >= 3

    for x, label, active in (
        (left_x, "model", False),
        (right_x, "python fn", fn_active),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x - 13, top_y),
                26,
                5,
                boxstyle="round,pad=0.3",
                facecolor="#3a2f14" if active else NODE_FILL,
                edgecolor=NEW_COLOR if active else NODE_EDGE,
                linewidth=2.2 if active else 1.8,
            )
        )
        ax.text(
            x,
            top_y + 2.5,
            label,
            fontsize=10,
            family="monospace",
            color=NEW_COLOR if active else INK,
            ha="center",
            va="center",
            fontweight="bold" if active else "normal",
        )
    ax.plot([left_x, left_x], [bottom_y, top_y], color="#2a2c31", linewidth=1.6)
    ax.plot([right_x, right_x], [bottom_y, top_y], color="#2a2c31", linewidth=1.6)

    sy = top_y - 6.0
    if schema:
        ax.text(left_x - 13, sy, "tools bound to model:", fontsize=7.2, color=MUTED)
        sy -= 2.4
        for name, desc in schema.items():
            ax.text(
                left_x - 13,
                sy,
                f"{name}: {_short(desc, 34)}",
                fontsize=6.6,
                family="monospace",
                color=INK,
            )
            sy -= 2.0

    if phase >= 2 and call:
        y_req = 30.0
        call_str = f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call.get('args', {}).items())})"
        ax.add_patch(
            FancyArrowPatch(
                (left_x, y_req),
                (right_x, y_req),
                arrowstyle="-|>",
                mutation_scale=13,
                color=BLUE,
                linewidth=1.6,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                ((left_x + right_x) / 2 - 20, y_req + 0.6),
                40,
                2.6,
                boxstyle="round,pad=0.15",
                facecolor=PANEL,
                edgecolor=BLUE,
                linewidth=1.0,
            )
        )
        ax.text(
            (left_x + right_x) / 2,
            y_req + 1.9,
            _short(call_str, 46),
            fontsize=6.6,
            family="monospace",
            color=BLUE,
            ha="center",
            va="center",
        )
        ax.text(
            (left_x + right_x) / 2,
            y_req - 1.4,
            f"id: {call.get('id', '')}",
            fontsize=6.0,
            family="monospace",
            color=NEW_COLOR,
            ha="center",
        )

    if have_world:
        wx0, wx1, wy0, wy1 = _panel_frame(
            ax, world_title, right_x + 12, 98.0, bottom_y, top_y
        )
        wy = wy1 - 5.0
        for line in world:
            hot = isinstance(line, str) and line.startswith("*")
            ax.text(
                wx0 + 1.5,
                wy,
                line.lstrip("*").strip(),
                fontsize=7.4,
                family="monospace",
                color=NEW_COLOR if hot else INK,
                fontweight="bold" if hot else "normal",
            )
            wy -= 2.6

    if phase >= 4 and result:
        y_res = 17.0
        res_str = f"ToolMessage: {_short(str(result.get('content', '')), 40)}"
        ax.add_patch(
            FancyArrowPatch(
                (right_x, y_res),
                (left_x, y_res),
                arrowstyle="-|>",
                mutation_scale=13,
                color=GREEN,
                linewidth=1.6,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                ((left_x + right_x) / 2 - 20, y_res + 0.6),
                40,
                2.6,
                boxstyle="round,pad=0.15",
                facecolor=PANEL,
                edgecolor=GREEN,
                linewidth=1.0,
            )
        )
        ax.text(
            (left_x + right_x) / 2,
            y_res + 1.9,
            _short(res_str, 46),
            fontsize=6.6,
            family="monospace",
            color=GREEN,
            ha="center",
            va="center",
        )
        ax.text(
            (left_x + right_x) / 2,
            y_res - 1.4,
            f"id: {result.get('tool_call_id', '')}",
            fontsize=6.0,
            family="monospace",
            color=NEW_COLOR,
            ha="center",
        )
    _save(fig, path)


# ---- scorecard --------------------------------------------------------------------


def draw_scorecard(
    rows: list[dict],
    path: Path,
    title: str,
    note: str = "",
    columns: list | None = None,
):
    fig, ax = _new_fig(title, note)
    n = len(rows)
    ncells = max((len(r.get("cells", [])) for r in rows), default=0)
    x_label, x_cells0, x_verdict = 4.0, 26.0, 88.0
    cell_w = (x_verdict - 2 - x_cells0) / max(ncells, 1)
    row_h = min(4.4, 40 / max(n, 1))
    y = 48.0
    if columns:
        for j, c in enumerate(columns):
            ax.text(
                x_cells0 + j * cell_w + cell_w / 2,
                y + 2.2,
                c,
                fontsize=7,
                color=MUTED,
                ha="center",
                fontweight="bold",
            )
        y -= 2.8
    ax.plot([x_label, x_verdict + 8], [y + 1.4, y + 1.4], color="#2a2c31", linewidth=1)
    for r in rows:
        ax.text(
            x_label,
            y,
            _short(r["label"], 26),
            fontsize=8.0,
            family="monospace",
            color=INK,
        )
        for j, c in enumerate(r.get("cells", [])):
            ax.text(
                x_cells0 + j * cell_w + cell_w / 2,
                y,
                _short(str(c), 14),
                fontsize=7.4,
                family="monospace",
                color=MUTED,
                ha="center",
            )
        v = r.get("verdict")
        if v:
            vcolor = GREEN if v == "pass" else PINK
            ax.add_patch(
                FancyBboxPatch(
                    (x_verdict, y - 1.5),
                    9.5,
                    2.8,
                    boxstyle="round,pad=0.15",
                    facecolor=PANEL,
                    edgecolor=vcolor,
                    linewidth=1.4,
                )
            )
            ax.text(
                x_verdict + 4.75,
                y - 0.1,
                "PASS" if v == "pass" else "FAIL",
                fontsize=7,
                family="monospace",
                color=vcolor,
                ha="center",
                va="center",
                fontweight="bold",
            )
        y -= row_h
    _save(fig, path)


# ---- one big card -------------------------------------------------------------------


_TONE_COLOR = {"neutral": MUTED, "bad": PINK, "good": GREEN}


def draw_card(
    text: str,
    path: Path,
    title: str,
    note: str = "",
    tone: str = "neutral",
    subtitle: str = "",
):
    fig, ax = _new_fig(title, note)
    color = _TONE_COLOR.get(tone, MUTED)
    y1 = 54.0
    if subtitle:
        ax.text(4, y1, subtitle, fontsize=8.5, color=MUTED)
        y1 -= 3.0
    ax.add_patch(
        FancyBboxPatch(
            (4, 8),
            92,
            y1 - 8,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=color,
            linewidth=1.8 if tone != "neutral" else 1.0,
        )
    )
    y = y1 - 3.4
    for para in text.split("\n"):
        for ln in _wrap(para, 76):
            if y < 9.5:
                break
            ax.text(7, y, ln, fontsize=9.5, family="monospace", color=INK)
            y -= 2.6
    _save(fig, path)


# ---- eval-specific: test set, judge panel, span timeline -----------------------


def draw_test_set(cases: list[dict], path: Path, title: str, note: str = ""):
    """A fixed test-set table: id, input, expected -- the eval's ground
    truth, drawn once so every later figure can point back at "case 3".
    Each case: {"id", "input", "expected", "verdict" (optional "pass"/"fail")}."""
    fig, ax = _new_fig(title, note)
    n = len(cases)
    x_id, x_in, x_exp, x_verdict = 4.0, 12.0, 52.0, 88.0
    row_h = min(4.6, 40 / max(n, 1))
    y = 48.0
    for label, x in (("id", x_id), ("input", x_in), ("expected", x_exp)):
        ax.text(
            x, y + 2.2, label, fontsize=7, color=MUTED, ha="left", fontweight="bold"
        )
    y -= 2.8
    ax.plot([x_id, x_verdict + 8], [y + 1.4, y + 1.4], color="#2a2c31", linewidth=1)
    for c in cases:
        ax.text(
            x_id,
            y,
            str(c["id"]),
            fontsize=8.0,
            family="monospace",
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x_in,
            y,
            _short(c["input"], 42),
            fontsize=7.4,
            family="monospace",
            color=BLUE,
        )
        ax.text(
            x_exp,
            y,
            _short(c["expected"], 38),
            fontsize=7.4,
            family="monospace",
            color=GREEN,
        )
        v = c.get("verdict")
        if v:
            vcolor = GREEN if v == "pass" else PINK
            ax.add_patch(
                FancyBboxPatch(
                    (x_verdict, y - 1.5),
                    9.5,
                    2.8,
                    boxstyle="round,pad=0.15",
                    facecolor=PANEL,
                    edgecolor=vcolor,
                    linewidth=1.4,
                )
            )
            ax.text(
                x_verdict + 4.75,
                y - 0.1,
                "PASS" if v == "pass" else "FAIL",
                fontsize=7,
                family="monospace",
                color=vcolor,
                ha="center",
                va="center",
                fontweight="bold",
            )
        y -= row_h
    _save(fig, path)


def draw_judge_panel(
    shown_text: str,
    verdict_label: str,
    rationale: str,
    path: Path,
    title: str,
    note: str = "",
    tone: str = "neutral",
    flag: str = "",
):
    """The LLM-as-judge pattern: shown_text is exactly what the judge model
    saw (candidate answer + rubric + any fed-in premise); verdict_label and
    rationale are the scripted judge's own reply. flag, if set, names a
    fed-in premise the judge rubber-stamped instead of checking -- the
    deterministic illustration of that failure mode."""
    fig, ax = _new_fig(title, note)
    lx0, lx1, y0, y1 = 4.0, 47.0, 8.0, 52.0
    rx0, rx1 = 53.0, 96.0
    _text_card(ax, shown_text, lx0, lx1, y0, y1, title="shown to the judge")

    color = _TONE_COLOR.get(tone, MUTED)
    ax.add_patch(
        FancyBboxPatch(
            (rx0, y0),
            rx1 - rx0,
            y1 - y0,
            boxstyle="round,pad=0.3",
            facecolor=PANEL,
            edgecolor=color,
            linewidth=1.8 if tone != "neutral" else 1.2,
        )
    )
    ax.text(
        rx0 + 1.5, y1 - 2.4, "judge verdict", fontsize=9, color=MUTED, fontweight="bold"
    )
    ax.add_patch(
        FancyBboxPatch(
            (rx0 + 1.5, y1 - 8.5),
            24,
            4.0,
            boxstyle="round,pad=0.15",
            facecolor=NODE_FILL,
            edgecolor=color,
            linewidth=1.6,
        )
    )
    ax.text(
        rx0 + 13.5,
        y1 - 6.5,
        verdict_label,
        fontsize=9.5,
        family="monospace",
        color=color,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ry = y1 - 12.0
    width_chars = max(int((rx1 - rx0) * 1.55), 10)
    floor = y0 + (7.0 if flag else 2.0)
    for line in rationale.splitlines():
        for ln in _wrap(line, width_chars):
            if ry < floor:
                break
            ax.text(rx0 + 1.5, ry, ln, fontsize=7.6, family="monospace", color=INK)
            ry -= 2.1
    if flag:
        ax.add_patch(
            FancyBboxPatch(
                (rx0 + 1.0, y0 + 1.0),
                rx1 - rx0 - 2.0,
                4.0,
                boxstyle="round,pad=0.15",
                facecolor="#3a2f14",
                edgecolor=NEW_COLOR,
                linewidth=1.4,
            )
        )
        ax.text(
            rx0 + 2.0,
            y0 + 3.0,
            _short(flag, 52),
            fontsize=6.8,
            family="monospace",
            color=NEW_COLOR,
            va="center",
            fontweight="bold",
        )
    _save(fig, path)


def draw_span_timeline(
    spans: list[dict],
    path: Path,
    title: str,
    note: str = "",
    root_label: str = "trace",
):
    """A nested span timeline: one horizontal bar per span, positioned on a
    shared time axis and indented/colored by nesting depth. Each span dict:
    {"name", "parent" (a name or None), "start_ms", "duration_ms",
    "tokens" (optional), "cost" (optional)}."""
    fig, ax = _new_fig(title, note)
    n = len(spans)
    if not n:
        _save(fig, path)
        return

    depth: dict[str, int] = {}
    for s in spans:
        p = s.get("parent")
        depth[s["name"]] = 0 if p is None else depth.get(p, 0) + 1

    t_end = max(s["start_ms"] + s["duration_ms"] for s in spans) or 1
    tx0, tx1 = 34.0, 72.0
    row_h = min(5.6, 44 / n)
    y = 50.0
    ax.text(tx0, y + 2.4, root_label, fontsize=7, color=MUTED, fontweight="bold")
    ax.text(80.0, y + 2.4, "tokens / cost", fontsize=7, color=MUTED, fontweight="bold")
    for s in spans:
        d = depth[s["name"]]
        color = LANE_COLORS[d % len(LANE_COLORS)]
        label_x = 4.0 + d * 3.0
        ax.text(
            label_x,
            y,
            _short(s["name"], max(22 - d * 2, 8)),
            fontsize=7.2,
            family="monospace",
            color=color,
            va="center",
        )
        bx0 = tx0 + (s["start_ms"] / t_end) * (tx1 - tx0)
        bw = max((s["duration_ms"] / t_end) * (tx1 - tx0), 0.6)
        ax.add_patch(
            FancyBboxPatch(
                (bx0, y - row_h * 0.32),
                bw,
                row_h * 0.64,
                boxstyle="round,pad=0.04",
                facecolor=NODE_FILL,
                edgecolor=color,
                linewidth=1.6,
            )
        )
        ax.text(
            bx0 + bw + 1.0,
            y,
            f"{s['duration_ms']}ms",
            fontsize=6.2,
            family="monospace",
            color=MUTED,
            va="center",
        )
        meta = []
        if s.get("tokens") is not None:
            meta.append(f"{s['tokens']}tok")
        if s.get("cost") is not None:
            meta.append(f"${s['cost']:.4f}")
        if meta:
            ax.text(
                80.0,
                y,
                "  ".join(meta),
                fontsize=6.6,
                family="monospace",
                color=INK,
                va="center",
            )
        y -= row_h
    _save(fig, path)


# ---- self-check ------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    from beanline import Stock, make_tools
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    out = Path(tempfile.mkdtemp(prefix="langviz-check-"))

    # ---- messages (real langchain BaseMessage + plain dicts) ----
    tool_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_stock",
                "args": {"item": "oat milk"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
    )
    msgs = [
        SystemMessage(content="You run the register at Beanline Coffee."),
        HumanMessage(content="One large oat milk latte to go, please."),
        tool_ai,
        ToolMessage(content="2 left", tool_call_id="c1"),
        AIMessage(content="That's a large oat milk latte, $6.25 total."),
    ]
    draw_messages(msgs, out / "messages.png", "Messages: order flow", new=(4,))
    draw_messages(
        [{"role": "human", "text": "cheapest large drink with an extra shot?"}],
        out / "messages_dict.png",
        "Messages: dict form",
        right_text='{\n  "drink": "espresso",\n  "size": "large",\n  "extras": ["extra shot"]\n}',
        right_title="parsed order (JSON)",
    )
    draw_dual_messages(
        msgs[:3],
        msgs[:3],
        out / "dual.png",
        "Dual: before vs after",
        "before",
        "after",
        left_verdict="bad",
        right_verdict="ok",
    )

    # ---- pipeline (LCEL chain) ----
    draw_pipeline(
        [
            {
                "label": "prompt",
                "type_label": "ChatPromptTemplate",
                "payload": "menu + order",
                "state": "done",
            },
            {
                "label": "model",
                "type_label": "ScriptedChatModel",
                "payload": "AIMessage(...)",
                "state": "active",
            },
            {
                "label": "parser",
                "type_label": "PydanticOutputParser",
                "payload": "",
                "state": "pending",
            },
        ],
        out / "pipeline.png",
        "Pipeline: LCEL chain",
    )

    # ---- graph (real langgraph) ----
    class OrderState(TypedDict):
        drink: str
        confirmed: bool

    def take_order(state: OrderState) -> dict:
        return {"drink": "latte"}

    def confirm(state: OrderState) -> dict:
        return {"confirmed": True}

    def route(state: OrderState) -> str:
        return "confirm" if state.get("drink") else END

    g = StateGraph(OrderState)
    g.add_node("take_order", take_order)
    g.add_node("confirm", confirm)
    g.add_edge(START, "take_order")
    g.add_conditional_edges("take_order", route, {"confirm": "confirm", END: END})
    g.add_edge("confirm", END)
    compiled = g.compile()
    drawable = compiled.get_graph()

    positions = {
        "__start__": (7, 30),
        "take_order": (24, 30),
        "confirm": (46, 30),
        "__end__": (60, 30),
    }
    draw_graph(
        drawable,
        positions,
        out / "graph.png",
        "Graph: order flow",
        active="confirm",
        visited=("take_order",),
        taken_edges=(("take_order", "confirm"),),
        edge_label=("take_order", "confirm", "confirm"),
        state_rows=[
            {
                "channel": "drink",
                "value": "latte",
                "reducer": "overwrite",
                "delta": "espresso",
                "changed": True,
            },
            {
                "channel": "confirmed",
                "value": "True",
                "reducer": None,
                "delta": None,
                "changed": False,
            },
        ],
    )

    # ---- state ----
    draw_state(
        [
            {
                "channel": "messages",
                "value": "[sys, human, ai, tool, ai]",
                "reducer": "add_messages",
                "delta": "ai",
                "changed": True,
            },
            {
                "channel": "total",
                "value": "6.25",
                "reducer": "add",
                "delta": "6.25",
                "changed": True,
            },
            {
                "channel": "to_go",
                "value": "True",
                "reducer": None,
                "delta": None,
                "changed": False,
            },
        ],
        out / "state.png",
        "State: order channels",
    )

    # ---- stream ----
    chunks = list("one large oat milk latte, six twenty five".split(" "))
    chunks = [c + " " for c in chunks]
    draw_stream(chunks, upto=4, path=out / "stream.png", title="Stream: token arrival")

    # ---- thread lanes ----
    draw_thread_lanes(
        [
            {
                "label": "thread-maya",
                "state": "active",
                "checkpoints": [
                    {"step": "order", "hint": "latte"},
                    {"step": "confirm", "hint": "6.25"},
                ],
                "resume_from": 0,
            },
            {
                "label": "thread-sam",
                "state": "done",
                "checkpoints": [{"step": "order", "hint": "cappuccino"}],
                "resume_from": None,
            },
            {
                "label": "thread-alex",
                "state": "dimmed",
                "checkpoints": [],
                "resume_from": None,
            },
        ],
        out / "lanes.png",
        "Lanes: threads share one checkpointer",
    )

    # ---- tool wire ----
    stock = Stock()
    check_stock, get_menu, compute_price = make_tools(stock)
    draw_tool_wire(
        4,
        out / "wire.png",
        "Wire: check_stock",
        call={"name": "check_stock", "args": {"item": "oat milk"}, "id": "c1"},
        result={"content": "1 left", "tool_call_id": "c1"},
        world=["oat milk: 1", "*just taken: oat milk", "croissant: 3"],
        world_title="stock room",
        schema={
            "check_stock": "how many units are left",
            "get_menu": "the full menu board",
        },
    )

    # ---- scorecard ----
    draw_scorecard(
        [
            {"label": "invoke", "cells": ["120ms", "$0.001"], "verdict": "pass"},
            {"label": "stream", "cells": ["340ms", "$0.001"], "verdict": "pass"},
            {"label": "agent loop", "cells": ["1200ms", "$0.004"], "verdict": "fail"},
        ],
        out / "scorecard.png",
        "Scorecard: latency + cost",
        columns=["latency", "cost"],
    )

    # ---- card ----
    draw_card(
        "compute_price(drink='latte', size='large', extras=['oat milk'])\n-> $6.25",
        out / "card.png",
        "Card: tool result",
        tone="good",
        subtitle="compute_price",
    )

    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 11, [p.name for p in pngs]
    for p in pngs:
        assert p.stat().st_size > 20_000, (p.name, p.stat().st_size)
    print("langviz self-check: all figures rendered.")
