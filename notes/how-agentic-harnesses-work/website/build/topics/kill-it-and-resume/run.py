"""Kill it, and resume: nothing precious lives in the process.

A REAL child process is spawned and SIGKILLed mid-session, in the middle of
writing a feature. The disk survives (it always did: files, commits, the
feature board). A fresh session wakes, reads three real things, and knows
exactly where the old one died, then redoes only the interrupted feature.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import PYTHON, build_workspace, todo_source  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    Harness,
    ScriptedModel,
    Turn,
    builtin_tools,
    mark_feature,
    passing_count,
    read_features,
)
from harnessviz import clear, draw_frame, draw_session_lanes  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Kill it, and resume"
WORKER = HERE / "worker.py"
DISK_FILES = ["todo.py", "progress.md", "feature_list.json", "init.sh"]


def main():
    clear(FIGURES)
    ws = build_workspace(stage=10, memory=True)
    h = Harness(ws)

    sessions_before = [
        {"label": "S1", "state": "done", "blocks": 5, "note": "F01-F04"},
        {"label": "S2", "state": "done", "blocks": 5, "note": "F05-F07"},
        {"label": "S3", "state": "done", "blocks": 5, "note": "F08-F10"},
        {"label": "S4", "state": "active", "blocks": 6, "note": "F11-F12"},
    ]

    # ---- step-01: before the kill, the real disk up to the last commit ----
    draw_session_lanes(
        sessions_before,
        FIGURES / "step-01.png",
        TITLE,
        note="A session mid-feature. The disk is current up to its last commit.",
        features=read_features(ws),
        commits=ws.git_log(),
        disk_files=DISK_FILES,
    )

    # ---- spawn a REAL child on the same workspace, read to the mid-tool-call
    # checkpoint, then really SIGKILL it ----
    proc = subprocess.Popen(
        [PYTHON, str(WORKER), str(ws.root)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    saw_checkpoint1 = False
    saw_mid = False
    for line in proc.stdout:
        line = line.strip()
        if line == "CHECKPOINT-1":
            saw_checkpoint1 = True
        elif line == "MID-TOOL-CALL":
            saw_mid = True
            break
    assert saw_checkpoint1 and saw_mid, "the child must really reach both checkpoints"

    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    assert proc.returncode == -9, f"expected a real SIGKILL, got {proc.returncode}"

    passing_after_kill = passing_count(read_features(ws))

    # ---- step-02: the kill ----
    sessions_dead = sessions_before[:-1] + [
        {"label": "S4", "state": "dead", "blocks": 6, "note": "F11 landed, F12 killed"}
    ]
    draw_session_lanes(
        sessions_dead,
        FIGURES / "step-02.png",
        TITLE,
        note="SIGKILL. A real dead process, returncode -9. No goodbye, no flush.",
        features=read_features(ws),
        commits=ws.git_log(),
        disk_files=DISK_FILES,
    )

    # ---- step-03: what survived ----
    compiles = subprocess.run(
        [PYTHON, "-m", "py_compile", str(ws.path("todo.py"))], capture_output=True
    )
    assert compiles.returncode != 0, "the half-written todo.py must not compile"

    throwaway = Agent(
        ScriptedModel([Turn(text="Surveying the workspace after the crash.")]),
        builtin_tools(ws),
        ws,
        budget=20000,
    )
    throwaway.run("Survey the workspace.")
    snap_survive = h.snap("survived", throwaway, loop_node="done")
    draw_frame(
        snap_survive,
        FIGURES / "step-03.png",
        TITLE,
        right="files",
        note=(
            f"The disk is intact: {passing_after_kill} features green, the "
            f"commits, progress.md. todo.py is the half-written casualty "
            f"(py_compile rc={compiles.returncode}). Do not adopt a pet, the "
            "process was disposable, the disk was not."
        ),
    )

    # ---- step-04 + step-05: a new session re-orients, then resumes ----
    resume_model = ScriptedModel(
        [
            Turn(
                text="Reading progress.md.",
                tool="read_file",
                args={"path": "progress.md"},
            ),
            Turn(
                text="Checking feature_list.json.",
                tool="read_file",
                args={"path": "feature_list.json"},
            ),
            Turn(
                text="Checking git status.",
                tool="bash",
                args={"command": "git status"},
            ),
            Turn(tool="run_tests", args={}),
            Turn(
                text=(
                    "I see exactly where the old session died: F11 landed and "
                    "committed for real, F12 is a half-written todo.py that "
                    "does not even import. Resuming from there."
                )
            ),
            Turn(
                text="Rewriting todo.py with a clean F12.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(12)},
            ),
            Turn(tool="run_tests", args={}),
            Turn(text="F12 shipped: usage text now covers an unknown command. 12/12."),
        ]
    )
    resumer = Agent(resume_model, builtin_tools(ws), ws, budget=20000)

    def on_event(event, msg):
        if event == "tool_result" and "12 passed" in msg.content:
            mark_feature(ws, "F12", True)
            ws.git_commit("F12: usage text")

    resumer.run("Investigate where the previous session left off.", on_event=on_event)
    snap_reoriented = h.snap("reoriented", resumer, loop_node="done")
    draw_frame(
        snap_reoriented,
        FIGURES / "step-04.png",
        TITLE,
        right="none",
        note=(
            "Three reads and one test run. The new session now knows exactly "
            "where the old one died."
        ),
    )

    resumer.run("Resume the interrupted feature.", on_event=on_event)
    snap_resumed = h.snap("resumed", resumer, loop_node="done")
    draw_frame(
        snap_resumed,
        FIGURES / "step-05.png",
        TITLE,
        right="files",
        note=(
            "It redid only the interrupted feature. Everything already "
            "committed survived untouched."
        ),
    )

    # ---- step-06: the full arc ----
    sessions_final = sessions_before[:-1] + [
        {"label": "S4", "state": "dead", "blocks": 6, "note": "F11 landed, F12 killed"},
        {"label": "S5", "state": "done", "blocks": 4, "note": "F12 resumed"},
    ]
    draw_session_lanes(
        sessions_final,
        FIGURES / "step-06.png",
        TITLE,
        note=(
            "Sessions are cattle, the repo is the pet. The commit dots are "
            "the heartbeat you can watch from a phone."
        ),
        features=read_features(ws),
        commits=ws.git_log(),
        disk_files=DISK_FILES,
    )

    # ---- oracle ----
    assert proc.returncode == -9, "the child must have been really SIGKILLed"
    assert passing_after_kill == 11, (
        f"expected 11/12 right after the kill, got {passing_after_kill}"
    )
    rc, passed, failed, summary = ws.run_tests()
    assert rc == 0 and passed == 12 and failed == 0, summary
    log_subjects = " ".join(msg for _, msg in ws.git_log())
    assert "F11" in log_subjects and "F12" in log_subjects, log_subjects
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, child killed (rc={proc.returncode}), "
        f"board {passing_after_kill}/12 right after the kill -> {passed}/12 "
        "at the end. All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
