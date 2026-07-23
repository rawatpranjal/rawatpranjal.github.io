"""worker.py -- a real child process for the "kill it, and resume" topic.

Spawned by run.py as a subprocess, pointed at a REAL workspace directory
(the same tinyharness workspace run.py is drawing from). It writes real
files, makes a real git commit for F11, then starts F12 with a half-written
todo.py and goes to sleep. run.py SIGKILLs it right there. This process
never gets to say goodbye.

No matplotlib import: this process only touches the workspace, never draws.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import todo_source  # noqa: E402
from harness import (  # noqa: E402
    Agent,
    ScriptedModel,
    Turn,
    Workspace,
    builtin_tools,
    mark_feature,
)


def main():
    ws = Workspace(root=Path(sys.argv[1]))
    tools = builtin_tools(ws)
    half_f12 = todo_source(12)[: todo_source(12).index("def main") + 8]

    model = ScriptedModel(
        [
            Turn(
                text="Reading progress.md.",
                tool="read_file",
                args={"path": "progress.md"},
            ),
            Turn(
                text="Implementing F11.",
                tool="write_file",
                args={"path": "todo.py", "content": todo_source(11)},
            ),
            Turn(
                text="Starting F12.",
                tool="write_file",
                args={"path": "todo.py", "content": half_f12},
            ),
        ]
    )
    agent = Agent(model, tools, ws, budget=20000)
    write_count = {"n": 0}

    def on_event(event, msg):
        if event == "tool_result" and msg.tool_name == "write_file":
            write_count["n"] += 1
            if write_count["n"] == 1:
                mark_feature(ws, "F11", True)
                ws.git_commit("F11: missing-id errors")
                print("CHECKPOINT-1", flush=True)
            elif write_count["n"] == 2:
                print("MID-TOOL-CALL", flush=True)
                time.sleep(60)  # never wakes: the parent SIGKILLs it here

    agent.run("Build F11, then start F12.", on_event=on_event)


if __name__ == "__main__":
    main()
