"""The running example every topic photographs: build todo.py from a
12-item feature_list.json, with a real unittest suite as the oracle.

todo_source(upto) assembles the app with features F01..F<upto> implemented,
so a "session" that builds feature k really changes the code, really flips
that feature's test from red to green, and the diff between stages is the
session's real diff. The staging is verified below: at stage k, exactly k
of the 12 feature tests pass.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import subprocess

from harness import MCPServer, Workspace

PYTHON = "/Users/pranjal/.local/bin/python"

# ---- the feature list (Anthropic effective-harnesses schema) ----------------

FEATURES = [
    ("F01", "functional", "todo add TEXT saves a task to todos.json"),
    ("F02", "functional", "todo list shows open tasks with their ids"),
    ("F03", "functional", "todo done N marks a task complete"),
    ("F04", "functional", "todo delete N removes a task"),
    ("F05", "functional", "todo count reports open and done totals"),
    ("F06", "functional", "todo search TERM filters tasks by substring"),
    ("F07", "functional", "todo priority N high|normal|low sets a priority"),
    ("F08", "functional", "todo list orders high-priority tasks first"),
    ("F09", "functional", "todo due N YYYY-MM-DD attaches a due date"),
    ("F10", "functional", "todo clear-done deletes completed tasks"),
    ("F11", "robustness", "done/delete on a missing id fails with a clear message"),
    ("F12", "robustness", "an unknown command prints usage and exits nonzero"),
]

TEST_NAMES = {
    "F01": "test_f01_add",
    "F02": "test_f02_list",
    "F03": "test_f03_done",
    "F04": "test_f04_delete",
    "F05": "test_f05_count",
    "F06": "test_f06_search",
    "F07": "test_f07_priority",
    "F08": "test_f08_priority_order",
    "F09": "test_f09_due",
    "F10": "test_f10_clear_done",
    "F11": "test_f11_missing_id",
    "F12": "test_f12_usage",
}


def feature_list_json() -> str:
    return json.dumps(
        [
            {
                "id": fid,
                "category": cat,
                "description": desc,
                "steps": [
                    f"implement {fid}",
                    f"run {TEST_NAMES[fid]}",
                    "verify end-to-end",
                ],
                "passes": False,
            }
            for fid, cat, desc in FEATURES
        ],
        indent=1,
    )


# ---- todo.py, assembled feature by feature ----------------------------------

HEADER = '''"""todo.py -- a tiny CLI todo app, built feature by feature."""
import json, os, sys

DB = "todos.json"


def load():
    if os.path.exists(DB):
        with open(DB) as f:
            return json.load(f)
    return []


def save(items):
    with open(DB, "w") as f:
        json.dump(items, f, indent=1)
'''

USAGE_BLOCK = '''
USAGE = """usage: todo <command>
commands: add TEXT | list | done N | delete N | count | search TERM
          priority N high|normal|low | due N DATE | clear-done"""
'''


def _find_block(upto: int) -> str:
    if upto >= 11:
        return """
def _find(items, n):
    for item in items:
        if item["id"] == n:
            return item
    print(f"no task with id {n}")
    sys.exit(1)
"""
    return """
def _find(items, n):
    return next(item for item in items if item["id"] == n)
"""


BLOCKS = {
    "F01": """
def cmd_add(args):
    items = load()
    next_id = max((i["id"] for i in items), default=0) + 1
    items.append({"id": next_id, "text": " ".join(args), "done": False,
                  "priority": "normal", "due": None})
    save(items)
    print(f"added {next_id}")
""",
    "F02": None,  # cmd_list is assembled below (F08/F09 modify it)
    "F03": """
def cmd_done(args):
    items = load()
    _find(items, int(args[0]))["done"] = True
    save(items)
    print("done")
""",
    "F04": """
def cmd_delete(args):
    items = load()
    items.remove(_find(items, int(args[0])))
    save(items)
    print("deleted")
""",
    "F05": """
def cmd_count(args):
    items = load()
    open_n = sum(1 for i in items if not i["done"])
    print(f"{open_n} open, {len(items) - open_n} done")
""",
    "F06": """
def cmd_search(args):
    term = " ".join(args).lower()
    for i in load():
        if term in i["text"].lower():
            print(f"{i['id']}. {i['text']}")
""",
    "F07": """
def cmd_priority(args):
    n, level = int(args[0]), args[1]
    if level not in ("high", "normal", "low"):
        print("priority must be high|normal|low")
        sys.exit(1)
    items = load()
    _find(items, n)["priority"] = level
    save(items)
    print(f"priority of {n} set to {level}")
""",
    "F09": """
def cmd_due(args):
    n, date = int(args[0]), args[1]
    items = load()
    _find(items, n)["due"] = date
    save(items)
    print(f"due date of {n} set to {date}")
""",
    "F10": """
def cmd_clear_done(args):
    items = [i for i in load() if not i["done"]]
    save(items)
    print("cleared")
""",
}

CMD_NAMES = {
    "F01": ("add", "cmd_add"),
    "F02": ("list", "cmd_list"),
    "F03": ("done", "cmd_done"),
    "F04": ("delete", "cmd_delete"),
    "F05": ("count", "cmd_count"),
    "F06": ("search", "cmd_search"),
    "F07": ("priority", "cmd_priority"),
    "F09": ("due", "cmd_due"),
    "F10": ("clear-done", "cmd_clear_done"),
}


def _list_block(upto: int) -> str:
    sort = (
        """
    order = {"high": 0, "normal": 1, "low": 2}
    items.sort(key=lambda i: order[i["priority"]])"""
        if upto >= 8
        else ""
    )
    due = (
        '''
        tag = f" (due {i['due']})" if i["due"] else ""'''
        if upto >= 9
        else '''
        tag = ""'''
    )
    return f"""
def cmd_list(args):
    items = [i for i in load() if not i["done"]]{sort}
    for i in items:{due}
        print(f"{{i['id']}}. [{{i['priority']}}] {{i['text']}}{{tag}}")
"""


def todo_source(upto: int) -> str:
    """The real todo.py with features F01..F<upto> implemented."""
    ids = [fid for fid, _, _ in FEATURES[:upto]]
    parts = [HEADER]
    if upto >= 12:
        parts.append(USAGE_BLOCK)
    if upto >= 3:
        parts.append(_find_block(upto))
    for fid in ids:
        if fid == "F02":
            parts.append(_list_block(upto))
        elif BLOCKS.get(fid):
            parts.append(BLOCKS[fid])
    cmds = ", ".join(
        f'"{cmd}": {fn}' for fid, (cmd, fn) in CMD_NAMES.items() if fid in ids
    )
    guard = (
        """
    if not argv or argv[0] not in COMMANDS:
        print(USAGE)
        sys.exit(1)"""
        if upto >= 12
        else ""
    )
    parts.append(f"""

COMMANDS = {{{cmds}}}


def main(argv):{guard}
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
""")
    return "".join(parts)


# ---- the test suite (the oracle) --------------------------------------------

TESTS_SRC = '''"""Feature tests for todo.py -- the oracle. One test per feature.
It is unacceptable to remove or edit these tests to make them pass."""
import io, json, os, sys, tempfile, unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.getcwd())
import todo


def run(*argv):
    out = io.StringIO()
    code = 0
    try:
        with redirect_stdout(out):
            todo.main(list(argv))
    except SystemExit as e:
        code = e.code or 0
    return out.getvalue(), code


class TodoTests(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp(prefix="todo-test-")
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._cwd)

    def test_f01_add(self):
        run("add", "buy milk")
        data = json.load(open("todos.json"))
        self.assertEqual(data[0]["text"], "buy milk")

    def test_f02_list(self):
        run("add", "buy milk")
        out, _ = run("list")
        self.assertIn("buy milk", out)

    def test_f03_done(self):
        run("add", "buy milk")
        run("done", "1")
        out, _ = run("list")
        self.assertNotIn("buy milk", out)

    def test_f04_delete(self):
        run("add", "buy milk")
        run("delete", "1")
        self.assertEqual(json.load(open("todos.json")), [])

    def test_f05_count(self):
        run("add", "a")
        run("add", "b")
        run("done", "1")
        out, _ = run("count")
        self.assertIn("1 open, 1 done", out)

    def test_f06_search(self):
        run("add", "buy milk")
        run("add", "walk dog")
        out, _ = run("search", "milk")
        self.assertIn("buy milk", out)
        self.assertNotIn("walk dog", out)

    def test_f07_priority(self):
        run("add", "a")
        out, code = run("priority", "1", "urgent")
        self.assertEqual(code, 1)
        run("priority", "1", "high")
        self.assertEqual(json.load(open("todos.json"))[0]["priority"], "high")

    def test_f08_priority_order(self):
        run("add", "low thing")
        run("add", "big thing")
        run("priority", "2", "high")
        out, _ = run("list")
        self.assertLess(out.index("big thing"), out.index("low thing"))

    def test_f09_due(self):
        run("add", "a")
        run("due", "1", "2026-02-01")
        out, _ = run("list")
        self.assertIn("due 2026-02-01", out)

    def test_f10_clear_done(self):
        run("add", "a")
        run("add", "b")
        run("done", "1")
        run("clear-done")
        self.assertEqual(len(json.load(open("todos.json"))), 1)

    def test_f11_missing_id(self):
        run("add", "a")
        out, code = run("done", "99")
        self.assertEqual(code, 1)
        self.assertIn("no task with id 99", out)

    def test_f12_usage(self):
        out, code = run("frobnicate")
        self.assertEqual(code, 1)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
'''


# ---- memory files, skills, docs ---------------------------------------------

CLAUDE_MD = """# todo.py

A tiny CLI todo app, built one feature per session from feature_list.json.

- Run the app: python todo.py <command>
- Run the oracle: sh init.sh
- Never edit tests/ or feature descriptions. Flip "passes" only after the
  feature verifies end-to-end.
"""

INIT_SH = """# how to run this project -- read me first, don't rediscover it
python -m unittest discover -s tests -v
"""

AGENTS_MD = """# gotchas learned by earlier sessions

- todos.json ids must survive deletes: use max(id)+1, never len(items)+1.
- unittest discover needs tests/__init__.py absent and cwd at repo root.
- priority must validate its level BEFORE loading the db, or a bad call
  half-writes state.
"""

PROGRESS_SEED = """# progress log (append-only)

2026-01-01 S1: initializer -- wrote feature_list.json (12 features, all red),
init.sh, this file. No app code yet.
"""

FAILED_APPROACH_LINE = (
    "2026-01-02 S2: F01-F05 done. FAILED APPROACH: tried argparse subparsers "
    "for the command table; nested help text broke test_f12_usage; switched "
    "to a plain dict dispatch. Do not retry argparse here."
)

SKILLS = {
    "cli-conventions": (
        "House rules for CLI apps: dispatch, exit codes, usage text",
        """---
name: cli-conventions
description: House rules for CLI apps: dispatch, exit codes, usage text
---

# CLI conventions

- Commands dispatch through a plain dict, no framework.
- Every failure path prints one clear line and exits 1; success exits 0.
- Usage text lists every command on two lines, no prose.
- State lives in one JSON file next to the script, loaded fresh per command,
  written back whole. No partial writes: validate args BEFORE load().
- ids are stable: max(id)+1, never len+1 (deletes must not recycle ids).
- Output is grep-friendly: one record per line, id first.
- See references/exit-codes.md for the exit-code table.
""",
    ),
    "release-checklist": (
        "Pre-release gates: tests green, feature list current, clean git",
        """---
name: release-checklist
description: Pre-release gates: tests green, feature list current, clean git
---

# release checklist

1. python -m unittest discover -s tests -- all green, no skips.
2. feature_list.json: every shipped feature passes:true, none overclaimed.
3. git status clean; last commit message names the feature.
""",
    ),
}

SKILL_SPOKE = (
    "cli-conventions/references/exit-codes.md",
    """# exit codes

0 success | 1 user error (bad args, missing id) | 2 internal bug
Never exit 0 on a failure path: scripts chain on exit codes.
""",
)

DOCS = {
    "docs/priority-spec.md": "Priorities are high, normal, low. list orders "
    "high first, then normal, then low. Invalid levels are rejected with exit 1.",
    "docs/due-dates.md": "Due dates attach to a task as YYYY-MM-DD and render "
    "in list output as (due DATE) after the text.",
    "docs/persistence.md": "All state lives in todos.json next to the script, "
    "one JSON array, rewritten whole on every mutation.",
    "docs/team-lunch.md": "The team lunch rotates between the taqueria and "
    "the noodle bar. Vegetarians pick first.",
}


# ---- workspace builders ------------------------------------------------------


def build_workspace(
    stage: int = 5,
    git: bool = True,
    features: bool = True,
    skills: bool = False,
    docs: bool = False,
    memory: bool = False,
) -> Workspace:
    """The canonical starting state: todo.py at `stage` features, tests, and
    whichever harness furniture the topic needs. Feature list marks F01..F<stage>
    passing, matching the code."""
    ws = Workspace()
    if stage > 0:
        ws.write("todo.py", todo_source(stage))
    ws.write("tests/test_todo.py", TESTS_SRC)
    ws.write("CLAUDE.md", CLAUDE_MD)
    ws.write("init.sh", INIT_SH)
    if features:
        ws.write("feature_list.json", feature_list_json())
        for fid, _, _ in FEATURES[:stage]:
            _mark(ws, fid, True)
    if skills:
        for name, (_, body) in SKILLS.items():
            ws.write(f".skills/{name}/SKILL.md", body)
        ws.write(f".skills/{SKILL_SPOKE[0]}", SKILL_SPOKE[1])
    if docs:
        for path, text in DOCS.items():
            ws.write(path, text)
    if memory:
        ws.write("AGENTS.md", AGENTS_MD)
        ws.write("progress.md", PROGRESS_SEED)
    if git:
        ws.git_init()
        ws.git_commit(f"scaffold at stage {stage}")
    return ws


def _mark(ws: Workspace, fid: str, passes: bool):
    feats = json.loads(ws.read("feature_list.json"))
    for f in feats:
        if f["id"] == fid:
            f["passes"] = passes
    ws.write("feature_list.json", json.dumps(feats, indent=1))


def run_one_test(ws: Workspace, fid: str) -> tuple[bool, str]:
    """Really run one feature's test, end to end, in the workspace."""
    proc = subprocess.run(
        [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-k", TEST_NAMES[fid]],
        cwd=ws.root,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1]
    return ok, tail


def make_checker_server(ws: Workspace) -> MCPServer:
    """The toy MCP server: end-to-end feature verification as a service."""

    def verify_feature(feature_id: str) -> str:
        ok, tail = run_one_test(ws, feature_id)
        return f"{'PASS' if ok else 'FAIL'}: {feature_id} -- {tail}"

    def feature_status() -> str:
        feats = json.loads(ws.read("feature_list.json"))
        done = sum(1 for f in feats if f["passes"])
        return f"{done}/{len(feats)} features marked passing"

    return MCPServer(
        "checker",
        {
            "verify_feature": ("Run one feature's test end-to-end", verify_feature),
            "feature_status": ("Report feature_list.json progress", feature_status),
        },
    )


# ---- a matchbox retrieval index (for the one RAG topic) ---------------------


def mini_search(
    query: str, corpus: dict[str, str], k: int = 2
) -> list[tuple[str, float]]:
    """Real TF-IDF weighted cosine similarity, stdlib only."""

    def tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9-]+", text.lower())

    docs = {path: tokens(text) for path, text in corpus.items()}
    n = len(docs)
    df: dict[str, int] = {}
    for toks in docs.values():
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + n / d) for w, d in df.items()}

    def vec(toks: list[str]) -> dict[str, float]:
        v: dict[str, float] = {}
        for w in toks:
            v[w] = v.get(w, 0.0) + idf.get(w, math.log(1 + n))
        return v

    def cos(a: dict, b: dict) -> float:
        dot = sum(a[w] * b.get(w, 0.0) for w in a)
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    q = vec(tokens(query))
    scored = [(path, cos(q, vec(toks))) for path, toks in docs.items()]
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


# ---- fixture self-check ------------------------------------------------------

if __name__ == "__main__":
    # The staging oracle: at stage k, exactly k feature tests pass.
    for k in range(1, 13):
        ws = build_workspace(stage=k, git=False, features=False)
        rc, passed, failed, summary = ws.run_tests()
        assert passed == k, f"stage {k}: expected {k} passing, got {summary}"
        assert failed == 12 - k, f"stage {k}: {summary}"
        ws.cleanup()

    ws = build_workspace(stage=12, skills=True, docs=True, memory=True)
    rc, passed, failed, _ = ws.run_tests()
    assert rc == 0 and passed == 12

    ok, tail = run_one_test(ws, "F08")
    assert ok, tail
    server = make_checker_server(ws)
    import json as _json

    res = _json.loads(
        server.handle(
            _json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "verify_feature",
                        "arguments": {"feature_id": "F12"},
                    },
                }
            )
        )
    )
    assert "PASS" in res["result"]["content"][0]["text"]

    hits = mini_search("how should priorities order the list", DOCS)
    assert hits[0][0] == "docs/priority-spec.md", hits

    d = unified_diff(todo_source(5), todo_source(6), "todo.py")
    assert "+def cmd_search" in d

    print(
        "ok: 12 stages verified 1:1 against the oracle, checker MCP live, "
        f"mini_search top hit {hits[0][0]} @ {hits[0][1]:.2f}"
    )
    ws.cleanup()
