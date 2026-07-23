"""tinyharness: a minimal but real agentic harness, built to be photographed.

Every figure in this deck is a snapshot of this harness actually executing:
the context window is a real list of messages with real token counts, tools
really read and write files in a real sandbox directory, hooks are real
callbacks that really block or inject, the MCP client and server exchange
real JSON-RPC 2.0 strings, compaction really shrinks the context, commits
are real git commits, and the multi-session runs really burn down a real
feature_list.json with a real test suite deciding pass/fail.

The one scripted part is the model. A ScriptedModel replays turns supplied
by each topic's scenario, so every run is deterministic, replayable, and
oracle-assertable -- the same role the toy corpus plays in the RAG deck.
The deck says this out loud on its "meet tinyharness" slide.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Pinned dates make git shas deterministic across regenerations.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "tinyharness",
    "GIT_AUTHOR_EMAIL": "agent@tinyharness.local",
    "GIT_COMMITTER_NAME": "tinyharness",
    "GIT_COMMITTER_EMAIL": "agent@tinyharness.local",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "HOME": "/tmp",
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
}


def count_tokens(text: str) -> int:
    """Honest-enough tokenizer: ~4 characters per token, deterministic."""
    return max(1, round(len(text) / 4))


# ---- the context window ------------------------------------------------------


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    kind: str  # text | tool_use | tool_result | summary | skill_index |
    #            skill_body | hook_inject | memory
    content: str
    tool_name: str | None = None
    pinned: bool = False  # survives compaction verbatim
    tokens: int = 0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = count_tokens(self.content)


def total_tokens(context: list[Message]) -> int:
    return sum(m.tokens for m in context)


# ---- the scripted model ------------------------------------------------------


@dataclass
class Turn:
    """One scripted model response: plain text, or a tool call."""

    text: str = ""
    tool: str | None = None
    args: dict = field(default_factory=dict)


class ScriptedModel:
    """Replays scenario-supplied turns in order. It genuinely receives the
    context (and records the token count it saw, for oracle asserts), but
    its output is the script. Raises loudly if the loop asks for more turns
    than the scenario wrote -- a drifted scenario must kill the build, never
    ship a wrong figure."""

    def __init__(self, turns: list[Turn]):
        self.turns = list(turns)
        self.i = 0
        self.contexts_seen: list[int] = []

    def complete(self, context: list[Message]) -> Turn:
        self.contexts_seen.append(total_tokens(context))
        if self.i >= len(self.turns):
            raise RuntimeError(
                f"ScriptedModel exhausted after {self.i} turns -- "
                "the scenario script and the loop disagree"
            )
        turn = self.turns[self.i]
        self.i += 1
        return turn


# ---- the workspace (a real sandbox directory, with real git) ----------------


class Workspace:
    """A real temp directory the tools act on. Files are real files, commits
    are real git commits, tests really run. Figures read truth back from here."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(tempfile.mkdtemp(prefix="tinyharness-"))

    def path(self, rel: str) -> Path:
        return self.root / rel

    def read(self, rel: str) -> str:
        return self.path(rel).read_text()

    def write(self, rel: str, content: str):
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def files(self) -> dict[str, str]:
        """Read back every text file (relative path -> content), skipping .git."""
        out: dict[str, str] = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_dir() or ".git" in p.parts:
                continue
            rel = str(p.relative_to(self.root))
            try:
                out[rel] = p.read_text()
            except UnicodeDecodeError:
                out[rel] = "<binary>"
        return out

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=self.root, env=GIT_ENV, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")
        return proc.stdout

    def git_init(self):
        self._git("init", "-q", "-b", "main")

    def git_commit(self, message: str) -> str:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self._git("rev-parse", "--short", "HEAD").strip()

    def git_log(self) -> list[tuple[str, str]]:
        """[(sha7, subject)] oldest first; [] before git init."""
        if not (self.root / ".git").exists():
            return []
        out = self._git("log", "--reverse", "--format=%h%x09%s")
        return [tuple(line.split("\t", 1)) for line in out.splitlines() if line]

    def git_dirty(self) -> bool:
        return bool(self._git("status", "--porcelain").strip())

    def run_tests(self) -> tuple[int, int, int, str]:
        """Really run the workspace's unittest suite (stdlib, no pytest dep).
        Returns (returncode, passed, failed, one_line_summary)."""
        proc = subprocess.run(
            [
                "/Users/pranjal/.local/bin/python",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            env={**GIT_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        text = proc.stdout + proc.stderr
        ran = 0
        for line in text.splitlines():
            if line.startswith("Ran ") and " test" in line:
                ran = int(line.split()[1])
        failed = text.count("FAIL: ") + text.count("ERROR: ")
        passed = ran - failed
        ok = proc.returncode == 0
        summary = f"{passed} passed, {failed} failed" if ran else "no tests ran"
        return (0 if ok else 1, passed, failed, summary)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ---- tools -------------------------------------------------------------------


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[str, callable]] = {}
        self.call_counts: dict[str, int] = {}

    def register(self, name: str, description: str, fn):
        self._tools[name] = (description, fn)
        self.call_counts.setdefault(name, 0)

    def names(self) -> list[str]:
        return list(self._tools)

    def description(self, name: str) -> str:
        return self._tools[name][0]

    def execute(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            raise RuntimeError(f"unknown tool {name!r} -- scenario drifted")
        self.call_counts[name] += 1
        try:
            return self._tools[name][1](**args)
        except TypeError as e:
            raise RuntimeError(f"bad args for tool {name!r}: {e}") from e


def builtin_tools(ws: Workspace) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "read_file",
        "Read a file from the workspace",
        lambda path: ToolResult(ws.read(path))
        if ws.exists(path)
        else ToolResult(f"file not found: {path}", is_error=True),
    )
    reg.register(
        "write_file",
        "Write a file in the workspace",
        lambda path, content: (
            ws.write(path, content),
            ToolResult(f"wrote {path} ({len(content.splitlines())} lines)"),
        )[1],
    )
    reg.register(
        "list_files",
        "List workspace files",
        lambda: ToolResult("\n".join(ws.files())),
    )

    def _run_tests() -> ToolResult:
        rc, passed, failed, summary = ws.run_tests()
        return ToolResult(f"tests: {summary}", is_error=rc != 0)

    reg.register("run_tests", "Run the workspace test suite", _run_tests)

    def _bash(command: str) -> ToolResult:
        # bash-lite: a small whitelist, run for real inside the workspace.
        allowed = ("ls", "cat ", "pwd", "git log", "git status", "python ")
        if not command.startswith(allowed):
            return ToolResult(f"command not allowed: {command}", is_error=True)
        proc = subprocess.run(
            command,
            shell=True,
            cwd=ws.root,
            capture_output=True,
            text=True,
            env=GIT_ENV,
            timeout=30,
        )
        out = (proc.stdout + proc.stderr).strip() or "(no output)"
        return ToolResult(out, is_error=proc.returncode != 0)

    reg.register("bash", "Run a whitelisted shell command", _bash)
    return reg


# ---- hooks -------------------------------------------------------------------


@dataclass
class HookDecision:
    action: str = "allow"  # allow | deny | inject | continue_
    reason: str = ""
    text: str = ""  # injected context, when action == "inject"


class Hooks:
    """Real callbacks around the loop. A deny on pre_tool_use means the tool
    function is never called; the denial reason enters the context as a tool
    result -- the same semantics Claude Code's hooks have."""

    EVENTS = (
        "session_start",
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "stop",
        "pre_compact",
    )

    def __init__(self):
        self._hooks: dict[str, list[tuple[str, callable]]] = {
            e: [] for e in self.EVENTS
        }
        self.fired: list[dict] = []  # {step, event, hook, action, detail}

    def on(self, event: str, name: str, fn):
        self._hooks[event].append((name, fn))

    def registered(self) -> dict[str, list[str]]:
        return {e: [n for n, _ in fns] for e, fns in self._hooks.items() if fns}

    def fire(self, event: str, step: int, **payload) -> list[HookDecision]:
        decisions = []
        for name, fn in self._hooks[event]:
            d = fn(**payload) or HookDecision()
            self.fired.append(
                {
                    "step": step,
                    "event": event,
                    "hook": name,
                    "action": d.action,
                    "detail": d.reason or d.text[:60],
                }
            )
            decisions.append(d)
        return decisions


# ---- skills (progressive disclosure over real files) ------------------------


class SkillLibrary:
    """Scans workspace/.skills/<name>/SKILL.md -- real files. The index (one
    name+description line per skill) is the cheap tier every session pays;
    load() pulls one full body into the context on demand."""

    def __init__(self, ws: Workspace):
        self.ws = ws

    def skills(self) -> dict[str, str]:
        """name -> one-line description, parsed from each SKILL.md frontmatter."""
        out = {}
        skills_dir = self.ws.path(".skills")
        if not skills_dir.exists():
            return out
        for p in sorted(skills_dir.iterdir()):
            f = p / "SKILL.md"
            if f.exists():
                desc = ""
                for line in f.read_text().splitlines():
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip()
                        break
                out[p.name] = desc
        return out

    def index_message(self) -> Message:
        lines = [f"{name}: {desc}" for name, desc in self.skills().items()]
        return Message(
            "system", "skill_index", "Available skills:\n" + "\n".join(lines)
        )

    def body(self, name: str) -> str:
        return self.ws.read(f".skills/{name}/SKILL.md")

    def load(self, name: str) -> Message:
        return Message("system", "skill_body", self.body(name))


# ---- MCP: a real JSON-RPC 2.0 client/server pair ----------------------------


class MCPServer:
    """An in-process MCP server that speaks real JSON-RPC 2.0 strings. The
    transport is a synchronous pump instead of a socket, which keeps runs
    deterministic; the messages on the wire are the real protocol."""

    def __init__(self, name: str, tools: dict[str, tuple[str, callable]]):
        self.name = name
        self.tools = tools  # tool -> (description, fn)

    def handle(self, raw: str) -> str:
        req = json.loads(raw)
        method, params, rid = req["method"], req.get("params", {}), req["id"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": self.name, "version": "1.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": t, "description": d[0]} for t, d in self.tools.items()
                ]
            }
        elif method == "tools/call":
            fn = self.tools[params["name"]][1]
            out = fn(**params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": out}]}
        else:
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32601, "message": method},
                }
            )
        return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})


class MCPClient:
    """Pumps JSON-RPC strings to the server and keeps every literal message
    for the wire-panel figure. Discovered tools register into the harness's
    ToolRegistry under mcp__<server>__<tool> -- foreign code looks native."""

    def __init__(self, server: MCPServer):
        self.server = server
        self.wire: list[tuple[str, dict]] = []  # ("->" | "<-", parsed message)
        self._id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params:
            req["params"] = params
        raw_req = json.dumps(req)
        self.wire.append(("->", json.loads(raw_req)))
        raw_res = self.server.handle(raw_req)
        res = json.loads(raw_res)
        self.wire.append(("<-", res))
        return res["result"]

    def initialize(self) -> dict:
        return self.request("initialize", {"clientInfo": {"name": "tinyharness"}})

    def list_tools(self) -> list[dict]:
        return self.request("tools/list")["tools"]

    def attach(self, registry: ToolRegistry):
        for tool in self.list_tools():
            name = f"mcp__{self.server.name}__{tool['name']}"

            def _call(_tool=tool["name"], **kwargs) -> ToolResult:
                result = self.request(
                    "tools/call", {"name": _tool, "arguments": kwargs}
                )
                return ToolResult(result["content"][0]["text"])

            registry.register(name, tool["description"], _call)


# ---- the agent ---------------------------------------------------------------


DEFAULT_SYSTEM = "You are a coding agent. Work in the workspace with the tools."


class Agent:
    """The loop: model -> tool_use -> execute -> tool_result -> model ...
    until a text turn (or a stop hook bounces it back). One Agent = one
    session = one context window."""

    def __init__(
        self,
        model: ScriptedModel,
        tools: ToolRegistry,
        workspace: Workspace,
        hooks: Hooks | None = None,
        skills: SkillLibrary | None = None,
        budget: int = 2000,
        keep_last: int = 4,
        system_prompt: str = DEFAULT_SYSTEM,
        name: str = "agent",
    ):
        self.model = model
        self.tools = tools
        self.ws = workspace
        self.hooks = hooks or Hooks()
        self.skills = skills
        self.budget = budget
        self.keep_last = keep_last
        self.name = name
        self.context: list[Message] = [
            Message("system", "text", system_prompt, pinned=True)
        ]
        self.step = 0
        self.compactions = 0
        self.done_text = ""

    # -- context plumbing --

    def append(self, msg: Message, on_event=None, event: str = ""):
        self.context.append(msg)
        self.maybe_compact(on_event)
        if on_event:
            on_event(event or f"{msg.role}:{msg.kind}", msg)

    def maybe_compact(self, on_event=None):
        if total_tokens(self.context) <= self.budget:
            return
        self.hooks.fire("pre_compact", self.step)
        if on_event:
            on_event("pre_compact", None)
        head = [m for m in self.context if m.pinned]
        tail = [m for m in self.context if not m.pinned][-self.keep_last :]
        middle = [m for m in self.context if not m.pinned and m not in tail]
        lines = [
            f"- [{m.role}/{m.kind}] {m.content.splitlines()[0][:64]}" for m in middle
        ]
        summary = Message(
            "system", "summary", "Summary of earlier conversation:\n" + "\n".join(lines)
        )
        self.context = head + [summary] + tail
        self.compactions += 1
        if on_event:
            on_event("compact", summary)

    # -- the loop --

    def run(self, user_prompt: str, on_event=None, max_iterations: int = 40) -> str:
        self.step = 0
        if len(self.context) == 1:  # fresh session: fire session_start once
            for d in self.hooks.fire("session_start", self.step):
                if d.action == "inject":
                    self.append(
                        Message("system", "hook_inject", d.text, pinned=True),
                        on_event,
                        "hook:session_start:inject",
                    )
            if self.skills:
                self.append(self.skills.index_message(), on_event, "skill_index")

        for d in self.hooks.fire("user_prompt_submit", self.step, prompt=user_prompt):
            if d.action == "inject":
                self.append(
                    Message("system", "hook_inject", d.text),
                    on_event,
                    "hook:user_prompt_submit:inject",
                )
        self.append(Message("user", "text", user_prompt, pinned=True), on_event, "user")

        for _ in range(max_iterations):
            self.step += 1
            turn = self.model.complete(self.context)

            if turn.tool:
                call_repr = (
                    f"{turn.tool}({json.dumps(turn.args, sort_keys=True)[1:-1]})"
                )
                self.append(
                    Message(
                        "assistant",
                        "tool_use",
                        (turn.text + "\n" if turn.text else "") + call_repr,
                        tool_name=turn.tool,
                    ),
                    on_event,
                    "assistant:tool_use",
                )
                denied = None
                for d in self.hooks.fire(
                    "pre_tool_use", self.step, tool=turn.tool, args=turn.args
                ):
                    if d.action == "deny":
                        denied = d
                        break
                if denied:
                    self.append(
                        Message(
                            "tool",
                            "tool_result",
                            f"BLOCKED by hook: {denied.reason}",
                            tool_name=turn.tool,
                        ),
                        on_event,
                        "hook:pre_tool_use:deny",
                    )
                    continue
                result = self.tools.execute(turn.tool, turn.args)
                extra = ""
                for d in self.hooks.fire(
                    "post_tool_use", self.step, tool=turn.tool, result=result
                ):
                    if d.action == "inject":
                        extra = "\n" + d.text
                self.append(
                    Message(
                        "tool",
                        "tool_result",
                        ("ERROR: " if result.is_error else "") + result.output + extra,
                        tool_name=turn.tool,
                    ),
                    on_event,
                    "tool_result",
                )
                continue

            # text turn: the model wants to stop
            self.append(
                Message("assistant", "text", turn.text), on_event, "assistant:text"
            )
            bounced = False
            for d in self.hooks.fire("stop", self.step, text=turn.text):
                if d.action == "continue_":
                    self.append(
                        Message("user", "hook_inject", d.text),
                        on_event,
                        "hook:stop:continue",
                    )
                    bounced = True
            if not bounced:
                self.done_text = turn.text
                return turn.text

        raise RuntimeError("max_iterations exhausted without a final text turn")

    # -- checkpoint / resume: nothing precious lives in the process --

    def checkpoint(self, rel: str = ".checkpoint.json"):
        state = {
            "name": self.name,
            "budget": self.budget,
            "keep_last": self.keep_last,
            "compactions": self.compactions,
            "context": [
                {
                    "role": m.role,
                    "kind": m.kind,
                    "content": m.content,
                    "tool_name": m.tool_name,
                    "pinned": m.pinned,
                }
                for m in self.context
            ],
        }
        self.ws.write(rel, json.dumps(state, indent=1))

    @classmethod
    def resume(
        cls,
        ws: Workspace,
        model: ScriptedModel,
        tools: ToolRegistry,
        rel: str = ".checkpoint.json",
        **kw,
    ) -> "Agent":
        state = json.loads(ws.read(rel))
        agent = cls(
            model,
            tools,
            ws,
            budget=state["budget"],
            keep_last=state["keep_last"],
            name=state["name"],
            **kw,
        )
        agent.context = [
            Message(m["role"], m["kind"], m["content"], m["tool_name"], m["pinned"])
            for m in state["context"]
        ]
        agent.compactions = state["compactions"]
        return agent


# ---- feature list (the Anthropic burn-down pattern) -------------------------


def read_features(ws: Workspace, rel: str = "feature_list.json") -> list[dict]:
    return json.loads(ws.read(rel)) if ws.exists(rel) else []


def passing_count(features: list[dict]) -> int:
    return sum(1 for f in features if f.get("passes"))


def mark_feature(
    ws: Workspace, feature_id: str, passes: bool, rel: str = "feature_list.json"
):
    features = read_features(ws, rel)
    for f in features:
        if f["id"] == feature_id:
            f["passes"] = passes
            break
    else:
        raise RuntimeError(f"unknown feature {feature_id!r}")
    ws.write(rel, json.dumps(features, indent=1))


# ---- snapshots: the photograph mechanism ------------------------------------


@dataclass
class Snapshot:
    label: str
    note: str
    context: list[Message]
    total_tokens: int
    budget: int
    loop_node: str  # user | model | hooks | tool | done
    files: dict[str, str]
    commits: list[tuple[str, str]]
    events: list[dict]  # hooks.fired so far
    wire: list[tuple[str, dict]]
    features: list[dict]
    tool_names: list[str]
    new_msgs: set[int] = field(default_factory=set)
    changed_files: set[str] = field(default_factory=set)


class Harness:
    """Owns the workspace and the snapshot history. snap() re-reads truth
    from disk and the live agent, and diffs against the previous snapshot so
    whatever is new draws amber -- the gitviz `now - known` mechanic."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.snapshots: list[Snapshot] = []

    def snap(
        self,
        label: str,
        agent: Agent | None = None,
        note: str = "",
        loop_node: str = "model",
        wire: list | None = None,
    ) -> Snapshot:
        context = copy.deepcopy(agent.context) if agent else []
        files = self.ws.files()
        prev = self.snapshots[-1] if self.snapshots else None
        prev_n = len(prev.context) if prev else 0
        new_msgs = set(range(prev_n, len(context)))
        if (
            prev
            and prev.context
            and context[: len(prev.context)] != prev.context[: len(context)]
        ):
            # compaction rewrote history: everything is "new"
            new_msgs = set(range(len(context)))
        changed = {
            k for k, v in files.items() if prev is None or prev.files.get(k) != v
        }
        s = Snapshot(
            label=label,
            note=note,
            context=context,
            total_tokens=total_tokens(context),
            budget=agent.budget if agent else 0,
            loop_node=loop_node,
            files=files,
            commits=self.ws.git_log(),
            events=list(agent.hooks.fired) if agent else [],
            wire=list(wire) if wire else [],
            features=read_features(self.ws),
            tool_names=agent.tools.names() if agent else [],
            new_msgs=new_msgs,
            changed_files=changed,
        )
        self.snapshots.append(s)
        return s


# ---- self-check --------------------------------------------------------------

if __name__ == "__main__":
    ws = Workspace()
    ws.git_init()
    ws.write("app.py", "def hello():\n    return 'hi'\n")
    ws.write(
        "tests/test_app.py",
        "import sys, unittest\nsys.path.insert(0, '..')\n"
        "sys.path.insert(0, '.')\nfrom app import hello\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_hello(self):\n        self.assertEqual(hello(), 'hi')\n",
    )
    ws.git_commit("initial")

    tools = builtin_tools(ws)
    hooks = Hooks()
    blocked_calls = {"n": 0}

    def guard(tool, args):
        if tool == "write_file" and args.get("path") == ".env":
            return HookDecision("deny", reason=".env is protected")
        return HookDecision()

    hooks.on("pre_tool_use", "protect-env", guard)

    model = ScriptedModel(
        [
            Turn(
                text="Reading app.py first.", tool="read_file", args={"path": "app.py"}
            ),
            Turn(tool="write_file", args={"path": ".env", "content": "SECRET=1"}),
            Turn(tool="run_tests", args={}),
            Turn(text="Done: tests pass, .env write was blocked."),
        ]
    )
    agent = Agent(model, tools, ws, hooks=hooks, budget=5000)
    h = Harness(ws)
    agent.run(
        "Check the app and run the tests.", on_event=lambda e, m: h.snap(e, agent)
    )

    assert not ws.exists(".env"), "hook must really block the write"
    assert any("BLOCKED" in m.content for m in agent.context), (
        "denial must enter context"
    )
    rc, passed, failed, _ = ws.run_tests()
    assert rc == 0 and passed == 1 and failed == 0

    # MCP round-trip
    server = MCPServer("checker", {"ping": ("Ping the checker", lambda: "pong")})
    client = MCPClient(server)
    client.initialize()
    client.attach(tools)
    assert "mcp__checker__ping" in tools.names()
    assert tools.execute("mcp__checker__ping", {}).output == "pong"
    assert all(m.get("jsonrpc") == "2.0" for _, m in client.wire)

    # compaction: a middle-heavy context really shrinks into head+summary+tail
    a2 = Agent(ScriptedModel([]), tools, ws, budget=200, keep_last=2)
    for i in range(8):
        a2.context.append(Message("tool", "tool_result", f"result {i}: " + "x " * 80))
    before = total_tokens(a2.context)
    a2.maybe_compact()
    after = total_tokens(a2.context)
    assert a2.compactions == 1 and after < before
    assert any(m.kind == "summary" for m in a2.context)

    # checkpoint / resume
    agent.checkpoint()
    fresh = Agent.resume(ws, ScriptedModel([]), tools)
    assert [m.content for m in fresh.context] == [m.content for m in agent.context]

    print(f"ok: {len(h.snapshots)} snapshots, tests green, hook blocked, MCP live")
    ws.cleanup()
