"""MCP, the universal port: a real JSON-RPC handshake turns foreign code into
a tool that looks exactly like a built-in one.

Every message on the wire is a real json.dumps/json.loads round trip through
an in-process MCP server; the only scripted part is the model deciding to
call the checker.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from fixtures import build_workspace, make_checker_server  # noqa: E402
from harness import Agent, Harness, MCPClient, ScriptedModel, Turn, builtin_tools  # noqa: E402
from harnessviz import clear, draw_frame, draw_wire  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "MCP, the universal port"


def main():
    clear(FIGURES)
    ws = build_workspace(stage=8)
    tools = builtin_tools(ws)
    server = make_checker_server(ws)
    client = MCPClient(server)

    # step 1: two separate programs, no wire traffic yet, the belt holds only
    # built-ins.
    draw_wire(
        [],
        FIGURES / "step-01.png",
        TITLE,
        client_tools=tools.names(),
        note="Before the handshake: two separate programs, no wire traffic, the belt holds only built-in tools.",
    )

    # step 2: the real initialize handshake.
    client.initialize()
    draw_wire(
        client.wire,
        FIGURES / "step-02.png",
        TITLE,
        client_tools=tools.names(),
        note="A real JSON-RPC initialize request, and the server's capabilities response.",
    )

    # step 3: discovery, run manually so the wire panel shows the exchange
    # plainly before any registration happens.
    listed = client.request("tools/list")
    draw_wire(
        client.wire,
        FIGURES / "step-03.png",
        TITLE,
        client_tools=tools.names(),
        note=f"tools/list discovers {', '.join(t['name'] for t in listed['tools'])} on the server.",
    )

    # step 4: attach for real -- the two foreign tools register into the same
    # registry as the built-ins. The wire slice stays the one already shown;
    # only the belt grows.
    client.attach(tools)
    draw_wire(
        client.wire[:4],
        FIGURES / "step-04.png",
        TITLE,
        client_tools=tools.names(),
        note="Attached: the foreign tools now live in the same registry as the built-ins, amber-flagged.",
    )

    # step 5: the model calls the foreign tool exactly like a native one.
    model = ScriptedModel(
        [
            Turn(
                text="Verifying F08 through the checker.",
                tool="mcp__checker__verify_feature",
                args={"feature_id": "F08"},
            ),
            Turn(text="F08 is confirmed."),
        ]
    )
    agent = Agent(model, tools, ws, budget=1500)
    h = Harness(ws)
    agent.run("Confirm F08 (priority ordering) is really done.")
    draw_wire(
        client.wire,
        FIGURES / "step-05.png",
        TITLE,
        client_tools=tools.names(),
        note="tools/call runs the real test end to end and comes back PASS, over the same wire.",
    )

    # step 6: in the context column, the result reads like any native tool result.
    snap = h.snap("done", agent, loop_node="done")
    draw_frame(
        snap,
        FIGURES / "step-06.png",
        TITLE,
        right="none",
        note="The model cannot tell local from remote, every connected server's schemas are index tax paid up front.",
    )

    # ---- oracle ----
    assert all(m.get("jsonrpc") == "2.0" for _, m in client.wire)
    req_ids = [m["id"] for d, m in client.wire if d == "->"]
    res_ids = [m["id"] for d, m in client.wire if d == "<-"]
    assert req_ids == res_ids, (req_ids, res_ids)
    assert "mcp__checker__verify_feature" in tools.names()
    tool_results = [m.content for m in agent.context if m.kind == "tool_result"]
    assert any("PASS: F08" in c for c in tool_results), tool_results
    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 6, f"expected 6 figures, got {figs}"
    print(
        f"{len(figs)} figures, {len(client.wire)} wire messages paired 1:1, "
        f"tool result {[c for c in tool_results if 'PASS' in c][0]!r}. "
        "All checks passed."
    )
    ws.cleanup()


if __name__ == "__main__":
    main()
