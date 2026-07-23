"""Meet Smolagents: tinyharness's hand-built loop, run by a real framework.

Everything here is real smolagents except the model. A real ToolCallingAgent
(smolagents.agents.ToolCallingAgent, a real smolagents.MultiStepAgent) is
built with one real, deterministic tool (check_stock, a plain Python
function against a real dict). The model is a ScriptedModel subclassing
smolagents' own Model interface: it returns a fixed, scripted tool call
instead of thinking, but the agent has no idea -- it parses the real
ChatMessageToolCall, executes the real check_stock function, records a real
smolagents.ActionStep in agent.memory.steps, and feeds the real observation
back for the next turn. Same honesty contract as the rest of the deck (see
harness.py's docstring): only the model is scripted, the loop and the tool
are the real library's own code, not a mock.

The four figures below pause the real smolagents generator (agent.run(...,
stream=True)) at four real points and photograph tinyharness's own
Message/Snapshot column at each: MODEL (the scripted model is asked for the
next action over this exact context), ACTION (smolagents has parsed a real
ToolCall, before anything executes), TOOL (the real check_stock function has
now executed and returned its observation), OBSERVATION (that observation
is durably recorded in agent.memory.steps -- tinyharness's RECORD, smolagents'
own memory).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from harness import Message, Snapshot, total_tokens  # noqa: E402
from harnessviz import clear, draw_frame  # noqa: E402

from smolagents import (  # noqa: E402
    ActionStep,
    FinalAnswerStep,
    Model,
    MultiStepAgent,
    TaskStep,
    ToolCallingAgent,
    tool,
)
from smolagents.agents import ActionOutput, ToolCall, ToolOutput  # noqa: E402
from smolagents.models import (  # noqa: E402
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)
from smolagents.monitoring import LogLevel  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Meet Smolagents"
BUDGET = 1200
LOOP_LABELS = {
    "model": "MODEL",
    "hooks": "ACTION",
    "tool": "TOOL",
    "context": "OBSERVATION",
}
TASK = "How much oat milk is left in stock?"

# ---- the one real, deterministic tool ----------------------------------------

STOCK = {"oat milk": 2, "espresso beans": 12}


@tool
def check_stock(item: str) -> str:
    """Check how many units of an item are left in the stock room.

    Args:
        item: the item name to check.
    """
    return f"{STOCK.get(item, 0)} left"


# ---- the one scripted part: the model -----------------------------------------


class ScriptedModel(Model):
    """Replays a fixed list of (tool_name, args) turns. Real smolagents.Model
    subclass -- ToolCallingAgent cannot tell it apart from a real API model
    except that its answer is scripted, not generated."""

    def __init__(self, calls, on_call=None):
        super().__init__()
        self.calls = list(calls)
        self.i = 0
        self.on_call = on_call

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        if self.i >= len(self.calls):
            raise RuntimeError(
                f"ScriptedModel exhausted after {self.i} turns -- "
                "the scenario script and the smolagents loop disagree"
            )
        name, args = self.calls[self.i]
        self.i += 1
        if self.on_call:
            self.on_call(name, args, messages)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id=f"call_{self.i}",
                    type="function",
                    function=ChatMessageToolCallFunction(name=name, arguments=args),
                )
            ],
        )


def call_repr(name: str, args: dict) -> str:
    return f"{name}({json.dumps(args, sort_keys=True)[1:-1]})"


def main():
    clear(FIGURES)
    snaps: dict[str, Snapshot] = {}

    def snap(label: str, note: str, context: list[Message], loop_node: str) -> Snapshot:
        return Snapshot(
            label=label,
            note=note,
            context=list(context),
            total_tokens=total_tokens(context),
            budget=BUDGET,
            loop_node=loop_node,
            files={},
            commits=[],
            events=[],
            wire=[],
            features=[],
            tool_names=list(agent.tools.keys()),
        )

    context: list[Message] = []

    def on_model_call(name, args, messages):
        if "model" not in snaps:
            snaps["model"] = snap(
                "model",
                "MODEL: the real ToolCallingAgent asks the scripted model for the "
                "next action over this exact context; only the answer is scripted.",
                context,
                "model",
            )

    model = ScriptedModel(
        [
            ("check_stock", {"item": "oat milk"}),
            ("final_answer", {"answer": "2 left"}),
        ],
        on_call=on_model_call,
    )
    agent = ToolCallingAgent(
        tools=[check_stock], model=model, max_steps=5, verbosity_level=LogLevel.OFF
    )

    system_prompt = agent.memory.system_prompt.system_prompt
    context.append(Message("system", "text", system_prompt, pinned=True))
    context.append(Message("user", "text", TASK, pinned=True))

    first_tool_call: ToolCall | None = None
    first_tool_output: ToolOutput | None = None
    first_action_step: ActionStep | None = None
    second_action_step: ActionStep | None = None
    final_answer_step: FinalAnswerStep | None = None

    for event in agent.run(TASK, stream=True):
        if isinstance(event, ToolCall) and first_tool_call is None:
            first_tool_call = event
            context.append(
                Message(
                    "assistant",
                    "tool_use",
                    call_repr(event.name, event.arguments),
                    tool_name=event.name,
                )
            )
            snaps["action"] = snap(
                "action",
                "ACTION: smolagents has parsed a real ToolCall from the model's "
                "tool_calls -- check_stock has not executed yet.",
                context,
                "hooks",
            )
        elif isinstance(event, ToolOutput) and first_tool_output is None:
            first_tool_output = event
            context.append(
                Message(
                    "tool",
                    "tool_result",
                    event.observation,
                    tool_name=event.tool_call.name,
                )
            )
            snaps["tool"] = snap(
                "tool",
                "TOOL: the real check_stock function has just executed against "
                "the real STOCK dict and returned this observation.",
                context,
                "tool",
            )
        elif isinstance(event, ActionOutput):
            pass
        elif isinstance(event, ActionStep):
            if first_action_step is None:
                first_action_step = event
                snaps["context"] = snap(
                    "context",
                    "OBSERVATION: the ActionStep is finalized and appended to "
                    "agent.memory.steps -- durably recorded, exactly like RECORD.",
                    context,
                    "context",
                )
            else:
                second_action_step = event
        elif isinstance(event, FinalAnswerStep):
            final_answer_step = event

    draw_frame(
        snaps["model"],
        FIGURES / "step-01.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note=snaps["model"].note,
    )
    draw_frame(
        snaps["action"],
        FIGURES / "step-02.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note=snaps["action"].note,
    )
    draw_frame(
        snaps["tool"],
        FIGURES / "step-03.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note=snaps["tool"].note,
    )
    draw_frame(
        snaps["context"],
        FIGURES / "step-04.png",
        TITLE,
        right="loop",
        loop_labels=LOOP_LABELS,
        note=snaps["context"].note,
    )

    # ---- oracle ----
    assert isinstance(agent, MultiStepAgent), (
        "must be a real smolagents agent, not a mock"
    )
    assert isinstance(agent, ToolCallingAgent)

    assert first_tool_call is not None and first_tool_call.name == "check_stock"
    assert first_tool_output is not None
    assert first_tool_output.observation == "2 left", (
        "the real check_stock function must have run"
    )
    assert first_tool_output.tool_call.name == "check_stock"

    assert final_answer_step is not None
    assert final_answer_step.output == first_tool_output.observation, (
        "the scripted final answer must match the real tool's real observation"
    )

    recorded = agent.memory.steps
    assert len(recorded) == 3, (
        f"expected [TaskStep, ActionStep, ActionStep], got {recorded}"
    )
    assert isinstance(recorded[0], TaskStep)
    assert isinstance(recorded[1], ActionStep) and isinstance(recorded[2], ActionStep)
    assert recorded[1].tool_calls[0].name == "check_stock"
    assert recorded[1].observations == "2 left"
    assert recorded[2].tool_calls[0].name == "final_answer"
    assert recorded[2].is_final_answer is True
    assert first_action_step is recorded[1] and second_action_step is recorded[2]

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 4, f"expected 4 figures, got {figs}"

    print(
        f"{len(figs)} figures, real agent {type(agent).__name__}, real tool "
        f"check_stock -> {first_tool_output.observation!r}, "
        f"final answer {final_answer_step.output!r}, "
        f"{len(recorded)} real memory steps recorded. All checks passed."
    )


if __name__ == "__main__":
    main()
