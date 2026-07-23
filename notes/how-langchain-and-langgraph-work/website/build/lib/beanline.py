"""Beanline Coffee: the running example every topic photographs.

A small coffee shop with a real menu, a real (mutable) stock room, and a
real till. Every topic's run.py builds its scene from these fixtures, so
figures are photographs of real state, never mockups.

The honesty contract, same as the sibling decks: everything is real except
the model. ScriptedChatModel replays a fixed list of AIMessages through the
real langchain code paths (invoke, stream, tool calling, graphs), records
every context it was actually shown, and dies loudly if the script and the
code path disagree. Prompts really render, parsers really parse, tools
really execute against the real Stock/Till, graphs really route/checkpoint.

Self-check: run this file directly.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- the shop

MENU = {
    "latte": 4.50,
    "cappuccino": 4.00,
    "espresso": 3.00,
    "mocha": 5.00,
    "cold brew": 4.25,
}
SIZES = {"small": 0.00, "medium": 0.50, "large": 1.00}
EXTRAS = {"oat milk": 0.75, "extra shot": 1.00, "vanilla": 0.50}
FOOD = {"croissant": 3.50, "cookie": 2.75}


def menu_board() -> str:
    """The menu board as text -- the data a prompt template injects."""
    drinks = ", ".join(f"{d} ${p:.2f}" for d, p in MENU.items())
    sizes = ", ".join(f"{s} +${p:.2f}" for s, p in SIZES.items())
    extras = ", ".join(f"{e} +${p:.2f}" for e, p in EXTRAS.items())
    food = ", ".join(f"{f} ${p:.2f}" for f, p in FOOD.items())
    return f"drinks: {drinks}. sizes: {sizes}. extras: {extras}. food: {food}."


class Stock:
    """The stock room. Real mutable counts; tools read and take from it."""

    def __init__(self, counts: dict[str, int] | None = None):
        self.counts = dict(
            counts
            if counts is not None
            else {"oat milk": 2, "almond milk": 0, "whole milk": 10, "croissant": 3}
        )

    def check(self, item: str) -> int:
        return self.counts.get(item, 0)

    def take(self, item: str, n: int = 1) -> None:
        if self.counts.get(item, 0) < n:
            raise ValueError(f"out of {item}")
        self.counts[item] -= n


class Till:
    """The till. Real arithmetic, real mutation, a ledger for figures."""

    def __init__(self, balance: float = 200.00):
        self.balance = round(balance, 2)
        self.ledger: list[
            tuple[str, float, float]
        ] = []  # (kind, amount, balance-after)

    def charge(self, amount: float) -> None:
        self.balance = round(self.balance + amount, 2)
        self.ledger.append(("charge", round(amount, 2), self.balance))

    def refund(self, amount: float) -> None:
        self.balance = round(self.balance - amount, 2)
        self.ledger.append(("refund", round(amount, 2), self.balance))


# ------------------------------------------------------------- the order


class OrderItem(BaseModel):
    drink: str = Field(description="one of the menu drinks")
    size: str = Field(default="medium", description="small, medium or large")
    extras: list[str] = Field(default_factory=list, description="extras, e.g. oat milk")


class Order(BaseModel):
    items: list[OrderItem] = Field(description="the drinks ordered")
    food: list[str] = Field(
        default_factory=list, description="food items, e.g. croissant"
    )
    to_go: bool = Field(default=False, description="takeaway or not")


def price(item: OrderItem) -> float:
    """Real arithmetic for one drink. Raises on unknown names -- no guessing."""
    return round(
        MENU[item.drink] + SIZES[item.size] + sum(EXTRAS[e] for e in item.extras), 2
    )


def total(order: Order) -> float:
    return round(
        sum(price(i) for i in order.items) + sum(FOOD[f] for f in order.food), 2
    )


# --------------------------------------------------------------- the tools


def make_tools(stock: Stock):
    """The tool belt: real functions closed over a real Stock instance."""

    @tool
    def check_stock(item: str) -> str:
        """How many units of an ingredient or food item are left in the stock room."""
        return f"{stock.check(item)} left"

    @tool
    def get_menu() -> str:
        """The full menu board: drinks, sizes, extras, food, with prices."""
        return menu_board()

    @tool
    def compute_price(drink: str, size: str, extras: list[str]) -> str:
        """Price one drink: base price plus size plus extras."""
        p = price(OrderItem(drink=drink, size=size, extras=extras))
        return f"${p:.2f}"

    return check_stock, get_menu, compute_price


# ----------------------------------------------------------- the one fake


class ScriptExhausted(RuntimeError):
    """The scenario asked for more model turns than were scripted.

    A drifted scenario must kill the build, never ship a wrong figure.
    """


class ScriptedChatModel(GenericFakeChatModel):
    """The deck's single fake: a deterministic scripted model.

    Real langchain in every other respect -- the scripted AIMessages travel
    through the real invoke/stream/tool-calling/graph code paths. Two
    additions over GenericFakeChatModel:

    * ``calls`` records every message list the model was actually shown
      (the load-bearing hook for oracles: "the rendered menu really reached
      the model").
    * ``bind_tools`` records the bound tool schemas and returns self
      (the base class raises NotImplementedError; agents need it).
    """

    calls: list = Field(default_factory=list)
    bound_tools: list = Field(default_factory=list)

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(list(messages))
        try:
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except StopIteration:
            raise ScriptExhausted(
                "script exhausted -- the scenario and the code path disagree"
            ) from None

    def bind_tools(self, tools: Sequence, **kwargs: Any):
        self.bound_tools = list(tools)
        return self


def scripted(*msgs: str | AIMessage) -> ScriptedChatModel:
    """Build a ScriptedChatModel from plain strings and AIMessages."""
    out: list[AIMessage] = []
    for m in msgs:
        out.append(AIMessage(content=m) if isinstance(m, str) else m)
    return ScriptedChatModel(messages=iter(out))


def tool_call_msg(name: str, args: dict, call_id: str, content: str = "") -> AIMessage:
    """An AIMessage that requests one tool call, the shape real models emit."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------- self-check

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # the recurring number: a large oat-milk latte is 6.25, plus croissant 9.75
    maya = Order(
        items=[OrderItem(drink="latte", size="large", extras=["oat milk"])],
        food=["croissant"],
        to_go=True,
    )
    assert price(maya.items[0]) == 6.25
    assert total(maya) == 9.75

    stock = Stock()
    check_stock, get_menu, compute_price = make_tools(stock)
    assert check_stock.invoke({"item": "oat milk"}) == "2 left"
    assert "latte $4.50" in get_menu.invoke({})
    assert (
        compute_price.invoke(
            {"drink": "espresso", "size": "large", "extras": ["extra shot"]}
        )
        == "$5.00"
    )
    stock.take("oat milk")
    assert stock.check("oat milk") == 1

    till = Till()
    till.charge(9.75)
    till.refund(6.25)
    assert till.balance == 203.50
    assert till.ledger == [("charge", 9.75, 209.75), ("refund", 6.25, 203.50)]

    # the scripted model records contexts and replays deterministically
    m = scripted("hello Maya", tool_call_msg("check_stock", {"item": "oat milk"}, "c1"))
    r1 = m.invoke([HumanMessage(content="hi")])
    assert r1.content == "hello Maya" and m.calls[0][0].content == "hi"
    r2 = m.invoke([HumanMessage(content="got oat?")])
    assert r2.tool_calls[0]["name"] == "check_stock"
    try:
        m.invoke([HumanMessage(content="one too many")])
        raise AssertionError("script exhaustion did not raise")
    except ScriptExhausted:
        pass
    assert len(m.calls) == 3  # the fatal third call was still recorded

    # bind_tools records and returns self; streaming word-splits for real
    m2 = scripted("one large latte")
    assert m2.bind_tools([check_stock]) is m2 and m2.bound_tools == [check_stock]
    chunks = list(m2.stream([HumanMessage(content="read it back")]))
    assert "".join(c.content for c in chunks) == "one large latte"
    assert len(chunks) == 5  # words and spaces are separate chunks

    # the current agent constructor accepts the scripted model
    from langchain.agents import create_agent

    m3 = scripted(
        tool_call_msg("get_menu", {}, "c1"),
        "cheapest large drink with an extra shot: espresso, $5.00",
    )
    agent = create_agent(m3, [get_menu, compute_price])
    out = agent.invoke(
        {"messages": [HumanMessage(content="cheapest large + extra shot?")]}
    )
    kinds = [type(x).__name__ for x in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"], kinds
    assert out["messages"][2].content == menu_board()  # the tool really ran
    print("beanline self-check: all checks passed.")
