from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import anyio
import pytest
from pushary import PusharyError
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, RunContext, models
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pushary_pydantic_ai import ARGUMENTS_TOO_LONG, PusharyApprovals, _wait_in_thread, pushary_tool

models.ALLOW_MODEL_REQUESTS = False
API_KEY = "pk_test.sk_test"

EVALUATIONS: dict[str, dict[str, Any]] = {
    "requires_human": {"verdict": "requires_human", "policy": None, "reason": "No rule names this action.", "authorizationId": None},
    "allow": {"verdict": "allow", "policy": "refund", "reason": "Allowed by policy rule refund.", "authorizationId": "az_1"},
    "deny": {"verdict": "deny", "policy": "refund", "reason": "Denied by policy rule refund.", "authorizationId": "az_2"},
}


@contextmanager
def phone(
    answers: list[str], verdict: str = "requires_human", failure: Exception | None = None,
) -> Iterator[list[tuple[str, str, dict[str, Any]]]]:
    requests: list[tuple[str, str, dict[str, Any]]] = []
    queue = list(answers)

    def request(_client: object, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        requests.append((method, path, deepcopy(kwargs)))
        if path == "/authorize":
            return EVALUATIONS[verdict]
        if method == "POST" and path == "/decisions":
            if failure is not None:
                raise failure
            body = kwargs["body"]
            value = queue.pop(0) if queue else None
            return {
                "decisionId": body["idempotencyKey"], "status": "answered" if value else "pending",
                "answered": value is not None, "value": value, "type": body["type"],
            }
        return {"decisionId": path.rsplit("/", 1)[1], "status": "pending", "answered": False, "value": None, "type": "confirm"}

    with patch("pushary.client.PusharyServer._request", request):
        yield requests


def decisions(requests: list[tuple[str, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [kwargs["body"] for method, path, kwargs in requests if method == "POST" and path == "/decisions"]


def approvals(timeout_seconds: int = 0, **overrides: Any) -> PusharyApprovals[None]:
    return PusharyApprovals(external_id="customer_1", api_key=API_KEY, timeout_seconds=timeout_seconds, **overrides)


def refund_agent(
    capability: PusharyApprovals[Any], model: Model | None = None,
) -> tuple[Agent[None, str], list[int]]:
    executions: list[int] = []
    agent: Agent[None, str] = Agent(model or TestModel(), capabilities=[capability])

    @agent.tool_plain(requires_approval=True)
    def refund(amount: int, note: str = "") -> str:
        executions.append(amount)
        return "refunded"

    return agent, executions


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def test_a_yes_on_the_phone_runs_the_tool_inside_the_same_run() -> None:
    with phone(["yes"]) as requests:
        agent, executions = refund_agent(approvals())
        result = agent.run_sync("Refund order 1234")
    assert executions == [0]
    assert isinstance(result.output, str)
    [asked] = decisions(requests)
    assert asked["type"] == "confirm"
    assert asked["externalId"] == "customer_1"
    assert asked["toolName"] == "refund"


def test_a_no_blocks_the_tool_and_tells_the_model() -> None:
    with phone(["no"]):
        agent, executions = refund_agent(approvals())
        result = agent.run_sync("Refund order 1234")
    assert executions == []
    [denied] = tool_returns(result.all_messages())
    assert "denied" in str(denied.content).lower()


def test_no_answer_never_reads_as_a_yes() -> None:
    with phone([]):
        agent, executions = refund_agent(approvals())
        agent.run_sync("Refund order 1234")
    assert executions == []


def test_a_rule_that_allows_skips_the_phone() -> None:
    with phone([], verdict="allow") as requests:
        agent, executions = refund_agent(approvals())
        agent.run_sync("Refund order 1234")
    assert executions == [0]
    assert decisions(requests) == []


def test_a_rule_that_denies_blocks_without_paging_anyone() -> None:
    with phone(["yes"], verdict="deny") as requests:
        agent, executions = refund_agent(approvals())
        result = agent.run_sync("Refund order 1234")
    assert executions == []
    assert decisions(requests) == []
    [denied] = tool_returns(result.all_messages())
    assert "Denied by policy rule refund." in str(denied.content)


def test_every_approval_in_one_turn_is_asked() -> None:
    executions: list[str] = []
    with phone(["yes", "yes"]) as requests:
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[approvals()])

        @agent.tool_plain(requires_approval=True)
        def refund(amount: int) -> str:
            executions.append("refund")
            return "refunded"

        @agent.tool_plain(requires_approval=True)
        def cancel(order: str) -> str:
            executions.append("cancel")
            return "cancelled"

        agent.run_sync("Refund and cancel")
    assert sorted(executions) == ["cancel", "refund"]
    assert sorted(body["toolName"] for body in decisions(requests)) == ["cancel", "refund"]


def call_then_finish(tool_name: str, args: dict[str, Any]) -> FunctionModel:
    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returned = tool_returns(messages)
        retried = [part for message in messages for part in message.parts if isinstance(part, RetryPromptPart)]
        if returned:
            return ModelResponse(parts=[TextPart(str(returned[-1].content))])
        if retried:
            return ModelResponse(parts=[TextPart("gave up")])
        return ModelResponse(parts=[ToolCallPart(tool_name, args, f"{tool_name}_call")])

    return FunctionModel(respond)


def ask_then_finish(args: dict[str, Any]) -> FunctionModel:
    return call_then_finish("ask_human", args)


def test_ask_human_gets_the_person_answer_back() -> None:
    with phone(["Monday"]) as requests:
        agent: Agent[None, str] = Agent(
            ask_then_finish({"question": "Ship on which day?", "kind": "select", "options": ["Friday", "Monday"]}),
            tools=[pushary_tool()],
            capabilities=[approvals()],
        )
        result = agent.run_sync("Plan the release")
    [returned] = tool_returns(result.all_messages())
    assert returned.content == {"kind": "select", "status": "answered", "value": "Monday", "approved": None}
    [asked] = decisions(requests)
    assert asked["type"] == "select"
    assert asked["options"] == ["Friday", "Monday"]
    assert not any(path == "/authorize" for _method, path, _kwargs in requests)


def test_an_unreadable_question_goes_back_to_the_model_without_paging_anyone() -> None:
    with phone(["only"]) as requests:
        agent: Agent[None, str] = Agent(
            ask_then_finish({"question": "Pick one?", "kind": "select", "options": ["only"]}),
            tools=[pushary_tool()],
            capabilities=[approvals()],
        )
        result = agent.run_sync("Plan the release")
    assert result.output == "gave up"
    assert decisions(requests) == []


def test_other_deferred_calls_are_left_for_the_caller_without_looking_up_anyone() -> None:
    agent: Agent[None, str | DeferredToolRequests] = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[PusharyApprovals(external_id=lambda _ctx: None, api_key=API_KEY)],
    )

    @agent.tool_plain
    def run_elsewhere(job: str) -> str:
        raise CallDeferred()

    with phone([]) as requests:
        result = agent.run_sync("Run the job")
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.calls] == ["run_elsewhere"]
    assert decisions(requests) == []


@dataclass
class Customer:
    user_id: str


def test_the_person_is_resolved_per_run_from_trusted_deps() -> None:
    with phone(["yes"]) as requests:
        capability: PusharyApprovals[Customer] = PusharyApprovals(
            external_id=lambda ctx: ctx.deps.user_id, api_key=API_KEY, timeout_seconds=0,
        )
        agent: Agent[Customer, str] = Agent(TestModel(), deps_type=Customer, capabilities=[capability])

        @agent.tool_plain(requires_approval=True)
        def refund(amount: int) -> str:
            return "refunded"

        agent.run_sync("Refund", deps=Customer(user_id="customer_42"))
    assert decisions(requests)[0]["externalId"] == "customer_42"


def test_a_missing_key_fails_where_the_capability_is_defined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUSHARY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="PUSHARY_API_KEY"):
        PusharyApprovals(external_id="customer_1")


def test_nobody_to_ask_refuses_rather_than_approves() -> None:
    with phone(["yes"]), pytest.raises(ValueError, match="no end-user to ask"):
        agent, executions = refund_agent(PusharyApprovals(external_id=lambda _ctx: None, api_key=API_KEY))
        agent.run_sync("Refund order 1234")
    assert executions == []


def test_the_approver_sees_every_argument_when_the_question_is_cut_short() -> None:
    note = "x" * 400
    with phone(["yes"]) as requests:
        agent, executions = refund_agent(approvals(), call_then_finish("refund", {"note": note, "amount": 99999}))
        agent.run_sync("Refund order 1234")
    [asked] = decisions(requests)
    assert asked["question"].endswith("...")
    assert "99999" not in asked["question"]
    assert asked["context"] == f'Tool arguments: {{"amount": 99999, "note": "{note}"}}'
    assert executions == [99999]


def test_arguments_too_long_to_show_in_full_are_denied_without_paging_anyone() -> None:
    with phone(["yes"]) as requests:
        agent, executions = refund_agent(approvals(), call_then_finish("refund", {"note": "x" * 2000, "amount": 5}))
        result = agent.run_sync("Refund order 1234")
    assert executions == []
    assert requests == []
    [denied] = tool_returns(result.all_messages())
    assert denied.content == ARGUMENTS_TOO_LONG


def test_a_failed_request_surfaces_as_itself_and_nobody_else_is_asked() -> None:
    with phone(["yes", "yes"], failure=PusharyError("Payment required", 402)) as requests:
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[approvals()])

        @agent.tool_plain(requires_approval=True)
        def refund(amount: int) -> str:
            return "refunded"

        @agent.tool_plain(requires_approval=True)
        def cancel(order: str) -> str:
            return "cancelled"

        with pytest.raises(PusharyError, match="Payment required"):
            agent.run_sync("Refund and cancel")
    assert len(decisions(requests)) == 1


def test_the_prompt_closes_when_the_wait_ends() -> None:
    with phone(["yes", "Monday"]) as requests:
        agent, _executions = refund_agent(approvals(timeout_seconds=30))
        agent.run_sync("Refund order 1234")
        Agent(
            ask_then_finish({"question": "Ship on which day?", "kind": "select", "options": ["Friday", "Monday"]}),
            tools=[pushary_tool()],
            capabilities=[approvals(timeout_seconds=30)],
        ).run_sync("Plan the release")
    assert [body["expiresInSeconds"] for body in decisions(requests)] == [30, 30]


def test_a_question_nobody_answered_in_time_reads_as_expired() -> None:
    with phone([]):
        agent: Agent[None, str] = Agent(
            ask_then_finish({"question": "Ship today?"}), tools=[pushary_tool()], capabilities=[approvals()],
        )
        result = agent.run_sync("Plan the release")
    [returned] = tool_returns(result.all_messages())
    assert returned.content == {"kind": "confirm", "status": "expired", "value": None, "approved": False}


def test_a_choice_outside_the_offered_options_is_refused() -> None:
    with phone(["Sunday"]), pytest.raises(ValueError, match="outside the requested options"):
        Agent(
            ask_then_finish({"question": "Ship on which day?", "kind": "select", "options": ["Friday", "Monday"]}),
            tools=[pushary_tool()],
            capabilities=[approvals()],
        ).run_sync("Plan the release")


def test_an_async_resolver_is_awaited() -> None:
    async def customer(ctx: RunContext[Customer]) -> str:
        return ctx.deps.user_id

    with phone(["yes"]) as requests:
        capability: PusharyApprovals[Customer] = PusharyApprovals(
            external_id=customer, api_key=API_KEY, timeout_seconds=0,
        )
        agent: Agent[Customer, str] = Agent(TestModel(), deps_type=Customer, capabilities=[capability])

        @agent.tool_plain(requires_approval=True)
        def refund(amount: int) -> str:
            return "refunded"

        agent.run_sync("Refund", deps=Customer(user_id="customer_7"))
    assert decisions(requests)[0]["externalId"] == "customer_7"


def test_a_waiting_approval_leaves_the_app_thread_pool_free() -> None:
    asked = threading.Event()
    release = threading.Event()

    def request(_client: object, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/authorize":
            return EVALUATIONS["requires_human"]
        asked.set()
        release.wait(5)
        return {
            "decisionId": kwargs["body"]["idempotencyKey"], "status": "answered", "answered": True,
            "value": "yes", "type": "confirm",
        }

    async def scenario() -> list[int]:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        agent, executions = refund_agent(approvals())
        async with anyio.create_task_group() as group:
            group.start_soon(agent.run, "Refund order 1234")
            while not asked.is_set():
                await anyio.sleep(0.01)
            with anyio.fail_after(2):
                assert await anyio.to_thread.run_sync(lambda: "free") == "free"
            release.set()
        return executions

    with patch("pushary.client.PusharyServer._request", request):
        assert anyio.run(scenario) == [0]


def test_run_stream_events_asks_on_the_phone() -> None:
    async def scenario() -> list[int]:
        agent, executions = refund_agent(approvals())
        async with agent.run_stream_events("Refund order 1234") as stream:
            async for _event in stream:
                pass
        return executions

    with phone(["yes"]) as requests:
        assert anyio.run(scenario) == [0]
    assert len(decisions(requests)) == 1


def test_run_stream_ends_before_approvals_so_the_tool_never_runs() -> None:
    async def scenario() -> list[int]:
        agent, executions = refund_agent(approvals())
        with pytest.raises(UserError, match="DeferredToolRequests"):
            async with agent.run_stream("Refund order 1234") as stream:
                await stream.get_output()
        return executions

    with phone(["yes"]) as requests:
        assert anyio.run(scenario) == []
    assert decisions(requests) == []


def test_cancelled_waits_keep_their_workers_and_cancelled_queued_calls_never_start() -> None:
    release = threading.Event()
    lock = threading.Lock()
    started = 0

    def blocked() -> None:
        nonlocal started
        with lock:
            started += 1
        release.wait(10)

    async def scenario() -> None:
        try:
            for index in range(100):
                task = asyncio.create_task(_wait_in_thread(blocked))
                with anyio.fail_after(2):
                    while started <= index:
                        await asyncio.sleep(0.001)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            queued = asyncio.create_task(_wait_in_thread(blocked))
            await asyncio.sleep(0.05)
            assert started == 100
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            release.set()
            await _wait_in_thread(lambda: None)
            assert started == 100
        finally:
            release.set()

    asyncio.run(scenario())


def test_worker_receives_the_callers_context_without_changing_it() -> None:
    request_id: ContextVar[str] = ContextVar("request_id", default="unset")

    def worker() -> str:
        received = request_id.get()
        request_id.set("worker-only")
        return received

    async def scenario() -> None:
        request_id.set("customer-request")
        assert await _wait_in_thread(worker) == "customer-request"
        assert request_id.get() == "customer-request"

    asyncio.run(scenario())
