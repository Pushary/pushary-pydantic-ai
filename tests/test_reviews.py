from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent, DeferredToolRequests, ToolDenied, models
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pushary import PusharyServer

from pushary_pydantic_ai import HumanQuestion, ReviewBatch, create_reviews, pushary_tool, resolve_reviews

models.ALLOW_MODEL_REQUESTS = False
API_KEY = "pk_test.sk_test"


@contextmanager
def decision_transport():
    records = {}
    requests = []

    def request(_client, method, path, **kwargs):
        requests.append((method, path, deepcopy(kwargs)))
        if method == "POST":
            body = kwargs["body"]
            key = body["idempotencyKey"]
            records.setdefault(key, {
                "decisionId": key, "status": "pending", "answered": False, "value": None,
                "type": body["type"], "question": body["question"], "options": body.get("options"),
                "context": body.get("context"),
                "externalId": body["externalId"],
            })
            return records[key]
        return records[path.rsplit("/", 1)[1]]

    with patch("pushary.client.PusharyServer._request", request):
        yield records, requests


def approval_request(amount=10, call_id="refund_call"):
    return DeferredToolRequests(approvals=[ToolCallPart("refund", {"amount": amount}, call_id)])


def question_request(kind="confirm", options=None):
    args = {"question": "What should we do?", "kind": kind}
    if options is not None:
        args["options"] = options
    return DeferredToolRequests(calls=[ToolCallPart("ask_human", args, "question_call")])


def create(requests, run_id="run_1", external_id="customer_1"):
    return create_reviews(requests, run_id=run_id, external_id=external_id, api_key=API_KEY)


def resolve(requests, batch):
    return resolve_reviews(requests, batch, api_key=API_KEY)


@pytest.mark.parametrize("value,executed", [("yes", True), ("no", False)])
def test_native_approval_pauses_restores_and_resumes(value, executed):
    executions = []
    agent = Agent(TestModel(custom_output_text="Done"), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def refund(amount: int) -> str:
        executions.append(amount)
        return "refunded"

    result = agent.run_sync("Refund")
    assert isinstance(result.output, DeferredToolRequests)
    assert not executions
    with decision_transport() as (records, transport):
        batch = create(result.output, run_id=result.run_id)
        assert resolve(result.output, batch) is None
        assert transport[0][2]["body"]["wait"] is False
        stored_batch = batch.model_dump_json()
        stored_requests = TypeAdapter(DeferredToolRequests).dump_json(result.output)
        stored_messages = result.all_messages_json()
        records[batch.reviews[0].decision_id].update(status="answered", answered=True, value=value)
        restored = TypeAdapter(DeferredToolRequests).validate_json(stored_requests)
        answers = resolve(restored, ReviewBatch.model_validate_json(stored_batch))
        assert answers is not None
        resumed = agent.run_sync(
            message_history=ModelMessagesTypeAdapter.validate_json(stored_messages),
            deferred_tool_results=answers,
        )
        assert resumed.output == "Done"
        assert bool(executions) is executed


@pytest.mark.parametrize("kind,options,value,approved", [
    ("confirm", None, "yes", True),
    ("confirm", None, "no", False),
    ("select", ["yes", "later"], "yes", None),
    ("input", None, "yes", None),
    ("input", None, "Use quantity 12", None),
])
def test_real_question_tool_returns_typed_data_never_approves_another_tool(kind, options, value, approved):
    args = question_request(kind, options).calls[0].args_as_dict()

    def model(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("Answered")])
        return ModelResponse(parts=[ToolCallPart("ask_human", args, "question_call")])

    tool = pushary_tool()
    assert "external_id" not in tool.function_schema.json_schema["properties"]
    assert tool.function_schema.json_schema["additionalProperties"] is False
    assert tool.function_schema.json_schema["properties"]["kind"]["enum"] == ["confirm", "select", "input"]
    agent = Agent(FunctionModel(model), tools=[tool], output_type=[str, DeferredToolRequests])
    initial = agent.run_sync("Ask the customer")
    assert isinstance(initial.output, DeferredToolRequests)
    with decision_transport() as (records, _):
        batch = create(initial.output, run_id=initial.run_id)
        records[batch.reviews[0].decision_id].update(status="answered", answered=True, value=value)
        answers = resolve(initial.output, batch)
        assert answers.approvals == {}
        assert answers.calls["question_call"] == {
            "kind": kind, "status": "answered", "value": value, "approved": approved,
        }
        assert agent.run_sync(message_history=initial.all_messages(), deferred_tool_results=answers).output == "Answered"


@pytest.mark.parametrize("args", [
    {"question": " "},
    {"question": "Q", "kind": "unknown"},
    {"question": "Q", "kind": "select"},
    {"question": "Q", "kind": "select", "options": ["one"]},
    {"question": "Q", "kind": "select", "options": ["same", "same"]},
    {"question": "Q", "kind": "input", "options": ["one", "two"]},
    {"question": "Q", "kind": "confirm", "external_id": "attacker"},
    {"question": "Q" * 501},
])
def test_question_schema_rejects_invalid_requests_before_network(args):
    with pytest.raises(ValidationError):
        HumanQuestion.model_validate(args)
    with decision_transport() as (_, transport):
        requests = DeferredToolRequests(calls=[ToolCallPart("ask_human", args, "question")])
        with pytest.raises(ValidationError):
            create(requests)
        assert transport == []


def test_retry_identity_is_scoped_to_run_customer_call_and_full_arguments():
    with decision_transport() as (records, _):
        first = create(approval_request())
        assert create(approval_request()) == first
        assert create(approval_request(), run_id="run_2") != first
        assert create(approval_request(), external_id="customer_2") != first
        assert create(approval_request(11)) != first
        assert create(approval_request(call_id="call_2")) != first
        assert len(records) == 5


@pytest.mark.parametrize("external_id,run_id", [("", "run"), (" ", "run"), ("u" * 257, "run"), ("🙂" * 129, "run"), ("u", "")])
def test_identity_validation_before_network(external_id, run_id):
    with decision_transport() as (_, transport):
        with pytest.raises(ValidationError):
            create(approval_request(), external_id=external_id, run_id=run_id)
        assert not transport


def test_changed_call_and_foreign_batches_fail_before_reading():
    with decision_transport() as (_, transport):
        batch = create(approval_request())
        count = len(transport)
        for requests in (approval_request(99), question_request(), DeferredToolRequests()):
            with pytest.raises(ValueError):
                resolve(requests, batch)
        assert len(transport) == count


@pytest.mark.parametrize("status", ["expired", "cancelled"])
def test_unanswered_terminal_reviews_deny_tools_and_return_question_status(status):
    requests = approval_request()
    requests.calls = question_request("input").calls
    with decision_transport() as (records, _):
        batch = create(requests)
        for decision in records.values():
            decision.update(status=status)
        answers = resolve(requests, batch)
        assert isinstance(answers.approvals["refund_call"], ToolDenied)
        assert answers.calls["question_call"] == {"kind": "input", "status": status, "value": None, "approved": None}


@pytest.mark.parametrize("corruption", [
    {"decisionId": "another_decision"},
    {"type": "input"},
    {"externalId": "another_customer"},
    {"question": "Another question"},
    {"context": "Tool arguments: {\"amount\": 999}"},
    {"context": None},
    {"options": ["one", "two"]},
    {"status": "answered", "answered": False, "value": "yes"},
    {"status": "answered", "answered": "true", "value": "yes"},
    {"status": "answered", "answered": True, "value": None},
])
def test_invalid_or_mismatched_api_responses_never_resume(corruption):
    requests = approval_request()
    with decision_transport() as (records, _):
        batch = create(requests)
        records[batch.reviews[0].decision_id].update(corruption)
        with pytest.raises(ValueError):
            resolve(requests, batch)


def test_invalid_selection_never_resumes_and_nullable_live_recipient_is_supported():
    requests = question_request("select", ["one", "two"])
    with decision_transport() as (records, _):
        batch = create(requests)
        records[batch.reviews[0].decision_id].update(status="answered", answered=True, value="three", externalId=None)
        with pytest.raises(ValueError, match="outside"):
            resolve(requests, batch)
        records[batch.reviews[0].decision_id]["value"] = "two"
        assert resolve(requests, batch).calls["question_call"]["value"] == "two"


def test_unknown_external_calls_duplicates_and_oversize_approval_fail_before_any_create():
    unknown = approval_request()
    unknown.calls.append(ToolCallPart("other_external_tool", {}, "external"))
    duplicate = approval_request()
    duplicate.calls.append(ToolCallPart("ask_human", {"question": "Q"}, "refund_call"))
    oversized = DeferredToolRequests(approvals=[ToolCallPart("refund", {"data": "x" * 2000}, "refund")])
    for requests in (unknown, duplicate, oversized):
        with decision_transport() as (_, transport):
            with pytest.raises(ValueError):
                create(requests)
            assert not transport


def test_pending_batch_waits_for_all_answers_and_api_errors_propagate():
    requests = approval_request()
    requests.calls = question_request("input").calls
    with decision_transport() as (records, _):
        batch = create(requests)
        records[batch.reviews[0].decision_id].update(status="answered", answered=True, value="yes")
        assert resolve(requests, batch) is None
        with patch("pushary.client.PusharyServer._request", side_effect=TimeoutError("offline")):
            with pytest.raises(TimeoutError):
                resolve(requests, batch)


def test_partial_create_retry_reuses_existing_decisions():
    requests = approval_request()
    requests.calls = question_request("input").calls
    with decision_transport() as (records, transport):
        original = PusharyServer._request

        def fail_second(client, method, path, **kwargs):
            if len(transport) == 1:
                raise TimeoutError("second create unavailable")
            return original(client, method, path, **kwargs)

        with patch("pushary.client.PusharyServer._request", fail_second):
            with pytest.raises(TimeoutError):
                create(requests)
        assert len(records) == 1
        batch = create(requests)
        assert len(records) == 2
        assert len(batch.reviews) == 2


@pytest.mark.parametrize("args", [{1: "one"}, {"items": (1, 2)}, {"value": float("nan")}, {"value": object()}])
def test_shared_fingerprint_rejects_non_json_approval_arguments(args):
    requests = DeferredToolRequests(approvals=[ToolCallPart("refund", args, "refund_call")])
    with decision_transport() as (_, transport):
        with pytest.raises(ValueError, match="JSON"):
            create(requests)
        assert not transport


def test_creation_forwards_expiry_and_reachability_requirements():
    with decision_transport() as (_, transport):
        create_reviews(
            approval_request(), external_id="customer_1", run_id="run_1", api_key=API_KEY,
            expires_in_seconds=3600, require_reachable=True,
        )
        assert transport[0][2]["body"]["expiresInSeconds"] == 3600
        assert transport[0][2]["body"]["requireReachable"] is True
