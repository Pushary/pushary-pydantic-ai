from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from pydantic_ai import CallDeferred, DeferredToolRequests, DeferredToolResults, Tool, ToolDenied
from pydantic_ai.messages import ToolCallPart
from pushary import DecisionStatus, DecisionType
from pushary.adapters import AdapterKernel, decision_fingerprint, is_affirmative

__version__ = "0.1.1"
__all__ = [
    "HumanQuestion", "HumanAnswer", "PendingReview", "ReviewBatch",
    "pushary_tool", "create_reviews", "resolve_reviews", "connect",
]

_kernel = AdapterKernel("the Pydantic AI helpers")
connect = _kernel.connect
_Identity = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]
_Option = Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"\S")]


class HumanQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    question: Annotated[str, StringConstraints(min_length=1, max_length=500, pattern=r"\S")]
    kind: DecisionType = "confirm"
    options: Annotated[list[_Option], Field(min_length=2, max_length=20)] | None = None

    @model_validator(mode="after")
    def valid_options(self) -> HumanQuestion:
        if self.kind == "select":
            if self.options is None or len(set(self.options)) != len(self.options):
                raise ValueError("Select questions require distinct options.")
        elif self.options is not None:
            raise ValueError("Only select questions accept options.")
        return self


class HumanAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: DecisionType
    status: Literal["answered", "expired", "cancelled"]
    value: str | None
    approved: bool | None


class PendingReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    call_id: _Identity
    fingerprint: _Identity
    decision_id: _Identity


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: _Identity
    external_id: _Identity
    reviews: list[PendingReview]

    @field_validator("external_id")
    @classmethod
    def preserve_recipient(cls, value: str) -> str:
        if len(value.encode("utf-16-le")) > 512:
            raise ValueError("external_id exceeds the API's 256 UTF-16 character limit.")
        return value


class _Review(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    call_id: _Identity
    tool_name: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    fingerprint: str
    approval: bool
    request: HumanQuestion
    context: Annotated[str, StringConstraints(max_length=2000)]

    @field_validator("tool_name")
    @classmethod
    def preserve_tool_name(cls, value: str) -> str:
        if len(value.encode("utf-16-le")) > 200:
            raise ValueError("tool_name exceeds the API's 100 UTF-16 character limit.")
        return value


class _CreatedDecision(BaseModel):
    model_config = ConfigDict(strict=True)

    decisionId: _Identity


class _Decision(_CreatedDecision):
    status: DecisionStatus
    answered: bool
    type: DecisionType
    question: str
    context: str | None = None
    options: list[str] | None = None
    externalId: str | None = None
    value: Annotated[str, StringConstraints(max_length=2000)] | None

    @model_validator(mode="after")
    def valid_answer(self) -> _Decision:
        if self.answered != (self.status == "answered"):
            raise ValueError("Inconsistent decision status.")
        if self.answered and not self.value:
            raise ValueError("Answered decisions require a value.")
        return self


def pushary_tool() -> Tool[object]:
    def ask_human(request: HumanQuestion) -> HumanAnswer:
        raise CallDeferred()

    return Tool(
        ask_human,
        description="Ask a human for yes/no confirmation, a choice, or a written answer on their phone.",
    )


def _reviews(requests: DeferredToolRequests, run_id: str, external_id: str) -> list[_Review]:
    ReviewBatch(run_id=run_id, external_id=external_id, reviews=[])
    reviews: list[_Review] = []
    seen: set[str] = set()
    entries: list[tuple[ToolCallPart, bool]] = [
        *((call, True) for call in requests.approvals),
        *((call, False) for call in requests.calls),
    ]
    for call, approval in entries:
        if call.tool_call_id in seen:
            raise ValueError("Deferred tool call IDs must be unique.")
        seen.add(call.tool_call_id)
        args = call.args_as_dict()
        fingerprint = decision_fingerprint([
            "pydantic-ai", external_id, run_id, call.tool_call_id, call.tool_name, approval, args,
        ])
        serialized_args = json.dumps(args, sort_keys=True, allow_nan=False)
        if approval:
            question = HumanQuestion(question=f"Approve {call.tool_name}?")
            context = f"Tool arguments: {serialized_args}"
        else:
            if call.tool_name != "ask_human":
                raise ValueError(f"Pushary cannot resolve external tool {call.tool_name!r}.")
            question = HumanQuestion.model_validate(args)
            context = ""
        reviews.append(_Review(
            call_id=call.tool_call_id, tool_name=call.tool_name, fingerprint=fingerprint,
            approval=approval, request=question, context=context,
        ))
    return reviews


def create_reviews(
    requests: DeferredToolRequests,
    *,
    external_id: str,
    run_id: str,
    callback_url: str | None = None,
    agent_name: str | None = None,
    expires_in_seconds: int | None = None,
    require_reachable: bool | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ReviewBatch:
    reviews = _reviews(requests, run_id, external_id)
    pending: list[PendingReview] = []
    for review in reviews:
        created = _CreatedDecision.model_validate(_kernel.create_durable_decision(
            review.request.question,
            external_id=external_id,
            idempotency_key=review.fingerprint,
            callback_url=callback_url,
            type=review.request.kind,
            options=review.request.options,
            node=review.tool_name,
            context=review.context or None,
            agent_name=agent_name,
            expires_in_seconds=expires_in_seconds,
            require_reachable=require_reachable,
            api_key=api_key,
            base_url=base_url,
        ))
        pending.append(PendingReview(
            call_id=review.call_id, fingerprint=review.fingerprint, decision_id=created.decisionId,
        ))
    return ReviewBatch(run_id=run_id, external_id=external_id, reviews=pending)


def resolve_reviews(
    requests: DeferredToolRequests,
    batch: ReviewBatch,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> DeferredToolResults | None:
    expected = _reviews(requests, batch.run_id, batch.external_id)
    pending = {review.call_id: review for review in batch.reviews}
    if len(pending) != len(batch.reviews) or set(pending) != {review.call_id for review in expected}:
        raise ValueError("Saved reviews do not match the deferred tool calls.")
    if len({review.decision_id for review in batch.reviews}) != len(batch.reviews):
        raise ValueError("Each deferred call must have its own decision.")
    for review in expected:
        if pending[review.call_id].fingerprint != review.fingerprint:
            raise ValueError("The reviewed tool call changed; create a new review.")
    client = _kernel.client(api_key, base_url)
    results = DeferredToolResults()
    waiting = False
    for review in expected:
        saved = pending[review.call_id]
        decision = _Decision.model_validate(client.decisions.get(saved.decision_id))
        if (
            decision.decisionId != saved.decision_id
            or decision.type != review.request.kind
            or decision.question != review.request.question
            or (decision.context or "") != review.context
            or decision.options != review.request.options
            or decision.externalId not in (None, batch.external_id)
        ):
            raise ValueError("The returned decision does not match the saved review.")
        if decision.status == "pending":
            waiting = True
            continue
        value = decision.value if decision.answered else None
        if decision.answered and decision.type == "select" and value not in (review.request.options or []):
            raise ValueError("The human's choice is outside the requested options.")
        approved = decision.answered and is_affirmative(value)
        if review.approval:
            results.approvals[review.call_id] = True if approved else ToolDenied(
                "The human declined." if decision.answered else f"The review {decision.status}; do not proceed."
            )
        else:
            results.calls[review.call_id] = HumanAnswer(
                kind=decision.type, status=decision.status, value=value,
                approved=approved if decision.type == "confirm" else None,
            ).model_dump()
    return None if waiting else results
