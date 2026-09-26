from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, Literal, TypeVar

import anyio
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator
from pydantic_ai import (
    CallDeferred, DeferredToolRequests, DeferredToolResults, ModelRetry, Tool, ToolApproved, ToolDenied,
)
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pushary import DecisionStatus, DecisionType
from pushary.adapters import (
    DENIED_UNANSWERED, AdapterKernel, ApprovalAsk, ApprovalGate, decision_fingerprint, is_affirmative,
    render_approval_question,
)

__version__ = "0.2.0"
__all__ = [
    "HumanQuestion", "HumanAnswer", "PendingReview", "ReviewBatch", "PusharyApprovals",
    "pushary_tool", "create_reviews", "resolve_reviews", "connect",
]

_kernel = AdapterKernel("the Pydantic AI helpers")
connect = _kernel.connect
_Identity = Annotated[str, StringConstraints(min_length=1, max_length=256, pattern=r"\S")]
_Option = Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"\S")]
_TEXT_MAX_LENGTH = 2000
_Text = Annotated[str, StringConstraints(max_length=_TEXT_MAX_LENGTH)]
CHOICE_OUTSIDE_OPTIONS = "The human's choice is outside the requested options."
ARGUMENTS_TOO_LONG = (
    "The arguments are too long for the approver to read in full, so this was denied without asking. "
    "Shorten them to ask again."
)


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
    context: _Text

    @field_validator("tool_name")
    @classmethod
    def preserve_tool_name(cls, value: str) -> str:
        if len(value.encode("utf-16-le")) > 200:
            raise ValueError("tool_name exceeds the API's 100 UTF-16 character limit.")
        return value


class _CreatedDecision(BaseModel):
    model_config = ConfigDict(strict=True)

    decisionId: _Identity


class _Answer(BaseModel):
    model_config = ConfigDict(strict=True)

    status: DecisionStatus
    answered: bool
    value: _Text | None

    @model_validator(mode="after")
    def valid_answer(self) -> _Answer:
        if self.answered != (self.status == "answered"):
            raise ValueError("Inconsistent decision status.")
        if self.answered and not self.value:
            raise ValueError("Answered decisions require a value.")
        return self


class _Decision(_CreatedDecision, _Answer):
    type: DecisionType
    question: str
    context: str | None = None
    options: list[str] | None = None
    externalId: str | None = None


ASK_HUMAN_TOOL = "ask_human"


def pushary_tool() -> Tool[object]:
    def ask_human(request: HumanQuestion) -> HumanAnswer:
        raise CallDeferred()

    return Tool(
        ask_human,
        description="Ask a human for yes/no confirmation, a choice, or a written answer on their phone.",
    )


def _arguments_context(args: dict[str, Any]) -> str:
    return f"Tool arguments: {json.dumps(args, sort_keys=True, allow_nan=False)}"


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
        if approval:
            question = HumanQuestion(question=f"Approve {call.tool_name}?")
            context = _arguments_context(args)
        else:
            if call.tool_name != ASK_HUMAN_TOOL:
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
            raise ValueError(CHOICE_OUTSIDE_OPTIONS)
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


_Result = TypeVar("_Result")
ExternalIdResolver = Callable[[RunContext[AgentDepsT]], "str | None | Awaitable[str | None]"]
_SettledStatus = Literal["answered", "expired", "cancelled"]
_MAX_CONCURRENT_WAITS = 100
_waits = anyio.CapacityLimiter(_MAX_CONCURRENT_WAITS)


async def _wait_in_thread(function: Callable[[], _Result]) -> _Result:
    return await anyio.to_thread.run_sync(function, abandon_on_cancel=True, limiter=_waits)


def _settled_status(status: DecisionStatus) -> _SettledStatus:
    if status == "pending":
        return "expired"
    return status


def _human_answer(question: HumanQuestion, answer: _Answer) -> HumanAnswer:
    value = answer.value if answer.answered else None
    if question.kind == "select" and value is not None and value not in (question.options or []):
        raise ValueError(CHOICE_OUTSIDE_OPTIONS)
    return HumanAnswer(
        kind=question.kind,
        status=_settled_status(answer.status),
        value=value,
        approved=(value is not None and is_affirmative(value)) if question.kind == "confirm" else None,
    )


def _question_retry(error: ValidationError) -> ModelRetry:
    return ModelRetry(f"The question could not be sent: {error.errors(include_url=False)[0]['msg']}")


class _GatedCall(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    external_id: str
    session_id: str


def _gated_call(call: ToolCallPart, external_id: str, session_id: str) -> _GatedCall:
    return _GatedCall(
        tool_name=call.tool_name, tool_call_id=call.tool_call_id, args=call.args_as_dict(),
        external_id=external_id, session_id=session_id,
    )


@dataclass
class PusharyApprovals(AbstractCapability[AgentDepsT]):
    external_id: str | ExternalIdResolver[AgentDepsT]
    agent_name: str | None = None
    timeout_seconds: int = 55
    require_reachable: bool | None = None
    policy: bool = True
    api_key: str | None = None
    base_url: str | None = None
    _gate: ApprovalGate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = "pushary_approvals"
        self._gate = _kernel.create_gate(
            api_key=self.api_key,
            base_url=self.base_url,
            agent_name=self.agent_name,
            expires_in_seconds=self.timeout_seconds,
            timeout_seconds=self.timeout_seconds,
            require_reachable=self.require_reachable,
            policy=self.policy,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        questions = [call for call in requests.calls if call.tool_name == ASK_HUMAN_TOOL]
        if not requests.approvals and not questions:
            return None
        external_id = _kernel.require_external_id(await self._external_id_for(ctx))
        session_id = ctx.run_id or ""
        results = DeferredToolResults()
        for call in requests.approvals:
            results.approvals[call.tool_call_id] = await self._ask_approval(
                ctx, _gated_call(call, external_id, session_id),
            )
        for call in questions:
            results.calls[call.tool_call_id] = await self._answer(ctx, _gated_call(call, external_id, session_id))
        return results

    async def _external_id_for(self, ctx: RunContext[AgentDepsT]) -> str | None:
        if not callable(self.external_id):
            return self.external_id
        resolved = self.external_id(ctx)
        if isinstance(resolved, Awaitable):
            return await resolved
        return resolved

    @durable_operation(name="ask_approval")
    async def _ask_approval(self, ctx: RunContext[AgentDepsT], call: _GatedCall) -> ToolApproved | ToolDenied:
        context = _arguments_context(call.args)
        if len(context) > _TEXT_MAX_LENGTH:
            return ToolDenied(ARGUMENTS_TOO_LONG)
        ask = ApprovalAsk(
            tool_name=call.tool_name,
            call_id=call.tool_call_id,
            session_id=call.session_id,
            question=render_approval_question(call.tool_name, call.args),
            external_id=call.external_id,
            input=call.args,
            context=context,
        )
        decision = await _wait_in_thread(partial(self._gate, ask))
        return ToolApproved() if decision.approved else ToolDenied(decision.reason or DENIED_UNANSWERED)

    async def _answer(self, ctx: RunContext[AgentDepsT], call: _GatedCall) -> dict[str, Any] | ModelRetry:
        try:
            question = HumanQuestion.model_validate(call.args)
        except ValidationError as error:
            return _question_retry(error)
        answer = await self._ask_question(ctx, call, question)
        return answer.model_dump()

    @durable_operation(name="ask_question")
    async def _ask_question(
        self, ctx: RunContext[AgentDepsT], call: _GatedCall, question: HumanQuestion,
    ) -> HumanAnswer:
        decision = await _wait_in_thread(partial(
            _kernel.ask_human,
            question.question,
            external_id=call.external_id,
            idempotency_key=decision_fingerprint(
                ["pydantic-ai", call.external_id, call.session_id, call.tool_call_id, ASK_HUMAN_TOOL, False, call.args],
            ),
            type=question.kind,
            options=question.options,
            node=ASK_HUMAN_TOOL,
            agent_name=self.agent_name,
            expires_in_seconds=self.timeout_seconds,
            require_reachable=self.require_reachable,
            timeout_seconds=self.timeout_seconds,
            api_key=self.api_key,
            base_url=self.base_url,
        ))
        return _human_answer(question, _Answer.model_validate(decision))
