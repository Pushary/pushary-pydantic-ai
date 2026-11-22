from __future__ import annotations

from pydantic_ai.messages import ModelMessage, RetryPromptPart, ToolReturnPart


def is_retry(part: object) -> bool:
    return isinstance(part, RetryPromptPart) or (isinstance(part, ToolReturnPart) and getattr(part, "outcome", None) == "retried")


def retries(messages: list[ModelMessage]) -> list[object]:
    return [part for message in messages for part in message.parts if is_retry(part)]


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart) and not is_retry(part)]
