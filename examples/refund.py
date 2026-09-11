from __future__ import annotations

from unittest.mock import patch

from pydantic import TypeAdapter
from pydantic_ai import Agent, DeferredToolRequests, models
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.test import TestModel

from pushary_pydantic_ai import ReviewBatch, create_reviews, resolve_reviews


def demo(approve: bool) -> None:
    executions: list[str] = []
    decisions: dict[str, dict[str, object]] = {}
    models.ALLOW_MODEL_REQUESTS = False
    agent = Agent(TestModel(custom_output_text="Finished."), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def issue_refund(order_id: str) -> str:
        executions.append(order_id)
        return "Refund recorded."

    def transport(_client: object, method: str, path: str, **kwargs: object) -> dict[str, object]:
        if method == "POST":
            body = kwargs["body"]
            assert isinstance(body, dict)
            key = str(body["idempotencyKey"])
            decisions.setdefault(key, {
                "decisionId": key, "type": body["type"], "question": body["question"],
                "options": body.get("options"), "externalId": body["externalId"],
                "context": body.get("context"),
                "status": "pending", "answered": False, "value": None,
            })
            return decisions[key]
        return decisions[path.rsplit("/", 1)[1]]

    with patch("pushary.client.PusharyServer._request", transport):
        result = agent.run_sync("Refund an order.")
        assert isinstance(result.output, DeferredToolRequests)
        assert executions == []
        batch = create_reviews(result.output, external_id="customer_demo", run_id=result.run_id, api_key="pk_demo.sk_demo")
        assert resolve_reviews(result.output, batch, api_key="pk_demo.sk_demo") is None
        saved_messages = result.all_messages_json()
        saved_requests = TypeAdapter(DeferredToolRequests).dump_json(result.output)
        saved_batch = batch.model_dump_json()

        for decision in decisions.values():
            decision.update(status="answered", answered=True, value="yes" if approve else "no")

        requests = TypeAdapter(DeferredToolRequests).validate_json(saved_requests)
        restored = ReviewBatch.model_validate_json(saved_batch)
        answers = resolve_reviews(requests, restored, api_key="pk_demo.sk_demo")
        assert answers is not None
        agent.run_sync(
            message_history=ModelMessagesTypeAdapter.validate_json(saved_messages),
            deferred_tool_results=answers,
        )
        assert len(executions) == int(approve)
        print(f"Simulated phone {'approval' if approve else 'denial'}: {len(executions)} refund executions.")


if __name__ == "__main__":
    demo(True)
    demo(False)
