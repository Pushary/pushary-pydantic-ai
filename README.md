# pushary-pydantic-ai

Your Pydantic AI agent asks. Your customer answers on their phone. The agent continues with the recorded answer.

Native tool approvals and `confirm`, `select`, and `input` questions, using Pydantic AI's deferred tools. Pushary's native mobile app is the main customer experience; existing Slack delivery and legacy web compatibility remain available. The adapter creates a decision and returns immediately, so the human wait holds no worker open.

This package is prepared for its first release and is not yet published. From the adapter package directory, install from this checkout:

```sh
uv pip install -e .
```

Requires Python 3.10+, Pydantic AI 2.42+, and the public Pushary SDK 2.1+. The source install resolves the released core SDK from PyPI. Model provider extras belong to your application; the adapter depends on the slim framework package.

## Approve a tool before it runs

```python
from pydantic_ai import Agent, DeferredToolRequests
from pushary_pydantic_ai import create_reviews, resolve_reviews

agent = Agent("openai:gpt-4.1-mini", output_type=[str, DeferredToolRequests])

@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str) -> str:
    return refund_order_once(order_id)

result = await agent.run("Refund order_123")
if isinstance(result.output, DeferredToolRequests):
    batch = create_reviews(
        result.output,
        external_id=authenticated_customer.id,
        run_id=result.run_id,
        agent_name="Support",
    )
```

`refund_order_once` is your application's idempotent business operation. `authenticated_customer` comes from your server's authentication, never a model argument. Set `PUSHARY_API_KEY` on the server or pass `api_key=` to the helpers. Live end-user delivery requires Partner access and an enrolled customer; `connect(external_id)` returns the SDK's single-use enrollment link. Use a customer-bound key when available and keep it scoped to that same customer.

Save the original run's message history, deferred requests, and `batch.model_dump_json()` in trusted storage **before scheduling further work**. Persist the original run ID; a retry of decision creation must reuse that ID, the same customer, and the same original requests. Changed arguments produce a new decision rather than reusing approval.

Later, from a worker or callback handler:

```python
from pushary_pydantic_ai import ReviewBatch

batch = ReviewBatch.model_validate_json(saved_batch_json)
answers = resolve_reviews(saved_requests, batch)
if answers is not None:
    result = await agent.run(
        message_history=saved_messages,
        deferred_tool_results=answers,
    )
```

`saved_requests` and `saved_messages` must come from that original run, not from a model or browser. Pydantic's `TypeAdapter(DeferredToolRequests)` and `ModelMessagesTypeAdapter` support JSON round trips; the runnable example shows both. The follow-up is a new Pydantic run; retain your application's conversation identity separately.

Set `expires_in_seconds=` on creation to choose the review lifetime, subject to server limits. Set `require_reachable=True` to fail creation when the customer has no reachable delivery channel. These options are forwarded to the shared SDK; they do not create another scheduler or notification service.

`resolve_reviews` reads each saved decision through the authenticated SDK once, checking its question, options, type, customer when present, and full approval context against the saved request. It returns `None` while any decision remains pending. It does not silently turn an unfinished answer into rejection. Once all answers are terminal, a native approval receives approval only for an answered confirm decision with an affirmative value. Declines, expiration, and cancellation return native `ToolDenied` results. Network/API failures and invalid/mismatched responses raise without resuming.

The helpers perform synchronous SDK I/O, not long waits for people. In an async web server, run them in `asyncio.to_thread`. Your job scheduler owns retry timing, persistence, and an atomic claim so duplicate callbacks cannot resume the same saved run concurrently. Business tools must also enforce their own operation idempotency: this adapter does **not** guarantee exactly-once side effects or consume an execution permit.

You can pass `callback_url=` to creation. Verify callbacks using the public Pushary SDK and use the event only to find the saved batch and wake your worker. `resolve_reviews` re-fetches authoritative state; it never trusts a callback's answer directly. A scheduled poll should recover a missed callback.

## Ask for a choice or written answer

```python
from pushary_pydantic_ai import pushary_tool

agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=[str, DeferredToolRequests],
    tools=[pushary_tool()],
)
```

The model sees these validated input fields:

```json
{
  "question": "Which shipping service should we use?",
  "kind": "select",
  "options": ["Standard", "Express"]
}
```

The same create/resolve flow handles these requests. `external_id` is supplied by trusted server code when creating the batch and is absent from the tool schema. Returned question results contain `kind`, `status`, `value`, and `approved`; `approved` is `null` for select/input. A written "yes" is data, not authorization to run a different tool. The model may choose not to call `ask_human`, so use `requires_approval=True` on tools that must be gated.

Only native approval requests and external calls named `ask_human` are accepted. Other external tools are rejected before creating any decisions. Selection questions require 2–20 unique options. Questions are limited to 500 characters. Full approval arguments are shown in decision context; inputs that exceed the context's 2,000-character limit are rejected rather than silently hidden. Do not send secrets in tool arguments being reviewed.

## Run without a model or phone

```sh
uv pip install -e '.[test]'
python examples/refund.py
python -m pytest tests
python -m mypy
```

The example uses the real framework and shared SDK with an in-memory decision transport. It checks pause, JSON restoration, and approved/denied execution with model networking disabled. This proves adapter behavior; it does not claim that live native push delivery was tested.

Official framework references: [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), and [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/). Durable runtimes remain the application's choice; this adapter supplies the human response.


## Source and CI

The monorepo owns this package and its public-mirror workflow. The mirror CI tests Python 3.10 and 3.13, runs the native deferred-review suite, strict typing and the model-free example, builds the wheel and source archive, and runs the installed wheel in a clean environment. CI never contacts a phone or model provider. It does not publish the package.

## Release prerequisites

This candidate is wired into `release-pypi.yml` (including its manual dry run) and
the existing adapter drift check. Publish `pushary>=2.1.0` before this package.
Before the first upload, a PyPI project owner must configure a pending trusted
publisher for `pushary-pydantic-ai`: GitHub owner `aadilghani1`, repository
`pushary`, workflow `release-pypi.yml`, with no environment name. This setup has
not been performed by this change. Do not publish until it is configured and the
workflow dry run passes; adding the matrix entry alone does not authorize a release.
