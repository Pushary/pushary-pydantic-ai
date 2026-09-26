# pushary-pydantic-ai

Phone approvals for Pydantic AI agents. Your agent asks, your user taps Approve or Deny.

[Published on PyPI](https://pypi.org/project/pushary-pydantic-ai/) · [Integration guide](https://pushary.com/docs/agents/adapters?utm_source=github&utm_medium=oss-adapter&utm_campaign=pushary-pydantic-ai&utm_content=readme)

## What you need

- A Pushary Partner plan, from $99 a month. [Start the trial](https://pushary.com/sign-up?from=agent&plan=partner&utm_source=github&utm_medium=oss-adapter&utm_campaign=pushary-pydantic-ai&utm_content=partner-start).
- An API key from [Partner onboarding](https://pushary.com/onboarding/partner), set as `PUSHARY_API_KEY`.
- Your users install the free Pushary app ([iPhone](https://apps.apple.com/us/app/pushary/id6785677563), [Android](https://play.google.com/store/apps/details?id=com.pushary.app)). They never sign up or pay.

## Quick start

```sh
uv pip install pushary-pydantic-ai
export PUSHARY_API_KEY=pk_xxx.sk_xxx
```

```python
from pydantic_ai import Agent
from pushary_pydantic_ai import PusharyApprovals, connect

link = connect(authenticated_customer.id)  # once per user: show them this link

agent = Agent(
    "openai:gpt-4.1-mini",
    capabilities=[PusharyApprovals(external_id=authenticated_customer.id)],
)

@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str) -> str:
    return refund_order_once(order_id)

result = await agent.run("Refund order_123")
```

`PusharyApprovals` is a Pydantic AI [capability](https://pydantic.dev/docs/ai/capabilities/overview/). When the model calls a tool marked `requires_approval=True`, it asks your user on their phone and waits. Approve runs the tool in the same run. Deny, or no answer in time, returns a denial the model can read, and the tool does not run. Several approvals in one turn are asked one after another.

The notification shows the tool name and a short view of its arguments. The decision page shows every argument. A call whose arguments come to more than 2,000 characters is denied without asking anyone, because the approver could not read all of it.

Rules on your site answer first when the key can read them, and can allow or deny without paging anyone. Set `policy=False` to always ask a person. Add `pushary_tool()` to the agent's tools and the same capability answers the model's `ask_human` questions too. Other deferred calls are left for your code.

For a multi-tenant agent, resolve the person per run from trusted deps, never from the model's tool input:

```python
PusharyApprovals(external_id=lambda ctx: ctx.deps.user_id)
```

The resolver can also be async.

`timeout_seconds` is how long the run waits for an answer, 55 by default. The prompt expires with the wait, so a late tap cannot approve a call that was already denied. Pushary keeps a prompt open for at least one minute. The other options are `agent_name`, `require_reachable`, `policy`, `api_key` and `base_url`.

It works with `run`, `run_sync`, `iter` and `run_stream_events`. `run_stream` stops at the first deferred call before any capability sees it, so the tool never runs there; stream with `run_stream_events` instead. Under Pydantic AI's [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) the phone wait runs as a durable operation, outside workflow code. On Temporal, keep the activity timeout (60 seconds by default) longer than `timeout_seconds`.

The Pushary Python SDK is synchronous, so each wait runs on a worker thread from a pool of 100 kept for these waits. A waiting approval never takes a thread your app needs for its own sync code. Cancelling a run stops the wait at once, and the prompt expires on its own.

Pydantic AI's own docs: [Deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/).

## Wait without holding a worker

The capability holds the run open while it waits. To wait minutes or hours instead, let the run pause and resume it later:

```python
from pydantic_ai import Agent, DeferredToolRequests
from pushary_pydantic_ai import create_reviews

agent = Agent("openai:gpt-4.1-mini", output_type=[str, DeferredToolRequests])

@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str) -> str:
    return refund_order_once(order_id)

result = await agent.run("Refund order_123")
if isinstance(result.output, DeferredToolRequests):
    batch = create_reviews(result.output, external_id=authenticated_customer.id, run_id=result.run_id)
```

`create_reviews` creates a decision for each deferred call and returns immediately, so the human wait holds no worker open.

`refund_order_once` is your application's idempotent business operation. `authenticated_customer` comes from your server's authentication, never a model argument. Set `PUSHARY_API_KEY` on the server or pass `api_key=` to the helpers. Live end-user delivery requires Partner access and an enrolled customer; `connect(external_id)` returns the SDK's single-use enrollment link. Use a customer-bound key when available and keep it scoped to that same customer.

Save the original run's message history, deferred requests, and `batch.model_dump_json()` in trusted storage **before scheduling further work**. Persist the original run ID; a retry of decision creation must reuse that ID, the same customer, and the same original requests. Changed arguments produce a new decision rather than reusing approval.

Later, from a worker or callback handler:

```python
from pushary_pydantic_ai import ReviewBatch, resolve_reviews

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

`PusharyApprovals` answers these inline, and the create/resolve flow handles them durably. `external_id` is supplied by trusted server code when creating the batch and is absent from the tool schema. Returned question results contain `kind`, `status`, `value`, and `approved`; `approved` is `null` for select/input. A written "yes" is data, not authorization to run a different tool. The model may choose not to call `ask_human`, so use `requires_approval=True` on tools that must be gated.

In the durable flow, only native approval requests and external calls named `ask_human` are accepted, and other external tools are rejected before creating any decisions. Selection questions require 2-20 unique options. Questions are limited to 500 characters. Full approval arguments are shown in decision context; inputs that exceed the context's 2,000-character limit are rejected rather than silently hidden. Do not send secrets in tool arguments being reviewed.

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

## Runtime requirements

For local development, run `uv pip install -e .` from this package directory. Requires Python 3.10+, Pydantic AI 2.42+, anyio 4.7+, and the public Pushary SDK 2.2+. Model provider extras belong to your application; the adapter depends on the slim framework package.
