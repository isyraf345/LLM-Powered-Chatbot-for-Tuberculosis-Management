# Phase 3: LangSmith observability

## Implementation and scope

The FastAPI path now emits one parent `chat` trace with four child runs:

```text
chat (same ID as the API request)
  load_resources        model loading/caching and artifact versions
  retrieval             ranked passage IDs, scores and pages
  context_construction  included ranks, character count and limit
  generation (llm)      model/provider metadata and token usage
```

No LangChain/LangGraph migration, new LLM call, prompt change, corpus edit, or
retrieval change is introduced. The reference Streamlit UI is not instrumented;
use the API for tracing. Tracing records observations, not answer correctness.

## Enable hosted tracing

Create a LangSmith account/project and API key in your own workspace. Add these
settings privately to the project-root `.env` (already Git-ignored):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=tb-chatbot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
TB_TRACE_CONTENT=false
```

For an EU workspace, use `https://eu.api.smith.langchain.com`. Those two hosted
endpoints are supported; other endpoints disable tracing rather than forwarding
credentials to an arbitrary host. Restart the server after changes. Existing
terminal environment values override `.env`. Keep your separate OpenAI key.

```powershell
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

1. Open `http://127.0.0.1:8000/docs` and execute `GET /health`.
2. Check `tracing.enabled=true`. This means export is configured, not authenticated
   or confirmed delivered. Missing keys/invalid endpoints disable tracing with a
   sanitized server warning.
3. Submit a public test question through `POST /chat`.
4. Copy `trace_id` from the response (also `request_id`/`X-Request-ID`). Find the
   matching trace in your LangSmith project's trace list after export completes.
5. Inspect child runs, original start/end times, model/index versions, passage IDs,
   actual included ranks, and token usage. Check errors and `/health` export counters.

Hosted connection verification succeeded on 2026-11-03 after the owner added a key.
A synthetic `connection_smoke` parent and one child were uploaded and read back
from LangSmith (ID `670ea295-aeb7-4706-a404-d2188113f1d5`). No OpenAI call or chat
content was sent. Dashboard/readback visibility had a short indexing delay.
Next, submit a real `/chat` request to verify the full five-run tree and token usage.

## Privacy and content logging

Default: tracing disabled; content logging disabled even when tracing is enabled.
Metadata-only runs contain random request IDs, passage IDs/ranks/pages/scores,
artifact SHA-256 hashes, prompt-version label, model identifiers, timings, token
counts, and error class names. They do not include question, context, answer text,
raw provider responses, exception messages/tracebacks, credentials, or runtime env.
The exporter receives only explicitly assembled fields, not arbitrary locals.

`TB_TRACE_CONTENT=true` explicitly exports questions, retrieved passages, actual
context, exact generation messages, and answers. This is **not a PII scrubber**.
Use public/synthetic test prompts only when enabling it; do not submit personal
medical information. Changing the flag does not delete already-exported data.
No images are supported or exported in this phase.

## Token usage and costs

The existing OpenAI response supplies input, output, total, and cached input token
counts when available. The generator still returns the same answer string. Counts
are reset for each request, so missing usage never reuses a previous request's data.

The LLM run uses `ls_provider=openai`, `ls_model_name=gpt-4o-mini`, and standard
`usage_metadata`. LangSmith may estimate costs using its configured pricing.
No hardcoded dollar estimate is calculated here. Missing usage/model pricing means
cost may be unavailable; a missing value must not be interpreted as zero. Exact
provider model snapshots and billing should be checked against the provider.

## Reliability and correlation

`app/observability.py` buffers up to 32 completed traces in memory and exports with
one daemon worker, independent of the response path. The SDK's own batch queue is
disabled. Export uses bounded connection/read timeouts and no configured transport
retries. Slow or failed export cannot delay generation on the request thread.

Full queues drop traces; failed exports can leave partial trace trees. There is no
durable replay, disk spool, or guaranteed delivery. Shutdown waits at most two
seconds for the worker and drops queued work; an in-flight daemon request may end
after shutdown. This is suitable for the current local prototype, not a telemetry
delivery guarantee. Traces appear after the request completes, not live mid-request.

`GET /health` includes enabled/content flags and pending/exported/failed/dropped
counters for this process. `pending` excludes the in-flight export. `exported` means
all SDK writes returned successfully, not that someone verified the dashboard.

All HTTP responses have a server-generated `X-Request-ID`. Successful chat responses
also include `request_id` and nullable `trace_id`. With tracing enabled, the two IDs
match; a returned trace ID does not guarantee delivery. Errors preserve the request
header and sanitize their body. Valid requests failing in the service (including
busy/model/provider failures) produce error traces. Request validation errors occur
before the service and have a request ID but no chat trace. Health/docs do not emit
chat traces. Conversation IDs/feedback persistence arrive in Phase 4.

## First investigation: symptoms retrieval failure

Use `What are the symptoms of tuberculosis?` with `single_nq`.
The Phase 1 baseline retrieves IDs 317, 69, 318, 68, 397. Passage 451 directly
describes symptoms but is not retrieved. The supplied live answer admitted falling
back to general knowledge. Passage 397 has unrelated text that predates the refactor;
it is excluded from the saved context by the size limit.

With tracing enabled, compare retrieval results against `included_ranks`. With
public-test content logging explicitly enabled, inspect actual context and answer.
This distinguishes retrieval failure, context exclusion, and unsupported generation.
Record failures for evaluation; do not assume observability fixes them. The corpus,
prompt, and ranking are intentionally unchanged. The known case is saved in
`evaluation/known_failures.json` without pretending its labels are fully reviewed.

## Verification

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m pip check
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -B scripts/verify_api_baseline.py
.\venv\Scripts\python.exe -B scripts/verify_api_baseline.py --trace-local
```

Both baseline modes force offline CPU retrieval and fake generation; they never
contact LangSmith or OpenAI regardless of `.env`. The enabled mode uses an in-memory
exporter. Unit tests also stub network transport when testing the real LangSmith SDK.
Tests cover privacy, opt-in content, parent/child structure, token metadata, request
correlation, export failures, slow export, queue saturation, enrichment failures,
configuration failures, TLS certificate verification, and unchanged prompt/settings.
These offline checks used no API key or paid provider call. Synthetic hosted
verification is recorded above; full live chat trace inspection remains pending.

Recorded results (2026-11-03): 29 automated tests passed; dependency checks passed;
27 actual retrieval/API cases passed with tracing disabled and another 27 with local
tracing enabled. The latter produced 27 parent traces and 108 child runs in memory.

References: [LangSmith manual LLM tracing](https://docs.langchain.com/langsmith/log-llm-trace)
and [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk/tree/main/python).

## Windows certificate configuration

The first live upload failed TLS certificate verification. The exporter now uses
`truststore` and a requests HTTPS adapter to consult the operating system certificate
store. The adapter is installed after SDK initialization (which mounts its own
adapters). TLS verification remains enabled and other clients are not modified.
The synthetic export and readback succeeded with this configuration.
