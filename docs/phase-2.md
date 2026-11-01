# Phase 2: FastAPI extraction

## Contract

- `GET /health`: process liveness, whether a generation key is configured, and lazy
  model-loading policy. This does not assert provider connectivity or loaded-model readiness.
- `GET /retrievers`: Single NQ (default), Multiset, and LFQA.
- `POST /chat`: synchronous question answering with the existing retrieval/prompt flow.

Example request:

```json
{"question": "What are the symptoms of tuberculosis?", "retriever": "single_nq"}
```

Response fields: `answer`, `retriever`, `generation_model`, and `passages`. Each
passage includes `text`, `source`, `page`, and `score`. These are retrieved sources,
not validated claim-level citations. As before, all five retrieved passages are
returned even if the context limit excludes some from generation.

Question text must contain non-whitespace characters and be at most 4,000 characters.
Unknown fields and retriever names are rejected with 422. Questions are not stripped
or rewritten. No history is accepted or sent to generation in this release.

## Errors

| HTTP | Meaning |
| --- | --- |
| 422 | Invalid request |
| 503 | Busy, unavailable model/index, missing/invalid provider configuration, or provider rate limit |
| 504 | Provider request timed out |
| 502 | Provider connection/status error |
| 500 | Unexpected internal failure |

Errors do not return provider bodies, secrets, or local filesystem paths. Busy and
rate-limit responses include a suggested `Retry-After`. Requests are not queued.

## Lifecycle and preserved behavior

FastAPI lifespan owns one `ChatService`. Heavy imports/models are lazy. One chat
runs per process; concurrent chat requests fail fast instead of piling up model
work. One retriever is cached; switching variants unloads the previous Python
model reference. Model downloads and cold loads can delay the first request.
Provider client resources are closed on shutdown. Additional workers multiply
model memory requirements, so use one worker for the initial local setup.

The unused context encoder/tokenizer is no longer loaded during serving. Question
encoder, tokenizer choice, query normalization, top-five search, model artifacts,
3,500-character context construction, prompts, and generation sampling settings
are preserved. Context construction now lives in `app/context.py` and is shared
with Streamlit. Generation HTTP timeout is explicitly 60 seconds and automatic
SDK retries are disabled so API failures return promptly without hidden retries;
this is a request timeout, not a total end-to-end deadline for model loading.

The Streamlit reference still manages its own session and caches; it does not call
the HTTP API. The future React frontend will use the new endpoints.

## Verification

`tests/test_api.py` uses an injected service with fake retrieval/generation for
validation, response contracts, caching, lifecycle, busy/error paths, and baseline
prompt/settings checks. It never calls the provider.

`scripts/verify_api_baseline.py` uses the actual API, actual cached DPR models on
CPU, and a recording fake generator. It checks all 27 Phase 1 examples for exact
ranked passage IDs and context strings, with small numerical tolerance for scores.
It never overwrites baseline files and never calls the generation provider.

Full live provider testing remains pending a newly rotated key. Successful offline
tests do not establish generated-answer accuracy. Auth, database persistence,
LangSmith, frontend replacement, and image analysis are outside this phase.

### Results from this implementation

- 17 unit/API tests passed, including the five existing retrieval metric tests.
- All 27 real-retrieval API cases matched baseline passage rankings and exact context;
  scores matched within the documented tolerance.
- A real Uvicorn process served `/health` and `/retrievers` without credentials or
  loaded models, then was stopped after verification.
- `pip check` found no broken requirements. TLS verification stayed enabled during
  installation; pip 24 required `--use-feature=truststore` to use Windows certificates.
- Existing DPR unused-pooler-weight warnings remain. The installed Starlette version
  also emits a deprecation warning for its supported httpx TestClient integration;
  all tests passed. A test-client dependency migration can be handled separately.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
and [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
