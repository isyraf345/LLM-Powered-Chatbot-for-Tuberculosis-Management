# Phase 4: React frontend and delivery

## Current milestone

The first milestone replaces the primary chat UI with React/TypeScript and Vite.
It calls the existing FastAPI API; retrieval, prompts, context limits, and generation
are unchanged. Streamlit remains available as a reference.

Implemented: model selection loaded from `/retrievers`, question suggestions,
Markdown answers, expandable retrieved passages, request/trace references,
loading states, manual retry, connection recovery, and responsive layouts.
Duplicate submission is prevented while a request is running. A failed request
can be retried with its original question/retriever without duplicating the turn.
Raw model HTML and images are not rendered. Scores are labeled as retrieval
similarity, not confidence. Source results are not presented as verified citations.

Messages are held only in React memory and lost on refresh. They are still sent
to the backend/provider as part of answering; LangSmith content export follows
the backend configuration. Only the current question is sent to generation.
There is no persistent history, authentication, feedback storage, or deployment yet.
Phase 4 as a whole is **in progress**, not complete.

UI preference (2026-11-05): restored the original green sidebar layout, headline,
and suggestion cards at the owner's request. No backend behavior changed.
Restoration verified with the production build and all five browser tests (using
port 5174 to avoid interrupting the existing development server).

## Run locally

Start the backend from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In a second terminal (Node 24 is the locally tested runtime):

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The Vite server proxies `/api/*` to the local
FastAPI server on port 8000. No browser API key or frontend `.env` is needed;
the root `.env` is read only by the backend. Do not put provider keys in Vite
variables, which can be exposed in browser builds.

If npm fails to trust a certificate on this Windows machine, use
`$env:NODE_USE_SYSTEM_CA="1"` with Node 24 and retry. Keep TLS verification enabled.

The first question may take longer while the DPR model loads. Requests time out
in the browser after five minutes; that does not cancel backend computation.
Retries are manual because a request may already have incurred provider cost.
API availability does not prove model/provider readiness.

## Validation

```powershell
cd frontend
npm run build
npx playwright install chromium
npm test
```

Browser tests mock API responses: successful selected-retriever requests, answer
and source rendering, request references, new conversation, failed-request retry,
blank/duplicate submission, API reconnect, mobile width, and unsafe Markdown.
These tests do not call OpenAI or LangSmith or prove live answer quality.

Verified on 2026-11-03: TypeScript check and production build passed; all five
Playwright tests passed with one worker. Desktop and mobile screenshots were
inspected. The built frontend's proxy successfully reached the real FastAPI
`/retrievers` and `/health` endpoints, with no generation calls. Full live chat
through the browser and hosted trace inspection remain manual checks.
The suite uses one worker because concurrent browser startup was unreliable on
this machine. In restricted agent environments, browser tests may require
permission to start and clean up local server/browser processes.

For manual integration, start both servers, ask a public test question, expand
passages and request details, and find its trace in LangSmith if tracing is enabled.
Stop the backend to inspect failure/retry behavior. Refresh to verify session clearing.

`npm run build` emits `frontend/dist`. `npm run preview` serves it locally at
http://127.0.0.1:4173 with the same API proxy. Preview is not a production server.
Production hosting needs an equivalent `/api` reverse proxy and authenticated API.

## Remaining Phase 4 milestones

1. PostgreSQL schema and migrations for owned conversations, messages, and feedback.
2. Authentication with server-side ownership checks; agree retention/deletion behavior.
3. Docker and CI checks; choose a cloud provider against budget, then deploy.
4. Verify the complete deployed interaction and document setup and limitations.

The known symptoms retrieval/grounding failure remains tracked separately.

Implementation references: [Vite setup](https://vite.dev/guide/),
[Vite server proxy](https://vite.dev/config/server-options.html#server-proxy).
