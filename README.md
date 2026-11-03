# Tuberculosis information chatbot

Current application: FastAPI + DPR retrieval + local FAISS indexes + GPT-4o-mini.
The Streamlit interface remains available as a reference during frontend migration.
This is an educational research project, not a diagnostic service.

## Start here: roadmap and agent handoff

**Phases 1 and 2 are pushed. Phase 3 tracing is implemented locally; a synthetic hosted trace has been verified.**
Reviewed evaluation labels and answer-quality improvements remain outstanding.

Read [the seven-phase roadmap and handoff](docs/ROADMAP.md) before continuing work.
It records agreed scope, current architecture, acceptance criteria, unresolved
decisions, and the next implementation steps. [AGENTS.md](AGENTS.md) directs coding
agents to that document. No previous chat history is needed to understand the plan.

Suggested message for a new coding agent:

> Read AGENTS.md, README.md, and docs/ROADMAP.md. Inspect the current repository and
> Git status, summarize what is already implemented, then verify a full Phase 3 chat trace using
> the handoff checklist. Preserve current chatbot behavior and the saved baselines.
> Do not redesign the stack or start later phases without discussing the scope.

## Run locally (Windows PowerShell)

Use Python 3.12. The direct dependency versions were captured from the existing
Windows environment; a fresh installation has not yet been independently verified.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-new-key"
Set-Location app
..\venv\Scripts\python.exe -m streamlit run app.py
```

### Run the API (from repository root)

```powershell
$env:OPENAI_API_KEY = "your-new-key"
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. `/health` and
`/retrievers` work without a key or loaded models. `/chat` requires a valid key.
Use one worker initially; each worker has its own model memory. The service handles
one chat at a time and returns 503 with `Retry-After` when busy. It is a local API
at this stage, without authentication, persistence, or a public deployment setup.

See [API contract and lifecycle](docs/phase-2.md) for requests, errors, and verification.

Alternatively, enter your key in the root `.env` file as `OPENAI_API_KEY=...`.
The application loads this file automatically. Existing terminal environment
variables take precedence. Restart the server after editing the file.
`.env` is Git-ignored; `.env.example` is the empty template. Revoke/rotate the previously
hardcoded key in the provider account before use. No credential is included here.
First use may download pretrained DPR weights. Model paths are resolved relative to the project, preserving the newer GitHub fix.

## Current execution flow

`app/app.py` selects a retriever, retrieves five passages, builds a context limited
to 3,500 characters, and calls `app/generator.py`. Conversation history is stored
only in Streamlit session state and is not sent to the language model. The default
retriever remains Single NQ; Multiset and LFQA remain available. The app displays
all retrieved passages even when the context limit excludes some from generation.

The three indexes use the same 836 passages from two guideline/report documents.
Their original PDFs and exact final indexing pipeline are not present. Existing
FAISS indexes are preserved. `experiments/` is historical, not runtime code.

## Validation (from repository root)

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -B scripts/capture_baseline.py
```

Install `requirements-dev.txt` for API tests. To verify the API against the existing
baseline without overwriting it or calling the provider:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -B scripts/verify_api_baseline.py
```

Baseline capture forces offline model loading, runs on CPU, and makes no generation
API calls. It requires cached model weights and overwrites the retrieval snapshots;
review differences before accepting a new baseline.

See [Phase 1 notes](docs/phase-1.md) for baseline scope and outstanding work, and
[evaluation instructions](evaluation/README.md) before running retrieval evaluation.

## LangSmith observability (Phase 3)

Tracing is implemented for the API and disabled by default. Configure the LangSmith
settings in your Git-ignored `.env`; see [Phase 3 setup and verification](docs/phase-3.md).
Default metadata-only traces omit questions, answers, and passage text. Explicit
`TB_TRACE_CONTENT=true` opts into content export for public/synthetic debugging.
Uploads happen in a bounded background worker; failures cannot interrupt chat.
`/health` reports trace export status, and responses include request/trace references.

## Next milestone

A synthetic hosted trace is verified. Verify a full chat trace in LangSmith, then build the React/TypeScript
frontend and first deployment (Phase 4). Image classification remains a later phase.
The documented symptoms retrieval/grounding issue is still unresolved; tracing
makes it inspectable but does not fix it automatically.
