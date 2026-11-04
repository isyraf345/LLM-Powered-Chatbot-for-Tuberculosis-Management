# Seven-phase roadmap and coding-agent handoff

Last updated: 2026-11-05. Phase 3 pushed; Phase 4 frontend implemented locally; original green layout restored. Full live chat trace inspection remains pending.

## Purpose and agreed scope

The owner is a graduating data science student building a resume-worthy project,
primarily for AI engineering, with ML engineering/data science evidence through
an experimental chest X-ray classifier. The objective is a finished, demonstrable,
measured application and practical engineering experience, not maximum tool count.

The original TB chatbot works well from the owner's perspective. Preserve its
retrieval/generation approach. Improve architecture, observability, evaluation,
and deployment, then add image classification and discussion of its stored result.
Do not force a data engineering component into this project.

### Decisions already made

- Python/FastAPI backend; React/TypeScript will replace Streamlit as the final UI.
- Preserve DPR + FAISS and all three retriever options initially.
- LangSmith is the selected chatbot observability tool. It does not require a
  LangChain rewrite or LangGraph.
- PostgreSQL is planned for persistent application records. Object storage is
  planned for uploaded images/model files when the image feature is built.
- PyTorch is planned for image-model training and inference.
- Docker, GitHub Actions, and one cloud provider are planned for delivery.
- Start with ordinary Python orchestration. Defer .NET, LangGraph, LangChain
  migration, multi-agent systems, Kubernetes, Kafka, Spark, and a warehouse.
- MLflow is not in the initial implementation. Reconsider only if image experiment
  tracking warrants it; LangSmith does not track classifier training on its behalf.
- A background inference worker will be selected when image integration begins.

## Current status

| Phase | Status | Remaining qualification |
| --- | --- | --- |
| 1. Baseline and organization | Implemented and pushed | Rotated key/live answer capture, reviewed relevance labels, clean-install validation, source/index provenance remain outstanding |
| 2. FastAPI extraction | Implemented and pushed | Real retrieval verified; owner supplied a live response, revealing a grounding failure |
| 3. LangSmith observability | Implemented and pushed; synthetic hosted trace verified | Full live chat trace and token usage inspection pending |
| 4. New frontend and first deployment | In progress: React/TypeScript frontend implemented locally | Persistence, auth, feedback, Docker, CI, and cloud deployment remain |
| 5. Image-classification experiment | Not started | No dataset selected, no feasibility review completed, no trained model |
| 6. Image-aware chat integration | Not started | No upload API, storage, inference jobs, or image results |
| 7. Portfolio evidence | Partially documented only | Final demo, measured results, and resume claims remain |

Implemented is not equivalent to clinically validated, production-ready, or fully
evaluated. Do not describe the outstanding items above as finished.

GitHub: https://github.com/isyraf345/LLM-Powered-Chatbot-for-Tuberculosis-Management

Branch: `main`. Last verified pushed commit: `c0bcba1e` (Phase 3 observability); Phase 2: `ea8fe68c`.
Phase 1 integration commit: `b8f36668`. Check live Git status/history because later
commits can supersede this snapshot. This handoff's own publication status is not
implied by those earlier pushes.

## Architecture and important behavior

```text
Current:
HTTP request -> FastAPI -> ChatService -> DPR/FAISS -> context builder -> GPT-4o-mini
                              |
                       structured answer + passages

Reference UI: Streamlit -> same retriever/context builder/generator
              (Streamlit does not currently call the HTTP API)

Planned:
React -> FastAPI -> chat service + LangSmith
             | -> PostgreSQL: conversations, feedback, analysis records
             | -> object storage + background image worker (later)
```

- `GET /health`: liveness and key-configured flag, not provider/model readiness.
- `GET /retrievers`: Single NQ (default), Multiset, LFQA.
- `POST /chat`: `{"question":"What is TB?","retriever":"single_nq"}`.
- API response: answer, retriever, generation model, and retrieved passages.
- One request at a time per process; excess chat requests return 503/Retry-After.
- Lazy model loading; only one retriever cached at a time. Switching variants
  evicts the prior model reference. Use one Uvicorn worker initially.
- Retrieve five passages; build context with a 3,500-character limit, stopping at
  the first passage that would overflow. Displayed results may include passages
  that never enter generation. Capture this distinction in observability.
- Generation: GPT-4o-mini, temperature 0.3, top-p 0.95, max output 600 tokens;
  request timeout 60 seconds and SDK automatic retries disabled.
- No conversation history is sent to generation. Streamlit history is display-only.
  Persisting history later does not automatically implement conversational memory.
- No NLI verification or validated claim-level citations exist in the live flow.
- Three indexes contain 836 vectors each; their passage JSON files are identical.
  Single NQ/Multiset use 768 dimensions; LFQA uses 128. Original source PDFs and
  the final build pipeline are absent. Do not rebuild from the legacy notebook.

### File map

| Path | Role |
| --- | --- |
| `app/api.py` | FastAPI routes, request/response schemas, error mapping, lifespan |
| `app/service.py` | Model caching, concurrency control, request orchestration |
| `app/context.py` | Shared context construction |
| `app/dpr_retriever.py` | Question encoding and FAISS retrieval; unused context encoder removed |
| `app/generator.py` | Existing prompts and generation call |
| `app/config.py` | Model definitions, portable artifact paths, environment/dotenv key |
| `app/observability.py` | Explicit LangSmith trace records and bounded background export |
| `app/app.py` | Reference Streamlit UI |
| `models/` | Preserved indexes and passage metadata |
| `docs/baseline/` | Original manifest, observed environment, nine retrieval snapshots per model |
| `evaluation/` | Smoke questions, unreviewed 99-example relevance worksheet, evaluation instructions |
| `app/evaluate_dpr.py` | Corrected reviewed-ID retrieval metrics |
| `experiments/tb_legacy.ipynb` | Archived Colab prototype; not current index builder |
| `tests/` | Metric/API/service/context/generation contract tests |
| `scripts/verify_api_baseline.py` | Real retrieval via API, fake generation; does not overwrite baselines |
| `scripts/capture_baseline.py` | Explicit baseline regeneration; overwrites snapshots |

## Phase 1 — Preserve and organize

Completed: saved configuration/artifact hashes; captured nine representative
questions across three models; pinned direct dependencies; archived notebook;
removed application NLI module/config; externalized API key; restored ignore rules;
corrected flawed evaluation logic; added labels worksheet and tests.

Important: the old evaluation made Top-5 always pass. Its scores cannot support
resume claims. New scoring requires reviewed passage IDs and rejects unreviewed
labels. The existing expected text is not automatically a relevance label.

Remaining: owner must rotate/revoke the exposed historical key; capture live
generation behavior after a new key is configured; review relevance labels; recover
index provenance if rebuilding is needed. See [Phase 1 notes](phase-1.md).

## Phase 2 — Extract the backend

Completed: FastAPI endpoints, shared context builder, service lifecycle, request
validation, sanitized errors, lazy loading, and removal of unused serving encoder.
Streamlit remains available during migration.

Verified at the Phase 2 revision: 17 tests passed; all 27 actual retrieval/API cases
matched ranked IDs and exact context, with score tolerance; real Uvicorn health and
retriever requests passed without a key; `pip check` passed. Provider calls were
mocked. See [API details](phase-2.md).

## Phase 3 — LangSmith observability (local implementation complete)

Implemented: metadata-only tracing by default, opt-in content, five-run trace trees,
background export with a bounded queue, sanitized errors, request correlation,
artifact versions, actual included ranks, and provider usage metadata. See
[Phase 3 setup](phase-3.md) and add a LangSmith key privately to `.env` to verify
one hosted trace. No LangSmith credential was available during initial implementation. The owner later added one; a synthetic parent/child trace was uploaded and read back successfully on 2026-11-03. TLS uses the OS certificate store with verification enabled.

Verified: 29 tests passed, `pip check` passed, and all 27 baseline cases matched with
tracing disabled and again with local tracing enabled (54 API requests total).
The enabled run exported 27 parent and 108 child records to an in-memory sink.
These offline checks made no provider or hosted LangSmith calls. SDK serialization was tested with
network transport mocked. No prompt/corpus repair or frontend work is included.

The checklist below records the agreed scope. Local implementation/testing is done;
synthetic hosted validation passed; full live chat trace inspection remains. Feedback storage remains Phase 4.

### First failure case to investigate

On 2026-11-02 the owner supplied a live Single NQ response about TB symptoms.
Generation returned an answer, but explicitly fell back to general knowledge
because the retrieved context did not list symptoms. This is evidence of a working
provider connection in that reported interaction, not an answer-quality pass.

The saved symptoms baseline retrieves IDs 317, 69, 318, 68, and 397. Passage 451
(management guideline, PDF page 18) directly describes symptoms but is absent from
those results. Passage 397 contains an unrelated literary sentence. Inspection
confirmed this text in all three passage files, whose hashes still match Phase 1.
The fifth passage is excluded from the saved generation context by its size limit.

Use this case to trace retrieval versus actual context versus answer grounding.
Audit corpus provenance and investigate ranking before selecting a fix. Do not
silently edit passage text while keeping embeddings/baselines unchanged. No corpus
or prompt fix has yet been made. Baseline parity preserves existing weaknesses.

Implementation checklist:

1. Inspect current API/service/generator and current official LangSmith Python SDK
   documentation. Select and pin a compatible SDK version.
2. Add environment-based enablement, project/endpoint/key settings, and safe
   defaults. Document exact variables in `.env.example`; never include real keys.
3. Instrument one parent chat trace with child retrieval, context construction,
   and generation runs. Record retriever/index/prompt/model versions, stage timings,
   retrieved passage IDs/scores, included-context distinction, and errors.
4. Capture provider token usage and available cost estimates without changing the
   returned answer or sending extra model requests. Do not invent cost precision.
5. Introduce a request/trace reference usable for debugging and later feedback.
   Define when feedback storage/endpoints are added; none exist today.
6. Make content export explicit: no raw medical images/identifying content in
   hosted traces by default. Metadata-only or redacted defaults must be documented;
   use public/synthetic prompts to validate richer traces.
7. Ensure disabled/misconfigured/unavailable tracing cannot break chat. Bound any
   telemetry work and avoid serializing credentials or raw provider error bodies.
8. Test disabled tracing, successful trace structure, failures/timeouts, redaction,
   and output parity. Reuse existing tests and cached-model baseline verification.
9. Validate one real hosted trace if credentials are available; otherwise record
   hosted verification as pending. Do not claim local mocks prove export works.
10. Add `docs/phase-3.md` with setup, example investigation, checks, and limitations.

Done when: an interaction can be traced through retrieval/context/generation,
failure handling is observable without changing user outputs, and failure examples
can feed regression cases. Continue label review separately; tracing is not accuracy.

## Phase 4 — Replace Streamlit and ship the chatbot

Build React/TypeScript chat with source passages, loading/error/retry states,
retriever selection, and feedback. Connect it to the existing API. Streaming is
optional and should not block a usable first version.

First milestone implemented: `frontend/` uses React/TypeScript and Vite with a local
`/api` proxy. Chat, retriever selection, Markdown answers, source passages,
request/trace details, loading/error/retry, and session reset are available.
Messages are in browser memory only; feedback is not implemented. Backend behavior
is unchanged. See [Phase 4 notes](phase-4.md) for startup and verification.

UI preference (2026-11-05): restored the original green sidebar, headline, and
suggestion cards at the owner's request. Retrieval controls remain in the sidebar;
sources and request details remain expandable. No API behavior changed.

Verified for this milestone: TypeScript/production build, five Playwright browser
tests, desktop/mobile visual inspection, and built frontend proxy access to the
real API health/retriever endpoints. Browser chat tests mocked generation; live
browser/provider/trace inspection remains outstanding.

Add PostgreSQL and migrations for retained conversations/messages/feedback. Select
an established authentication approach before exposing private persisted records.
Check ownership on all record access. Decide retention/deletion behavior. Treat
conversational memory as a separately evaluated behavior change.

Add Docker, relevant GitHub Actions checks, environment secrets, and deployment
to one provider selected against budget. Configure frontend/API origins explicitly.
Do not publicly deploy the current unauthenticated cost-incurring API as-is.

Done when: another person can use the deployed chatbot, inspect sources, submit
feedback, and see reliable loading/errors; developers can investigate traces.
Retire Streamlit from the primary user flow once the replacement is verified.

## Phase 5 — Develop the image classifier

A short feasibility check can happen before Phase 4 is finished; substantial model
training follows the first chatbot release. No dataset has been chosen yet.

Review terms, provenance, label meaning, patient identifiers, duplicates, overlap,
class balance, and independent-source testing options. Define the task the data
actually supports. Do not promise diagnosis or use generated labels as truth.

Audit -> fixed patient-aware splits where possible -> preprocessing -> transfer
learning baseline -> a few justified experiments -> validation-based selection ->
untouched test evaluation. Fit preprocessing/augmentation only in appropriate splits.

Record seeds, configurations, data/code/model versions, metrics, and artifacts.
Assess sensitivity, specificity, AUROC, PR-AUC, confusion matrices, uncertainty,
and errors. Prefer cross-source testing where feasible. Evaluate calibration before
describing scores as probabilities. Grad-CAM is optional, not clinical evidence.

Done when: training is reproducible and results/limitations are defensible. Poor
generalization is a finding to report, not something to hide behind random-split
accuracy. Decide whether MLflow earns a role after the experiment workflow exists.

## Phase 6 — Connect images and chat

Upload -> validate file size/format/decoding -> store asset -> create analysis job
-> worker applies versioned preprocessing/model -> persist result -> display result
-> explicit "Discuss this result" action attaches that analysis to conversation.

Select object storage and a worker/queue at implementation time. Start with polling
for pending/running/completed/failed status. Prevent duplicate submissions and
ensure failures are recoverable. File decoding does not prove X-ray suitability.

Store owner/image reference, analysis ID/status, timestamps, prediction scores,
threshold, model/preprocessing versions, and safe failure details. Enforce ownership
and coordinated deletion for files and records.

Chat retrieves the chosen completed result plus relevant TB information. It must
not infer anatomical findings from a classification score, mix images, treat failed
jobs as successful, or rerun inference unnecessarily. Add integration evaluations
for these cases. Connect analysis/model references to observability traces.

Done when: upload-to-result-to-conversation works reliably and its experimental
scope is clear. LangGraph is still optional, not a prerequisite.

## Phase 7 — Package the portfolio

Prepare a deployed demo, reproducible setup, architecture diagram, evaluation and
error-analysis reports, model limitations, a short demo video, and design tradeoffs.
Create resume bullets from actual measurements. Emphasize AI workflow integration,
observability, deployment, reproducibility, and model evaluation rather than tool count.

Done when: someone can understand/run the project and the owner can defend the
design and evidence in an interview. No invented metrics or clinical claims.

## Commands for the next agent (PowerShell, repository root)

```powershell
git status --short
git log -5 --oneline
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m pip check
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -B scripts/verify_api_baseline.py
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Create the environment with Python 3.12 if missing. Baseline verification requires
cached pretrained weights, forces offline mode, uses CPU, and makes no provider
calls. On a new machine, missing weights are a setup issue, not automatically a
regression. Model downloads and fresh installation have not been fully validated.
On this Windows machine pip 24 needed `--use-feature=truststore` to use system
certificates; do not disable TLS verification to work around certificate errors.

Set a newly rotated `OPENAI_API_KEY` privately before live `/chat` testing. The Git-ignored root `.env` is loaded automatically; existing environment values
take precedence. Restart the server after changes. No account credential should be read
from the old Git history or pasted into documentation.

## Open decisions and limits

- Available weeks/hours, monthly cloud/API budget, and GPU access are unknown.
- Cloud provider, auth provider, frontend styling/component choices, image dataset,
  object store, and worker/queue are not finalized.
- LangSmith defaults are in `.env.example` and missing settings were added to local `.env`.
  A key is now configured and a synthetic hosted trace verified; full chat inspection is next. Content logging remains false.
- There is no evidence that the owner has rotated the old key yet.
- The original CSV is unreviewed; do not run scoring by marking all rows reviewed.
- Local reports/manuals are untracked user material; leave them alone by default.
- Historical phase notes describe their respective revisions. This roadmap is the
  current cross-phase status; source code and new verified results take precedence.

Update this document after future work so another agent can continue without the
conversation. Do not mark a milestone fully validated when external checks remain.
