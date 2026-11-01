# Tuberculosis information chatbot

Current application: Streamlit + DPR retrieval + local FAISS indexes + GPT-4o-mini.
Phase 1 preserves the live retrieval and generation behavior before API migration.
This is an educational research project, not a diagnostic service.

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

Set the key privately in your local terminal. `.env.example` documents the variable;
the application does not automatically load `.env`. Revoke/rotate the previously
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

Baseline capture forces offline model loading, runs on CPU, and makes no generation
API calls. It requires cached model weights and overwrites the retrieval snapshots;
review differences before accepting a new baseline.

See [Phase 1 notes](docs/phase-1.md) for baseline scope and outstanding work, and
[evaluation instructions](evaluation/README.md) before running retrieval evaluation.

## Next milestone

Extract FastAPI endpoints and add LangSmith traces before replacing Streamlit with
React/TypeScript. Image classification is a later milestone. These components have
not been implemented in Phase 1.
