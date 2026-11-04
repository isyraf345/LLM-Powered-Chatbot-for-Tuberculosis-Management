# Anonymous saved history (Phase 4)

No sign-in is required. The backend issues a random 256-bit browser credential
in the `tb_session` cookie (HttpOnly, SameSite=Strict, Path=/). PostgreSQL stores
only its SHA-256 hash. All history routes verify ownership; knowing a conversation
or message UUID is not enough to access it. The cookie is an access credential:
someone using the same browser profile can access that profile's history.

Sessions have a fixed 30-day expiry from creation, not a sliding extension. History
survives refreshes and API restarts, but clearing cookies removes access. There is
no recovery or cross-device sync. Localhost and 127.0.0.1 are different cookie hosts;
use the same frontend URL consistently.

## Setup

The current machine has PostgreSQL running in Docker on `127.0.0.1:5433`, database
`tb_chatbot`, with a named volume. Local `.env` settings were added without changing
existing provider keys. The initial migration was applied.

For a fresh installation, install `requirements.txt`, choose a URL-safe database
password, and privately add these values to root `.env`:

```dotenv
POSTGRES_PASSWORD=YOUR_URL_SAFE_PASSWORD
DATABASE_URL=postgresql+psycopg://tb_app:YOUR_URL_SAFE_PASSWORD@127.0.0.1:5433/tb_chatbot
COOKIE_SECURE=false
```

Start Docker Desktop, then from repository root:

```powershell
docker compose up -d --wait db
.\venv\Scripts\python.exe -m alembic upgrade head
.\venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Restart an already-running API to load the new routes
and database configuration. No provider credentials belong in the frontend.
The dev proxy defaults to port 8000; `TB_API_TARGET` can override its target for
isolated integration testing. `COOKIE_SECURE=true` is required for HTTPS hosting.
Do not enable permissive credentialed CORS: cookie routes require the custom
`X-TB-Client: web` header on mutations and reject cross-site Fetch Metadata.

`docker compose stop db` preserves the data. Do not run `docker compose down -v`
unless intentionally deleting the database. Changing POSTGRES_PASSWORD in `.env`
does not reset the password of an already-initialized volume.

## Data and request lifecycle

- `browser_sessions`: token hash and expiry.
- `conversations`: session owner, title, creation/update times.
- `messages`: one question/answer exchange per row, retriever, status, attempt,
  saved answer JSON (including passages and trace ID), request ID, and timestamps.
- `feedback`: one updatable rating (-1 or 1) and optional comment per message.

Conversation titles come from the first question without a model call. History
does not become model memory: generation still receives only the current question
and the existing retrieved context. Answer content is stored in PostgreSQL even
when `TB_TRACE_CONTENT=false`; that flag controls LangSmith exports only.

The browser chooses UUIDs for conversations and messages. Idempotent PUT requests
reuse those UUIDs. A short transaction checks session ownership and reserves a
pending message, then releases database locks before inference. A second transaction
saves the answer or sanitized failure. Duplicate requests return the existing state;
an explicit retry increments the attempt only for failed messages. A delayed older
attempt cannot overwrite a newer attempt. This prevents duplicate records and
ordinary duplicate generation, but cannot guarantee exactly-once provider billing
across process crashes or failures between generation and the final database write.

Pending messages are polled by the browser. After ten minutes, a pending attempt
is treated as interrupted on read/retry and can be retried explicitly. A still-running
old worker may have made a provider call; the warning does not promise cost-free retry.
Deleting during inference prevents the completion from recreating deleted records.

Deleting a conversation cascades to messages and feedback. Clear all history is
scoped to the current session. Expired sessions are rejected immediately; cleanup
runs on API startup and hourly while the API runs. If the API is stopped, expired
rows remain until it restarts or a scheduled cleanup runs:

```powershell
.\venv\Scripts\python.exe -m app.cleanup_history
```

Deletion removes application database rows, not existing LangSmith traces or provider
records. Those have separate retention controls. No durable backup or recovery
procedure is implemented yet.

## Routes

| Method/path | Purpose |
| --- | --- |
| POST `/session` | Create/reuse the anonymous cookie session |
| GET `/conversations` | List this session's history |
| PUT `/conversations/{id}` | Create a conversation idempotently |
| GET `/conversations/{id}` | Read messages and feedback |
| PATCH `/conversations/{id}` | Rename |
| DELETE `/conversations/{id}` | Delete one conversation |
| DELETE `/conversations` | Clear this session's history |
| PUT `/conversations/{id}/messages/{message_id}` | Submit/replay an attempt |
| PUT `/conversations/{id}/messages/{message_id}/feedback` | Save/update feedback |

The original stateless `/chat` endpoint remains for baseline compatibility. The
React app uses saved-message routes. Neither endpoint is ready for unrestricted
public use; rate limits, abuse controls, deployment hardening, CI, and app containers
remain later Phase 4 work. The Compose file currently starts only the database.

## Verification

Run the standard Python tests. PostgreSQL integration tests require an explicit
`TEST_DATABASE_URL`; they create/migrate/drop a randomly named `tb_test_*` schema,
never the application's tables. A database role with CREATE SCHEMA is needed.

To use the local database privately from PowerShell without displaying its URL:

```powershell
$env:TEST_DATABASE_URL = & .\venv\Scripts\python.exe -c "from dotenv import dotenv_values; print(dotenv_values('.env')['DATABASE_URL'])"
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
Remove-Item Env:TEST_DATABASE_URL
.\venv\Scripts\python.exe -m alembic check
.\venv\Scripts\python.exe -m pip check
```

Frontend: `npm run build` and `npm test` in `frontend`. Set
`$env:TEST_FRONTEND_PORT="5174"` if your dev server already uses 5173.

Manual check: ask a public test question, refresh, reopen it from History, save
feedback, rename, and delete it. A private browser window should have separate
history. Test provider calls and inspect hosted LangSmith traces separately;
automated tests use fake generation.

Recorded on 2026-11-05: 41 Python tests including 12 PostgreSQL integration tests
passed, as did seven browser tests, `alembic check`, `pip check`, and the frontend
production build. An additional real browser/Vite/FastAPI/PostgreSQL check passed
for persistence after refresh, feedback, separate browser sessions, deletion, and
mobile width with fake generation. No paid provider call was made.

References: [SQLAlchemy sessions](https://docs.sqlalchemy.org/en/20/orm/session_basics.html),
[Alembic migrations](https://alembic.sqlalchemy.org/en/latest/tutorial.html),
[PostgreSQL row locks](https://www.postgresql.org/docs/current/explicit-locking.html).
