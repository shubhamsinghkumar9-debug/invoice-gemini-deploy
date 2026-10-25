# Invoice Action Agent — Gemini deployment

This version uses the supplied Gemini REST API code, not OpenAI. It exposes a FastAPI ASGI app as `app` for Render.

## Render deployment

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

Set these Render environment variables:

- `GEMINI_API_KEY`: API key from Google AI Studio.
- `GEMINI_MODEL`: model ID available to your Google AI Studio key. The supplied default is `gemini-3.5-flash-lite`; if Google reports that model is unavailable, change this to a model ID enabled for your account.
- `A2A_BEARER_TOKEN`: long random secret used by API clients. The Render blueprint can generate one automatically.
- `A2A_BASE_URL`: optional; set to the service's public origin, e.g. `https://your-service.onrender.com`. The Agent Card appends `/a2a/`.
- `SQLITE_PATH`: defaults to `state.db`.

## Endpoints

- `GET /health` — health/configuration status (does not reveal the API key)
- `GET /.well-known/agent-card.json` — Agent Card
- `POST /a2a/message:send` — submit invoice batch or send approval results
- `GET /a2a/tasks` — list tasks
- `GET /a2a/tasks/{tid}` — get task
- `POST /a2a/tasks/{tid}:cancel` — cancel task

Protected endpoints require headers:

```http
Authorization: Bearer YOUR_A2A_BEARER_TOKEN
A2A-Version: 1.0
Content-Type: application/a2a+json
```

The Gemini key is read from environment variables. Do not commit secrets to Git.

## SQLite persistence note

The supplied implementation stores data in SQLite and writes `decisions.jsonl` locally. On Render Free, local filesystem data may be lost on redeploy/restart and instances do not provide durable shared storage. For durable production state, attach a persistent disk on a plan that supports it or adapt the storage layer to PostgreSQL. This package intentionally retains SQLite to match the supplied code.
