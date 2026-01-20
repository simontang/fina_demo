# Fina Demo

Demo repository that combines:
- `agent/`: Node.js (Fastify) gateway + agent orchestration
- `prediction_app/`: Python (FastAPI) dataset + prediction APIs (including sales forecasting)
- `ai_web/`: React (Refine) admin UI

Extras:
- Voice Agent supports Volcengine RTC VoiceChat via `agent` (`/api/rtc/*`) and UI at `/admin/agents/voice/rtc`

## Quickstart (Dev)

### 1) Start the Python API (`prediction_app/`, port 8000)

```bash
cd prediction_app
./start_api.sh
```

### 2) Start the gateway (`agent/`, port 6203)

```bash
cd agent
pnpm install
pnpm dev
```

Environment (`agent/.env`):
```bash
PORT=6203
VOLCENGINE_API_KEY2=...
UPLOAD_DIR=./uploads
VOLCENGINE_APP_ID=...
VOLCENGINE_APP_KEY=...
VOLC_ACCESSKEY=...
VOLC_SECRETKEY=...
```

### 3) Start the admin UI (`ai_web/`, port 5173)

```bash
cd ai_web
pnpm install
pnpm dev
```

Notes:
- Vite proxies:
  - `/api/v1/*` -> `http://localhost:6203` (gateway, reverse-proxies to Python)
  - `/api/*` -> `http://localhost:6203` (gateway)

## Docker (Compose)

Bring up the full demo stack (CSV + model assets are loaded from disk; no DB required):

```bash
docker compose up --build
```

Notes:
- `raw_data/` and `models/` are baked into the `prediction_app` image during build (no host bind mounts needed).
- If you update CSVs or model files locally, re-run `docker compose up --build`.

Services:
- Admin UI: http://localhost:3201/admin/
- Agent gateway: http://localhost:6203
- Python API (debug): http://localhost:8000

Optional (AI insights / explanations):
- Export `VOLCENGINE_API_KEY2` before running compose, or create a local `.env` file (not committed) at repo root:

```bash
export VOLCENGINE_API_KEY2=...
docker compose up --build
```
