# Fina Demo

Demo repository that combines:
- `agent/`: Node.js (Fastify) gateway + agent orchestration
- `prediction_app/`: Python (FastAPI) dataset + prediction APIs (including sales forecasting)
- `ai_web/`: React (Refine) admin UI

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
```

### 3) Start the admin UI (`ai_web/`, port 5173)

```bash
cd ai_web
pnpm install
pnpm dev
```

Notes:
- Vite proxies:
  - `/api/v1/*` -> `http://localhost:8000` (Python API)
  - `/api/*` -> `http://localhost:6203` (gateway)
