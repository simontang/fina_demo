# Fina Demo: Microservices, Ports, APIs, and URL Design

This doc explains two runtime issues and proposes a clean, consistent URL/API design for the whole demo stack.

## 1) Current Services (What Runs Where)

### 1. `ai_web` (React Admin UI + Nginx)
- Role: serves the SPA under `/admin/`, and reverse-proxies API calls to the backend.
- Container/service name: `ai_web`
- Default port:
  - Host: `3201`
  - Container: `3201`
- Key config:
  - Build-time env: `VITE_API_URL` defaults to `/api` in `ai_web/Dockerfile:20` which becomes the base for BFF calls.

### 2. `agent` (Node/Fastify + LatticeGateway + API gateway)
- Role:
  - Single backend entrypoint for the UI
  - LatticeGateway endpoints (assistants/threads/runs/state)
  - Reverse-proxy `/api/v1/*` to Python (`prediction_app`)
  - RTC voice chat proxy endpoints
  - Optional “BFF” endpoints for datasets/files/agents
- Container/service name: `agent`
- Default port:
  - Host: `6203`
  - Container: `6203`

### 3. `prediction_app` (Python/FastAPI)
- Role:
  - Dataset endpoints (list/detail/preview/stats)
  - RFM analysis (CSV-based)
  - Sales forecast, stock allocation simulation, model assets
- Container/service name: `prediction_app`
- Default port:
  - Host: `8000`
  - Container: `8000`

### 4. Optional `postgres` (not currently in docker-compose.yml)
- Role: persistence for:
  - dataset table storage (if you want DB-backed preview/stats)
  - agent checkpointing/memory (if enabled in the future)
- Default port:
  - Host: `5432`
  - Container: `5432`

## 2) Issue Analysis

### Issue A
`Route GET:/api/bff/api/assistants/deep_research_agent/deep_research_agent_thread_1/state not found`

#### What happens
1. The UI chat component builds a BFF base URL from `VITE_API_URL` and appends `/bff`:
   - `ai_web/src/components/chating/index.tsx:14` ~ `ai_web/src/components/chating/index.tsx:21`
   - If `VITE_API_URL=/api`, then `baseURL` becomes `/api/bff`.
2. The `@axiom-lattice/react-sdk` then requests Lattice endpoints under:
   - `/api/bff/api/assistants/.../state`
3. `ai_web/nginx.conf` currently proxies **all** `/api/*` to `agent:6203` without rewrite:
   - `ai_web/nginx.conf:8`
4. The agent **does not** register routes under `/api/bff/*`. It registers them under `/bff/*`:
   - `agent/src/gateway.ts:29` (LatticeGateway under `/bff`)
   - `agent/src/gateway.ts:64` (custom BFF endpoints under `/bff`)

Result: the browser calls `/api/bff/...`, Nginx forwards that to the agent as `/api/bff/...`, but the agent only has `/bff/...` → Fastify returns `Route GET:/api/bff/... not found`.

#### Fix options (pick one)

Option A (proxy rewrite; minimal backend change):
- Update `ai_web/nginx.conf` to rewrite:
  - `/api/bff/*` → `http://agent:6203/bff/*`
  - keep `/api/v1/*` → `http://agent:6203/api/v1/*`
  - keep `/api/rtc/*` → `http://agent:6203/api/rtc/*`

Option B (backend alias; minimal proxy change) **recommended**:
- Keep Nginx simple (still proxy `/api/*` as-is), but register BFF routes in the agent under **both**:
  - `/bff/*` (existing, backward-compatible)
  - `/api/bff/*` (new, matches UI default `VITE_API_URL=/api`)

Rationale: the UI build already assumes `/api` as the API base (see `ai_web/Dockerfile:20`), so exposing `/api/bff/*` in the agent removes the need for proxy rewriting in both Nginx and Vite dev proxy.

---

### Issue B
`/var/run/postgresql/.s.PGSQL.5432 failed: No such file or directory ...`

#### What happens
Some dataset endpoints in `prediction_app` still attempt to read from Postgres tables, even when no Postgres is configured/running:
- DB connection uses env `DB_HOST/DB_NAME/...`:
  - `prediction_app/api/datasets.py:58`
  - When `DB_HOST` is missing/empty, `psycopg2` falls back to a local Unix socket:
    - `/var/run/postgresql/.s.PGSQL.5432`
- These endpoints call DB-backed helpers:
  - `GET /api/v1/datasets/{dataset_id}` → `prediction_app/api/datasets.py:1169`
  - `GET /api/v1/datasets/{dataset_id}/preview` → `prediction_app/api/datasets.py:1121`
  - `GET /api/v1/datasets/{dataset_id}/stats` → `prediction_app/api/datasets.py:1232`

Even though RFM analysis is already CSV-based:
- `POST /api/v1/datasets/{dataset_id}/rfm` explicitly reads CSV (no DB):
  - `prediction_app/api/datasets.py:1268`

Result: dataset “detail/preview/stats” fail whenever Postgres is absent.

#### Fix options (pick one)

Option A (CSV-first, no DB dependency) **recommended for this demo**
- Treat `prediction_app/config/datasets.json` (`csv_path`) as the source of truth.
- For dataset detail/preview/stats:
  - If DB env is not configured, load from CSV.
  - If DB env *is* configured, optionally allow DB-backed mode (for large tables).

Option B (DB-backed mode)
- Add a `postgres` service to `docker-compose.yml`.
- Provide `DB_HOST=postgres`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, and import CSV into tables.

Rationale: the product requirements earlier state “load CSV directly, do not read from DB”, so Option A matches the intended demo behavior.

## 3) Proposed Canonical URL / API Namespace

Goal: browser talks to **one origin** (the UI origin), and the gateway (`agent`) fans out to internal services.

### Public (browser-facing) URLs
- UI: `http://<host>:3201/admin/`
- APIs (single origin): `http://<host>:3201/api/...`

### Nginx (`ai_web`) routing
Recommended proxy rules:
- `/admin/*` → SPA static files
- `/api/*` → `agent:6203` (gateway)

### Agent (`agent`) routing (recommended target)

Namespace design:
- `GET /api/health` → gateway health (optional alias)
- `GET /api/bff/health` → LatticeGateway health
- `* /api/bff/api/*` → LatticeGateway API (assistants/threads/runs/state)
- `* /api/v1/*` → reverse-proxy to Python (`prediction_app:8000`)
- `POST /api/rtc/proxyFetch` → RTC proxy
- `POST /api/rtc/voice_chat` → RTC callback (optional)
- `POST /api/rtc/update-trigger` → RTC unicast trigger (optional)

Compatibility:
- Keep existing `/bff/*` endpoints for direct access via `http://localhost:6203/bff/*`
  - but the UI should standardize on `/api/bff/*` via Nginx.

### Python (`prediction_app`) routing
Keep Python endpoints under `/api/v1/*` and let the agent proxy them:
- `/api/v1/datasets/*`
- `/api/v1/models/*`
- `/api/v1/model-assets/*`
- `/api/v1/datasets/{dataset_id}/rfm`
- `/api/v1/datasets/{dataset_id}/sales-forecast`
- `/api/v1/datasets/{dataset_id}/stock-allocation/*`

## 4) Ports and Service-to-Service Connectivity

### Local dev (without Docker)
- `ai_web` dev server: `http://localhost:5173/admin/`
- `agent`: `http://localhost:6203`
- `prediction_app`: `http://localhost:8000`

Vite proxy (current):
- `/api/*` → `http://localhost:6203` (`ai_web/vite.config.ts:12`)
- `/api/v1/*` → `http://localhost:6203` (`ai_web/vite.config.ts:12`)

Standard: use `/api/bff/*` from the browser.
- The agent exposes `/api/bff/*` directly, so Vite/Nginx do not need rewrites.

### Docker compose (recommended demo runtime)
- UI: `http://localhost:3201/admin/`
- Agent (debug): `http://localhost:6203`
- Python (debug): `http://localhost:8000`

Inside the compose network:
- `ai_web` → `agent:6203`
- `agent` → `prediction_app:8000`

## 5) Configuration Checklist (Minimal)

### Required for “chat UI works”
- UI build arg: `VITE_API_URL=/api` (default in Dockerfile)
- Ensure `/api/bff/*` resolves to agent BFF routes (either via proxy rewrite or agent alias)

### Required for “dataset detail works without Postgres”
- Ensure dataset endpoints do not require DB (CSV-first), or configure Postgres + `DB_*` env.

## 6) Decisions Needed (please confirm)

Implemented / chosen design

1) Issue A (BFF routing): expose **both** prefixes
- Direct on agent: `http://localhost:6203/bff/*`
- Via UI `/api` proxy: `http://localhost:3201/api/bff/*`

This matches the frontend’s default `VITE_API_URL=/api` behavior, so nginx/vite do not need special rewrites for `/api/bff`.

2) Issue B (dataset detail): CSV-first (no Postgres required)
- Python dataset APIs under `/api/v1/datasets/*` read from the configured `csv_path` in `prediction_app/config/datasets.json`.
- This removes the dependency on a local Postgres socket (`/var/run/postgresql/.s.PGSQL.5432`).

Optional: you can still add Postgres later if you want DB-backed tables for very large datasets, but it is not required for this demo stack.
