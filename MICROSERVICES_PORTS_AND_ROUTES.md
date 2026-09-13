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

### 5. `cdp_service` (Spring Boot CDP segment service)
- Role:
  - Multi-tenant segment definition CRUD
  - Multi-tenant segment data CRUD
  - Processing endpoint that executes a definition SQL against a configured CDP PostgreSQL datasource and stores the result as a JSON snapshot
- Container/service name: `cdp_service`
- Current root compose port:
  - Host: `5706`
  - Container: `5706`
- Key APIs:
  - `GET/POST/PUT/DELETE /api/v1/segment-definitions`
  - `GET/POST/PUT/DELETE /api/v1/segment-data`
  - `POST /api/v1/segment-definitions/{id}/process`

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
- CDP service (debug): `http://localhost:5706`

Inside the compose network:
- `ai_web` → `agent:6203`
- `agent` → `prediction_app:8000`
- `cdp_service` → configured CDP PostgreSQL datasource via `t_datasource_config`

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

## 7) Document Service

Document parsing is now part of the `fina_demo` compose stack.

- Service: `document-api`
- Worker: `document-worker`
- Local API: `http://localhost:5710`
- Swagger: `http://localhost:5710/docs`
- Nginx route: `/api/documents/v1/*` → `http://127.0.0.1:5710/v1/*`
- Internal queue: `document-redis:6379`
- Local helper services: `document-postgres`, `document-redis`
- Optional local object storage: `document-minio`, only needed when
  `document_service/.env` points at the local MinIO endpoint.

Core endpoints:

- `POST /api/documents/v1/assets`
- `POST /api/documents/v1/runs`
- `GET /api/documents/v1/runs/{run_id}`
- `GET /api/documents/v1/runs/{run_id}/outputs/markdown`
- `GET /api/documents/v1/engines`

The service reads `document_service/.env` for third-party engine credentials,
object storage, Redis, and database configuration. For production-style runs in
this project, use Aliyun PostgreSQL and TOS-compatible object storage so
URL-based engines such as MinerU and Qwen OCR can fetch presigned HTTPS files.

## 8) Platform Service（files + webhooks）

Spring Boot 3.2 / Java 17 multi-tenant reusable service combining two modules
from the platform base architecture
(`docs/architecture-discussions/agentic-semantic-factory/`). Webhook backend:
self-hosted Svix (MIT) behind a tenant-model facade.

- Local API: `http://localhost:5707`
- Compose: built from `./platform-service`; sidecars `svix-server` (image
  `svix/svix-server`, internal only) and `platform-db-init` (creates the
  `file_service` + `svix` databases). Schema migrations are automatic:
  Flyway (`db/migration/V*`) runs at startup — fresh databases get V1 from
  scratch, pre-Flyway databases are baselined without touching existing
  tables; svix-server runs its own migrations on start
- Metadata in `document-postgres` shared `postgres` db (Flyway-managed,
  same pattern as metrics-server); objects in `document-minio` bucket `files`
  (override with `FILE_OBJECT_STORAGE_*` for TOS/S3); Svix state in its own
  `svix` database + dedicated `svix-redis` (prod) / `document-redis` db 3 (dev)
- Nginx routes: `/api/filesvc/*` → `5707 /api/v1/*` (agent BFF owns
  `/api/files/*`); `/api/webhooks/*` → `5707 /api/v1/webhooks/*`
- Auth: `X-Tenant-Id` header (required; `FILE_SERVICE_DEFAULT_TENANT` provides
  a dev default) + optional `X-Api-Key` (`FILE_SERVICE_API_KEY`)
- Contract: the tenant rides ONLY in the `X-Tenant-Id` header — never in the
  body, never in `path`, and it is not echoed back in responses. Caller paths
  are tenant-relative; the tenant prefix is applied internally to the storage
  key (never exposed); no sequential ids are exposed either
  (enumeration/information-leak risk)

### files 模块

- Addressing: full logical path `{dir}/{filename}`. Immutable versioning:
  re-upload appends a version, identical content dedupes; no folder entities —
  directories are path-prefix aggregates
- Key APIs — S3-style resource addressing (object key is the URL path),
  merged with fina-ai interactions; no sequential ids exposed:
  - `POST /api/v1/files/upload` — multipart (`path`, `fileName?`, `fileCategory?`, `usage?`, `meta?`)
  - `PUT /api/v1/files/{key…}` — raw-stream upload; metadata via `X-File-Category/Usage/Meta` headers
  - `GET /api/v1/files/{key…}?version=&bom=` — direct download
  - `HEAD /api/v1/files/{key…}` — metadata as headers (`ETag`=sha256, `X-File-Md5/Version/Meta`)
  - `DELETE /api/v1/files/{key…}?version=` — soft delete
  - `GET /api/v1/files?prefix=&delimiter=/` — pseudo-directory listing
  - `POST /api/v1/files/presign` {path, version?, ttlSeconds?} — download URL.
    Reachable/cloud storage → storage-native presigned URL (bandwidth bypasses
    this service); internal storage (self-hosted MinIO) → our own download URL.
    `FILE_LINK_MODE` forces `auto` (default) or `presign`;
    `PUBLIC_FILE_BASE_URL` rewrites the presigned host when storage is
    published behind another domain. Requires the tenant header.
- Smoke: `platform-service/scripts/smoke.sh` — 11/11 passed against a TOS
  S3-compatible bucket (2026-09-13)

### webhooks 模块（Svix facade）

Our tenant model on the outside, Svix on the inside (tenant=Svix application,
topic=event type auto-registered, destination=endpoint with auto-generated
`whsec_` secret). Deliveries use Standard Webhooks headers, at-least-once with
Svix's retry schedule and default SSRF protection.

- `POST /api/v1/webhooks/destinations` {url, topics[], description?} →
  {endpointId, secret, topics}
- `GET/DELETE /api/v1/webhooks/destinations[/{endpointId}]`
- `POST /api/v1/webhooks/publish` {topic, data} → {messageId, topic}
- `GET /api/v1/webhooks/messages?limit=`,
  `GET /api/v1/webhooks/messages/{messageId}/attempts`
- Topics: `import.completed`, `gate.passed`, `decision.captured`,
  `job.completed`, `run.published`
- Scripts (`platform-service/scripts/`): `provision-destination.sh`,
  `publish.py`, `mock-receiver.py` (signature-verifying receiver, accepts both
  `svix-*` and `webhook-*` header families),
  `webhook-smoke.sh` — **full delivery+signature pass verified locally on
  2026-09-13** (svix-server image pulled via the `docker.1panel.live` mirror;
  note `SVIX_WHITELIST_SUBNETS` must be a JSON array)
- Portal: `http://localhost:5707/portal` — self-contained static page (no JS
  toolchain): enter a tenant id to manage destinations, browse events, and
  inspect delivery attempts. Internal tool served by the service itself; add
  an nginx `/portal/` route only if external access is wanted
