# Document Service

Unified document IO service for remote document parsing engines.

The service stores files in S3-compatible object storage, records runs and
engine jobs in SQL storage, runs asynchronous parse jobs through Redis/Celery,
and dispatches to real remote engines only. Missing credentials mark an engine
as unavailable; there is no runtime mock fallback. Docker Compose defaults to
Postgres, and MySQL is supported through `mysql+pymysql://...` plus the DDL in
`sql/mysql/`.

Word documents are rendered to PDF with headless LibreOffice before engine
execution when `LOCAL_DOCX_TO_PDF_ENABLED=true`. The rendered PDF is saved as a
`preprocessed_pdf` asset and then passed to the selected remote engine.

## API

- `POST /v1/assets` uploads a file and returns an `asset_id`.
- `POST /v1/runs` creates an asynchronous `document.parse` run.
- `GET /v1/runs/{run_id}` returns status and output asset ids.
- `GET /v1/runs/{run_id}/outputs/markdown` downloads the Markdown output file.
- `GET /v1/assets/{asset_id}/download-url` returns a presigned download URL.
- `GET /v1/engines` returns engine capabilities and availability.
- `GET /v1/operations` returns supported operation contracts.
- `POST /v1/runs/{run_id}/cancel` marks a local queued/running run as cancelled.

## Run Locally

```bash
cp document_service/.env.example document_service/.env
docker compose up --build \
  document-postgres document-redis document-minio document-minio-init \
  document-api document-worker
```

API: <http://localhost:5710/docs>

In the main `fina_demo` compose stack the host port is fixed at `5710`. Service
credentials, Redis, database, and object storage settings still come from
`document_service/.env`.

If `document_service/.env` points to TOS or another managed S3-compatible
bucket, `document-minio` and `document-minio-init` are not required.

The service is also kept compatible with the standalone compose file:

```bash
docker compose -f docker-compose.document-service.yml --env-file document_service/.env up --build
```

Cloud engines that fetch files by URL require object storage presigned URLs to
be reachable from the public internet or the engine's private network.

## Engine Job Tracking

Each run creates a `document_engine_jobs` row. It tracks the selected engine,
local lifecycle status, remote job id/status when available, submit payload,
submit response, last poll summary, attempts, and terminal errors.

For Postgres deployments, apply:

```bash
psql "$DOCUMENT_SERVICE_POSTGRES_DSN" \
  -f document_service/sql/postgres/001_create_document_engine_jobs.sql
```

For MySQL deployments, apply:

```bash
mysql "$DOCUMENT_SERVICE_MYSQL_DSN" < document_service/sql/mysql/001_create_document_engine_jobs.sql
```

## Example

```bash
curl -F "file=@/path/to/工作说明书标准模板（双语）.docx" \
  http://localhost:5710/v1/assets

curl -X POST http://localhost:5710/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "document.parse",
    "engine": "auto",
    "inputs": {"source_asset_id": "asset_xxx"},
    "params": {
      "output_formats": ["markdown", "json"],
      "mode": "balanced",
      "language_hint": ["zh", "en"],
      "page_ranges": null
    }
  }'
```

## Development

```bash
cd document_service
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
