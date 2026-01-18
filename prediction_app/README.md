# Prediction App

Python service for dataset management, model inference/deployment, and sales forecasting.

Key features:
- Dataset APIs (`/api/v1/datasets/*`)
- Generic model inference (`/api/v1/predict`) and model deployment management (`/api/v1/models/*`)
- Sales forecasting (`/api/v1/datasets/{dataset_id}/sales-forecast`) using a **global LightGBM** model
- Model asset management that scans repo-root `models/` (`/api/v1/model-assets/*`)

## Repository Layout

```
prediction_app/
  api/                 FastAPI app + inference
  training/            training scripts
  config/              datasets.json, etc.
  scripts/             optional DB import tools
  shared/              shared utilities
  start_api.py
  start_api.sh

raw_data/              input CSVs
models/                model assets (repo-root)
  {model_name}/{version}/
    model.pkl
    metadata.json
```

## Quickstart

### 1) Create a venv + install dependencies

On Apple Silicon, prefer an arm64 venv (we use `.venv_py312` in this repo):

```bash
cd prediction_app
python3 -m venv .venv_py312
source .venv_py312/bin/activate
pip install -r api/requirements.txt
```

### 2) Start the API

```bash
cd prediction_app
./start_api.sh
```

The API runs at `http://localhost:8000`.

### 3) Train a Sales Forecast model (LightGBM, revenue)

This trains a global regressor and writes to repo-root:
`models/{model_name}/{version}/model.pkl` + `metadata.json`.

```bash
cd prediction_app/training
python train_sales_forecast_lgbm.py \
  --model-name sales_forecast \
  --version v1.0.0 \
  --holiday-country GB \
  --val-days 30
```

## API Endpoints (high level)

### Health
`GET /health`

### List available models (builtin + deployed + assets + local training)
`GET /api/v1/models/available`

### Model asset management (scans repo-root `models/`)
`GET /api/v1/model-assets`

### Dataset APIs
- `GET /api/v1/datasets`
- `GET /api/v1/datasets/{dataset_id}`
- `GET /api/v1/datasets/{dataset_id}/preview?page=1&pageSize=20`

### Sales Forecast (dataset-scoped)

Builds calendar + lag/rolling features from recent history and runs inference for the future horizon.

`POST /api/v1/datasets/{dataset_id}/sales-forecast`

```json
{
  "model_id": "sales_forecast:v1.0.0",
  "target_entity_id": "85123A",
  "forecast_horizon": 7,
  "context_window_days": 60,
  "sales_metric": "revenue",
  "promotion_factor": 1.0,
  "holiday_country": "GB",
  "rounding": "none"
}
```

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Common variables:
- `PORT` (default: `8000`)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (optional; some features use Postgres)
