# Python API Startup Notes

## 1) `[Errno 48] Address already in use` (port is busy)

Cause: Port `8000` is already used by another process.

Fix:
```bash
# Find the process using port 8000
lsof -i :8000

# Kill it (replace <PID>)
kill -9 <PID>

# Or start on a different port
PORT=8001 ./start_api.sh
```

## 2) Missing dependencies (e.g. `email-validator>=2.0`)

Cause: The active Python environment does not have required packages installed.

Fix (recommended): use a venv under `prediction_app/` and install requirements:
```bash
cd prediction_app
python3 -m venv .venv_py312
source .venv_py312/bin/activate
pip install -r api/requirements.txt

./start_api.sh
```

## 3) LightGBM issues on Apple Silicon (arm64)

Symptoms:
- `Sales forecast failed: No module named 'lightgbm'`
- or LightGBM import errors related to OpenMP (`libomp`)

Fix:
- Prefer the arm64 venv `prediction_app/.venv_py312`
- Start the API via `prediction_app/start_api.sh` (it prefers `.venv_py312` when present)

## 4) `Could not import module "app"`

Cause: Starting uvicorn from inside `api/` with `app:app` can break imports (`api.*`).

Fix: Start from the `prediction_app/` root using `api.app:app` (handled by `start_api.py` / `start_api.sh`).
