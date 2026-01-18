#!/bin/bash
# Start API service

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "${ROOT_DIR}/.venv_py312/bin/python" ]]; then
  PY="${ROOT_DIR}/.venv_py312/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY="${ROOT_DIR}/.venv/bin/python"
else
  PY="python3"
fi

"${PY}" "${ROOT_DIR}/start_api.py"
