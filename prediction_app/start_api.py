#!/usr/bin/env python3
"""
Convenience script to start the FastAPI service.

Note: This repo may contain multiple venvs. On Apple Silicon, we keep an arm64 venv
(`.venv_py312`) to ensure LightGBM works reliably. If it exists, we prefer it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prefer_arm64_venv() -> None:
    """Re-exec using prediction_app/.venv_py312 when present (avoids missing LightGBM)."""
    project_root = Path(__file__).parent
    preferred = project_root / ".venv_py312" / "bin" / "python"
    try:
        if preferred.exists():
            cur = Path(sys.executable).resolve()
            pref = preferred.resolve()
            if cur != pref:
                os.execv(str(pref), [str(pref), str(__file__), *sys.argv[1:]])
    except Exception:
        # Best-effort only; fall back to current interpreter.
        return


_prefer_arm64_venv()

from dotenv import load_dotenv  # noqa: E402

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded env file: {env_path}")
else:
    print(f"Env file not found (optional): {env_path}")

# Ensure project root is in sys.path so we can import api.xxx
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn  # noqa: E402

    port = int(os.getenv("PORT", 8000))
    print(f"Starting API server on port {port}...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=True)
