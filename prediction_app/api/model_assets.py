"""
Model Assets API

Reads models from the repo-level `models/` directory.

Expected layout:
  models/{model_name}/{version}/model.pkl
  models/{model_name}/{version}/metadata.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

# prediction_app/ is the project root for the Python service.
project_root = Path(__file__).parent.parent
repo_root = project_root.parent
models_root = repo_root / "models"

router = APIRouter(prefix="/api/v1/model-assets", tags=["model-assets"])


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            v = json.load(f)
            return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def scan_model_assets() -> List[Dict[str, Any]]:
    """Scan the repo-level models folder and return all discovered model versions."""
    items: List[Dict[str, Any]] = []
    if not models_root.exists():
        return items

    for name_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        for version_dir in sorted([p for p in name_dir.iterdir() if p.is_dir()]):
            meta_path = version_dir / "metadata.json"
            model_path = version_dir / "model.pkl"
            if not meta_path.exists() and not model_path.exists():
                continue

            meta = _safe_read_json(meta_path) if meta_path.exists() else {}

            name = str(meta.get("name") or name_dir.name)
            version = str(meta.get("version") or version_dir.name)

            metrics = meta.get("metrics")
            if not isinstance(metrics, dict):
                metrics = None

            items.append(
                {
                    "id": f"{name}:{version}",
                    "name": name,
                    "version": version,
                    "framework": meta.get("framework"),
                    "task": meta.get("task"),
                    "target_metric": meta.get("target_metric") or meta.get("target"),
                    "trained_at": meta.get("trained_at"),
                    "metrics": metrics,
                    "files": {
                        "model": str(model_path) if model_path.exists() else None,
                        "metadata": str(meta_path) if meta_path.exists() else None,
                    },
                }
            )

    return items


def _find_model_asset(name: str, version: str) -> tuple[Path, Path, Dict[str, Any]]:
    name_dir = models_root / name
    version_dir = name_dir / version
    if not version_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    meta_path = version_dir / "metadata.json"
    model_path = version_dir / "model.pkl"
    meta = _safe_read_json(meta_path) if meta_path.exists() else {}
    return model_path, meta_path, meta


@router.get("")
def list_model_assets(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=200),
):
    items = scan_model_assets()
    total = int(len(items))
    start = (page - 1) * pageSize
    end = start + pageSize
    page_items = items[start:end]
    total_pages = int((total + pageSize - 1) / pageSize) if pageSize else 1
    return {
        "success": True,
        "data": page_items,
        "total": total,
        "page": page,
        "pageSize": pageSize,
        "totalPages": total_pages,
    }


@router.get("/{model_name}/{version}")
def get_model_asset_detail(model_name: str, version: str):
    model_path, meta_path, meta = _find_model_asset(model_name, version)
    return {
        "success": True,
        "data": {
            "name": model_name,
            "version": version,
            "metadata": meta,
            "files": {
                "model": str(model_path) if model_path.exists() else None,
                "metadata": str(meta_path) if meta_path.exists() else None,
            },
        },
    }
