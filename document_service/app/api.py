from __future__ import annotations

import mimetypes
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.dispatcher import file_extension
from app.engines.base import EngineError
from app.engines.registry import EngineRegistry
from app.models import Asset, Run
from app.schemas import AssetResponse, CreateRunRequest, DownloadUrlResponse, EngineInfo, RunResponse
from app.storage import S3Storage, StorageError
from app.tasks import enqueue_process_run

router = APIRouter(prefix="/v1")


@router.post("/assets", response_model=AssetResponse, status_code=status.HTTP_201_CREATED)
async def create_asset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AssetResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    asset_id = f"asset_{uuid4().hex}"
    safe_filename = file.filename or "upload.bin"
    content_type = _content_type(safe_filename, file.content_type)
    key = f"assets/{asset_id}/{safe_filename}"
    try:
        storage = S3Storage(settings)
        stored = storage.upload_bytes(key, data, content_type)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    asset = Asset(
        id=asset_id,
        filename=safe_filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        storage_key=stored.key,
        kind="original",
        asset_metadata={"extension": file_extension(safe_filename)},
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return _asset_response(asset)


@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    request: CreateRunRequest,
    db: Session = Depends(get_db),
) -> RunResponse:
    asset = db.get(Asset, request.inputs.source_asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="source asset not found")
    run = Run(
        id=f"run_{uuid4().hex}",
        operation=request.operation,
        requested_engine=request.engine,
        status="queued",
        inputs=request.inputs.model_dump(),
        params=request.params.model_dump(),
        outputs={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    try:
        enqueue_process_run(run.id)
    except Exception as exc:
        run.status = "failed"
        run.error_code = "queue_unavailable"
        run.error_message = str(exc)
        db.commit()
        raise HTTPException(status_code=503, detail="run queue is unavailable") from exc
    return _run_response(run)


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, db: Session = Depends(get_db)) -> RunResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_response(run)


@router.get("/runs/{run_id}/outputs/markdown")
def get_run_markdown(
    run_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    markdown_asset_id = run.outputs.get("markdown")
    if not markdown_asset_id:
        raise HTTPException(status_code=404, detail="markdown output not found")
    asset = db.get(Asset, markdown_asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="markdown asset not found")
    try:
        data = S3Storage(settings).download_bytes(asset.storage_key)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{asset.filename}"'},
    )


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: str, db: Session = Depends(get_db)) -> RunResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status in {"succeeded", "failed"}:
        return _run_response(run)
    run.status = "cancelled"
    db.commit()
    db.refresh(run)
    return _run_response(run)


@router.get("/assets/{asset_id}/download-url", response_model=DownloadUrlResponse)
def get_download_url(
    asset_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DownloadUrlResponse:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="asset not found")
    storage = S3Storage(settings)
    return DownloadUrlResponse(
        asset_id=asset.id,
        download_url=storage.presigned_get_url(asset.storage_key),
        expires_in_seconds=settings.presigned_url_ttl_seconds,
    )


@router.get("/engines", response_model=list[EngineInfo])
def list_engines(settings: Settings = Depends(get_settings)) -> list[EngineInfo]:
    registry = EngineRegistry(settings)
    return [
        EngineInfo(
            name=cap.name,
            available=cap.available,
            unavailable_reason=cap.unavailable_reason,
            supported_extensions=sorted(cap.supported_extensions),
            max_size_mb=cap.max_size_mb,
            requires_public_url=cap.requires_public_url,
            best_for=cap.best_for,
        )
        for cap in (adapter.capability() for adapter in registry.list())
    ]


@router.get("/operations")
def list_operations() -> list[dict]:
    return [
        {
            "operation": "document.parse",
            "execution": "async_run",
            "inputs": ["source_asset_id"],
            "params": ["output_formats", "mode", "language_hint", "page_ranges"],
            "outputs": ["document_ir", "engine_raw"],
        }
    ]


@router.post("/engines/select")
def select_engine_preview(
    filename: str,
    size_bytes: int,
    engine: str = "auto",
    settings: Settings = Depends(get_settings),
) -> dict:
    registry = EngineRegistry(settings)
    try:
        adapter = registry.select(engine, file_extension(filename), size_bytes, {})
    except EngineError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
    return {"engine": adapter.name}


def _asset_response(asset: Asset) -> AssetResponse:
    return AssetResponse(
        asset_id=asset.id,
        filename=asset.filename,
        content_type=asset.content_type,
        size_bytes=asset.size_bytes,
        storage_key=asset.storage_key,
        kind=asset.kind,
    )


def _content_type(filename: str, provided: str | None) -> str | None:
    if provided and provided != "application/octet-stream":
        return provided
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or provided


def _run_response(run: Run) -> RunResponse:
    return RunResponse(
        run_id=run.id,
        operation=run.operation,
        requested_engine=run.requested_engine,
        selected_engine=run.selected_engine,
        status=run.status,  # type: ignore[arg-type]
        inputs=run.inputs,
        params=run.params,
        outputs=run.outputs,
        error_code=run.error_code,
        error_message=run.error_message,
    )
