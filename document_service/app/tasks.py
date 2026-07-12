from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.dispatcher import Dispatcher, file_extension
from app.engines.base import EngineError, EngineRequest
from app.models import Asset, EngineJob, Run, utcnow
from app.normalizer import normalize_engine_result
from app.preprocessors import PreprocessError, can_render_word_to_pdf, render_word_to_pdf
from app.storage import S3Storage
from app.worker import celery_app


@celery_app.task(name="document_service.process_run")
def process_run(run_id: str) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        run = db.get(Run, run_id)
        if run is None:
            return
        if run.status == "cancelled":
            return
        source_asset_id = run.inputs.get("source_asset_id")
        asset = db.get(Asset, source_asset_id)
        if asset is None:
            _fail(db, run, "asset_not_found", f"Source asset not found: {source_asset_id}")
            return

        run.status = "running"
        db.commit()

        storage = S3Storage(settings)
        dispatcher = Dispatcher(settings)
        try:
            adapter = dispatcher.select_engine(run.requested_engine, asset.filename, asset.size_bytes, run.params)
        except EngineError as exc:
            _fail(db, run, exc.code, exc.message)
            return

        run.selected_engine = adapter.name
        db.commit()
        engine_job = _create_engine_job(db, run, adapter.name, asset)

        effective_asset = asset
        effective_extension = file_extension(asset.filename)
        effective_source_bytes = None
        preprocessed_pdf_asset = None

        if can_render_word_to_pdf(effective_extension, settings) and "pdf" in adapter.capability().supported_extensions:
            _update_engine_job(db, engine_job, local_status="preprocessing")
            try:
                original_bytes = storage.download_bytes(asset.storage_key)
                preprocessed = render_word_to_pdf(original_bytes, asset.filename, settings)
            except PreprocessError as exc:
                _update_engine_job(
                    db,
                    engine_job,
                    local_status="failed",
                    error_code=exc.code,
                    error_message=exc.message,
                    completed_at=utcnow(),
                )
                _fail(db, run, exc.code, exc.message)
                return
            preprocessed_pdf_asset = _save_bytes_asset(
                db,
                storage,
                filename=f"{run.id}-{preprocessed.filename}",
                key=f"runs/{run.id}/preprocessed/{preprocessed.filename}",
                data=preprocessed.data,
                content_type=preprocessed.content_type,
                kind="preprocessed_pdf",
                metadata={"run_id": run.id, "source_asset_id": asset.id, "preprocessor": "libreoffice"},
            )
            run.outputs = {"preprocessed_pdf": preprocessed_pdf_asset.id}
            db.commit()
            effective_asset = preprocessed_pdf_asset
            effective_extension = preprocessed.extension
            effective_source_bytes = preprocessed.data

            if not adapter.supports(effective_extension, effective_asset.size_bytes):
                _update_engine_job(
                    db,
                    engine_job,
                    local_status="failed",
                    error_code="file_too_large",
                    error_message=f"{adapter.name} does not support the rendered PDF size",
                    completed_at=utcnow(),
                )
                _fail(db, run, "file_too_large", f"{adapter.name} does not support the rendered PDF size")
                return

        source_url = storage.presigned_get_url(effective_asset.storage_key)
        source_bytes = None
        if not adapter.capability().requires_public_url:
            source_bytes = effective_source_bytes or storage.download_bytes(effective_asset.storage_key)
        engine_request = EngineRequest(
            source_url=source_url,
            filename=effective_asset.filename,
            content_type=effective_asset.content_type,
            size_bytes=effective_asset.size_bytes,
            extension=effective_extension,
            params=run.params,
            source_bytes=source_bytes,
        )
        _update_engine_job(
            db,
            engine_job,
            local_status="submitted",
            submit_payload={
                "operation": run.operation,
                "requested_engine": run.requested_engine,
                "selected_engine": adapter.name,
                "source_asset_id": asset.id,
                "effective_asset_id": effective_asset.id,
                "filename": effective_asset.filename,
                "content_type": effective_asset.content_type,
                "size_bytes": effective_asset.size_bytes,
                "extension": effective_extension,
                "params": run.params,
                "used_source_bytes": source_bytes is not None,
            },
        )

        try:
            engine_result = adapter.parse(engine_request)
        except EngineError as exc:
            if _is_cancelled(db, run):
                _update_engine_job(db, engine_job, local_status="cancelled", completed_at=utcnow())
                return
            _update_engine_job(
                db,
                engine_job,
                local_status="failed",
                remote_status="failed",
                error_code=exc.code,
                error_message=exc.message,
                last_poll_response={"error_code": exc.code, "error_message": exc.message},
                completed_at=utcnow(),
            )
            _fail(db, run, exc.code, exc.message)
            return
        except Exception as exc:  # defensive boundary for third-party adapters
            if _is_cancelled(db, run):
                _update_engine_job(db, engine_job, local_status="cancelled", completed_at=utcnow())
                return
            _update_engine_job(
                db,
                engine_job,
                local_status="failed",
                remote_status="failed",
                error_code="engine_exception",
                error_message=str(exc),
                last_poll_response={"error_code": "engine_exception", "error_message": str(exc)},
                completed_at=utcnow(),
            )
            _fail(db, run, "engine_exception", str(exc))
            return
        if _is_cancelled(db, run):
            _update_engine_job(db, engine_job, local_status="cancelled", completed_at=utcnow())
            return

        raw_asset = _save_json_asset(
            db,
            storage,
            filename=f"{run.id}-{adapter.name}-raw.json",
            key=f"runs/{run.id}/raw/{adapter.name}.json",
            payload=engine_result.raw,
            kind="engine_raw",
            metadata={"run_id": run.id, "engine": adapter.name},
        )
        document_ir = normalize_engine_result(
            source_asset_id=asset.id,
            result=engine_result,
            raw_asset_id=raw_asset.id,
        )
        ir_asset = _save_json_asset(
            db,
            storage,
            filename=f"{run.id}-document-ir.json",
            key=f"runs/{run.id}/document_ir.json",
            payload=document_ir.model_dump(),
            kind="document_ir",
            metadata={"run_id": run.id, "engine": adapter.name},
        )
        markdown_asset = None
        markdown = document_ir.content.get("markdown") or ""
        if markdown.strip():
            markdown_asset = _save_text_asset(
                db,
                storage,
                filename=f"{run.id}.md",
                key=f"runs/{run.id}/document.md",
                text=markdown,
                content_type="text/markdown; charset=utf-8",
                kind="document_markdown",
                metadata={"run_id": run.id, "engine": adapter.name},
            )

        run.status = "succeeded"
        run.outputs = {
            "preprocessed_pdf": preprocessed_pdf_asset.id if preprocessed_pdf_asset else run.outputs.get("preprocessed_pdf"),
            "document_ir": ir_asset.id,
            "markdown": markdown_asset.id if markdown_asset else None,
            "engine_raw": raw_asset.id,
            "engine": adapter.name,
            "summary": {
                "block_count": len(document_ir.blocks),
                "table_count": len(document_ir.tables),
                "has_markdown": bool(document_ir.content.get("markdown")),
            },
        }
        _update_engine_job(
            db,
            engine_job,
            local_status="succeeded",
            remote_status=_remote_status(engine_result.raw) or "succeeded",
            remote_job_id=_remote_job_id(engine_result.raw),
            remote_poll_url=_remote_poll_url(engine_result.raw),
            submit_response=_submit_response(engine_result.raw),
            last_poll_response={
                "engine_raw_asset_id": raw_asset.id,
                "document_ir_asset_id": ir_asset.id,
                "markdown_asset_id": markdown_asset.id if markdown_asset else None,
                "summary": run.outputs["summary"],
            },
            last_polled_at=utcnow(),
            completed_at=utcnow(),
        )
        db.commit()
    finally:
        db.close()


def enqueue_process_run(run_id: str) -> None:
    process_run.delay(run_id)


def _create_engine_job(db: Session, run: Run, engine: str, source_asset: Asset) -> EngineJob:
    job = EngineJob(
        id=f"ejob_{uuid4().hex}",
        run_id=run.id,
        engine=engine,
        local_status="selected",
        attempt_count=1,
        submit_payload={
            "operation": run.operation,
            "requested_engine": run.requested_engine,
            "source_asset_id": source_asset.id,
            "source_filename": source_asset.filename,
            "source_content_type": source_asset.content_type,
            "source_size_bytes": source_asset.size_bytes,
            "params": run.params,
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _update_engine_job(db: Session, job: EngineJob, **changes: Any) -> None:
    for key, value in changes.items():
        setattr(job, key, value)
    db.add(job)
    db.commit()
    db.refresh(job)


def _fail(db: Session, run: Run, code: str, message: str) -> None:
    run.status = "failed"
    run.error_code = code
    run.error_message = message
    db.commit()


def _is_cancelled(db: Session, run: Run) -> bool:
    db.refresh(run)
    return run.status == "cancelled"


def _save_json_asset(
    db: Session,
    storage: S3Storage,
    *,
    filename: str,
    key: str,
    payload: dict | list,
    kind: str,
    metadata: dict,
) -> Asset:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    stored = storage.upload_bytes(key, data, "application/json")
    asset = Asset(
        id=f"asset_{uuid4().hex}",
        filename=filename,
        content_type="application/json",
        size_bytes=stored.size_bytes,
        storage_key=stored.key,
        kind=kind,
        asset_metadata=metadata,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def _save_bytes_asset(
    db: Session,
    storage: S3Storage,
    *,
    filename: str,
    key: str,
    data: bytes,
    content_type: str,
    kind: str,
    metadata: dict,
) -> Asset:
    stored = storage.upload_bytes(key, data, content_type)
    asset = Asset(
        id=f"asset_{uuid4().hex}",
        filename=filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        storage_key=stored.key,
        kind=kind,
        asset_metadata=metadata,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def _save_text_asset(
    db: Session,
    storage: S3Storage,
    *,
    filename: str,
    key: str,
    text: str,
    content_type: str,
    kind: str,
    metadata: dict,
) -> Asset:
    data = text.encode("utf-8")
    stored = storage.upload_bytes(key, data, content_type)
    asset = Asset(
        id=f"asset_{uuid4().hex}",
        filename=filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        storage_key=stored.key,
        kind=kind,
        asset_metadata=metadata,
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def _submit_response(raw: dict[str, Any]) -> dict[str, Any] | None:
    submitted = raw.get("submitted") if isinstance(raw, dict) else None
    if isinstance(submitted, dict):
        return submitted
    if isinstance(raw, dict) and any(key in raw for key in ("request_check_url", "check_url", "task_id", "taskId", "id")):
        return raw
    return None


def _remote_job_id(raw: dict[str, Any]) -> str | None:
    submitted = _submit_response(raw) or {}
    data = submitted.get("data") if isinstance(submitted.get("data"), dict) else {}
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    result_data = result.get("data") if isinstance(result.get("data"), dict) else {}
    candidates = [
        data.get("jobId"),
        data.get("task_id"),
        data.get("taskId"),
        data.get("id"),
        submitted.get("jobId"),
        submitted.get("task_id"),
        submitted.get("taskId"),
        submitted.get("request_id"),
        submitted.get("id"),
        result_data.get("jobId"),
        result_data.get("task_id"),
        result_data.get("taskId"),
        result_data.get("id"),
        raw.get("request_id"),
        raw.get("task_id"),
        raw.get("taskId"),
        raw.get("id"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _remote_status(raw: dict[str, Any]) -> str | None:
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    result_data = result.get("data") if isinstance(result.get("data"), dict) else {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    candidates = [
        result_data.get("state"),
        result_data.get("status"),
        result.get("status"),
        data.get("state"),
        data.get("status"),
        raw.get("status"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate).lower()
    return None


def _remote_poll_url(raw: dict[str, Any]) -> str | None:
    submitted = _submit_response(raw) or {}
    for key in ("request_check_url", "check_url", "poll_url", "pollUrl"):
        value = submitted.get(key) or raw.get(key)
        if value:
            return str(value)
    return None
