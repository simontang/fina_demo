from sqlalchemy import create_engine, inspect

from app.db import Base
from app.models import Asset, EngineJob, Run  # noqa: F401


def test_engine_jobs_table_schema_is_created():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("document_engine_jobs")}
    indexes = {index["name"] for index in inspector.get_indexes("document_engine_jobs")}

    assert "document_engine_jobs" in inspector.get_table_names()
    assert {
        "id",
        "run_id",
        "engine",
        "local_status",
        "remote_status",
        "remote_job_id",
        "remote_poll_url",
        "submit_payload",
        "submit_response",
        "last_poll_response",
        "attempt_count",
        "error_code",
        "error_message",
        "last_polled_at",
        "completed_at",
        "created_at",
        "updated_at",
    }.issubset(columns)
    assert {
        "ix_document_engine_jobs_run_id",
        "ix_document_engine_jobs_remote_job",
        "ix_document_engine_jobs_local_status",
    }.issubset(indexes)
