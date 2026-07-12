import pytest

from app.config import Settings
from app.engines.base import EngineError
from app.engines.registry import EngineRegistry


def make_settings(**overrides) -> Settings:
    values = {
        "database_url": "sqlite:///./test.db",
        "redis_url": "redis://localhost:6379/0",
        "object_storage_endpoint": "http://document-minio:9000",
        "object_storage_bucket": "documents",
        "object_storage_access_key": "minio",
        "object_storage_secret_key": "minio123",
        "public_presign_base_url": None,
        "datalab_api_key": None,
        "mineru_token": None,
        "textin_app_id": None,
        "textin_secret_code": None,
        "qwen_api_key": None,
        "paddleocr_api_url": None,
        "paddleocr_token": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def select(settings: Settings, filename: str, params: dict | None = None, size_bytes: int = 1024):
    extension = filename.rsplit(".", 1)[-1]
    return EngineRegistry(settings).select("auto", extension, size_bytes, params or {})


def test_auto_docx_prefers_datalab_for_general_parse():
    settings = make_settings(
        datalab_api_key="dl",
        mineru_token="mu",
        textin_app_id="app",
        textin_secret_code="secret",
        public_presign_base_url="https://files.example.com",
    )

    adapter = select(settings, "sow.docx", {"output_formats": ["markdown", "json"], "language_hint": ["zh", "en"]})

    assert adapter.name == "datalab"


def test_auto_accurate_prefers_mineru_when_public_url_exists():
    settings = make_settings(
        datalab_api_key="dl",
        mineru_token="mu",
        public_presign_base_url="https://files.example.com",
    )

    adapter = select(settings, "complex.pdf", {"mode": "accurate", "output_formats": ["markdown", "json"]})

    assert adapter.name == "mineru"


def test_auto_accurate_falls_back_when_public_url_is_local_minio():
    settings = make_settings(datalab_api_key="dl", mineru_token="mu")

    adapter = select(settings, "complex.pdf", {"mode": "accurate", "output_formats": ["markdown", "json"]})

    assert adapter.name == "datalab"


def test_auto_chinese_markdown_only_prefers_textin():
    settings = make_settings(
        datalab_api_key="dl",
        textin_app_id="app",
        textin_secret_code="secret",
    )

    adapter = select(settings, "rag.docx", {"output_formats": ["markdown"], "language_hint": ["zh"]})

    assert adapter.name == "textin"


def test_auto_image_prefers_qwen_when_public_url_exists():
    settings = make_settings(
        datalab_api_key="dl",
        qwen_api_key="qw",
        public_presign_base_url="https://files.example.com",
    )

    adapter = select(settings, "scan.png", {"output_formats": ["markdown"]})

    assert adapter.name == "qwen_ocr"


def test_auto_image_falls_back_when_qwen_cannot_fetch_local_url():
    settings = make_settings(datalab_api_key="dl", qwen_api_key="qw")

    adapter = select(settings, "scan.png", {"output_formats": ["markdown"]})

    assert adapter.name == "datalab"


def test_forced_unavailable_engine_fails():
    settings = make_settings()

    with pytest.raises(EngineError) as exc:
        EngineRegistry(settings).select("datalab", "docx", 1024, {})

    assert exc.value.code == "engine_unavailable"


def test_forced_public_url_engine_without_public_url_fails():
    settings = make_settings(mineru_token="mu")

    with pytest.raises(EngineError) as exc:
        EngineRegistry(settings).select("mineru", "pdf", 1024, {})

    assert exc.value.code == "public_url_required"


def test_no_supported_engine_fails():
    settings = make_settings()

    with pytest.raises(EngineError) as exc:
        EngineRegistry(settings).select("auto", "docx", 1024, {})

    assert exc.value.code == "no_supported_engine"


def test_forced_unsupported_extension_fails():
    settings = make_settings(datalab_api_key="dl")

    with pytest.raises(EngineError) as exc:
        EngineRegistry(settings).select("datalab", "exe", 1024, {})

    assert exc.value.code == "unsupported_format"


def test_forced_pdf_engine_can_accept_docx_via_local_preprocessing():
    settings = make_settings(paddleocr_api_url="https://paddle.example/jobs", paddleocr_token="token")

    adapter = EngineRegistry(settings).select("paddleocr_remote", "docx", 23 * 1024 * 1024, {})

    assert adapter.name == "paddleocr_remote"
