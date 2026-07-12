from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    database_url: str = Field(
        default="sqlite:///./document_service.db",
        validation_alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    object_storage_endpoint: str | None = Field(default=None, validation_alias="OBJECT_STORAGE_ENDPOINT")
    object_storage_region: str = Field(default="us-east-1", validation_alias="OBJECT_STORAGE_REGION")
    object_storage_bucket: str = Field(default="documents", validation_alias="OBJECT_STORAGE_BUCKET")
    object_storage_access_key: str = Field(default="", validation_alias="OBJECT_STORAGE_ACCESS_KEY")
    object_storage_secret_key: str = Field(default="", validation_alias="OBJECT_STORAGE_SECRET_KEY")
    object_storage_force_path_style: bool = Field(default=True, validation_alias="OBJECT_STORAGE_FORCE_PATH_STYLE")
    public_presign_base_url: str | None = Field(default=None, validation_alias="PUBLIC_PRESIGN_BASE_URL")
    presigned_url_ttl_seconds: int = Field(default=3600, validation_alias="PRESIGNED_URL_TTL_SECONDS")

    datalab_api_key: str | None = Field(default=None, validation_alias="DATALAB_API_KEY")
    datalab_base_url: str = Field(default="https://www.datalab.to", validation_alias="DATALAB_BASE_URL")

    mineru_token: str | None = Field(default=None, validation_alias="MINERU_TOKEN")
    mineru_base_url: str = Field(default="https://mineru.net", validation_alias="MINERU_BASE_URL")
    mineru_model_version: str = Field(default="vlm", validation_alias="MINERU_MODEL_VERSION")

    textin_app_id: str | None = Field(default=None, validation_alias="TEXTIN_APP_ID")
    textin_secret_code: str | None = Field(default=None, validation_alias="TEXTIN_SECRET_CODE")
    textin_base_url: str = Field(default="https://api.textin.com", validation_alias="TEXTIN_BASE_URL")

    qwen_api_key: str | None = Field(default=None, validation_alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="QWEN_BASE_URL",
    )
    qwen_model: str = Field(default="qwen3.5-ocr", validation_alias="QWEN_MODEL")

    paddleocr_api_url: str | None = Field(default=None, validation_alias="PADDLEOCR_API_URL")
    paddleocr_token: str | None = Field(default=None, validation_alias="PADDLEOCR_TOKEN")
    paddleocr_model: str = Field(default="PaddleOCR-VL-1.6", validation_alias="PADDLEOCR_MODEL")

    local_docx_to_pdf_enabled: bool = Field(default=True, validation_alias="LOCAL_DOCX_TO_PDF_ENABLED")
    libreoffice_binary: str = Field(default="soffice", validation_alias="LIBREOFFICE_BINARY")

    default_parse_operation: Literal["document.parse"] = "document.parse"


@lru_cache
def get_settings() -> Settings:
    return Settings()
