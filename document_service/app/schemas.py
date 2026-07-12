from typing import Any, Literal

from pydantic import BaseModel, Field


EngineName = Literal["auto", "datalab", "mineru", "textin", "qwen_ocr", "paddleocr_remote"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class AssetResponse(BaseModel):
    asset_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    storage_key: str
    kind: str


class RunInputs(BaseModel):
    source_asset_id: str


class RunParams(BaseModel):
    output_formats: list[str] = Field(default_factory=lambda: ["markdown", "json"])
    mode: Literal["fast", "balanced", "accurate"] = "balanced"
    language_hint: list[str] = Field(default_factory=list)
    page_ranges: list[str] | None = None


class CreateRunRequest(BaseModel):
    operation: Literal["document.parse"] = "document.parse"
    engine: EngineName = "auto"
    inputs: RunInputs
    params: RunParams = Field(default_factory=RunParams)


class RunResponse(BaseModel):
    run_id: str
    operation: str
    requested_engine: str
    selected_engine: str | None
    status: RunStatus
    inputs: dict[str, Any]
    params: dict[str, Any]
    outputs: dict[str, Any]
    error_code: str | None = None
    error_message: str | None = None


class DownloadUrlResponse(BaseModel):
    asset_id: str
    download_url: str
    expires_in_seconds: int


class EngineInfo(BaseModel):
    name: str
    available: bool
    unavailable_reason: str | None = None
    supported_extensions: list[str]
    max_size_mb: int | None = None
    requires_public_url: bool
    best_for: list[str]


class DocumentBlock(BaseModel):
    id: str
    type: str
    text: str = ""
    page: int | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentTable(BaseModel):
    id: str
    rows: list[list[str]]
    page: int | None = None


class DocumentIR(BaseModel):
    source_asset_id: str
    engine: str
    content: dict[str, str]
    blocks: list[DocumentBlock]
    tables: list[DocumentTable]
    raw_outputs: dict[str, str]
