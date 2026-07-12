from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings


class EngineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EngineCapability:
    name: str
    available: bool
    unavailable_reason: str | None
    supported_extensions: set[str]
    max_size_mb: int | None
    requires_public_url: bool
    best_for: list[str]


@dataclass
class EngineRequest:
    source_url: str
    filename: str
    content_type: str | None
    size_bytes: int
    extension: str
    params: dict[str, Any]
    source_bytes: bytes | None = None


@dataclass
class EngineResult:
    engine: str
    raw: dict[str, Any]
    markdown: str | None = None
    json_content: dict[str, Any] | list[Any] | None = None
    plain_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EngineAdapter:
    name: str

    def __init__(self, settings: Settings):
        self.settings = settings

    def capability(self) -> EngineCapability:
        raise NotImplementedError

    def parse(self, request: EngineRequest) -> EngineResult:
        raise NotImplementedError

    def supports(self, extension: str, size_bytes: int) -> bool:
        cap = self.capability()
        if extension.lower() not in cap.supported_extensions:
            return False
        if cap.max_size_mb is not None and size_bytes > cap.max_size_mb * 1024 * 1024:
            return False
        return True
