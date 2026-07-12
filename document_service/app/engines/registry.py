from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from app.config import Settings
from app.engines.base import EngineAdapter, EngineError
from app.engines.datalab import DatalabAdapter
from app.engines.mineru import MinerUAdapter
from app.engines.paddleocr_remote import PaddleOcrRemoteAdapter
from app.engines.qwen_ocr import QwenOcrAdapter
from app.engines.textin import TextInAdapter
from app.preprocessors import WORD_EXTENSIONS


class EngineRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.adapters: dict[str, EngineAdapter] = {
            adapter.name: adapter
            for adapter in [
                DatalabAdapter(settings),
                MinerUAdapter(settings),
                TextInAdapter(settings),
                QwenOcrAdapter(settings),
                PaddleOcrRemoteAdapter(settings),
            ]
        }

    def list(self) -> list[EngineAdapter]:
        return list(self.adapters.values())

    def get(self, name: str) -> EngineAdapter | None:
        return self.adapters.get(name)

    def select(self, requested: str, extension: str, size_bytes: int, params: dict) -> EngineAdapter:
        extension = extension.lower().lstrip(".")
        if requested != "auto":
            adapter = self.adapters.get(requested)
            if adapter is None:
                raise EngineError("unknown_engine", f"Unknown engine: {requested}")
            cap = adapter.capability()
            if not cap.available:
                raise EngineError("engine_unavailable", cap.unavailable_reason or f"{requested} is unavailable")
            if not self._supports_input(adapter, extension, size_bytes):
                raise EngineError("unsupported_format", f"{requested} does not support .{extension} or file size")
            if cap.requires_public_url and not self._public_url_available():
                raise EngineError("public_url_required", f"{requested} requires a public HTTPS presigned URL")
            return adapter

        for name in self._auto_order(extension, params):
            adapter = self.adapters[name]
            cap = adapter.capability()
            if cap.available and self._supports_input(adapter, extension, size_bytes) and (
                not cap.requires_public_url or self._public_url_available()
            ):
                return adapter
        raise EngineError("no_supported_engine", f"No available engine supports .{extension} with current credentials and limits")

    def _auto_order(self, extension: str, params: dict) -> Iterable[str]:
        image_exts = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
        output_formats = set(params.get("output_formats") or [])
        language_hint = {str(x).lower() for x in params.get("language_hint") or []}
        mode = params.get("mode") or "balanced"

        if extension in image_exts:
            return ["qwen_ocr", "datalab", "textin", "paddleocr_remote", "mineru"]
        if mode == "accurate":
            return ["mineru", "datalab", "textin", "paddleocr_remote", "qwen_ocr"]
        if output_formats == {"markdown"} and language_hint.intersection({"zh", "cn", "chinese"}):
            return ["textin", "datalab", "mineru", "paddleocr_remote", "qwen_ocr"]
        return ["datalab", "mineru", "textin", "paddleocr_remote", "qwen_ocr"]

    def _public_url_available(self) -> bool:
        url = self.settings.public_presign_base_url or self.settings.object_storage_endpoint
        if not url:
            return True
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https":
            return False
        if host in {"localhost", "127.0.0.1", "0.0.0.0", "minio", "document-minio"}:
            return False
        return True

    def _supports_input(self, adapter: EngineAdapter, extension: str, size_bytes: int) -> bool:
        if adapter.supports(extension, size_bytes):
            return True
        if (
            self.settings.local_docx_to_pdf_enabled
            and extension in WORD_EXTENSIONS
            and "pdf" in adapter.capability().supported_extensions
        ):
            return True
        return False
