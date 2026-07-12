from __future__ import annotations

import time
import mimetypes
from typing import Any

import httpx

from app.engines.base import EngineAdapter, EngineCapability, EngineError, EngineRequest, EngineResult


class DatalabAdapter(EngineAdapter):
    name = "datalab"

    def capability(self) -> EngineCapability:
        available = bool(self.settings.datalab_api_key)
        return EngineCapability(
            name=self.name,
            available=available,
            unavailable_reason=None if available else "DATALAB_API_KEY is not configured",
            supported_extensions={"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "png", "jpg", "jpeg", "webp", "tif", "tiff"},
            max_size_mb=200,
            requires_public_url=False,
            best_for=["general document conversion", "markdown", "json", "tables", "word documents"],
        )

    def parse(self, request: EngineRequest) -> EngineResult:
        if not self.settings.datalab_api_key:
            raise EngineError("engine_unavailable", "Datalab API key is not configured")
        output_formats = request.params.get("output_formats") or ["markdown"]
        output_format = "markdown" if "markdown" in output_formats else "json"
        mode = request.params.get("mode") or "balanced"
        headers = {"X-API-Key": self.settings.datalab_api_key}
        payload = {
            "file_url": request.source_url,
            "output_format": output_format,
            "mode": mode,
        }
        timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=300.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            files = None
            if request.source_bytes is not None:
                payload.pop("file_url")
                upload_filename = f"document.{request.extension}" if request.extension else request.filename
                content_type = request.content_type
                if not content_type or content_type == "application/octet-stream":
                    content_type = mimetypes.guess_type(upload_filename)[0] or "application/octet-stream"
                files = {"file": (upload_filename, request.source_bytes, content_type)}
            response = client.post(
                f"{self.settings.datalab_base_url.rstrip('/')}/api/v1/convert",
                data=payload,
                files=files,
                headers=headers,
            )
            self._raise_for_response(response)
            submitted = response.json()
            check_url = submitted.get("request_check_url") or submitted.get("check_url")
            if not check_url:
                return self._result(submitted)
            result = self._poll(client, check_url, headers)
            return self._result(result)

    def _poll(self, client: httpx.Client, check_url: str, headers: dict[str, str]) -> dict[str, Any]:
        for _ in range(300):
            response = client.get(check_url, headers=headers)
            self._raise_for_response(response)
            result = response.json()
            status = str(result.get("status", "")).lower()
            if status in {"complete", "completed", "succeeded", "success"}:
                return result
            if status in {"failed", "error"}:
                raise EngineError("third_party_failed", str(result.get("error") or result))
            time.sleep(2)
        raise EngineError("third_party_timeout", "Datalab conversion did not complete within polling window")

    def _result(self, raw: dict[str, Any]) -> EngineResult:
        markdown = raw.get("markdown") or raw.get("content")
        json_content = raw.get("json") or raw.get("blocks") or raw.get("document")
        plain_text = raw.get("text")
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, json_content=json_content, plain_text=plain_text)

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise EngineError("auth_failed", f"Datalab authorization failed: {response.text[:500]}")
        if response.status_code == 429:
            raise EngineError("quota_exceeded", f"Datalab rate limit/quota exceeded: {response.text[:500]}")
        if response.status_code >= 400:
            raise EngineError("third_party_error", f"Datalab error {response.status_code}: {response.text[:500]}")
