from __future__ import annotations

import httpx

from app.engines.base import EngineAdapter, EngineCapability, EngineError, EngineRequest, EngineResult


class TextInAdapter(EngineAdapter):
    name = "textin"

    def capability(self) -> EngineCapability:
        available = bool(self.settings.textin_app_id and self.settings.textin_secret_code)
        return EngineCapability(
            name=self.name,
            available=available,
            unavailable_reason=None if available else "TEXTIN_APP_ID and TEXTIN_SECRET_CODE are required",
            supported_extensions={"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "png", "jpg", "jpeg", "bmp", "tif", "tiff"},
            max_size_mb=10,
            requires_public_url=False,
            best_for=["Chinese RAG markdown", "business documents", "x_to_markdown"],
        )

    def parse(self, request: EngineRequest) -> EngineResult:
        if not (self.settings.textin_app_id and self.settings.textin_secret_code):
            raise EngineError("engine_unavailable", "TextIn credentials are not configured")
        if request.source_bytes is None:
            raise EngineError("internal_error", "TextIn adapter requires source bytes")
        headers = {
            "x-ti-app-id": self.settings.textin_app_id,
            "x-ti-secret-code": self.settings.textin_secret_code,
            "Content-Type": "application/octet-stream",
        }
        params = {"filename": request.filename}
        timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=300.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.settings.textin_base_url.rstrip('/')}/ai/service/v1/x_to_markdown",
                params=params,
                content=request.source_bytes,
                headers=headers,
            )
            self._raise_for_response(response)
            raw = response.json()
            self._raise_for_business_error(raw)
        data = raw.get("result") if isinstance(raw.get("result"), dict) else raw.get("data") if isinstance(raw.get("data"), dict) else raw
        markdown = data.get("markdown") or data.get("md") or data.get("content")
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, json_content=data)

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise EngineError("auth_failed", f"TextIn authorization failed: {response.text[:500]}")
        if response.status_code == 429:
            raise EngineError("quota_exceeded", f"TextIn rate limit/quota exceeded: {response.text[:500]}")
        if response.status_code >= 400:
            raise EngineError("third_party_error", f"TextIn error {response.status_code}: {response.text[:500]}")

    def _raise_for_business_error(self, raw: dict) -> None:
        code = raw.get("code")
        if code in {None, 0, "0", 200, "200"}:
            return
        message = str(raw.get("message") or raw.get("msg") or raw)
        if str(code) == "40302" or "文件大小" in message:
            raise EngineError("file_too_large", f"TextIn rejected the file: {message}")
        raise EngineError("third_party_error", f"TextIn business error {code}: {message}")
