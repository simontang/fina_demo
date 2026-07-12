from __future__ import annotations

import httpx

from app.engines.base import EngineAdapter, EngineCapability, EngineError, EngineRequest, EngineResult


class QwenOcrAdapter(EngineAdapter):
    name = "qwen_ocr"

    def capability(self) -> EngineCapability:
        available = bool(self.settings.qwen_api_key)
        return EngineCapability(
            name=self.name,
            available=available,
            unavailable_reason=None if available else "QWEN_API_KEY is not configured",
            supported_extensions={"pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"},
            max_size_mb=None,
            requires_public_url=True,
            best_for=["OCR-heavy images", "scanned pages", "receipts", "single-page high-precision extraction"],
        )

    def parse(self, request: EngineRequest) -> EngineResult:
        if not self.settings.qwen_api_key:
            raise EngineError("engine_unavailable", "Qwen OCR API key is not configured")
        headers = {
            "Authorization": f"Bearer {self.settings.qwen_api_key}",
            "Content-Type": "application/json",
        }
        if request.extension == "pdf":
            return self._parse_pdf(request, headers)
        return self._parse_image(request, headers)

    def _parse_pdf(self, request: EngineRequest, headers: dict[str, str]) -> EngineResult:
        payload = {
            "model": self.settings.qwen_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_url": request.source_url,
                        }
                    ],
                }
            ],
            "ocr_options": {"task": "document_parsing"},
        }
        with httpx.Client(timeout=600) as client:
            response = client.post(f"{self.settings.qwen_base_url.rstrip('/')}/responses", json=payload, headers=headers)
            self._raise_for_response(response)
            raw = response.json()
        markdown = self._extract_response_content(raw)
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, plain_text=markdown)

    def _parse_image(self, request: EngineRequest, headers: dict[str, str]) -> EngineResult:
        payload = {
            "model": self.settings.qwen_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all document text and structure. Return concise Markdown only."},
                        {"type": "image_url", "image_url": {"url": request.source_url}},
                    ],
                }
            ],
            "temperature": 0,
        }
        with httpx.Client(timeout=180) as client:
            response = client.post(f"{self.settings.qwen_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            self._raise_for_response(response)
            raw = response.json()
        markdown = self._extract_content(raw)
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, plain_text=markdown)

    def _extract_content(self, raw: dict) -> str:
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        return raw.get("text") or ""

    def _extract_response_content(self, raw: dict) -> str:
        output = raw.get("output")
        if isinstance(output, list):
            for item in output:
                content_items = item.get("content") if isinstance(item, dict) else None
                if not isinstance(content_items, list):
                    continue
                for content in content_items:
                    if not isinstance(content, dict):
                        continue
                    ocr_result = content.get("ocr_result")
                    if isinstance(ocr_result, str):
                        return ocr_result
                    if isinstance(ocr_result, dict):
                        for key in ("markdown", "text", "content"):
                            value = ocr_result.get(key)
                            if isinstance(value, str):
                                return value
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
        if isinstance(raw.get("text"), str):
            return raw["text"]
        return ""

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise EngineError("auth_failed", f"Qwen OCR authorization failed: {response.text[:500]}")
        if response.status_code == 429:
            raise EngineError("quota_exceeded", f"Qwen OCR rate limit/quota exceeded: {response.text[:500]}")
        if response.status_code >= 400:
            raise EngineError("third_party_error", f"Qwen OCR error {response.status_code}: {response.text[:500]}")
