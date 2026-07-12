from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.engines.base import EngineAdapter, EngineCapability, EngineError, EngineRequest, EngineResult


class PaddleOcrRemoteAdapter(EngineAdapter):
    name = "paddleocr_remote"

    def capability(self) -> EngineCapability:
        available = bool(self.settings.paddleocr_api_url and self.settings.paddleocr_token)
        return EngineCapability(
            name=self.name,
            available=available,
            unavailable_reason=None if available else "PADDLEOCR_API_URL and PADDLEOCR_TOKEN are required",
            supported_extensions={"pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"},
            max_size_mb=None,
            requires_public_url=False,
            best_for=["self-hosted OCR", "internal network", "cost-priority PDF/image parsing"],
        )

    def parse(self, request: EngineRequest) -> EngineResult:
        if not (self.settings.paddleocr_api_url and self.settings.paddleocr_token):
            raise EngineError("engine_unavailable", "PaddleOCR remote endpoint is not configured")
        if request.source_bytes is None:
            raise EngineError("internal_error", "PaddleOCR adapter requires source bytes")
        timeout = httpx.Timeout(600.0, connect=30.0, read=600.0, write=300.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            headers = {"Authorization": f"bearer {self.settings.paddleocr_token}"}
            data = {
                "model": self.settings.paddleocr_model,
                "optionalPayload": json.dumps(
                    {
                        "useDocOrientationClassify": False,
                        "useDocUnwarping": False,
                        "useChartRecognition": False,
                    }
                ),
            }
            files = {"file": (request.filename, request.source_bytes, request.content_type or "application/octet-stream")}
            response = client.post(self.settings.paddleocr_api_url, data=data, files=files, headers=headers)
            self._raise_for_response(response)
            submitted = response.json()
            self._raise_for_business_error(submitted)
            job_id = self._job_id(submitted)
            if not job_id:
                return self._result(submitted)
            result = self._poll(client, job_id, headers)
            return self._result({"submitted": submitted, "result": result})

    def _poll(self, client: httpx.Client, job_id: str, headers: dict[str, str]) -> dict[str, Any]:
        poll_url = f"{self.settings.paddleocr_api_url.rstrip('/')}/{job_id}"
        last_result: dict[str, Any] = {}
        for _ in range(240):
            response = client.get(poll_url, headers=headers)
            self._raise_for_response(response)
            result = response.json()
            self._raise_for_business_error(result)
            last_result = result
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            state = str(data.get("state") or "").lower()
            if state == "done":
                result_urls = data.get("resultUrl") if isinstance(data.get("resultUrl"), dict) else {}
                markdown = self._download_text(client, result_urls.get("markdownUrl"))
                json_content = self._download_jsonl(client, result_urls.get("jsonUrl"))
                return {**result, "downloaded_markdown": markdown, "downloaded_json": json_content}
            if state == "failed":
                raise EngineError("third_party_failed", str(data.get("errorMsg") or result))
            time.sleep(5)
        raise EngineError("third_party_timeout", f"PaddleOCR job did not complete: {last_result}")

    def _result(self, raw: dict[str, Any]) -> EngineResult:
        data = raw.get("result") if isinstance(raw.get("result"), dict) else raw
        markdown = data.get("downloaded_markdown")
        json_content = data.get("downloaded_json")
        if not markdown and isinstance(json_content, list):
            markdown_parts = []
            for line in json_content:
                result = line.get("result") if isinstance(line, dict) else None
                if not isinstance(result, dict):
                    continue
                for item in result.get("layoutParsingResults") or []:
                    item_markdown = item.get("markdown") if isinstance(item, dict) else None
                    if isinstance(item_markdown, dict) and isinstance(item_markdown.get("text"), str):
                        markdown_parts.append(item_markdown["text"])
            markdown = "\n\n".join(markdown_parts)
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, json_content=json_content or data)

    def _download_text(self, client: httpx.Client, url: str | None) -> str | None:
        if not url:
            return None
        response = client.get(url)
        response.raise_for_status()
        return response.text

    def _download_jsonl(self, client: httpx.Client, url: str | None) -> list[dict[str, Any]] | None:
        if not url:
            return None
        response = client.get(url)
        response.raise_for_status()
        rows = []
        for line in response.text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _job_id(self, raw: dict[str, Any]) -> str | None:
        data = raw.get("data")
        if isinstance(data, dict):
            return data.get("jobId")
        return None

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise EngineError("auth_failed", f"PaddleOCR authorization failed: {response.text[:500]}")
        if response.status_code == 429:
            raise EngineError("quota_exceeded", f"PaddleOCR rate limit/quota exceeded: {response.text[:500]}")
        if response.status_code >= 400:
            raise EngineError("third_party_error", f"PaddleOCR error {response.status_code}: {response.text[:500]}")

    def _raise_for_business_error(self, raw: dict[str, Any]) -> None:
        code = raw.get("code")
        if code in {None, 0, "0"}:
            return
        message = str(raw.get("msg") or raw.get("errorMsg") or raw)
        code_text = str(code)
        if code_text in {"10003"}:
            raise EngineError("file_too_large", f"PaddleOCR rejected the file: {message}")
        if code_text in {"10004", "10005", "10006", "10007", "10008"}:
            raise EngineError("unsupported_format", f"PaddleOCR rejected the request: {message}")
        if code_text in {"12001"}:
            raise EngineError("quota_exceeded", f"PaddleOCR quota exceeded: {message}")
        if code_text in {"12002"}:
            raise EngineError("rate_limited", f"PaddleOCR rate limited: {message}")
        raise EngineError("third_party_error", f"PaddleOCR business error {code}: {message}")
