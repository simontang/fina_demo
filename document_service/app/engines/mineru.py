from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import httpx

from app.engines.base import EngineAdapter, EngineCapability, EngineError, EngineRequest, EngineResult


class MinerUAdapter(EngineAdapter):
    name = "mineru"

    def capability(self) -> EngineCapability:
        available = bool(self.settings.mineru_token)
        return EngineCapability(
            name=self.name,
            available=available,
            unavailable_reason=None if available else "MINERU_TOKEN is not configured",
            supported_extensions={"pdf", "png", "jpg", "jpeg", "jp2", "webp", "gif", "bmp", "doc", "docx", "ppt", "pptx", "xls", "xlsx"},
            max_size_mb=200,
            requires_public_url=True,
            best_for=["complex Chinese layouts", "tables", "formulas", "multi-format structured output"],
        )

    def parse(self, request: EngineRequest) -> EngineResult:
        if not self.settings.mineru_token:
            raise EngineError("engine_unavailable", "MinerU token is not configured")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.mineru_token}",
        }
        payload = {"url": request.source_url, "model_version": self.settings.mineru_model_version}
        with httpx.Client(timeout=60) as client:
            response = client.post(f"{self.settings.mineru_base_url.rstrip('/')}/api/v4/extract/task", json=payload, headers=headers)
            self._raise_for_response(response)
            submitted = response.json()
            self._raise_for_business_error(submitted)
            task_id = self._find_task_id(submitted)
            if not task_id:
                return self._result(submitted)
            result = self._poll(client, task_id, headers)
            return self._result({"submitted": submitted, "result": result})

    def _poll(self, client: httpx.Client, task_id: str, headers: dict[str, str]) -> dict[str, Any]:
        poll_urls = [
            f"{self.settings.mineru_base_url.rstrip('/')}/api/v4/extract/task/{task_id}",
            f"{self.settings.mineru_base_url.rstrip('/')}/api/v4/extract/task?task_id={task_id}",
        ]
        last_result: dict[str, Any] = {}
        for _ in range(300):
            for url in poll_urls:
                response = client.get(url, headers=headers)
                if response.status_code == 404:
                    continue
                if response.status_code == 405:
                    continue
                self._raise_for_response(response)
                result = response.json()
                self._raise_for_business_error(result)
                last_result = result
                status = str(
                    self._dig(result, "data.state")
                    or self._dig(result, "data.status")
                    or result.get("state")
                    or result.get("status")
                    or ""
                ).lower()
                if status in {"done", "complete", "completed", "succeeded", "success"}:
                    return self._download_result_files(client, result)
                if status in {"failed", "error"}:
                    raise EngineError("third_party_failed", str(self._dig(result, "data.err_msg") or result.get("msg") or result.get("error") or result))
                break
            time.sleep(2)
        raise EngineError("third_party_timeout", f"MinerU task did not complete: {last_result}")

    def _result(self, raw: dict[str, Any]) -> EngineResult:
        result = raw.get("result") if isinstance(raw.get("result"), dict) else raw
        downloaded = result.get("downloaded") if isinstance(result.get("downloaded"), dict) else {}
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        if isinstance(result.get("data"), dict):
            data = result["data"]
        markdown = downloaded.get("markdown") or data.get("markdown") or data.get("md") or data.get("content")
        json_content = downloaded.get("json") or data.get("json") or data.get("layout") or data.get("blocks")
        return EngineResult(engine=self.name, raw=raw, markdown=markdown, json_content=json_content)

    def _find_task_id(self, raw: dict[str, Any]) -> str | None:
        data = raw.get("data")
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("task_id") or data.get("taskId") or data.get("id")
        return raw.get("task_id") or raw.get("taskId") or raw.get("id")

    def _dig(self, raw: dict[str, Any], path: str) -> Any:
        current: Any = raw
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.status_code in {401, 403}:
            raise EngineError("auth_failed", f"MinerU authorization failed: {response.text[:500]}")
        if response.status_code == 429:
            raise EngineError("quota_exceeded", f"MinerU rate limit/quota exceeded: {response.text[:500]}")
        if response.status_code >= 400:
            raise EngineError("third_party_error", f"MinerU error {response.status_code}: {response.text[:500]}")

    def _raise_for_business_error(self, raw: dict[str, Any]) -> None:
        code = raw.get("code")
        if code in {None, 0, "0"}:
            return
        message = str(raw.get("msg") or raw)
        raise EngineError("third_party_error", f"MinerU business error {code}: {message}")

    def _download_result_files(self, client: httpx.Client, result: dict[str, Any]) -> dict[str, Any]:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        zip_url = data.get("full_zip_url")
        if not zip_url:
            return result
        response = client.get(zip_url)
        response.raise_for_status()
        downloaded: dict[str, Any] = {"full_zip_url": zip_url}
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            md_name = self._find_archive_member(names, "full.md", ".md")
            if md_name:
                downloaded["markdown"] = archive.read(md_name).decode("utf-8", errors="replace")
            json_payloads: dict[str, Any] = {}
            for name in names:
                if name.endswith(".json") and any(marker in name for marker in ("content_list", "middle", "layout", "model")):
                    try:
                        import json

                        json_payloads[name] = json.loads(archive.read(name).decode("utf-8", errors="replace"))
                    except Exception:
                        continue
            if json_payloads:
                downloaded["json"] = json_payloads
        return {**result, "downloaded": downloaded}

    def _find_archive_member(self, names: list[str], exact_suffix: str, fallback_suffix: str) -> str | None:
        for name in names:
            if name.endswith(exact_suffix):
                return name
        for name in names:
            if name.endswith(fallback_suffix):
                return name
        return None
