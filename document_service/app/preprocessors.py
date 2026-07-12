from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


class PreprocessError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PreprocessedDocument:
    filename: str
    content_type: str
    extension: str
    data: bytes


WORD_EXTENSIONS = {"doc", "docx"}


def can_render_word_to_pdf(extension: str, settings: Settings) -> bool:
    return settings.local_docx_to_pdf_enabled and extension.lower() in WORD_EXTENSIONS


def render_word_to_pdf(source_bytes: bytes, filename: str, settings: Settings) -> PreprocessedDocument:
    extension = Path(filename).suffix.lower() or ".docx"
    with tempfile.TemporaryDirectory(prefix="document-service-") as tmp:
        workdir = Path(tmp)
        profile_dir = workdir / "lo-profile"
        profile_dir.mkdir()
        input_path = workdir / f"source{extension}"
        input_path.write_bytes(source_bytes)
        command = [
            settings.libreoffice_binary,
            f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            "pdf",
            "--outdir",
            str(workdir),
            str(input_path),
        ]
        try:
            completed = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=180, check=False)
        except FileNotFoundError as exc:
            raise PreprocessError("preprocessor_unavailable", f"LibreOffice binary not found: {settings.libreoffice_binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PreprocessError("preprocessor_timeout", "LibreOffice docx-to-pdf conversion timed out") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise PreprocessError("preprocessor_failed", f"LibreOffice docx-to-pdf conversion failed: {detail[:1000]}")
        output_path = input_path.with_suffix(".pdf")
        if not output_path.exists():
            candidates = list(workdir.glob("*.pdf"))
            if not candidates:
                raise PreprocessError("preprocessor_failed", "LibreOffice did not produce a PDF")
            output_path = candidates[0]
        return PreprocessedDocument(
            filename=f"{Path(filename).stem or 'document'}.pdf",
            content_type="application/pdf",
            extension="pdf",
            data=output_path.read_bytes(),
        )
