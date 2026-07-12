from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from app.engines.base import EngineResult
from app.schemas import DocumentBlock, DocumentIR, DocumentTable


def normalize_engine_result(
    *,
    source_asset_id: str,
    result: EngineResult,
    raw_asset_id: str,
) -> DocumentIR:
    markdown = result.markdown or _extract_markdown(result.raw) or ""
    plain_text = result.plain_text or _markdown_to_text(markdown) or _extract_text(result.raw)
    blocks, tables = _blocks_from_json(result.json_content)
    if not blocks:
        blocks, tables_from_markdown = _blocks_from_markdown(markdown)
        tables.extend(tables_from_markdown)
    html_tables = _tables_from_html(markdown, start_index=len(tables) + 1)
    if html_tables:
        tables.extend(html_tables)
    return DocumentIR(
        source_asset_id=source_asset_id,
        engine=result.engine,
        content={"markdown": markdown, "plain_text": plain_text},
        blocks=blocks,
        tables=tables,
        raw_outputs={"engine_result_asset_id": raw_asset_id},
    )


def _extract_markdown(raw: dict[str, Any]) -> str | None:
    for path in [
        ("markdown",),
        ("md",),
        ("content",),
        ("data", "markdown"),
        ("data", "md"),
        ("data", "content"),
        ("result", "markdown"),
        ("result", "md"),
        ("result", "content"),
    ]:
        value = _dig(raw, path)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _extract_text(raw: dict[str, Any]) -> str:
    for path in [("text",), ("data", "text"), ("result", "text")]:
        value = _dig(raw, path)
        if isinstance(value, str):
            return value
    return ""


def _dig(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = raw
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _markdown_to_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`#>|-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _blocks_from_json(json_content: dict[str, Any] | list[Any] | None) -> tuple[list[DocumentBlock], list[DocumentTable]]:
    if json_content is None:
        return [], []
    candidate_blocks: list[Any]
    if isinstance(json_content, list):
        candidate_blocks = json_content
    elif isinstance(json_content, dict):
        candidate_blocks = (
            json_content.get("blocks")
            or json_content.get("pages")
            or json_content.get("layout")
            or json_content.get("elements")
            or []
        )
    else:
        return [], []

    blocks: list[DocumentBlock] = []
    tables: list[DocumentTable] = []
    for item in _flatten_blocks(candidate_blocks):
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("id") or f"b{len(blocks) + 1:03d}")
        block_type = str(item.get("type") or item.get("category") or item.get("block_type") or "paragraph").lower()
        text = str(item.get("text") or item.get("content") or item.get("html") or "")
        page = _safe_int(item.get("page") or item.get("page_num") or item.get("page_index"))
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else item.get("box") if isinstance(item.get("box"), list) else None
        rows = item.get("rows") or item.get("cells")
        if block_type == "table" and isinstance(rows, list):
            table_rows = _coerce_table_rows(rows)
            table_id = block_id.replace("b", "t", 1) if block_id.startswith("b") else f"t{len(tables) + 1:03d}"
            tables.append(DocumentTable(id=table_id, rows=table_rows, page=page))
        blocks.append(DocumentBlock(id=block_id, type=block_type, text=text, page=page, bbox=bbox, metadata=_metadata_without_large_fields(item)))
    return blocks, tables


def _flatten_blocks(items: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("blocks"), list):
            flattened.extend(_flatten_blocks(item["blocks"]))
        elif isinstance(item, dict) and isinstance(item.get("elements"), list):
            flattened.extend(_flatten_blocks(item["elements"]))
        else:
            flattened.append(item)
    return flattened


def _metadata_without_large_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in {"text", "content", "html", "rows", "cells", "blocks", "elements"}}


def _coerce_table_rows(rows: list[Any]) -> list[list[str]]:
    result: list[list[str]] = []
    for row in rows:
        if isinstance(row, list):
            result.append([_cell_text(cell) for cell in row])
        elif isinstance(row, dict) and isinstance(row.get("cells"), list):
            result.append([_cell_text(cell) for cell in row["cells"]])
    return result


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("text") or cell.get("content") or cell.get("value") or "")
    return str(cell)


def _blocks_from_markdown(markdown: str) -> tuple[list[DocumentBlock], list[DocumentTable]]:
    blocks: list[DocumentBlock] = []
    tables: list[DocumentTable] = []
    lines = [line.rstrip() for line in markdown.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if _looks_like_markdown_table_start(lines, i):
            table_lines: list[str] = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i].strip())
                i += 1
            rows = _parse_markdown_table(table_lines)
            table_id = f"t{len(tables) + 1:03d}"
            tables.append(DocumentTable(id=table_id, rows=rows, page=None))
            blocks.append(DocumentBlock(id=f"b{len(blocks) + 1:03d}", type="table", text="\n".join(table_lines), page=None))
            continue
        block_type = "heading" if line.startswith("#") else "paragraph"
        text = line.lstrip("#").strip() if block_type == "heading" else line
        blocks.append(DocumentBlock(id=f"b{len(blocks) + 1:03d}", type=block_type, text=text, page=None))
        i += 1
    return blocks, tables


def _looks_like_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(re.search(r"\|\s*:?-{3,}:?\s*(\||$)", lines[index + 1]))


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.search(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _tables_from_html(markdown: str, start_index: int = 1) -> list[DocumentTable]:
    parser = _HtmlTableParser()
    parser.feed(markdown)
    tables: list[DocumentTable] = []
    for offset, rows in enumerate(parser.tables):
        if rows:
            tables.append(DocumentTable(id=f"t{start_index + offset:03d}", rows=rows, page=None))
    return tables


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_rows = []
        elif self._table_depth and tag == "tr":
            self._current_row = []
        elif self._table_depth and tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            text = re.sub(r"\s+", " ", "".join(self._current_cell)).strip()
            self._current_row.append(text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self._current_rows.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0:
                self.tables.append(self._current_rows)
                self._current_rows = []


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
