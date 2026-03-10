#!/usr/bin/env python3
"""
Import model/table meta from a Woongjin-style Excel (.xlsx) into metrics-server meta/.

Goals:
  - Only *add missing* meta (CSV column lists + table-catalog entries).
  - Never overwrite existing meta files.
  - Use loose matching for "already exists" (case/underscore-insensitive).

This script uses only Python standard library (no pandas/openpyxl).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def norm_name(name: str) -> str:
    return re.sub(r"[_\W]+", "", (name or "")).upper()


def sanitize_tsv_cell(value: Optional[str]) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return s.strip()


def col_letters_to_index(col: str) -> int:
    n = 0
    for ch in col.upper():
        if not ("A" <= ch <= "Z"):
            continue
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def cell_ref_to_row_col(ref: str) -> Tuple[int, int]:
    col = "".join(c for c in ref if c.isalpha())
    row = "".join(c for c in ref if c.isdigit())
    if not row:
        return (0, 0)
    return (int(row), col_letters_to_index(col))


def first_nonblank_prefix(text: str) -> str:
    if not text:
        return ""
    # delimiters: ； ， 。 （ (
    m = re.split(r"[；，。（(]", text, maxsplit=1)
    prefix = (m[0] if m else text).strip()
    return prefix or text.strip()


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    name: str
    desc: str


class XlsxReader:
    def __init__(self, xlsx_path: Path):
        self.xlsx_path = xlsx_path
        self._zip: Optional[zipfile.ZipFile] = None
        self._shared_strings: List[str] = []
        self._sheet_name_to_path: Dict[str, str] = {}

    def __enter__(self) -> "XlsxReader":
        self._zip = zipfile.ZipFile(self.xlsx_path)
        self._shared_strings = self._read_shared_strings()
        self._sheet_name_to_path = self._read_sheet_name_to_path()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zip is not None:
            self._zip.close()
        self._zip = None

    @property
    def sheet_names(self) -> List[str]:
        return list(self._sheet_name_to_path.keys())

    def sheet_xml_path(self, sheet_name: str) -> Optional[str]:
        return self._sheet_name_to_path.get(sheet_name)

    def read_sheet_cells(self, sheet_name: str) -> Dict[Tuple[int, int], str]:
        if self._zip is None:
            raise RuntimeError("XlsxReader not opened")
        sheet_path = self.sheet_xml_path(sheet_name)
        if not sheet_path:
            raise KeyError(f"Sheet not found: {sheet_name}")
        xml_bytes = self._zip.read(sheet_path)
        root = ET.fromstring(xml_bytes)
        cells: Dict[Tuple[int, int], str] = {}
        for c in root.findall(".//s:c", NS):
            ref = c.attrib.get("r")
            if not ref:
                continue
            row, col = cell_ref_to_row_col(ref)
            if row <= 0 or col <= 0:
                continue
            value = self._cell_value(c)
            if value is None:
                continue
            cells[(row, col)] = value
        return cells

    def read_sheet_rows(
        self, sheet_name: str, max_rows: int = 50000, max_cols: int = 64
    ) -> List[List[Optional[str]]]:
        cells = self.read_sheet_cells(sheet_name)
        if not cells:
            return []
        rows: List[List[Optional[str]]] = []
        for r in range(1, max_rows + 1):
            row_vals: List[Optional[str]] = []
            any_nonempty = False
            for c in range(1, max_cols + 1):
                v = cells.get((r, c))
                row_vals.append(v)
                if v not in (None, ""):
                    any_nonempty = True
            if not any_nonempty:
                continue
            while row_vals and row_vals[-1] is None:
                row_vals.pop()
            rows.append(row_vals)
        return rows

    def _read_shared_strings(self) -> List[str]:
        if self._zip is None:
            raise RuntimeError("XlsxReader not opened")
        try:
            data = self._zip.read("xl/sharedStrings.xml")
        except KeyError:
            return []
        root = ET.fromstring(data)
        out: List[str] = []
        for si in root.findall("s:si", NS):
            # sharedStrings may contain multiple <t> across runs
            parts: List[str] = []
            for t in si.findall(".//s:t", NS):
                if t.text:
                    parts.append(t.text)
            out.append("".join(parts))
        return out

    def _read_sheet_name_to_path(self) -> Dict[str, str]:
        if self._zip is None:
            raise RuntimeError("XlsxReader not opened")
        wb = ET.fromstring(self._zip.read("xl/workbook.xml"))
        rels = ET.fromstring(self._zip.read("xl/_rels/workbook.xml.rels"))
        rel_ns = rels.tag.split("}")[0].strip("{")
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{rel_ns}}}Relationship")
        }

        out: Dict[str, str] = {}
        for sh in wb.findall("s:sheets/s:sheet", NS):
            name = sh.attrib.get("name")
            rid = sh.attrib.get(f"{{{NS['r']}}}id")
            if not name or not rid:
                continue
            target = rid_to_target.get(rid)
            if not target:
                continue
            if not target.startswith("xl/"):
                target = "xl/" + target
            out[name] = target
        return out

    def _cell_value(self, c: ET.Element) -> Optional[str]:
        t = c.attrib.get("t")

        if t == "inlineStr":
            is_node = c.find("s:is", NS)
            if is_node is None:
                return None
            text_parts: List[str] = []
            for tn in is_node.findall(".//s:t", NS):
                if tn.text:
                    text_parts.append(tn.text)
            return "".join(text_parts) if text_parts else None

        v = c.find("s:v", NS)
        if v is None or v.text is None:
            return None
        raw = v.text

        if t == "s":
            try:
                return self._shared_strings[int(raw)]
            except Exception:
                return raw
        return raw


def parse_models_sheet(rows: Sequence[Sequence[Optional[str]]]) -> List[ModelRow]:
    if not rows:
        return []
    header = [sanitize_tsv_cell(x) for x in rows[0]]
    # expected: 模型ID, 名称, 描述
    def find_col(name: str) -> int:
        for i, h in enumerate(header):
            if h == name:
                return i
        return -1

    id_idx = find_col("模型ID")
    name_idx = find_col("名称")
    desc_idx = find_col("描述")
    if name_idx < 0:
        raise ValueError("Models sheet missing required column: 名称")

    out: List[ModelRow] = []
    for r in rows[1:]:
        rid = sanitize_tsv_cell(r[id_idx]) if 0 <= id_idx < len(r) else ""
        name = sanitize_tsv_cell(r[name_idx]) if name_idx < len(r) else ""
        desc = sanitize_tsv_cell(r[desc_idx]) if 0 <= desc_idx < len(r) else ""
        if not name:
            continue
        out.append(ModelRow(model_id=rid, name=name, desc=desc))
    return out


def parse_fields_sheet(rows: Sequence[Sequence[Optional[str]]]) -> List[Tuple[str, str, str, str]]:
    if not rows:
        return []
    header = [sanitize_tsv_cell(x) for x in rows[0]]
    header_norm = [h.strip() for h in header]
    # expected: 字段ID, 字段名, 描述, 类型 (or bilingual)
    def idx_of(*names: str) -> int:
        for n in names:
            if n in header_norm:
                return header_norm.index(n)
        return -1

    id_idx = idx_of("字段ID", "Field ID")
    name_idx = idx_of("字段名", "Field Name")
    desc_idx = idx_of("描述", "Description")
    type_idx = idx_of("类型", "Type")
    if name_idx < 0:
        raise ValueError("Fields sheet missing required column: 字段名/Field Name")

    out: List[Tuple[str, str, str, str]] = []
    for r in rows[1:]:
        field_id = sanitize_tsv_cell(r[id_idx]) if 0 <= id_idx < len(r) else ""
        field_name = sanitize_tsv_cell(r[name_idx]) if name_idx < len(r) else ""
        desc = sanitize_tsv_cell(r[desc_idx]) if 0 <= desc_idx < len(r) else ""
        typ = sanitize_tsv_cell(r[type_idx]) if 0 <= type_idx < len(r) else ""
        if not field_name:
            continue
        out.append((field_id, field_name, desc, typ))
    return out


def find_header_row(rows: List[List[Optional[str]]]) -> int:
    for i, r in enumerate(rows):
        nonempty = sum(1 for x in r if sanitize_tsv_cell(x))
        if nonempty >= 2:
            return i
    return -1


def load_existing_view_names(meta_dir: Path) -> List[str]:
    names: List[str] = []
    for p in meta_dir.glob("view-*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            view_name = data.get("viewName")
            if isinstance(view_name, str) and view_name.strip():
                names.append(view_name.strip())
        except Exception:
            continue
    return names


def load_existing_catalog_names(meta_dir: Path) -> List[str]:
    p = meta_dir / "table-catalog.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    names: List[str] = []
    for item in data:
        if isinstance(item, dict):
            n = item.get("tableName")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
    return names


def update_table_catalog(
    meta_dir: Path, models: Sequence[ModelRow], dry_run: bool
) -> Tuple[int, List[str]]:
    catalog_path = meta_dir / "table-catalog.json"
    existing: List[dict] = []
    if catalog_path.exists():
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError("table-catalog.json must be a JSON array")

    existing_norm = {norm_name(d.get("tableName", "")) for d in existing if isinstance(d, dict)}
    added_names: List[str] = []

    for m in models:
        n = norm_name(m.name)
        if not n or n in existing_norm:
            continue
        doc_type = first_nonblank_prefix(m.desc) or m.name
        entry = {
            "tableName": m.name,
            "docType": doc_type,
            "docTypeEn": doc_type,
            "shortDesc": m.desc or None,
        }
        # Remove null shortDesc to keep JSON compact and avoid implying changes
        if entry["shortDesc"] is None:
            entry.pop("shortDesc", None)
        existing.append(entry)
        existing_norm.add(n)
        added_names.append(m.name)

    if added_names and not dry_run:
        catalog_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    return len(added_names), added_names


def write_model_csv(
    meta_dir: Path, model_name: str, rows: List[Tuple[str, str, str, str]], dry_run: bool
) -> bool:
    out_path = meta_dir / f"{model_name}.csv"
    if out_path.exists():
        return False
    header = "Field ID\tField Name\tDescription\tType\n"
    lines = [header]
    for field_id, field_name, desc, typ in rows:
        lines.append(
            "\t".join(
                [
                    sanitize_tsv_cell(field_id),
                    sanitize_tsv_cell(field_name),
                    sanitize_tsv_cell(desc),
                    sanitize_tsv_cell(typ),
                ]
            )
            + "\n"
        )
    if not dry_run:
        out_path.write_text("".join(lines), encoding="utf-8")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import missing model meta (CSV + table-catalog) from an .xlsx into metrics-server meta/."
    )
    parser.add_argument("--xlsx", required=True, help="Path to models_*.xlsx")
    parser.add_argument(
        "--meta-dir",
        required=True,
        help="Path to metrics-server/src/main/resources/meta directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="After running, assert every Excel model is covered by view-*.json or MTC_VW_AI_*.csv",
    )
    args = parser.parse_args(argv)

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    meta_dir = Path(args.meta_dir).expanduser().resolve()
    dry_run = bool(args.dry_run)

    if not xlsx_path.exists():
        print(f"ERROR: xlsx not found: {xlsx_path}", file=sys.stderr)
        return 2
    if not meta_dir.exists() or not meta_dir.is_dir():
        print(f"ERROR: meta dir not found: {meta_dir}", file=sys.stderr)
        return 2

    existing_view_names = load_existing_view_names(meta_dir)
    existing_csv_names = [p.stem for p in meta_dir.glob("MTC_VW_AI_*.csv")]
    existing_norm = {norm_name(n) for n in (existing_view_names + existing_csv_names)}

    created_csv = 0
    skipped_existing = 0
    missing_sheet: List[str] = []
    created_csv_names: List[str] = []

    with XlsxReader(xlsx_path) as xr:
        if "Models" not in xr.sheet_names:
            print("ERROR: sheet 'Models' not found in xlsx", file=sys.stderr)
            return 2

        models_rows_all = xr.read_sheet_rows("Models", max_rows=5000, max_cols=8)
        hdr_idx = find_header_row(models_rows_all)
        if hdr_idx < 0:
            print("ERROR: Models sheet seems empty", file=sys.stderr)
            return 2
        models = parse_models_sheet(models_rows_all[hdr_idx:])

        # Only add missing (by loose match)
        missing_models = [m for m in models if norm_name(m.name) not in existing_norm]

        # 1) CSV per missing model
        for m in missing_models:
            if m.name not in xr.sheet_names:
                missing_sheet.append(m.name)
                continue
            sheet_rows_all = xr.read_sheet_rows(m.name, max_rows=200000, max_cols=64)
            hdr = find_header_row(sheet_rows_all)
            if hdr < 0:
                print(f"WARN: sheet empty: {m.name}")
                continue
            fields = parse_fields_sheet(sheet_rows_all[hdr:])
            if not fields:
                print(f"WARN: no fields parsed: {m.name}")
                continue
            if write_model_csv(meta_dir, m.name, fields, dry_run=dry_run):
                created_csv += 1
                created_csv_names.append(m.name)
            else:
                skipped_existing += 1

        # 2) table-catalog.json append for missing models
        added_catalog_count, added_catalog_names = update_table_catalog(
            meta_dir, missing_models, dry_run=dry_run
        )

        print(f"Excel models: {len(models)}")
        print(f"Existing meta coverage (view/csv): {len(existing_norm)}")
        print(f"Missing models (need meta): {len(missing_models)}")
        print(f"CSV created: {created_csv} (dry_run={dry_run})")
        if created_csv_names:
            print("CSV created for:")
            for n in created_csv_names:
                print(f"  - {n}")
        if skipped_existing:
            print(f"CSV skipped (already exists by filename): {skipped_existing}")

        print(f"table-catalog entries added: {added_catalog_count} (dry_run={dry_run})")
        if added_catalog_names:
            print("table-catalog added for:")
            for n in added_catalog_names:
                print(f"  - {n}")

        if missing_sheet:
            print("WARN: model listed in Models sheet but worksheet not found:")
            for n in missing_sheet:
                print(f"  - {n}")

    if args.check:
        # Re-check with updated filesystem state (and still loose-match).
        view_names = load_existing_view_names(meta_dir)
        csv_names = [p.stem for p in meta_dir.glob("MTC_VW_AI_*.csv")]
        coverage = {norm_name(n) for n in (view_names + csv_names)}

        # reload excel models quickly
        with XlsxReader(xlsx_path) as xr:
            models_rows_all = xr.read_sheet_rows("Models", max_rows=5000, max_cols=8)
            hdr_idx = find_header_row(models_rows_all)
            models = parse_models_sheet(models_rows_all[hdr_idx:]) if hdr_idx >= 0 else []

        uncovered = [m.name for m in models if norm_name(m.name) not in coverage]
        if uncovered:
            print("ERROR: uncovered models remain after import:", file=sys.stderr)
            for n in uncovered:
                print(f"  - {n}", file=sys.stderr)
            return 1
        print("CHECK OK: all Excel models are covered by meta view/csv (loose match).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

