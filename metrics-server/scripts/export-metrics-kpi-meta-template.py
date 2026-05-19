#!/usr/bin/env python3
"""
Export metrics KPI meta JSON into a fillable Excel template (.xlsx).

The workbook is designed for human editing:
  - one sheet for core metric metadata
  - one sheet each for dimensions / thresholds / diagnostic rules / base filters
  - a README sheet explaining how rows map back to metric JSON

This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from xml.sax.saxutils import escape


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
DCMITYPE_NS = "http://purl.org/dc/dcmitype/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
EXTENDED_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"


@dataclass(frozen=True)
class SheetSpec:
    name: str
    rows: List[List[Any]]
    auto_filter: bool = True


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_index = repo_root / "metrics-server" / "src" / "main" / "resources" / "meta" / "metrics-index-meta.json"
    default_detail = repo_root / "metrics-server" / "src" / "main" / "resources" / "meta" / "metrics-detail-meta.json"
    default_output = repo_root / "outputs" / "metrics_kpi_meta_template.xlsx"

    ap = argparse.ArgumentParser(description="Export metrics KPI meta to a fillable Excel template.")
    ap.add_argument("--index", type=Path, default=default_index, help="Path to metrics-index-meta.json")
    ap.add_argument("--detail", type=Path, default=default_detail, help="Path to metrics-detail-meta.json")
    ap.add_argument("--output", type=Path, default=default_output, help="Output .xlsx file path")
    return ap.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def csv_text(values: Iterable[Any]) -> str:
    parts = [as_text(v).strip() for v in values]
    return ", ".join(p for p in parts if p)


def col_ref(col_num: int) -> str:
    label = []
    n = col_num
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label.append(chr(ord("A") + rem))
    return "".join(reversed(label))


def xml_text(text: Any) -> str:
    s = as_text(text)
    return escape(s)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_readme_rows(index_meta: dict, detail_items: Sequence[dict]) -> List[List[Any]]:
    domain_categories = csv_text(index_meta.get("domain_categories", []))
    rows = [
        ["Field", "Value"],
        ["Workbook Purpose", "Fillable KPI meta template exported from metrics JSON files."],
        ["Source Files", "metrics-index-meta.json + metrics-detail-meta.json"],
        ["Metric Count", len(detail_items)],
        ["Catalog Version", index_meta.get("metric_catalog_version", "")],
        ["Domain Categories", domain_categories],
        ["How To Use", "Edit existing rows or add new rows. Keep metric_name consistent across all sheets."],
        ["Sheet: kpi_core", "One row per KPI. Core business meaning, time context, calculation, source, and owner notes."],
        ["Sheet: dimensions", "One row per supported dimension for each KPI."],
        ["Sheet: thresholds", "One row per threshold rule in ai_agent_context.thresholds."],
        ["Sheet: diag_triggers", "One row per trigger condition in diagnostic_workflow.trigger.any_of."],
        ["Sheet: diag_actions", "One row per action in diagnostic_workflow.actions."],
        ["Sheet: base_filters", "One row per source.base_filters item."],
        ["Editable Columns", "owner, review_status, last_review_date, business_notes, implementation_notes and any copied/new rows."],
        ["Data Entry Rule", "If you add a new metric, create matching rows in kpi_core first, then add related dimensions / thresholds / actions."],
    ]
    return rows


def build_kpi_core_rows(index_meta: dict, detail_items: Sequence[dict]) -> List[List[Any]]:
    index_by_name = {
        item.get("metric_name"): item
        for item in index_meta.get("metrics_index", [])
    }
    header = [
        "metric_name",
        "display_name",
        "domain",
        "short_desc",
        "description",
        "search_keywords",
        "data_type",
        "format",
        "time_dimension",
        "time_label",
        "default_granularity",
        "default_window",
        "supported_grains",
        "calculation_type",
        "sql_expression",
        "dependencies",
        "source_table_view",
        "source_base_filters_json",
        "behavior_role",
        "analysis_priority",
        "polarity",
        "synonyms",
        "human_readable_explanation",
        "owner",
        "review_status",
        "last_review_date",
        "business_notes",
        "implementation_notes",
    ]
    rows: List[List[Any]] = [header]

    for item in detail_items:
        metric_name = item.get("metric_name", "")
        index_item = index_by_name.get(metric_name, {})
        time_ctx = item.get("default_time_context", {})
        calc = item.get("calculation", {})
        source = item.get("source", {})
        behavior = item.get("behavior_profile", {})
        ai_ctx = item.get("ai_agent_context", {})
        rows.append([
            metric_name,
            item.get("display_name", ""),
            item.get("domain", ""),
            index_item.get("short_desc", ""),
            item.get("description", ""),
            csv_text(index_item.get("search_keywords", [])),
            item.get("data_type", ""),
            item.get("format", ""),
            time_ctx.get("time_dimension", ""),
            time_ctx.get("label", ""),
            time_ctx.get("granularity", ""),
            time_ctx.get("window", ""),
            csv_text(time_ctx.get("supported_grains", [])),
            calc.get("type", ""),
            calc.get("sql_expression", ""),
            csv_text(calc.get("dependencies", [])),
            source.get("table_view", ""),
            json.dumps(source.get("base_filters", []), ensure_ascii=False),
            behavior.get("role", ""),
            behavior.get("analysis_priority", ""),
            ai_ctx.get("polarity", ""),
            csv_text(ai_ctx.get("synonyms", [])),
            ai_ctx.get("human_readable_explanation", ""),
            "",
            "",
            "",
            "",
            "",
        ])
    return rows


def build_dimensions_rows(detail_items: Sequence[dict]) -> List[List[Any]]:
    header = [
        "metric_name",
        "dim_order",
        "dim_id",
        "field_name",
        "type",
        "filter_operators",
        "group_by_example",
        "filter_example_dimension",
        "filter_example_operator",
        "filter_example_values",
        "notes",
    ]
    rows: List[List[Any]] = [header]
    for item in detail_items:
        metric_name = item.get("metric_name", "")
        for idx, dim in enumerate(item.get("supported_dimensions", []), start=1):
            filter_example = dim.get("filter_example", {})
            rows.append([
                metric_name,
                idx,
                dim.get("dim_id", ""),
                dim.get("field_name", ""),
                dim.get("type", ""),
                csv_text(dim.get("filter_operators", [])),
                dim.get("group_by_example", ""),
                filter_example.get("dimension", ""),
                filter_example.get("operator", ""),
                csv_text(filter_example.get("values", [])),
                "",
            ])
    return rows


def build_threshold_rows(detail_items: Sequence[dict]) -> List[List[Any]]:
    header = ["metric_name", "threshold_order", "metric", "operator", "value", "level", "notes"]
    rows: List[List[Any]] = [header]
    for item in detail_items:
        metric_name = item.get("metric_name", "")
        thresholds = item.get("ai_agent_context", {}).get("thresholds", [])
        for idx, threshold in enumerate(thresholds, start=1):
            rows.append([
                metric_name,
                idx,
                threshold.get("metric", ""),
                threshold.get("operator", ""),
                threshold.get("value", ""),
                threshold.get("level", ""),
                "",
            ])
    return rows


def build_trigger_rows(detail_items: Sequence[dict]) -> List[List[Any]]:
    header = ["metric_name", "trigger_order", "metric", "operator", "value", "notes"]
    rows: List[List[Any]] = [header]
    for item in detail_items:
        metric_name = item.get("metric_name", "")
        any_of = (
            item.get("ai_agent_context", {})
            .get("diagnostic_workflow", {})
            .get("trigger", {})
            .get("any_of", [])
        )
        for idx, trigger in enumerate(any_of, start=1):
            rows.append([
                metric_name,
                idx,
                trigger.get("metric", ""),
                trigger.get("operator", ""),
                trigger.get("value", ""),
                "",
            ])
    return rows


def build_action_rows(detail_items: Sequence[dict]) -> List[List[Any]]:
    header = ["metric_name", "action_order", "action_type", "compare_metric", "dimensions", "intent", "notes"]
    rows: List[List[Any]] = [header]
    for item in detail_items:
        metric_name = item.get("metric_name", "")
        actions = (
            item.get("ai_agent_context", {})
            .get("diagnostic_workflow", {})
            .get("actions", [])
        )
        for idx, action in enumerate(actions, start=1):
            rows.append([
                metric_name,
                idx,
                action.get("type", ""),
                action.get("metric", ""),
                csv_text(action.get("dimensions", [])),
                action.get("intent", ""),
                "",
            ])
    return rows


def build_base_filter_rows(detail_items: Sequence[dict]) -> List[List[Any]]:
    header = ["metric_name", "filter_order", "field", "operator", "value", "notes"]
    rows: List[List[Any]] = [header]
    for item in detail_items:
        metric_name = item.get("metric_name", "")
        base_filters = item.get("source", {}).get("base_filters", [])
        if not base_filters:
            rows.append([metric_name, "", "", "", "", ""])
            continue
        for idx, base_filter in enumerate(base_filters, start=1):
            rows.append([
                metric_name,
                idx,
                base_filter.get("field", ""),
                base_filter.get("operator", ""),
                as_text(base_filter.get("value", "")),
                "",
            ])
    return rows


def compute_widths(rows: Sequence[Sequence[Any]]) -> List[float]:
    col_count = max((len(row) for row in rows), default=0)
    widths: List[float] = []
    for col_idx in range(col_count):
        max_len = 10
        for row in rows:
            value = row[col_idx] if col_idx < len(row) else ""
            text = as_text(value)
            max_len = max(max_len, min(len(text) + 2, 60))
        widths.append(float(max_len))
    return widths


def cell_xml(ref: str, value: Any, style_id: int) -> str:
    style_attr = f' s="{style_id}"' if style_id else ""
    if value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if is_number(value):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_text(value)}</t></is></c>'


def sheet_xml(sheet: SheetSpec) -> str:
    rows = sheet.rows
    widths = compute_widths(rows)
    col_count = max((len(row) for row in rows), default=0)
    last_col = col_ref(col_count) if col_count else "A"
    last_row = len(rows) if rows else 1

    parts = [
        f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{DOC_REL_NS}">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<sheetFormatPr defaultRowHeight="15"/>',
        "<cols>",
    ]
    for idx, width in enumerate(widths, start=1):
        parts.append(f'<col min="{idx}" max="{idx}" width="{width:.2f}" customWidth="1"/>')
    parts.append("</cols><sheetData>")

    for row_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{row_idx}">')
        for col_idx in range(1, col_count + 1):
            value = row[col_idx - 1] if col_idx - 1 < len(row) else ""
            ref = f"{col_ref(col_idx)}{row_idx}"
            style_id = 1 if row_idx == 1 else 0
            parts.append(cell_xml(ref, value, style_id))
        parts.append("</row>")

    parts.append("</sheetData>")
    if sheet.auto_filter and rows:
        parts.append(f'<autoFilter ref="A1:{last_col}{last_row}"/>')
    parts.append("</worksheet>")
    return "".join(parts)


def workbook_xml(sheets: Sequence[SheetSpec]) -> str:
    parts = [
        f'<workbook xmlns="{MAIN_NS}" xmlns:r="{DOC_REL_NS}">',
        "<workbookPr/>",
        "<bookViews><workbookView activeTab=\"0\"/></bookViews>",
        "<sheets>",
    ]
    for idx, sheet in enumerate(sheets, start=1):
        parts.append(
            f'<sheet name="{escape(sheet.name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
    parts.append("</sheets></workbook>")
    return "".join(parts)


def workbook_rels_xml(sheets: Sequence[SheetSpec]) -> str:
    parts = [f'<Relationships xmlns="{REL_NS}">']
    for idx, _sheet in enumerate(sheets, start=1):
        parts.append(
            f'<Relationship Id="rId{idx}" '
            f'Type="{DOC_REL_NS}/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    styles_id = len(sheets) + 1
    parts.append(
        f'<Relationship Id="rId{styles_id}" Type="{DOC_REL_NS}/styles" Target="styles.xml"/>'
    )
    parts.append("</Relationships>")
    return "".join(parts)


def root_rels_xml() -> str:
    return (
        f'<Relationships xmlns="{REL_NS}">'
        f'<Relationship Id="rId1" Type="{DOC_REL_NS}/officeDocument" Target="xl/workbook.xml"/>'
        f'<Relationship Id="rId2" Type="{DOC_REL_NS}/metadata/core-properties" Target="docProps/core.xml"/>'
        f'<Relationship Id="rId3" Type="{DOC_REL_NS}/extended-properties" Target="docProps/app.xml"/>'
        f"</Relationships>"
    )


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{MAIN_NS}">
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>"""


def content_types_xml(sheet_count: int) -> str:
    parts = [
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        parts.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    parts.append("</Types>")
    return "".join(parts)


def core_props_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" '
        f'xmlns:dcterms="{DCTERMS_NS}" xmlns:dcmitype="{DCMITYPE_NS}" xmlns:xsi="{XSI_NS}">'
        "<dc:title>Metrics KPI Meta Template</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_props_xml(sheet_names: Sequence[str]) -> str:
    titles = "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name in sheet_names)
    count = len(sheet_names)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Properties xmlns="{EXTENDED_NS}" xmlns:vt="{VT_NS}">'
        "<Application>Codex</Application>"
        "<DocSecurity>0</DocSecurity>"
        "<ScaleCrop>false</ScaleCrop>"
        "<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\">"
        "<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>"
        f"<vt:variant><vt:i4>{count}</vt:i4></vt:variant>"
        "</vt:vector></HeadingPairs>"
        f"<TitlesOfParts><vt:vector size=\"{count}\" baseType=\"lpstr\">{titles}</vt:vector></TitlesOfParts>"
        "<Company></Company>"
        "<LinksUpToDate>false</LinksUpToDate>"
        "<SharedDoc>false</SharedDoc>"
        "<HyperlinksChanged>false</HyperlinksChanged>"
        "<AppVersion>16.0300</AppVersion>"
        "</Properties>"
    )


def write_workbook(output_path: Path, sheets: Sequence[SheetSpec]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("docProps/core.xml", core_props_xml())
        zf.writestr("docProps/app.xml", app_props_xml([sheet.name for sheet in sheets]))
        zf.writestr("xl/workbook.xml", workbook_xml(sheets))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(sheets))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, sheet in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(sheet))


def build_sheets(index_meta: dict, detail_items: Sequence[dict]) -> List[SheetSpec]:
    return [
        SheetSpec("README", build_readme_rows(index_meta, detail_items), auto_filter=False),
        SheetSpec("kpi_core", build_kpi_core_rows(index_meta, detail_items)),
        SheetSpec("dimensions", build_dimensions_rows(detail_items)),
        SheetSpec("thresholds", build_threshold_rows(detail_items)),
        SheetSpec("diag_triggers", build_trigger_rows(detail_items)),
        SheetSpec("diag_actions", build_action_rows(detail_items)),
        SheetSpec("base_filters", build_base_filter_rows(detail_items)),
    ]


def main() -> int:
    args = parse_args()
    index_meta = load_json(args.index)
    detail_items = load_json(args.detail)
    sheets = build_sheets(index_meta, detail_items)
    write_workbook(args.output, sheets)
    print(f"Wrote KPI meta template: {args.output}")
    print("Sheets:", ", ".join(sheet.name for sheet in sheets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
