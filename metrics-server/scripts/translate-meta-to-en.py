#!/usr/bin/env python3
"""
Translate metrics-server meta resources to English:
  - meta/table-catalog.json: docType/docTypeEn/shortDesc
  - meta/MTC_VW_AI_*.csv: Description column

Constraints:
  - Python standard library only.
  - Only rewrites when needed.
  - Ensures there are no CJK characters left in translated fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def has_cjk(s: Optional[str]) -> bool:
    return bool(s) and bool(CJK_RE.search(s))


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_punct(s: str) -> str:
    return (
        s.replace("；", ";")
        .replace("：", ":")
        .replace("，", ",")
        .replace("。", ".")
        .replace("（", "(")
        .replace("）", ")")
        .replace("、", "; ")
    )


@dataclass(frozen=True)
class CatalogTranslation:
    doc_type: str
    short_desc: str


CATALOG_BY_TABLE: Dict[str, CatalogTranslation] = {
    "MTC_VW_AI_OPCH": CatalogTranslation(
        doc_type="AP Invoice",
        short_desc="AP invoice view (purchasing). Positive purchase amounts; used for AP and procurement analysis.",
    ),
    "MTC_VW_AI_ORPC": CatalogTranslation(
        doc_type="AP Credit Memo",
        short_desc="AP credit memo view (purchasing). Negative purchase amounts (returns/adjustments).",
    ),
    "MTC_VW_AI_OPKL": CatalogTranslation(
        doc_type="Picking List",
        short_desc="Picking list view. Picking header and status for warehouse operations.",
    ),
    "MTC_VW_AI_CUSTBAL": CatalogTranslation(
        doc_type="Customer Balance Details",
        short_desc="Customer balance details (AR): aging/terms, payments received, and outstanding receivables.",
    ),
    "MTC_VW_AI_CUSTCREDIT": CatalogTranslation(
        doc_type="Customer Credit Exposure",
        short_desc="Customer credit exposure: credit limit, AR account balance (local currency), open orders, and open deliveries.",
    ),
    "MTC_VW_AI_CUSTBPID": CatalogTranslation(
        doc_type="Customer Branch Mapping",
        short_desc="Customer to branch/company mapping (organizational assignment).",
    ),
    "MTC_VW_AI_DEALBAL": CatalogTranslation(
        doc_type="Vendor Balance Details",
        short_desc="Vendor balance details (AP): aging/terms, payments made, and outstanding payables.",
    ),
    "MTC_VW_AI_DEALCREDIT": CatalogTranslation(
        doc_type="Vendor Credit Exposure",
        short_desc="Vendor credit exposure: credit limit, AP account balance (local currency), open purchase orders, and open goods receipts.",
    ),
    "MTC_VW_AI_DEALBPID": CatalogTranslation(
        doc_type="Vendor Branch Mapping",
        short_desc="Vendor to branch/company mapping (organizational assignment).",
    ),
    "MTC_VW_AI_FUNDBAL": CatalogTranslation(
        doc_type="Fund Balance",
        short_desc="Fund/cash balance view.",
    ),
    "MTC_VW_AI_OWOR": CatalogTranslation(
        doc_type="Production Order",
        short_desc="Production order header view.",
    ),
    "MTC_VW_AI_OWORDetail": CatalogTranslation(
        doc_type="Production Order Details",
        short_desc="Production order detail/line view.",
    ),
    "MTC_VW_AI_ODRF_OIGE": CatalogTranslation(
        doc_type="Inventory Goods Issue Draft",
        short_desc="Draft inventory goods issue (stock outbound) view.",
    ),
    "MTC_VW_AI_ODRF_OIGN": CatalogTranslation(
        doc_type="Inventory Goods Receipt Draft",
        short_desc="Draft inventory goods receipt (stock inbound) view.",
    ),
    "MTC_VW_AI_ODRF_OWTQ": CatalogTranslation(
        doc_type="Inventory Transfer Request Draft",
        short_desc="Draft inventory transfer request view.",
    ),
    "MTC_VW_AI_OWTQ": CatalogTranslation(
        doc_type="Inventory Transfer Request",
        short_desc="Inventory transfer request view.",
    ),
    "MTC_VW_AI_OWTR": CatalogTranslation(
        doc_type="Inventory Transfer",
        short_desc="Inventory transfer view.",
    ),
    "MTC_VW_AI_OACT": CatalogTranslation(
        doc_type="G/L Account Master",
        short_desc="G/L account master data view.",
    ),
    "MTC_VW_AI_OBPL": CatalogTranslation(
        doc_type="Branch Master",
        short_desc="Branch (BPL) master data view.",
    ),
    "MTC_VW_AI_OIGE": CatalogTranslation(
        doc_type="Inventory Goods Issue",
        short_desc="Inventory goods issue (stock outbound) view.",
    ),
    "MTC_VW_AI_OIGN": CatalogTranslation(
        doc_type="Inventory Goods Receipt",
        short_desc="Inventory goods receipt (stock inbound) view.",
    ),
    "MTC_VW_AI_OITM": CatalogTranslation(
        doc_type="Item Master",
        short_desc="Item master data view.",
    ),
    "MTC_VW_AI_OWHS": CatalogTranslation(
        doc_type="Warehouse Master",
        short_desc="Warehouse master data view.",
    ),
    "MTC_VW_AI_OPLN": CatalogTranslation(
        doc_type="Price List",
        short_desc="Price list master view.",
    ),
    "MTC_VW_AI_ITM1": CatalogTranslation(
        doc_type="Item Price List",
        short_desc="Item price list view.",
    ),
    "MTC_VW_AI_OSPP": CatalogTranslation(
        doc_type="BP Special Prices",
        short_desc="Business partner special prices view.",
    ),
    "MTC_VW_AI_FixedAssets": CatalogTranslation(
        doc_type="Fixed Assets Master",
        short_desc="Fixed assets master data view.",
    ),
    "MTC_VW_AI_ORCT": CatalogTranslation(
        doc_type="Incoming Payment",
        short_desc="Incoming payment (receipt) view.",
    ),
    "MTC_VW_AI_OVPM": CatalogTranslation(
        doc_type="Outgoing Payment",
        short_desc="Outgoing payment view.",
    ),
    "MTC_VW_AI_OPOR": CatalogTranslation(
        doc_type="Purchase Order",
        short_desc="Purchase order view for procurement analytics, open POs, and vendor delivery/receipt tracking.",
    ),
    "MTC_VW_AI_ODRF_OWTR": CatalogTranslation(
        doc_type="Inventory Transfer Draft",
        short_desc="Draft inventory transfer view.",
    ),
    "MTC_VW_AI_BOM_LIST": CatalogTranslation(
        doc_type="Bill of Materials",
        short_desc="Bill of materials (BOM) view.",
    ),
    "MTC_VW_AI_Customer": CatalogTranslation(
        doc_type="Customer Master",
        short_desc="Customer master data view.",
    ),
    "MTC_VW_AI_Supplier": CatalogTranslation(
        doc_type="Vendor Master",
        short_desc="Vendor master data view.",
    ),
    "MTC_VW_AI_OINC": CatalogTranslation(
        doc_type="Inventory Counting",
        short_desc="Inventory counting (stocktaking) view.",
    ),
    "MTC_VW_AI_OIVL": CatalogTranslation(
        doc_type="Inventory Movements",
        short_desc="Item inventory movements; used to identify inactive/slow-moving items.",
    ),
}


def translate_table_catalog(meta_dir: Path, dry_run: bool) -> Tuple[int, List[str]]:
    path = meta_dir / "table-catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("table-catalog.json must be a JSON array")

    changed = 0
    changed_tables: List[str] = []

    for entry in data:
        if not isinstance(entry, dict):
            continue
        table = entry.get("tableName")
        if not isinstance(table, str) or not table:
            continue

        trans = CATALOG_BY_TABLE.get(table)
        if not trans:
            continue

        need = (
            has_cjk(entry.get("docType"))
            or has_cjk(entry.get("docTypeEn"))
            or has_cjk(entry.get("shortDesc"))
            or not entry.get("docType")
            or not entry.get("docTypeEn")
        )
        if not need:
            continue

        entry["docType"] = trans.doc_type
        entry["docTypeEn"] = trans.doc_type
        entry["shortDesc"] = trans.short_desc

        if has_cjk(entry["docType"]) or has_cjk(entry["docTypeEn"]) or has_cjk(entry["shortDesc"]):
            raise ValueError(f"Catalog translation still contains CJK for table {table}")

        changed += 1
        changed_tables.append(table)

    if changed and not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return changed, changed_tables


FIELD_DESC_FALLBACK: Dict[str, str] = {
    "DocNum": "Document no. e.g. 1029",
    "DocEntry": "Internal ID e.g. 1029",
    "DocDate": "Document date e.g. 2024-05-30",
    "DocStatus": "Document status e.g. O-Open; C-Closed",
    "DocType": "Document type e.g. I-Item; S-Service",
    "ObjType": "Object type code",
    "Canceled": "Canceled flag e.g. Y-Yes; N-No",
    "CardCode": "Business partner code e.g. C19001; V10000",
    "CardName": "Business partner name e.g. Acme Ltd",
    "GroupName": "Business partner group e.g. Key Account; Standard",
    "Province": "Province e.g. Guangdong",
    "Region": "Region e.g. South China",
    "DocCur": "Currency e.g. RMB; USD; EUR",
    "Currency": "Currency e.g. RMB; USD; EUR",
    "IsLocCurrency": "Is local currency e.g. Y-Yes; N-No",
    "DiscPrcnt": "Document discount % e.g. 2",
    "SlpName": "Salesperson/buyer name e.g. John Smith",
    "LineNum": "Line no. e.g. 0; 1; 2",
    "LineStatus": "Line status e.g. O-Open; C-Closed",
    "BaseType": "Base object type code",
    "BaseEntry": "Base document internal ID",
    "BaseLine": "Base line no.",
    "ItemCode": "Item code e.g. A00001",
    "ItemName": "Item name/description e.g. Inkjet Printer",
    "Dscription": "Item description e.g. Inkjet Printer",
    "ItmsGrpNam": "Item group name e.g. 203-Printer",
    "ItmsGrpCod": "Item group code e.g. 101",
    "WhsCode": "Warehouse code e.g. 01",
    "WhsName": "Warehouse name e.g. Main WH",
    "Quantity": "Quantity e.g. 12",
    "OpenQty": "Open quantity e.g. 2",
    "OpenCreQty": "Open quantity e.g. 2",
    "ShipDate": "Ship date e.g. 2024-05-30",
    "VatGroup": "Tax code e.g. X1; X2",
    "VatPrcnt": "Tax/VAT rate % e.g. 13.0",
    "Rate": "Exchange rate e.g. 7.9076",
    "Price": "Unit price (excl. tax) e.g. 2310.00",
    "PriceAfVAT": "Unit price (incl. tax) e.g. 2310.00",
    "LineTotal": "Amount (excl. tax) e.g. 2702.70",
    "GTotal": "Amount (incl. tax) e.g. 23100.00",
    "TotalFrgn": "Amount (excl. tax, FC) e.g. 3281.25",
    "GTotalFC": "Amount (incl. tax, FC) e.g. 3281.25",
    "Project": "Project code",
    "PrjName": "Project name",
}


EXAMPLE_MAP: Dict[str, str] = {
    "Y-是": "Y-Yes",
    "N-否": "N-No",
    "O-未清": "O-Open",
    "C-已结算": "C-Closed",
    "I-物料": "I-Item",
    "S-服务": "S-Service",
    "C-客户": "C-Customer",
    "S-供应商": "S-Vendor",
    "L-潜在客户": "L-Lead",
    "可用": "Available",
}


PHRASE_REPLACEMENTS: List[Tuple[str, str]] = [
    ("业务伙伴参考编号", "BP reference"),
    ("业务伙伴目录编号", "BP catalog no."),
    ("业务伙伴BIC/SWIFT 代码", "BP BIC/SWIFT code"),
    ("业务伙伴银行所再国家地区", "BP bank country/region"),
    ("业务伙伴银行账号", "BP bank account"),
    ("业务伙伴银行分支", "BP bank branch"),
    ("业务伙伴银行名称", "BP bank name"),
    ("业务伙伴银行代码", "BP bank code"),
    ("业务伙伴类型", "BP type"),
    ("业务伙伴组", "BP group"),
    ("业务伙伴名称", "BP name"),
    ("业务伙伴代码", "BP code"),
    ("上级科目代码", "Parent account code"),
    ("内部标识号", "Internal ID"),
    ("单据日期", "Document date"),
    ("单据状态", "Document status"),
    ("单据类型", "Document type"),
    ("单据折扣", "Document discount"),
    ("汇率", "Exchange rate"),
    ("本位币", "local currency"),
    ("外币", "foreign currency"),
    ("不含税单价", "Unit price (excl. tax)"),
    ("含税单价", "Unit price (incl. tax)"),
    ("不含税金额", "Amount (excl. tax)"),
    ("含税金额", "Amount (incl. tax)"),
    ("不含税外币金额", "Amount (excl. tax, FC)"),
    ("含税外币金额", "Amount (incl. tax, FC)"),
    ("仓库代码", "Warehouse code"),
    ("仓库名称", "Warehouse name"),
    ("物料代码", "Item code"),
    ("物料名称/物料描述", "Item name/description"),
    ("物料名称", "Item name"),
    ("物料描述", "Item description"),
    ("物料组代码", "Item group code"),
    ("物料组名称", "Item group name"),
    ("物料组", "Item group"),
    ("项目代码", "Project code"),
    ("项目名称", "Project name"),
    ("成本中心", "Cost center"),
    ("科目代码", "Account code"),
    ("科目名称", "Account name"),
    ("科目", "Account"),
    ("取消", "Canceled"),
    ("备注", "Remarks"),
    ("状态", "Status"),
    ("类型", "Type"),
    ("编号", "No."),
    ("名称", "Name"),
    ("代码", "Code"),
    ("日期", "Date"),
    ("数量", "Quantity"),
    ("金额", "Amount"),
    ("货币", "Currency"),
]


def translate_prefix(prefix: str) -> str:
    s = norm_ws(normalize_punct(prefix))

    # Regex-based patterns
    m = re.fullmatch(r"成本中心(\d+)代码", s)
    if m:
        return f"Cost center {m.group(1)} code"
    m = re.fullmatch(r"成本中心(\d+)名称", s)
    if m:
        return f"Cost center {m.group(1)} name"
    m = re.fullmatch(r"成本利润中心(\d+)代码", s)
    if m:
        return f"Profit center {m.group(1)} code"
    m = re.fullmatch(r"成本利润中心(\d+)名称", s)
    if m:
        return f"Profit center {m.group(1)} name"

    m = re.fullmatch(r"使用年限\s*(\d+)", s)
    if m:
        return f"Useful life {m.group(1)}"

    m = re.fullmatch(r"折旧类型\s*([A-Za-z0-9_\-]+)", s)
    if m:
        return f"Depreciation type {m.group(1)}"

    if s.startswith("折旧类型名称"):
        rest = s[len("折旧类型名称") :].strip()
        rest = rest.replace("直线折旧", "Straight-line depreciation")
        return norm_ws(f"Depreciation type name {rest}".strip())

    m = re.fullmatch(r"折旧范围\s*([A-Za-z0-9_\-]+)", s)
    if m:
        return f"Depreciation area {m.group(1)}"

    if s.startswith("折旧范围名称"):
        rest = s[len("折旧范围名称") :].strip()
        rest = rest.replace("过账到总账", "Post to G/L")
        return norm_ws(f"Depreciation area name {rest}".strip())

    if s.startswith("发货方式"):
        out = normalize_punct(s)
        out = out.replace("手动", "Manual").replace("倒冲", "Backflush")
        return norm_ws(out.replace("发货方式", "Issue method"))

    out = s
    for cn, en in PHRASE_REPLACEMENTS:
        out = out.replace(cn, en)
    return norm_ws(out)


def english_example(field_name: str, typ: str) -> str:
    fb = FIELD_DESC_FALLBACK.get(field_name)
    if fb and "e.g." in fb:
        return fb.split("e.g.", 1)[1].strip()

    t = (typ or "").upper()
    if "TIMESTAMP" in t or "DATE" in t:
        return "2024-05-30"
    if "INT" in t:
        return "1029"
    if "DECIMAL" in t or "NUM" in t or "DOUBLE" in t or "FLOAT" in t:
        return "123.45"
    if "CHAR" in t or "VARCHAR" in t or "NVARCHAR" in t:
        if field_name.lower().endswith("name"):
            return "Acme Ltd"
        if field_name.lower().endswith("code"):
            return "C19001"
        return "Sample"
    return "Sample"


def translate_desc(desc: str, field_name: str, typ: str) -> str:
    d = norm_ws(normalize_punct(desc))
    if not has_cjk(d):
        return d

    # Special case: "货币 例如；RMB；EUR"
    if d.startswith("货币") and "例如" in d:
        d = d.replace("例如", "e.g.")

    m = re.match(r"^(.*?)\s*例如\s*[:：]\s*(.*)$", d)
    label = d
    example: Optional[str] = None
    if m:
        label = m.group(1).strip()
        example = m.group(2).strip()

    label_en = translate_prefix(label)

    example_en: Optional[str] = None
    if example is not None:
        ex = norm_ws(normalize_punct(example))
        for cn, en in EXAMPLE_MAP.items():
            ex = ex.replace(cn, en)
        ex = norm_ws(ex)
        if has_cjk(ex):
            ex = english_example(field_name, typ)
        example_en = ex

    out = label_en
    if example_en:
        out = f"{label_en} e.g. {example_en}"

    out = norm_ws(out)
    if has_cjk(out):
        out = FIELD_DESC_FALLBACK.get(field_name, field_name or "Field")

    if has_cjk(out):
        raise ValueError(f"Untranslated description (still has CJK): {desc} -> {out}")
    return out


def translate_csv_file(path: Path, dry_run: bool) -> Tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return (0, 0)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    changed = 0
    for row in rows:
        desc = (row.get("Description") or "").strip()
        if not has_cjk(desc):
            continue
        field_name = (row.get("Field Name") or "").strip()
        typ = (row.get("Type") or "").strip()
        new_desc = translate_desc(desc, field_name=field_name, typ=typ)
        if new_desc != desc:
            row["Description"] = new_desc
            changed += 1

    if changed and not dry_run:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    return (changed, len(rows))


def assert_meta_english(meta_dir: Path) -> None:
    cat = json.loads((meta_dir / "table-catalog.json").read_text(encoding="utf-8"))
    for e in cat:
        if not isinstance(e, dict):
            continue
        for k in ("docType", "docTypeEn", "shortDesc"):
            v = e.get(k)
            if isinstance(v, str) and has_cjk(v):
                raise SystemExit(f"CJK remains in table-catalog.json field {k}: {v}")

    for p in meta_dir.glob("MTC_VW_AI_*.csv"):
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                d = (row.get("Description") or "").strip()
                if has_cjk(d):
                    raise SystemExit(f"CJK remains in {p.name} Description: {d}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Translate metrics-server meta to English")
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="Fail if any CJK remains after translation")
    args = ap.parse_args(argv)

    meta_dir = Path(args.meta_dir).expanduser().resolve()
    if not meta_dir.is_dir():
        print(f"ERROR: meta dir not found: {meta_dir}", file=sys.stderr)
        return 2

    dry_run = bool(args.dry_run)

    cat_changed, cat_tables = translate_table_catalog(meta_dir, dry_run=dry_run)

    csv_files = sorted(meta_dir.glob("MTC_VW_AI_*.csv"))
    csv_changed_files: List[str] = []
    total_rows_changed = 0
    for p in csv_files:
        rows_changed, _rows_total = translate_csv_file(p, dry_run=dry_run)
        if rows_changed:
            csv_changed_files.append(p.name)
            total_rows_changed += rows_changed

    print(f"table-catalog entries updated: {cat_changed} (dry_run={dry_run})")
    for t in cat_tables:
        print(f"  - {t}")

    print(
        f"CSV files updated: {len(csv_changed_files)} "
        f"(rows changed: {total_rows_changed}, dry_run={dry_run})"
    )
    for n in csv_changed_files:
        print(f"  - {n}")

    if args.check:
        assert_meta_english(meta_dir)
        print("CHECK OK: meta table-catalog and CSV descriptions contain no CJK characters.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

