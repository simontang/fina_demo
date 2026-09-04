#!/usr/bin/env bash
set -euo pipefail

METRICS_BASE_URL="${METRICS_BASE_URL:-https://ada.alphafina.cn/api/metrics/api/v1}"
TENANT_ID="${TENANT_ID:-hankel}"
DATASOURCE_ID="${DATASOURCE_ID:-15}"
METRICS_INSECURE_SSL="${METRICS_INSECURE_SSL:-true}"

export METRICS_BASE_URL TENANT_ID DATASOURCE_ID METRICS_INSECURE_SSL

python3 - <<'PY'
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

base_url = os.environ["METRICS_BASE_URL"].rstrip("/")
tenant_id = os.environ["TENANT_ID"]
datasource_id = os.environ["DATASOURCE_ID"]
insecure_ssl = os.environ.get("METRICS_INSECURE_SSL", "true").lower() in {"1", "true", "yes"}
ssl_context = ssl._create_unverified_context() if insecure_ssl else None


def request(method, path, payload=None):
    url = base_url + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    for attempt in range(1, 4):
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "X-Tenant-Id": tenant_id},
        )
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            if attempt == 3:
                raise RuntimeError(f"{method} {url} failed after {attempt} attempts: {exc}") from exc
            time.sleep(attempt)
    data = json.loads(raw) if raw else {}
    if data.get("code") not in (None, 200):
        raise RuntimeError(f"{method} {url} failed: {json.dumps(data, ensure_ascii=False)}")
    return data


def quote(value):
    return urllib.parse.quote(str(value), safe="")


def existing(kind, key, object_type=None):
    path = f"/datasources/{datasource_id}/meta/{kind}/{quote(key)}"
    if object_type:
        path += f"?objectType={quote(object_type)}"
    data = request("GET", path).get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("items") or []
    return []


def upsert(kind, key, object_type, payload, access_grant=None):
    body = {
        "objectType": object_type,
        "objectKey": key,
        "payload": payload,
        "status": 1,
    }
    if access_grant is not None:
        body["accessGrant"] = access_grant
    if existing(kind, key, object_type):
        request("PUT", f"/datasources/{datasource_id}/meta/{kind}/{quote(key)}", body)
        print(f"updated {kind}/{object_type}: {key}")
    else:
        request("POST", f"/datasources/{datasource_id}/meta/{kind}", body)
        print(f"created {kind}/{object_type}: {key}")


def unpublish_table(key):
    if existing("tables", key):
        request("DELETE", f"/datasources/{datasource_id}/meta/tables/{quote(key)}")
        print(f"unpublished runtime table: {key}")


def dim(name, label, dtype="string"):
    return {"dim_id": name, "field_name": name, "label": label, "data_type": dtype}


def column(name, label, dtype="string", role="dimension"):
    return {"name": name, "label": label, "type": dtype, "role": role}


def grant(table_name):
    return {
        "schemaName": "public",
        "tablePattern": table_name,
        "patternType": "EXACT",
        "caseSensitive": False,
        "status": 1,
    }


def table_payload(table_name, display_name, description, grain, source_tables, columns, category):
    return {
        "schemaName": "public",
        "tableName": table_name,
        "displayName": display_name,
        "docType": "Hankel Distributor Performance",
        "docTypeEn": "Hankel Distributor Performance",
        "shortDesc": description,
        "grain": grain,
        "sourceSystem": "Hankel distributor review demo",
        "sourceTables": source_tables,
        "assetCategory": category,
        "columns": columns,
    }


def publish_table(payload):
    key = payload["tableName"]
    for object_type in ("table_catalog", "table_view_detail"):
        upsert("tables", key, object_type, payload, grant(key))


def time_context(field, label, window):
    return {
        "time_dimension": field,
        "label": label,
        "granularity": "month",
        "window": window,
        "supported_grains": ["month", "year"],
        "query_usage": {
            "filter": {
                "dimension_key": field,
                "supported_operators": ["BETWEEN", "EQ", "GTE", "LTE"],
                "value_format": "YYYY-MM-DD",
                "example": {
                    "dimension": field,
                    "operator": "BETWEEN",
                    "values": ["2026-01-01", "2026-12-31"],
                },
            },
            "group_by": {
                "pattern": f"{field}__{{grain}}",
                "examples": [f"{field}__month", f"{field}__year"],
            },
        },
    }


def publish_metric(name, display, description, source_view, calculation, dimensions,
                   fmt, synonyms, time_ctx):
    index_payload = {
        "metric_name": name,
        "display_name": display,
        "domain": "Hankel Distributor Performance",
        "short_desc": description,
        "search_keywords": synonyms + ["hankel", "distributor"],
        "source_type": "cdp_postgres",
        "source": {"table_view": source_view},
    }
    detail_payload = {
        "metric_name": name,
        "display_name": display,
        "domain": "Hankel Distributor Performance",
        "description": description,
        "data_type": "numeric",
        "format": fmt,
        "source_type": "cdp_postgres",
        "source": {"table_view": source_view, "base_filters": []},
        "calculation": calculation,
        "supported_dimensions": dimensions,
        "default_time_context": time_ctx,
        "ai_agent_context": {
            "polarity": "positive",
            "synonyms": synonyms,
            "human_readable_explanation": description,
        },
    }
    upsert("metrics", name, "metric_index", index_payload)
    upsert("metrics", name, "metric_detail", detail_payload)


sell_in_table = table_payload(
    "hankel_view_distr_sell_in",
    "Hankel 经销商 Sell-in 规范化明细",
    "Sell-in 流量明细，保留导入行并将 Excel 日期序列转换为可筛选的 posting_date。",
    "one imported sell-in allocation row",
    ["hankel_distr_sell_in"],
    [
        column("posting_year", "记账年份", "number"),
        column("posting_date", "记账日期", "date"),
        column("year_month", "账期"),
        column("sales_team", "销售团队"),
        column("sold_to", "经销商代码"),
        column("sold_to_name", "经销商名称"),
        column("product_idh", "产品 ID"),
        column("product_description", "产品描述"),
        column("gmm_l6_allocation", "GMM L6 分摊标记"),
        column("whether_year_sell_out", "是否计入当年 Sell-out"),
        column("whether_spl", "是否 SPL"),
        column("sell_in_quantity", "Sell-in 数量", "number", "measure"),
        column("nes", "NES 净外部销售额", "number", "measure"),
        column("gross_margin", "毛利额", "number", "measure"),
        column("product_contribution_15", "产品贡献额", "number", "measure"),
    ],
    "normalized flow view",
)

inventory_monthly_table = table_payload(
    "hankel_view_distr_inventory_monthly",
    "Hankel 经销商月度库存规范化明细",
    "按月保留销售团队分摊后的库存快照；raw_inventory_* 仅供审计，不用于汇总。",
    "one customer-product-month-sales-team allocation row",
    ["hankel_distr_inventory"],
    [
        column("snapshot_date", "库存快照日期", "date"),
        column("year", "年份", "number"),
        column("year_month", "账期"),
        column("sales_team", "销售团队"),
        column("customer_idh", "经销商 ID"),
        column("sold_to", "Sold-to"),
        column("product_idh", "产品 ID"),
        column("product_unit", "产品单位"),
        column("data_source", "数据来源"),
        column("whether_spl", "是否 SPL"),
        column("inventory_value", "分摊库存金额", "number", "measure"),
        column("inventory_quantity", "分摊库存数量", "number", "measure"),
        column("raw_inventory_value", "原始库存金额", "number", "audit"),
        column("raw_inventory_quantity", "原始库存数量", "number", "audit"),
        column("is_latest_snapshot", "是否最新快照", "boolean"),
    ],
    "normalized stock snapshot view",
)

inventory_current_table = table_payload(
    "hankel_view_distr_inventory_current",
    "Hankel 经销商当前库存明细",
    "仅包含全局最新可用月份，使用销售团队分摊后的库存金额和数量。",
    "one customer-product-sales-team allocation row at latest snapshot",
    ["hankel_view_distr_inventory_monthly"],
    inventory_monthly_table["columns"],
    "current stock snapshot view",
)

for table in (sell_in_table, inventory_monthly_table, inventory_current_table):
    publish_table(table)

sell_in_dims = [
    dim("posting_year", "记账年份", "number"),
    dim("posting_date", "记账日期", "date"),
    dim("year_month", "账期"),
    dim("sales_team", "销售团队"),
    dim("sold_to", "经销商代码"),
    dim("sold_to_name", "经销商名称"),
    dim("product_idh", "产品 ID"),
    dim("product_description", "产品描述"),
    dim("gmm_l6_allocation", "GMM L6 分摊标记"),
    dim("whether_year_sell_out", "是否计入当年 Sell-out"),
    dim("whether_spl", "是否 SPL"),
]
sell_in_time = time_context("posting_date", "Sell-in posting date", "all_loaded_periods")
sell_in_source = "public.hankel_view_distr_sell_in"

sell_in_metrics = [
    ("hankel_sell_in_nes", "Sell-in NES 净外部销售额", "Sell-in NES（人民币）。未提供日期筛选时汇总全部已加载期间。", "nes", "currency", ["sell-in nes", "净外部销售额"]),
    ("hankel_sell_in_quantity", "Sell-in 进货数量", "Sell-in 数量。未提供日期筛选时汇总全部已加载期间。", "sell_in_quantity", "number", ["sell-in quantity", "进货数量"]),
    ("hankel_sell_in_gross_margin", "Sell-in 毛利额", "Sell-in 毛利额（人民币）。未提供日期筛选时汇总全部已加载期间。", "gross_margin", "currency", ["sell-in gross margin", "毛利额"]),
    ("hankel_sell_in_contribution", "Sell-in 产品贡献额", "Sell-in 产品贡献额（人民币）。未提供日期筛选时汇总全部已加载期间。", "product_contribution_15", "currency", ["sell-in contribution", "产品贡献额"]),
]
for name, display, description, measure, fmt, synonyms in sell_in_metrics:
    publish_metric(
        name, display, description, sell_in_source,
        {"type": "aggregate", "aggregation": "sum", "measure": measure},
        sell_in_dims, fmt, synonyms, sell_in_time,
    )

publish_metric(
    "hankel_sell_in_gross_margin_rate",
    "Sell-in 毛利率",
    "Sell-in 毛利额除以 NES；按查询范围加权计算，不对行级百分比取平均。未提供日期筛选时使用全部已加载期间。",
    sell_in_source,
    {
        "type": "derived",
        "operator": "ratio",
        "numerator": "hankel_sell_in_gross_margin",
        "denominator": "hankel_sell_in_nes",
    },
    sell_in_dims,
    "percent",
    ["sell-in gross margin rate", "毛利率"],
    sell_in_time,
)

inventory_dims = [
    dim("snapshot_date", "库存快照日期", "date"),
    dim("year", "年份", "number"),
    dim("year_month", "账期"),
    dim("sales_team", "销售团队"),
    dim("customer_idh", "经销商 ID"),
    dim("sold_to", "Sold-to"),
    dim("product_idh", "产品 ID"),
    dim("product_unit", "产品单位"),
    dim("data_source", "数据来源"),
    dim("whether_spl", "是否 SPL"),
]
inventory_time = time_context("snapshot_date", "Inventory snapshot date", "latest_available_snapshot")
inventory_source = "public.hankel_view_distr_inventory_current"

publish_metric(
    "hankel_inventory_value",
    "当前库存金额",
    "最新可用月度快照的经销商库存金额（人民币），使用销售团队分摊值；不跨月份累计。",
    inventory_source,
    {"type": "aggregate", "aggregation": "sum", "measure": "inventory_value"},
    inventory_dims,
    "currency",
    ["current inventory value", "当前库存金额"],
    inventory_time,
)
publish_metric(
    "hankel_inventory_quantity",
    "当前库存数量",
    "最新可用月度快照的经销商库存数量，使用销售团队分摊值；不跨月份累计。",
    inventory_source,
    {"type": "aggregate", "aggregation": "sum", "measure": "inventory_quantity"},
    inventory_dims,
    "number",
    ["current inventory quantity", "当前库存数量"],
    inventory_time,
)

# Runtime exposes semantic views only. Builder/Admin datasource query can still
# inspect the underlying raw tables through the datasource grant scope.
for raw_table in ("hankel_distr_sell_in", "hankel_distr_inventory", "hankel_distr_sell_out"):
    unpublish_table(raw_table)

print("Done.")
PY
