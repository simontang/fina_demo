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
            headers={
                "Content-Type": "application/json",
                "X-Tenant-Id": tenant_id,
            },
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


def existing(kind, key, object_type):
    result = request(
        "GET",
        f"/datasources/{datasource_id}/meta/{kind}/{quote(key)}"
        f"?objectType={quote(object_type)}",
    ).get("data")
    if isinstance(result, list):
        return result[0] if result else None
    if isinstance(result, dict):
        items = result.get("items") or []
        return items[0] if items else None
    return None


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


def dimension(name, label, data_type="string"):
    return {
        "dim_id": name,
        "field_name": name,
        "label": label,
        "data_type": data_type,
    }


source_view = "public.hankel_view_distr_sell_out"
table_name = "hankel_view_distr_sell_out"

columns = [
    {"name": "customer_code", "label": "经销商代码", "type": "string", "role": "dimension"},
    {"name": "customer_name", "label": "经销商名称", "type": "string", "role": "dimension"},
    {"name": "end_customer_name", "label": "终端客户名称", "type": "string", "role": "dimension"},
    {"name": "product_code", "label": "产品代码", "type": "string", "role": "dimension"},
    {"name": "product_name", "label": "产品名称", "type": "string", "role": "dimension"},
    {"name": "product_unit", "label": "产品单位", "type": "string", "role": "dimension"},
    {"name": "sales_team", "label": "销售团队", "type": "string", "role": "dimension"},
    {"name": "combined_id", "label": "经销商 Combined ID", "type": "string", "role": "dimension"},
    {"name": "end_cust_num", "label": "终端客户编号", "type": "string", "role": "dimension"},
    {"name": "year", "label": "年份", "type": "number", "role": "dimension"},
    {"name": "year_month", "label": "账期", "type": "string", "role": "dimension"},
    {"name": "period_date", "label": "账期日期", "type": "date", "role": "dimension"},
    {"name": "platform", "label": "平台渠道", "type": "string", "role": "dimension"},
    {"name": "province", "label": "省份", "type": "string", "role": "dimension"},
    {"name": "city", "label": "城市", "type": "string", "role": "dimension"},
    {"name": "national_industry_l1", "label": "行业一级", "type": "string", "role": "dimension"},
    {"name": "national_industry_l2", "label": "行业二级", "type": "string", "role": "dimension"},
    {"name": "combined_national_industry_l2", "label": "国标中类行业组合", "type": "string", "role": "dimension"},
    {"name": "national_industry_l3", "label": "行业三级", "type": "string", "role": "dimension"},
    {"name": "henkel_key_mid_ind", "label": "Henkel Key Mid Industry", "type": "string", "role": "dimension"},
    {"name": "acm_sales_territory", "label": "ACM 销售区域", "type": "string", "role": "dimension"},
    {"name": "data_source", "label": "数据来源", "type": "string", "role": "dimension"},
    {"name": "whether_spl", "label": "是否 SPL", "type": "string", "role": "dimension"},
    {"name": "ta_flag", "label": "TA 标记", "type": "string", "role": "dimension"},
    {"name": "bian_end_cust", "label": "终端客户有效标记", "type": "string", "role": "dimension"},
    {"name": "sell_out_quantity", "label": "有效分摊出货数量", "type": "number", "role": "measure"},
    {"name": "sell_out_value", "label": "有效分摊出货金额", "type": "number", "role": "measure"},
    {"name": "raw_sales_quantity", "label": "原始出货数量", "type": "number", "role": "measure"},
    {"name": "raw_sell_out_value", "label": "原始出货金额", "type": "number", "role": "measure"},
    {"name": "allocated_sales_quantity", "label": "分摊出货数量", "type": "number", "role": "measure"},
    {"name": "allocated_sell_out_value", "label": "分摊出货金额", "type": "number", "role": "measure"},
    {"name": "excluded_sell_out_quantity", "label": "质量规则排除数量", "type": "number", "role": "measure"},
    {"name": "excluded_sell_out_value", "label": "质量规则排除金额", "type": "number", "role": "measure"},
    {"name": "is_quantity_outlier", "label": "数量异常", "type": "boolean", "role": "dimension"},
    {"name": "is_amount_outlier", "label": "金额异常", "type": "boolean", "role": "dimension"},
    {"name": "has_invalid_allocation_value", "label": "分摊金额无效", "type": "boolean", "role": "dimension"},
    {"name": "is_quantity_quality_excluded", "label": "数量质量排除", "type": "boolean", "role": "dimension"},
    {"name": "is_value_quality_excluded", "label": "金额质量排除", "type": "boolean", "role": "dimension"},
    {"name": "is_quality_excluded", "label": "质量规则排除", "type": "boolean", "role": "dimension"},
    {"name": "quality_issue_row_count", "label": "异常行计数", "type": "number", "role": "measure"},
    {"name": "quality_issues", "label": "质量问题", "type": "string", "role": "dimension"},
]

table_payload = {
    "schemaName": "public",
    "tableName": table_name,
    "displayName": "Hankel 经销商 Sell-out 规范化明细",
    "docType": "Distributor Sell-out",
    "docTypeEn": "Distributor Sell-out",
    "shortDesc": "按销售团队分摊的 Sell-out 明细；金额和数量分别应用质量规则，原始值与排除值保留供审计。",
    "grain": "one customer-product-month-sales-team allocation row",
    "sourceSystem": "Hankel distributor review demo",
    "sourceTables": ["hankel_distr_sell_out"],
    "assetCategory": "normalized quality view",
    "business_status": "customer_confirmed_semantic_foundation",
    "knowledge_base_refs": [
        "hankel-metrics-kb/01-business-context-and-terms.md",
        "hankel-metrics-kb/02-river-distributor-metrics.md",
    ],
    "known_limitations": [
        "Cross-fact Sales Type is not published until the customer mapping is available.",
        "Outlier thresholds are demo quality guardrails, not customer KPI rules.",
    ],
    "columns": columns,
}
grant = {
    "schemaName": "public",
    "tablePattern": table_name,
    "patternType": "EXACT",
    "caseSensitive": False,
    "status": 1,
}

dimensions = [
    dimension("customer_code", "经销商代码"),
    dimension("customer_name", "经销商名称"),
    dimension("end_customer_name", "终端客户名称"),
    dimension("product_code", "产品代码"),
    dimension("product_name", "产品名称"),
    dimension("product_unit", "产品单位"),
    dimension("sales_team", "销售团队"),
    dimension("combined_id", "经销商 Combined ID"),
    dimension("end_cust_num", "终端客户编号"),
    dimension("year", "年份", "number"),
    dimension("year_month", "账期"),
    dimension("period_date", "账期日期", "date"),
    dimension("platform", "平台渠道"),
    dimension("province", "省份"),
    dimension("city", "城市"),
    dimension("national_industry_l1", "行业一级"),
    dimension("national_industry_l2", "行业二级"),
    dimension("combined_national_industry_l2", "国标中类行业组合"),
    dimension("national_industry_l3", "行业三级"),
    dimension("henkel_key_mid_ind", "Henkel Key Mid Industry"),
    dimension("acm_sales_territory", "ACM 销售区域"),
    dimension("data_source", "数据来源"),
    dimension("whether_spl", "是否 SPL"),
    dimension("ta_flag", "TA 标记"),
    dimension("bian_end_cust", "终端客户有效标记"),
    dimension("is_quantity_outlier", "数量异常", "boolean"),
    dimension("is_amount_outlier", "金额异常", "boolean"),
    dimension("is_quality_excluded", "质量规则排除", "boolean"),
    dimension("is_quantity_quality_excluded", "数量质量排除", "boolean"),
    dimension("is_value_quality_excluded", "金额质量排除", "boolean"),
]

metric_specs = [
    {
        "name": "hankel_sell_out_value",
        "display": "Sell-out 出货金额",
        "description": "Sell-out 金额（人民币）：汇总 Territory 分摊金额，只应用金额质量规则，不受数量异常影响。",
        "measure": "sell_out_value",
        "format": "currency",
        "polarity": "positive",
        "synonyms": ["sell-out amount", "出货金额", "终端出货金额"],
        "business_status": "customer_confirmed",
        "business_note": "Territory amount is customer-confirmed; demo amount guardrails remain separately disclosed.",
    },
    {
        "name": "hankel_sell_out_quantity",
        "display": "Sell-out 出货数量",
        "description": "Sell-out 数量：汇总 Territory 分摊数量，只应用数量质量规则，不受金额异常影响。",
        "measure": "sell_out_quantity",
        "format": "number",
        "polarity": "positive",
        "synonyms": ["sell-out quantity", "出货数量", "终端出货数量"],
        "business_status": "customer_confirmed",
        "business_note": "Territory quantity is customer-confirmed; demo quantity guardrails remain separately disclosed.",
    },
    {
        "name": "hankel_sell_out_excluded_value",
        "display": "Sell-out 质量规则排除金额",
        "description": "因金额异常或 Territory 金额不可解析而被金额指标排除的 Sell-out 分摊金额。",
        "measure": "excluded_sell_out_value",
        "format": "currency",
        "polarity": "negative",
        "synonyms": ["excluded sell-out", "异常金额", "被排除金额"],
        "business_status": "demo_quality",
        "business_note": "Quality-control metric; not a customer business KPI.",
    },
    {
        "name": "hankel_sell_out_quality_issue_count",
        "display": "Sell-out 异常行数",
        "description": "触发 Sell-out 质量规则的分摊行数量。",
        "measure": "quality_issue_row_count",
        "format": "number",
        "polarity": "negative",
        "synonyms": ["sell-out quality issues", "异常行数", "数据质量问题"],
        "business_status": "demo_quality",
        "business_note": "Quality-control metric; not a customer business KPI.",
    },
]

print(f"Publishing distributor Sell-out meta to {base_url}, tenant={tenant_id}, datasource={datasource_id}")
for object_type in ("table_catalog", "table_view_detail"):
    upsert("tables", table_name, object_type, table_payload, grant)

for spec in metric_specs:
    index_payload = {
        "metric_name": spec["name"],
        "display_name": spec["display"],
        "domain": "Hankel Distributor Performance",
        "short_desc": spec["description"],
        "search_keywords": spec["synonyms"] + ["hankel", "distributor"],
        "source_type": "cdp_postgres",
        "source": {"table_view": source_view},
        "business_status": spec["business_status"],
        "business_note": spec["business_note"],
        "knowledge_base_refs": ["hankel-metrics-kb/02-river-distributor-metrics.md"],
    }
    detail_payload = {
        "metric_name": spec["name"],
        "display_name": spec["display"],
        "domain": "Hankel Distributor Performance",
        "description": spec["description"],
        "data_type": "numeric",
        "format": spec["format"],
        "source_type": "cdp_postgres",
        "source": {"table_view": source_view, "base_filters": []},
        "calculation": {
            "type": "aggregate",
            "aggregation": "sum",
            "measure": spec["measure"],
        },
        "supported_dimensions": dimensions,
        "default_time_context": {
            "time_dimension": "period_date",
            "label": "Sell-out period",
            "granularity": "month",
            "window": "all_loaded_periods",
            "supported_grains": ["month", "year"],
            "query_usage": {
                "filter": {
                    "dimension_key": "period_date",
                    "supported_operators": ["BETWEEN", "EQ", "GTE", "LTE"],
                    "value_format": "YYYY-MM-DD",
                    "example": {
                        "dimension": "period_date",
                        "operator": "BETWEEN",
                        "values": ["2026-01-01", "2026-12-31"]
                    }
                },
                "group_by": {
                    "pattern": "period_date__{grain}",
                    "examples": ["period_date__month", "period_date__year"]
                }
            }
        },
        "business_status": spec["business_status"],
        "business_note": spec["business_note"],
        "knowledge_base_refs": ["hankel-metrics-kb/02-river-distributor-metrics.md"],
        "ai_agent_context": {
            "polarity": spec["polarity"],
            "synonyms": spec["synonyms"],
            "human_readable_explanation": spec["description"],
        },
    }
    upsert("metrics", spec["name"], "metric_index", index_payload)
    upsert("metrics", spec["name"], "metric_detail", detail_payload)

print("Done.")
PY
