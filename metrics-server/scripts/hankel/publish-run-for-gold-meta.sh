#!/usr/bin/env bash
set -euo pipefail

METRICS_BASE_URL="${METRICS_BASE_URL:-https://ada.alphafina.cn/api/metrics/api/v1}"
TENANT_ID="${TENANT_ID:-hankel}"
DATASOURCE_ID="${DATASOURCE_ID:-15}"
METRICS_INSECURE_SSL="${METRICS_INSECURE_SSL:-true}"

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
    last_error = None
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
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {err_body}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            last_error = exc
            if attempt == 3:
                raise RuntimeError(f"{method} {url} failed after {attempt} attempts: {exc}") from exc
            time.sleep(attempt)
    else:
        raise RuntimeError(f"{method} {url} failed: {last_error}")
    if not raw:
        return {}
    data = json.loads(raw)
    if data.get("code") not in (None, 200):
        raise RuntimeError(f"{method} {url} failed: {json.dumps(data, ensure_ascii=False)}")
    return data


def quote(value):
    return urllib.parse.quote(str(value), safe="")


def existing(kind, key, object_type):
    path = (
        f"/datasources/{datasource_id}/meta/{kind}/{quote(key)}"
        f"?objectType={quote(object_type)}"
    )
    data = request("GET", path).get("data")
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        items = data.get("items") or []
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


def dim(dim_id, field_name=None, label=None, dtype="string"):
    return {
        "dim_id": dim_id,
        "field_name": field_name or dim_id,
        "label": label or dim_id.replace("_", " ").title(),
        "data_type": dtype,
    }


def col(name, dtype="string", desc=None):
    item = {"name": name, "dataType": dtype}
    if desc:
        item["description"] = desc
    return item


SOURCE_TABLES_BY_ASSET = {
    "hankel_view_run_for_gold_parameters": [],
    "hankel_view_sales_name_mapping": ["hankel_sales_name_mapping"],
    "hankel_view_project_opportunity_line": [
        "hankel_project_opportunity_lines",
        "hankel_view_sales_name_mapping",
        "hankel_view_run_for_gold_parameters",
    ],
    "hankel_view_new_project_opportunity": ["hankel_view_project_opportunity_line"],
    "hankel_view_new_order_line": [
        "hankel_new_order_lines",
        "hankel_view_sales_name_mapping",
        "hankel_view_run_for_gold_parameters",
    ],
    "hankel_view_won_validation_match_key": [
        "hankel_view_project_opportunity_line",
        "hankel_view_new_order_line",
    ],
    "hankel_view_validated_won_opportunity": [
        "hankel_view_project_opportunity_line",
        "hankel_view_won_validation_match_key",
    ],
    "hankel_view_run_for_gold_sales_summary": [
        "hankel_view_new_project_opportunity",
        "hankel_view_won_validation_match_key",
        "hankel_view_validated_won_opportunity",
    ],
    "hankel_view_run_for_gold_qualification_status": ["hankel_view_run_for_gold_sales_summary"],
    "hankel_view_run_for_gold_leaderboard": ["hankel_view_run_for_gold_qualification_status"],
    "hankel_view_run_for_gold_segment_qualification_status": [
        "hankel_view_project_opportunity_line",
        "hankel_view_won_validation_match_key",
        "hankel_view_validated_won_opportunity",
    ],
    "hankel_view_run_for_gold_segment_leaderboard": [
        "hankel_view_run_for_gold_segment_qualification_status"
    ],
    "hankel_view_run_for_gold_qualification_gap": [
        "hankel_view_run_for_gold_qualification_status",
        "hankel_view_run_for_gold_segment_qualification_status",
    ],
    "hankel_view_run_for_gold_report_reconciliation": [
        "hankel_view_run_for_gold_sales_summary",
        "hankel_view_run_for_gold_leaderboard",
        "hankel_view_run_for_gold_segment_leaderboard",
        "hankel_report_*",
    ],
}


def table_payload(table_name, display_name, doc_type, short_desc, grain, columns, category,
                  business_status="customer_confirmed_semantic_foundation", business_note=""):
    return {
        "schemaName": "public",
        "tableName": table_name,
        "displayName": display_name,
        "docType": doc_type,
        "docTypeEn": doc_type,
        "shortDesc": short_desc,
        "grain": grain,
        "sourceSystem": "Hankel Run for Gold demo",
        "sourceTables": SOURCE_TABLES_BY_ASSET.get(table_name, []),
        "assetCategory": category,
        "business_status": business_status,
        "business_note": business_note,
        "knowledge_base_refs": [
            "hankel-metrics-kb/03-caren-project-won-validation.md",
            "hankel-metrics-kb/05-analysis-governance-and-open-questions.md",
        ],
        "columns": columns,
    }


def access_grant(table_name):
    return {
        "schemaName": "public",
        "tablePattern": table_name,
        "patternType": "EXACT",
        "caseSensitive": False,
        "status": 1,
    }


def metric_index(metric_name, display_name, short_desc, domain, keywords, source_view,
                 business_status, business_note):
    return {
        "metric_name": metric_name,
        "display_name": display_name,
        "domain": domain,
        "short_desc": short_desc,
        "search_keywords": keywords,
        "source_type": "cdp_postgres",
        "source": {"table_view": source_view},
        "business_status": business_status,
        "business_note": business_note,
        "knowledge_base_refs": ["hankel-metrics-kb/03-caren-project-won-validation.md"],
    }


def metric_detail(metric_name, display_name, description, source_view, calculation,
                  supported_dimensions, fmt="number", data_type="numeric", synonyms=None,
                  business_status="written_spec_reference", business_note=""):
    return {
        "metric_name": metric_name,
        "display_name": display_name,
        "domain": "Hankel Run for Gold",
        "description": description,
        "data_type": data_type,
        "format": fmt,
        "source_type": "cdp_postgres",
        "source": {
            "table_view": source_view,
            "base_filters": [],
        },
        "calculation": calculation,
        "supported_dimensions": supported_dimensions,
        "default_time_context": {
            "time_dimension": "report_cutoff_date",
            "label": "Report cutoff",
            "granularity": "month",
            "window": "YTD Aug 2026",
            "supported_grains": ["month", "year"],
        },
        "business_status": business_status,
        "business_note": business_note,
        "knowledge_base_refs": ["hankel-metrics-kb/03-caren-project-won-validation.md"],
        "ai_agent_context": {
            "polarity": "positive" if "gap" not in metric_name else "negative",
            "synonyms": synonyms or [],
            "human_readable_explanation": description,
        },
    }


sales_summary_dims = [
    dim("canonical_sales_name", label="Canonical Sales Name"),
    dim("sales_name", label="Sales Name"),
    dim("team", label="Team"),
    dim("sales_team", label="Sales Team"),
    dim("sales_type", label="Sales Type"),
]

validation_dims = [
    dim("canonical_sales_name", label="Canonical Sales Name"),
    dim("team", label="Team"),
    dim("sales_team", label="Sales Team"),
    dim("sales_type", label="Sales Type"),
    dim("segment", label="Segment"),
    dim("primary_segment", label="Primary Segment"),
    dim("sold_to_idh", label="Sold-to IDH"),
    dim("product_idh", label="Product IDH"),
    dim("result", label="Validation Result"),
]

leaderboard_dims = [
    dim("award_pool", label="Award Pool"),
    dim("rank", label="Rank", dtype="number"),
    dim("position", label="Award Position"),
    dim("leader", label="Leader"),
    dim("canonical_sales_name", label="Canonical Sales Name"),
    dim("team", label="Team"),
    dim("sales_type", label="Sales Type"),
]

segment_leaderboard_dims = [
    dim("segment", label="Segment"),
    dim("award_pool", label="Award Pool"),
    dim("rank", label="Rank", dtype="number"),
    dim("position", label="Award Position"),
    dim("leader", label="Leader"),
    dim("canonical_sales_name", label="Canonical Sales Name"),
    dim("team", label="Team"),
    dim("sales_type", label="Sales Type"),
]

tables = [
    table_payload(
        "hankel_view_run_for_gold_parameters",
        "Hankel Run for Gold Parameters",
        "Run for Gold Parameters",
        "Single-row parameter source for the current reproducible YTD Aug 2026 demo run.",
        "one parameter set",
        [
            col("competition_start_date", "date"), col("competition_end_date", "date"),
            col("report_cutoff_date", "date"), col("target_year_start", "date"),
            col("target_year_end", "date"), col("target_year", "number"),
            col("validation_threshold", "number"),
        ],
        "parameter view",
        "demo_fixed_parameters",
        "Values are fixed for the current demo and must be replaced by run-scoped parameters for production.",
    ),
    table_payload(
        "hankel_view_sales_name_mapping",
        "Hankel Sales Name Mapping",
        "Run for Gold Sales Mapping",
        "Normalized mapping from raw project and new-order sales names to canonical sales names, sales type, and team.",
        "one raw mapping row",
        [
            col("canonical_sales_name"), col("project_sales_name_key"), col("new_order_sales_name_key"),
            col("sales_team"), col("sales_type"), col("source_file"), col("source_sheet"),
        ],
        "normalized view",
    ),
    table_payload(
        "hankel_view_project_opportunity_line",
        "Hankel Project Opportunity Line",
        "Run for Gold Project Opportunity",
        "Normalized opportunity-line facts with sales mapping, match-key quality flags, New Project flags, and Won Y1 recognition.",
        "one project opportunity product line",
        [
            col("canonical_sales_name"), col("team"), col("sales_type"), col("segment"),
            col("opportunity_id"), col("sold_to_idh"), col("product_idh"), col("status"),
            col("creation_date", "date"), col("close_date", "date"), col("y1_value", "number"),
            col("new_project_y1_value", "number"), col("check_period_won_y1", "number"),
            col("has_sales_mapping", "boolean"), col("is_valid_match_key", "boolean"),
            col("quality_issues"),
        ],
        "normalized view",
    ),
    table_payload(
        "hankel_view_new_project_opportunity",
        "Hankel New Project Opportunity",
        "Run for Gold New Project",
        "New Projects aggregated to canonical salesperson and Opportunity ID before higher-level totals.",
        "canonical_sales_name + opportunity_id",
        [
            col("canonical_sales_name"), col("opportunity_id"), col("team"),
            col("sales_team"), col("sales_type"), col("primary_segment"),
            col("segments"), col("new_project_y1", "number"),
            col("project_line_count", "number"), col("creation_date", "date"),
            col("report_cutoff_date", "date"), col("quality_issues"),
        ],
        "calculation view",
        "written_spec_reference",
        "New Project ranking metrics are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_new_order_line",
        "Hankel New Order Line",
        "Run for Gold New Order",
        "Normalized new-order line facts with canonical sales mapping, match-key quality flags, and rejection filtering.",
        "one new order line",
        [
            col("canonical_sales_name"), col("team"), col("sales_type"), col("segment"),
            col("order_number"), col("sales_document_item"), col("sold_to_idh"), col("product_idh"),
            col("created_on", "date"), col("order_quantity", "number"), col("order_value_cny", "number"),
            col("eligible_order_value_cny", "number"), col("is_eligible_new_order", "boolean"),
            col("quality_issues"),
        ],
        "normalized view",
    ),
    table_payload(
        "hankel_view_won_validation_match_key",
        "Hankel Won Validation Match Key",
        "Run for Gold Won Validation",
        "Won project to New Order validation at canonical sales + sold-to + product grain.",
        "canonical_sales_name + sold_to_idh + product_idh",
        [
            col("match_key"), col("canonical_sales_name"), col("team"), col("sales_type"),
            col("report_cutoff_date", "date"),
            col("segment"), col("primary_segment"), col("sold_to_idh"), col("product_idh"),
            col("opportunity_ids"), col("raw_won_y1", "number"), col("check_period_won_y1", "number"),
            col("required_new_order_value", "number"), col("matched_new_order_value", "number"),
            col("coverage", "number"), col("result"), col("counted_won_y1", "number"),
            col("new_order_gap", "number"), col("new_order_gap_raw", "number"),
            col("status_action"),
        ],
        "calculation view",
        "written_spec_reference",
        "Qualification thresholds are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_validated_won_opportunity",
        "Hankel Validated Won Opportunity",
        "Run for Gold Validated Won",
        "Opportunity-level validation result; an Opportunity passes when at least one exact product Match Key passes.",
        "canonical_sales_name + opportunity_id",
        [
            col("canonical_sales_name"), col("opportunity_id"), col("team"),
            col("sales_team"), col("sales_type"), col("primary_segment"),
            col("segments"), col("match_key_count", "number"), col("match_keys"),
            col("has_passed_match_key", "boolean"), col("result"),
            col("validated_opportunity_id"), col("raw_won_y1", "number"),
            col("report_cutoff_date", "date"), col("quality_issues"),
        ],
        "calculation view",
        "written_spec_reference",
        "Validated Won Count is defined in the written competition specification, not the current customer-confirmed POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_sales_summary",
        "Hankel Run for Gold Sales Summary",
        "Run for Gold Sales Summary",
        "Sales-level summary of New Project, validated Won, coverage, held Won, and New Order gap.",
        "canonical_sales_name",
        [
            col("canonical_sales_name"), col("sales_name"), col("team"), col("sales_type"),
            col("report_cutoff_date", "date"),
            col("new_project_count", "number"), col("new_y1", "number"),
            col("won_lines", "number"), col("validated_won_count", "number"),
            col("raw_won_y1", "number"), col("validation_won_y1", "number"),
            col("required_new_order_value", "number"), col("matched_new_order_value", "number"),
            col("competition_won_y1", "number"), col("held_won_y1", "number"),
            col("new_order_gap", "number"), col("order_coverage_rate", "number"),
        ],
        "calculation view",
    ),
    table_payload(
        "hankel_view_run_for_gold_leaderboard",
        "Hankel Run for Gold Leaderboard",
        "Run for Gold Overall Leaderboard",
        "Overall New and Experienced leaderboard with score, rank, award pool, position, and qualification fields.",
        "award_pool + canonical_sales_name",
        [
            col("award_pool"), col("rank", "number"), col("team"), col("leader"),
            col("canonical_sales_name"), col("sales_type"), col("report_cutoff_date", "date"),
            col("new_project_count", "number"),
            col("validated_won_count", "number"), col("new_y1", "number"), col("won_y1", "number"),
            col("score", "number"), col("position"), col("is_qualified", "boolean"),
        ],
        "calculation view",
        "written_spec_reference",
        "Ranking and awards are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_qualification_status",
        "Hankel Run for Gold Qualification Status",
        "Run for Gold Qualification",
        "All overall-pool candidates with qualification thresholds and status; only qualified rows enter the leaderboard.",
        "award_pool + canonical_sales_name",
        [
            col("award_pool"), col("team"), col("leader"), col("canonical_sales_name"),
            col("sales_type"), col("new_project_count", "number"),
            col("validated_won_count", "number"), col("new_y1", "number"),
            col("won_y1", "number"), col("new_project_required", "number"),
            col("validated_won_required", "number"), col("is_qualified", "boolean"),
            col("report_cutoff_date", "date"),
        ],
        "calculation view",
        "written_spec_reference",
        "Qualification and ranking are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_segment_leaderboard",
        "Hankel Run for Gold Segment Leaderboard",
        "Run for Gold Segment Leaderboard",
        "Key-segment leaderboard for Emotor, Fluid, and Medical award pools.",
        "segment + canonical_sales_name",
        [
            col("segment"), col("award_pool"), col("rank", "number"), col("team"), col("leader"),
            col("canonical_sales_name"), col("sales_type"), col("report_cutoff_date", "date"),
            col("new_project_count", "number"),
            col("validated_won_count", "number"), col("new_y1", "number"), col("won_y1", "number"),
            col("score", "number"), col("position"), col("is_qualified", "boolean"),
        ],
        "calculation view",
        "written_spec_reference",
        "Segment ranking and awards are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_segment_qualification_status",
        "Hankel Run for Gold Segment Qualification Status",
        "Run for Gold Segment Qualification",
        "All segment-pool candidates with qualification thresholds and status; only qualified rows enter segment leaderboards.",
        "segment + canonical_sales_name",
        [
            col("segment"), col("award_pool"), col("team"), col("leader"),
            col("canonical_sales_name"), col("sales_type"),
            col("new_project_count", "number"), col("validated_won_count", "number"),
            col("new_y1", "number"), col("won_y1", "number"),
            col("new_project_required", "number"), col("validated_won_required", "number"),
            col("is_qualified", "boolean"), col("report_cutoff_date", "date"),
        ],
        "calculation view",
        "written_spec_reference",
        "Segment qualification is outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_qualification_gap",
        "Hankel Run for Gold Qualification Gap",
        "Run for Gold Qualification Gap",
        "Overall and segment qualification gaps for sales who are short of New Project or validated Won thresholds.",
        "scope_type + award_pool + sales_name",
        [
            col("scope_type"), col("award_pool"), col("segment"), col("team"), col("sales_name"),
            col("sales_type"), col("new_project_current", "number"), col("new_project_required", "number"),
            col("new_project_gap", "number"), col("validated_won_current", "number"),
            col("validated_won_required", "number"), col("validated_won_gap", "number"),
            col("status"),
        ],
        "calculation view",
        "written_spec_reference",
        "Qualification thresholds are outside the customer-confirmed current POC.",
    ),
    table_payload(
        "hankel_view_run_for_gold_report_reconciliation",
        "Hankel Run for Gold Report Reconciliation",
        "Run for Gold QA Reconciliation",
        "Golden-report QA view comparing raw-derived semantic views with imported hankel_report_* tables.",
        "check_type + business_key",
        [
            col("check_type"), col("business_key"), col("calc_row_count", "number"),
            col("report_row_count", "number"), col("calc_amount", "number"),
            col("report_amount", "number"), col("amount_diff", "number"),
            col("status"), col("detail"),
        ],
        "QA view",
        "demo_quality",
        "Golden-report comparison is a QA asset and is not a business metric source.",
    ),
]

metric_specs = [
    (
        "hankel_new_projects_count",
        "Hankel New Projects Count",
        "Number of new projects created in the H2 competition window up to the report cutoff.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "new_project_count"},
        sales_summary_dims,
        "number",
        ["new project count", "new projects", "新项目数"],
    ),
    (
        "hankel_new_projects_y1",
        "Hankel New Projects Y1",
        "Total Y1 value from new projects created in the H2 competition window up to the report cutoff.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "new_y1"},
        sales_summary_dims,
        "currency",
        ["new project y1", "新项目Y1", "pipeline y1"],
    ),
    (
        "hankel_validated_won_count",
        "Hankel Validated Won Count",
        "Number of distinct Won Opportunity IDs with at least one exact product Match Key passing New Order validation.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "validated_won_count"},
        sales_summary_dims,
        "number",
        ["validated won", "pass count", "验证赢单数"],
    ),
    (
        "hankel_validation_won_y1",
        "Hankel Validation Won Y1",
        "Check-period Won Y1 amount used as the validation denominator.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "validation_won_y1"},
        sales_summary_dims,
        "currency",
        ["check won y1", "validation denominator", "验证期赢单Y1"],
    ),
    (
        "hankel_required_new_order_value",
        "Hankel Required New Order Value",
        "Required New Order amount for Won validation, calculated at 50% of check-period Won Y1.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "required_new_order_value"},
        sales_summary_dims,
        "currency",
        ["required new order", "50 percent threshold", "所需订单金额"],
    ),
    (
        "hankel_matched_new_order_value",
        "Hankel Matched New Order Value",
        "New Order amount matched to Won project match keys.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "matched_new_order_value"},
        sales_summary_dims,
        "currency",
        ["matched new order", "matched order value", "匹配订单金额"],
    ),
    (
        "hankel_order_coverage_rate",
        "Hankel Order Coverage Rate",
        "Uncapped matched New Order amount divided by validation Won Y1; values above 100% mean orders exceed the validation amount.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "derived", "operator": "ratio", "numerator": "hankel_matched_new_order_value", "denominator": "hankel_validation_won_y1"},
        sales_summary_dims,
        "percent",
        ["coverage", "order coverage", "订单覆盖率"],
    ),
    (
        "hankel_new_order_gap",
        "Hankel New Order Action Gap",
        "Non-negative remaining New Order amount needed for Below 50% Won validation match keys.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "new_order_gap"},
        sales_summary_dims,
        "currency",
        ["action gap", "new order gap", "还差多少订单", "订单行动缺口"],
    ),
    (
        "hankel_new_order_signed_gap",
        "Hankel New Order Signed Gap",
        "Signed required-minus-matched New Order amount; negative values represent over-coverage.",
        "public.hankel_view_won_validation_match_key",
        {"type": "aggregate", "aggregation": "sum", "measure": "new_order_gap_raw"},
        validation_dims,
        "currency",
        ["signed gap", "over or under coverage", "超额或不足", "订单有符号差额"],
    ),
    (
        "hankel_competition_won_y1",
        "Hankel Competition Won Y1",
        "Won Y1 counted into the competition after New Order validation.",
        "public.hankel_view_run_for_gold_sales_summary",
        {"type": "aggregate", "aggregation": "sum", "measure": "competition_won_y1"},
        sales_summary_dims,
        "currency",
        ["counted won y1", "competition won y1", "计入比赛赢单Y1"],
    ),
    (
        "hankel_final_score",
        "Hankel Final Score",
        "Final score at salesperson grain. Queries must group by canonical_sales_name; this metric is not additive across salespeople.",
        "public.hankel_view_run_for_gold_leaderboard",
        {"type": "aggregate", "aggregation": "avg", "measure": "score"},
        leaderboard_dims,
        "number",
        ["score", "leaderboard score", "综合分"],
    ),
    (
        "hankel_segment_final_score",
        "Hankel Segment Final Score",
        "Final score at segment and salesperson grain. Queries must group by segment and canonical_sales_name; this metric is not additive.",
        "public.hankel_view_run_for_gold_segment_leaderboard",
        {"type": "aggregate", "aggregation": "avg", "measure": "score"},
        segment_leaderboard_dims,
        "number",
        ["segment score", "segment leaderboard score", "细分奖分数"],
    ),
    (
        "hankel_match_key_count",
        "Hankel Won Match Key Count",
        "Number of Won validation match keys by validation result, customer, product, segment, or sales.",
        "public.hankel_view_won_validation_match_key",
        {"type": "aggregate", "aggregation": "count_distinct", "measure": "match_key"},
        validation_dims,
        "number",
        ["match keys", "won validation rows", "匹配键数量"],
    ),
]

metric_governance = {
    "hankel_new_projects_count": (
        "written_spec_reference",
        "Competition metric from the written specification; outside the customer-confirmed current POC.",
    ),
    "hankel_new_projects_y1": (
        "written_spec_reference",
        "Competition metric from the written specification; Opportunity-level SUM is applied before salesperson totals.",
    ),
    "hankel_validated_won_count": (
        "written_spec_reference",
        "Counts distinct Opportunity IDs, but the metric is outside the customer-confirmed current POC.",
    ),
    "hankel_validation_won_y1": (
        "pending_business_confirmation",
        "The formula is confirmed; the Won candidate start date still requires business confirmation.",
    ),
    "hankel_required_new_order_value": (
        "customer_confirmed",
        "Uses the exact 50% threshold; rounding is display-only.",
    ),
    "hankel_matched_new_order_value": (
        "customer_confirmed",
        "Uses exact Sales Name + Sold-to IDH + Product IDH Match Keys and target-calendar-year New Orders.",
    ),
    "hankel_order_coverage_rate": (
        "customer_confirmed",
        "Uncapped Matched New Order divided by Check-period Won Y1.",
    ),
    "hankel_new_order_gap": (
        "pending_business_confirmation",
        "Compatibility metric now explicitly represents non-negative Action Gap.",
    ),
    "hankel_new_order_signed_gap": (
        "pending_business_confirmation",
        "Explicit Signed Gap is published separately until the customer selects one external default.",
    ),
    "hankel_competition_won_y1": (
        "written_spec_reference",
        "Ranking input from the written specification; outside the customer-confirmed current POC.",
    ),
    "hankel_final_score": (
        "written_spec_reference",
        "Overall ranking is outside the customer-confirmed current POC.",
    ),
    "hankel_segment_final_score": (
        "written_spec_reference",
        "Segment ranking is outside the customer-confirmed current POC.",
    ),
    "hankel_match_key_count": (
        "technical_diagnostic",
        "Diagnostic Match Key count; do not use as Validated Won Opportunity count.",
    ),
}

print(f"Publishing Hankel Run for Gold meta to {base_url}, tenant={tenant_id}, datasource={datasource_id}")

for payload in tables:
    table_name = payload["tableName"]
    upsert("tables", table_name, "table_catalog", payload, access_grant(table_name))
    upsert("tables", table_name, "table_view_detail", payload, access_grant(table_name))

for spec in metric_specs:
    metric_name, display_name, description, source_view, calculation, dims, fmt, synonyms = spec
    business_status, business_note = metric_governance[metric_name]
    index_payload = metric_index(
        metric_name,
        display_name,
        description,
        "Hankel Run for Gold",
        synonyms + ["hankel", "run for gold"],
        source_view,
        business_status,
        business_note,
    )
    detail_payload = metric_detail(
        metric_name,
        display_name,
        description,
        source_view,
        calculation,
        dims,
        fmt=fmt,
        synonyms=synonyms,
        business_status=business_status,
        business_note=business_note,
    )
    if metric_name == "hankel_final_score":
        detail_payload["query_constraints"] = {
            "required_group_by": ["canonical_sales_name"],
            "non_additive": True,
        }
    elif metric_name == "hankel_segment_final_score":
        detail_payload["query_constraints"] = {
            "required_group_by": ["segment", "canonical_sales_name"],
            "non_additive": True,
        }
    upsert("metrics", metric_name, "metric_index", index_payload)
    upsert("metrics", metric_name, "metric_detail", detail_payload)

print("Done.")
PY
