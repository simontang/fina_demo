# Hankel Run for Gold Semantic Assets

本文档说明 Hankel Run for Gold demo 在 Metrics Server / datasource 15 上的语义资产建设方式。目标是把已导入的 raw/report 数据升级为 Agent runtime 可查询的 view 和 SQL-free metric meta。

## Scope

- Tenant: `hankel`
- Datasource: `15`
- Schema: `public`
- Raw tables:
  - `hankel_project_opportunity_lines`
  - `hankel_new_order_lines`
  - `hankel_sales_name_mapping`
- Golden report tables:
  - `hankel_report_won_reconciliation_detail`
  - `hankel_report_won_reconciliation_by_sales`
  - `hankel_report_current_leaders`
  - `hankel_report_overall_progress_leaderboard`
  - `hankel_report_overall_qualification_gap`
  - `hankel_report_segment_progress_leaderboard`
  - `hankel_report_segment_qualification_gap`

当前设计只做 PostgreSQL 普通 view，不做 materialized view，也不在 metric meta 中保存复杂 SQL。

## Asset Flow

1. Datasource admin/builder 使用 `GET /api/v1/datasources/15/table-grants` 获取可探查范围。
2. Builder 通过 `POST /api/v1/datasources/15/query` 对 datasource 做只读探查，并根据 `table-grants` 把建模范围收敛到 Hankel 表。
3. DBA 或部署脚本执行 `metrics-server/scripts/hankel/run-for-gold-views.sql`，创建 `hankel_view_` 语义视图。
4. 发布脚本调用 `meta/tables`，把 view 发布为 runtime 可用表，并为每个 view 写入 EXACT grant。
5. 发布脚本调用 `meta/metrics`，把 SQL-free metric detail 和 metric index 发布给 Agent。
6. 普通 Agent 只使用 `GET /api/v1/datasources/15/meta` 和 `POST /api/v1/metrics/query`。
7. `hankel_report_*` 表只作为 golden report 做对账验证，不作为主事实源。

## Competition Parameters

当前 demo 使用一个单行 `hankel_view_run_for_gold_parameters` 集中管理固定参数：

| Parameter | Value |
| --- | --- |
| Competition start date | `2026-07-01` |
| Competition end date | `2026-12-31` |
| Report cutoff date | `2026-08-31` |
| Target year | `2026` |
| Validation threshold | `0.5` |

后续生产化建议把这些参数放入配置表或 materialization job。

## Views

| View | Type | Grain | Purpose |
| --- | --- | --- | --- |
| `hankel_view_run_for_gold_parameters` | parameter view | one parameter set | 集中保存当前可复现 demo 的日期和验证阈值。 |
| `hankel_view_sales_name_mapping` | normalized view | one mapping row | 标准化销售姓名、team、sales type。 |
| `hankel_view_project_opportunity_line` | normalized view | opportunity line | 标准化 Project / Opportunity 行，补充 match-key 质量标记和 New/Won 计算字段。 |
| `hankel_view_new_project_opportunity` | calculation view | `canonical_sales_name + opportunity_id` | 先在 Opportunity 粒度汇总 New Project Y1。 |
| `hankel_view_new_order_line` | normalized view | order line | 标准化 New Order 行，过滤 rejected line，形成可匹配订单金额。 |
| `hankel_view_won_validation_match_key` | calculation view | `canonical_sales_name + sold_to_idh + product_idh` | 计算 Won 到 New Order 的 50% 验证结果。 |
| `hankel_view_validated_won_opportunity` | calculation view | `canonical_sales_name + opportunity_id` | Opportunity 至少一个精确 Match Key 通过时，将其计为 Validated Won。 |
| `hankel_view_run_for_gold_sales_summary` | calculation view | `canonical_sales_name` | 汇总 New Project、Validated Won、Won Y1、coverage、gap。 |
| `hankel_view_run_for_gold_qualification_status` | calculation view | `award_pool + canonical_sales_name` | 保存所有 overall 候选及资格状态。 |
| `hankel_view_run_for_gold_leaderboard` | calculation view | `award_pool + canonical_sales_name` | 只对已入围人员计算 overall score、rank、position。 |
| `hankel_view_run_for_gold_segment_qualification_status` | calculation view | `segment + canonical_sales_name` | 保存所有细分奖候选及资格状态。 |
| `hankel_view_run_for_gold_segment_leaderboard` | calculation view | `segment + canonical_sales_name` | 只对已入围人员计算细分奖 leaderboard。 |
| `hankel_view_run_for_gold_qualification_gap` | calculation view | `scope_type + award_pool + sales_name` | 输出 overall 和 segment 的入围差距。 |
| `hankel_view_run_for_gold_report_reconciliation` | QA view | `check_type + business_key` | 对比 raw-derived view 与 `hankel_report_*` golden report。 |

## Won Validation Logic

`hankel_view_won_validation_match_key` 是验证赢单的核心视图。

匹配粒度：

```text
canonical_sales_name + sold_to_idh + product_idh
```

Project 侧规则：

- `status = 'Won'`
- `close_date` 在 `2026-01-01` 到 `2026-08-31`
- 必须有 sales mapping、sold-to、product，才能进入有效 match key
- 纯数字标识符会移除 Excel 产生的尾部 `.0`，其余标识符保持原样

New Order 侧规则：

- `created_on` 在 `2026-01-01` 到 `2026-12-31`
- rejected line 不计入
- 必须有 sales mapping、sold-to、product，才能进入有效 match key

计算口径：

```text
check_period_won_y1 = y1_value / (13 - close_month) * recognized_months
required_new_order_value = check_period_won_y1 * 0.5
matched_new_order_value = sum(order_value_cny) by match key
coverage = matched_new_order_value / check_period_won_y1
result = Pass when matched_new_order_value >= required_new_order_value
action_gap = greatest(required_new_order_value - matched_new_order_value, 0)
signed_gap = required_new_order_value - matched_new_order_value
counted_won_y1 = raw_won_y1 when result = Pass else 0
```

Validated Won Count 不等于通过的 Match Key 数。系统先构建
`hankel_view_validated_won_opportunity`，同一个 Opportunity 只计一次，只要
其中至少一个产品 Match Key 为 `Pass` 即通过。

排行榜只读取 `is_qualified=true` 的资格状态行。奖池人数 `N` 因而等于实际
入围人数；New Y1 和 Won Y1 的分项排名使用纯金额降序 `RANK.EQ`，其他字段
只用于最终同分排序。

## Published Metrics

Published metrics are SQL-free. Complex logic is already represented in `hankel_view_` columns.

| Metric | Source View | Calculation |
| --- | --- | --- |
| `hankel_new_projects_count` | `public.hankel_view_run_for_gold_sales_summary` | `sum(new_project_count)` |
| `hankel_new_projects_y1` | same | `sum(new_y1)` |
| `hankel_validated_won_count` | same | `sum(validated_won_count)` |
| `hankel_validation_won_y1` | same | `sum(validation_won_y1)` |
| `hankel_required_new_order_value` | same | `sum(required_new_order_value)` |
| `hankel_matched_new_order_value` | same | `sum(matched_new_order_value)` |
| `hankel_order_coverage_rate` | same | ratio: `hankel_matched_new_order_value / hankel_validation_won_y1` |
| `hankel_new_order_gap` | same | `sum(new_order_gap)`，兼容名称，明确表示非负 Action Gap |
| `hankel_new_order_signed_gap` | `public.hankel_view_won_validation_match_key` | `sum(new_order_gap_raw)` |
| `hankel_competition_won_y1` | same | `sum(competition_won_y1)` |
| `hankel_final_score` | `public.hankel_view_run_for_gold_leaderboard` | `avg(score)` |
| `hankel_segment_final_score` | `public.hankel_view_run_for_gold_segment_leaderboard` | `avg(score)` |
| `hankel_match_key_count` | `public.hankel_view_won_validation_match_key` | `count_distinct(match_key)` |

## Runtime Query Examples

Query sales-type level progress:

```http
POST /api/v1/metrics/query
X-Tenant-Id: hankel
Content-Type: application/json

{
  "datasourceId": 15,
  "metrics": [
    "hankel_new_projects_count",
    "hankel_validated_won_count",
    "hankel_competition_won_y1",
    "hankel_order_coverage_rate"
  ],
  "groupBy": ["sales_type"],
  "orderBy": [
    {"field": "hankel_competition_won_y1", "direction": "DESC"}
  ],
  "limit": 20
}
```

Query sales gaps:

```http
POST /api/v1/metrics/query
X-Tenant-Id: hankel
Content-Type: application/json

{
  "datasourceId": 15,
  "metrics": ["hankel_new_order_gap", "hankel_matched_new_order_value"],
  "groupBy": ["canonical_sales_name", "sales_type"],
  "orderBy": [
    {"field": "hankel_new_order_gap", "direction": "DESC"}
  ],
  "limit": 20
}
```

Query leaderboard:

```http
POST /api/v1/metrics/query
X-Tenant-Id: hankel
Content-Type: application/json

{
  "datasourceId": 15,
  "metrics": ["hankel_final_score"],
  "groupBy": ["award_pool", "rank", "leader", "canonical_sales_name", "position"],
  "orderBy": [
    {"field": "hankel_final_score", "direction": "DESC"}
  ],
  "limit": 50
}
```

## Deployment

Create or replace views:

```bash
psql "$HANKEL_PG_URL" -f metrics-server/scripts/hankel/run-for-gold-views.sql
```

Publish table and metric meta:

```bash
METRICS_BASE_URL=https://ada.alphafina.cn/api/metrics/api/v1 \
TENANT_ID=hankel \
DATASOURCE_ID=15 \
metrics-server/scripts/hankel/publish-run-for-gold-meta.sh
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `METRICS_BASE_URL` | `https://ada.alphafina.cn/api/metrics/api/v1` | Metrics Server API base URL. |
| `TENANT_ID` | `hankel` | Tenant header for published meta and grants. |
| `DATASOURCE_ID` | `15` | Hankel datasource id. |
| `METRICS_INSECURE_SSL` | `true` | Allow self-signed or incomplete chain during demo deployment. |

## Verification

View row-count smoke checks:

```sql
select count(*) from hankel_view_project_opportunity_line;
select count(*) from hankel_view_new_order_line;
select count(*) from hankel_view_won_validation_match_key;
select count(*) from hankel_view_validated_won_opportunity;
select count(*) from hankel_view_run_for_gold_sales_summary;
select count(*) from hankel_view_run_for_gold_qualification_status;
select count(*) from hankel_view_run_for_gold_leaderboard;
select count(*) from hankel_view_run_for_gold_report_reconciliation;
```

Golden report QA:

```sql
select status, count(*)
from hankel_view_run_for_gold_report_reconciliation
group by status
order by status;
```

Expected behavior:

- `PASS` rows mean the raw-derived semantic view matches the imported report at that check grain.
- `KEY_MISMATCH`, `ROW_COUNT_MISMATCH`, `AMOUNT_MISMATCH`, and `RANK_MISMATCH` rows are not hidden. They identify where the current raw tables differ from the imported `hankel_report_*` snapshot.
- Earlier exploration found that the report and current raw data are not a perfect same-snapshot pair, so the QA view is intentionally diagnostic rather than forced to zero difference.

## Design Boundary

- Metrics runtime meta describes published business fields and SQL-free calculations.
- Complex transformation, matching, thresholds, score, and report reconciliation live in views.
- Datasource builder/admin may use `POST /datasources/15/query` for governed read-only exploration.
- Business Agent runtime should use `GET /datasources/15/meta` and `POST /metrics/query`.
- No raw physical table inventory is exposed to ordinary tenant Agent runtime.
- Metrics Server v1 treats `tenantId` as a weak runtime context: it first uses current tenant grants, then falls back to active grants on the same datasourceId when no tenant grants exist. This is only safe because Hankel datasource 15 is treated as a customer-scoped datasource.
