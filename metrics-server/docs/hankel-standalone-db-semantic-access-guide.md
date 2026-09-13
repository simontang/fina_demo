# Hankel 独立数据库访问与 Semantic Layer 机制整理

生成日期：2026-09-08
当前环境：tenant `hankel`，Metrics datasource `15`，PostgreSQL schema `public`

本文整理 Hankel 相关的数据库访问机制、Semantic Layer 访问机制，以及可落入独立 Hankel 数据库的数据质量洞察。目标是把 Hankel 从共享演示环境中拆出来时，仍然保留可治理的数据探查、指标查询、知识库和质量报告链路。

## 1. 总体建议

建议把独立 Hankel 数据库分为三类资产：

| 层级 | 放在哪里 | 作用 |
|---|---|---|
| Data Plane | Hankel 独立 PostgreSQL DB | 保存 `hankel_*` raw/report 表和 `hankel_view_*` 语义视图。 |
| Semantic Governance | Hankel DB + Metrics Server master DB | Hankel DB 保存可迁移的语义资产镜像；Metrics Server master DB 保存运行时发布状态。 |
| Knowledge & Quality | Hankel DB + Agent Knowledge Base | Hankel DB 保存结构化质量洞察和文档索引；Agent KB 保存 Markdown 知识文档供回答时检索。 |

当前 v1 不建议把数据库凭证明文放入 Hankel DB。Hankel DB 可以保存 `datasource_key`、`metrics_datasource_id`、schema、授权范围和 endpoint contract；实际密码、token、连接密钥继续放在部署环境变量、Secret Manager 或 Metrics Server 既有加密字段中。

## 2. 当前线上资产快照

线上 Metrics Server 当前状态：

| 项目 | 当前值 |
|---|---|
| Tenant | `hankel` |
| Datasource ID | `15` |
| Datasource 类型 | `cdp_postgres` |
| Schema | `public` |
| Active grants | `19` |
| Runtime tables | `18` |
| Runtime metrics | `24` |

Datasource grant 分为两类：

| 类型 | 例子 | 用途 |
|---|---|---|
| Builder scope | `public.hankel_` `PREFIX` | Admin/Builder Agent 可在该范围内做只读探查。 |
| Runtime table grant | `public.hankel_view_distr_sell_out` `EXACT` | 普通业务 Agent 只能访问已发布的语义视图。 |

## 3. 数据库访问机制

### 3.1 Datasource profile

Metrics Server 通过 `t_datasource_config` 管理外部 datasource。Hankel 当前 datasource 是 `15`，指向 PostgreSQL 数据源。

建议在独立 Hankel DB 中保存一个不含密码的 datasource profile：

```text
tenant_id: hankel
datasource_key: hankel_postgres
metrics_datasource_id: 15
engine: postgresql
default_schema: public
purpose: Hankel ACM analytics datasource
credential_ref: secret reference only
```

这样迁库时可以重新分配 datasource id，但业务文档、quality issue 和 semantic meta 仍然引用稳定的 `datasource_key`。

### 3.2 直连数据库进行分析

直连 Hankel PostgreSQL 适用于 DBA、数据工程师、Builder/Admin Agent 的离线分析、建模验证和质量排查。普通业务 Agent 不应直连数据库，而应通过 Metrics Runtime API。

建议至少区分两个数据库账号：

| 账号类型 | 权限 | 使用场景 |
|---|---|---|
| `hankel_analytics_ro` | 只读 `hankel_*` raw/report 表和 `hankel_view_*` views | 数据分析、质量排查、人工复核。 |
| `hankel_semantic_builder` | 只读 raw/report，并可 `CREATE OR REPLACE VIEW` 于指定 schema | 创建/更新 `hankel_view_*` 语义视图。 |

不要在文档、知识库或日志中保存明文密码。连接信息建议由 Secret Manager、部署环境变量或一次性安全通道提供。

本地或受信网络直连模板：

```bash
export HANKEL_DB_HOST="<pg-host>"
export HANKEL_DB_PORT="5432"
export HANKEL_DB_NAME="<hankel-db>"
export HANKEL_DB_USER="hankel_analytics_ro"
export PGPASSWORD="<read-from-secret-manager>"

psql "host=${HANKEL_DB_HOST} port=${HANKEL_DB_PORT} dbname=${HANKEL_DB_NAME} user=${HANKEL_DB_USER} sslmode=require"
```

如果数据库只允许从部署机或堡垒机访问，使用 SSH tunnel：

```bash
ssh -N -L 15432:<private-pg-host>:5432 deploy@ada.alphafina.cn

export PGPASSWORD="<read-from-secret-manager>"
psql "host=127.0.0.1 port=15432 dbname=<hankel-db> user=hankel_analytics_ro sslmode=require"
```

基础探查 SQL：

```sql
-- 只列出 Hankel 范围内对象，避免暴露其他租户或系统表。
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'hankel\_%' ESCAPE '\'
ORDER BY table_name;

-- 查看 Hankel 对象字段。
SELECT table_name, ordinal_position, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name LIKE 'hankel\_%' ESCAPE '\'
ORDER BY table_name, ordinal_position;

-- raw/report 行数抽检。
SELECT 'hankel_distr_sell_in' AS object_name, COUNT(*) AS row_count FROM public.hankel_distr_sell_in
UNION ALL
SELECT 'hankel_distr_sell_out', COUNT(*) FROM public.hankel_distr_sell_out
UNION ALL
SELECT 'hankel_distr_inventory', COUNT(*) FROM public.hankel_distr_inventory
UNION ALL
SELECT 'hankel_project_opportunity_lines', COUNT(*) FROM public.hankel_project_opportunity_lines
UNION ALL
SELECT 'hankel_new_order_lines', COUNT(*) FROM public.hankel_new_order_lines;
```

质量排查 SQL 示例：

```sql
-- Sell-out 默认指标排除情况。
SELECT
    is_quality_excluded,
    COUNT(*) AS rows,
    SUM(excluded_sell_out_value) AS excluded_value
FROM public.hankel_view_distr_sell_out
GROUP BY is_quality_excluded
ORDER BY is_quality_excluded DESC;

-- 业务指标烟测：按月查询 Sell-out 金额。
SELECT
    year_month,
    SUM(sell_out_value) AS sell_out_value
FROM public.hankel_view_distr_sell_out
WHERE period_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
GROUP BY year_month
ORDER BY sell_out_value DESC NULLS LAST
LIMIT 12;

-- Run for Gold golden report 对账状态。
SELECT status, COUNT(*) AS rows
FROM public.hankel_view_run_for_gold_report_reconciliation
GROUP BY status
ORDER BY status;
```

直连数据库分析的结果可以用于建模和质量报告，但不能自动变成普通 Agent 可查询资产。需要通过 `meta/tables` 和 `meta/metrics` 发布后，才进入 Runtime。

### 3.3 Builder/Admin 访问

Builder/Admin Agent 的访问流程：

1. 调用 `GET /api/v1/datasources/{dsId}/table-grants` 获取可探查范围。
2. 只在 grant 范围内调用 `POST /api/v1/datasources/{dsId}/query`。
3. 用 `/query` 做只读 SQL 探查，包括表结构、列、样例数据、分布、质量检查。
4. 根据探查结果创建或更新 `hankel_view_*` view、table meta、metric meta。

关键边界：

- `/query` 是 Builder/Admin 工具，不是普通业务 Agent 工具。
- `/query` 只允许 `SELECT` / `WITH`，禁止多语句、DDL、DML。
- 对 Hankel 来说，Builder scope 当前是 `public.hankel_` 前缀。
- 不向普通租户 Agent 暴露全库物理表清单。

Metrics Datasource Query 可以作为“不拿数据库密码”的分析方式，适合 Builder/Admin Agent：

```bash
export METRICS_BASE_URL="https://ada.alphafina.cn/api/metrics/api/v1"
export TENANT_ID="hankel"
export DATASOURCE_ID="15"

curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  "${METRICS_BASE_URL}/datasources/${DATASOURCE_ID}/table-grants"
```

通过 Metrics API 探查字段：

```bash
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  -H "Content-Type: application/json" \
  -X POST \
  "${METRICS_BASE_URL}/datasources/${DATASOURCE_ID}/query" \
  -d '{
    "sql": "select table_name, column_name, data_type from information_schema.columns where table_schema = :schemaName and table_name like :tablePrefix order by table_name, ordinal_position",
    "params": {
      "schemaName": "public",
      "tablePrefix": "hankel_%"
    },
    "maxRows": 1000,
    "debug": true
  }'
```

通过 Metrics API 做只读质量排查：

```bash
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  -H "Content-Type: application/json" \
  -X POST \
  "${METRICS_BASE_URL}/datasources/${DATASOURCE_ID}/query" \
  -d '{
    "sql": "select is_quality_excluded, count(*) as rows, sum(excluded_sell_out_value) as excluded_value from public.hankel_view_distr_sell_out group by is_quality_excluded order by is_quality_excluded desc",
    "maxRows": 20,
    "debug": true
  }'
```

### 3.4 Runtime 访问

普通 Agent 的访问流程：

1. 调用 `GET /api/v1/datasources/{dsId}/meta` 获取合并后的 runtime meta。
2. 根据 meta 中的 `metricsDetails`、`tablesDetails`、`supportedDimensions` 选择指标和维度。
3. 调用 `POST /api/v1/metrics/query` 执行 semantic metric query。

Runtime 只允许使用已发布 table/metric meta。即使 datasource 物理库中存在更多 `hankel_*` 表，普通 Agent 也不能绕过 meta 直接查询。

Metrics Runtime API 分析模板：

```bash
export METRICS_BASE_URL="https://ada.alphafina.cn/api/metrics/api/v1"
export TENANT_ID="hankel"
export DATASOURCE_ID="15"

# 1. 获取 runtime meta，确认可用指标和维度。
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  "${METRICS_BASE_URL}/datasources/${DATASOURCE_ID}/meta"
```

查询 Sell-out 月度金额：

```bash
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  -H "Content-Type: application/json" \
  -X POST \
  "${METRICS_BASE_URL}/metrics/query" \
  -d '{
    "datasourceId": 15,
    "metrics": ["hankel_sell_out_value"],
    "groupBy": ["period_date__month"],
    "filters": [
      {
        "dimension": "period_date",
        "operator": "BETWEEN",
        "values": ["2025-01-01", "2025-12-31"]
      }
    ],
    "orderBy": [
      {
        "field": "hankel_sell_out_value",
        "direction": "DESC"
      }
    ],
    "limit": 12,
    "debug": true
  }'
```

查询 Sell-out 数据质量指标：

```bash
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  -H "Content-Type: application/json" \
  -X POST \
  "${METRICS_BASE_URL}/metrics/query" \
  -d '{
    "datasourceId": 15,
    "metrics": [
      "hankel_sell_out_excluded_value",
      "hankel_sell_out_quality_issue_count"
    ],
    "groupBy": ["period_date__month"],
    "orderBy": [
      {
        "field": "hankel_sell_out_excluded_value",
        "direction": "DESC"
      }
    ],
    "limit": 20,
    "debug": true
  }'
```

查询 Run for Gold 覆盖率：

```bash
curl -sS \
  -H "X-Tenant-Id: ${TENANT_ID}" \
  -H "Content-Type: application/json" \
  -X POST \
  "${METRICS_BASE_URL}/metrics/query" \
  -d '{
    "datasourceId": 15,
    "metrics": [
      "hankel_required_new_order_value",
      "hankel_matched_new_order_value",
      "hankel_order_coverage_rate"
    ],
    "groupBy": ["canonical_sales_name", "sales_type"],
    "orderBy": [
      {
        "field": "hankel_order_coverage_rate",
        "direction": "ASC"
      }
    ],
    "limit": 20,
    "debug": true
  }'
```

Runtime 常见错误处理：

| 错误 | 含义 | 处理 |
|---|---|---|
| `Dimension '<name>' is not published in supported_dimensions` | 查询用了未发布维度 | 先读 `/meta`，改用已发布维度，或由 Builder 发布该维度。 |
| `Metric '<name>' not found` | 指标未发布或 datasource 不匹配 | 检查 `metricsDetails` 和 `index.metrics`。 |
| 未授权表错误 | `customSql` 引用了未发布 table meta | 使用已发布 view，或先走 `meta/tables` 发布。 |
| 日期类型错误 | 过滤值格式不符合维度类型 | 日期统一使用 `YYYY-MM-DD`。 |

### 3.5 现有接口边界

| 工具类型 | API | 用途 | 普通业务 Agent 是否可用 |
|---|---|---|---|
| Datasource Tool | `GET /api/v1/datasources/{dsId}/table-grants` | 查看当前租户 datasource 探查边界 | 否，Builder/Admin 用 |
| Datasource Tool | `POST /api/v1/datasources/{dsId}/query` | 在 datasource 内做只读探查 | 否，Builder/Admin 用 |
| Meta Tool | `GET/POST/PUT/DELETE /api/v1/datasources/{dsId}/meta/tables` | 发布 runtime table meta | 否，Builder/Admin 用 |
| Meta Tool | `GET/POST/PUT/DELETE /api/v1/datasources/{dsId}/meta/metrics` | 发布 runtime metric meta | 否，Builder/Admin 用 |
| Runtime Tool | `GET /api/v1/datasources/{dsId}/meta` | 获取 runtime 可查询语义资产 | 是 |
| Runtime Tool | `POST /api/v1/metrics/query` | 执行 semantic metric 查询 | 是 |

## 4. Semantic Layer 机制

Hankel Semantic Layer 分成四层：

| 层 | 对象 | 作用 |
|---|---|---|
| Raw / Report | `hankel_*` raw 表、`hankel_report_*` report 表 | 保留客户原始 Excel 导入结果和 golden report。 |
| Semantic View | `hankel_view_*` | 承载 join、标准化、质量标记、排名、验证、对账等复杂逻辑。 |
| Table Meta | `table_catalog`、`table_view_detail` | 描述哪些 view 可以被 runtime 使用、业务含义、粒度、字段。 |
| Metric Meta | `metric_index`、`metric_detail` | 描述指标名称、口径、source view、SQL-free calculation、可用维度。 |

设计原则：

- 复杂 SQL 不放在 metric meta 中，放在 `hankel_view_*`。
- metric meta 只保存 SQL-free calculation，例如 `sum(measure)`、ratio、公式。
- 知识库解释业务语义，不授予数据库访问权限。
- 数据质量规则要在 view 中显式输出 audit 字段，而不是把异常行从源表删除。

## 5. 当前 Hankel 数据对象

### 5.1 River Distributor Review

原始表：

| 表 | 行数/说明 |
|---|---|
| `hankel_distr_sell_in` | Sell-in，经销商进货/销售流量，当前导入 91,207 行。 |
| `hankel_distr_sell_out` | Sell-out，经销商向终端客户销售分摊行，当前导入 413,917 行。 |
| `hankel_distr_inventory` | Inventory，月末库存快照，当前导入 261,709 行。 |

语义视图：

| View | 粒度 | 说明 |
|---|---|---|
| `hankel_view_distr_sell_in` | 一条有效或被排除的 Sell-in 导入行 | 应用 Sales Team 白名单和 `GMM L6 allocation != Y`，保留 raw/excluded 审计字段。 |
| `hankel_view_distr_sell_out` | 一条客户-产品-月份-sales team 分摊行 | 使用 Territory 分摊字段，排除数量/金额异常进入默认指标。 |
| `hankel_view_distr_inventory_monthly` | 客户-产品-月份-sales team 库存快照 | 保存历史快照，不跨月累计。 |
| `hankel_view_distr_inventory_current` | 客户-产品-sales team 最新库存快照 | 当前库存指标默认使用最新有效月份。 |

### 5.2 Caren Run for Gold

原始与 report 表：

| 表 | 说明 |
|---|---|
| `hankel_project_opportunity_lines` | Project / Opportunity × Product 行。 |
| `hankel_new_order_lines` | New Order / Sales Document × Item 行。 |
| `hankel_sales_name_mapping` | 销售姓名映射维护表。 |
| `hankel_report_*` | Run for Gold golden report，只用于 QA 对账，不作为事实源。 |

语义视图：

| View | 粒度 | 说明 |
|---|---|---|
| `hankel_view_run_for_gold_parameters` | 单行参数 | 当前 demo 固定 `2026-07-01` 到 `2026-08-31`，threshold `0.5`。 |
| `hankel_view_sales_name_mapping` | mapping row | 标准化 Project/New Order sales name 到 canonical sales name。 |
| `hankel_view_project_opportunity_line` | Opportunity × Product | 标准化 ID、日期范围、match key、质量字段。 |
| `hankel_view_new_project_opportunity` | Sales × Opportunity | New Project 在 Opportunity 粒度先聚合。 |
| `hankel_view_new_order_line` | Order × Item | 标准化订单行，排除明确 rejected line。 |
| `hankel_view_won_validation_match_key` | Sales × Sold-to × Product | Won 到 New Order 的 50% 验证核心 view。 |
| `hankel_view_validated_won_opportunity` | Sales × Opportunity | 至少一个 match key pass，则 Opportunity 计入 validated won。 |
| `hankel_view_run_for_gold_sales_summary` | Canonical Sales Name | 汇总 New Project、Validated Won、Coverage、Gap。 |
| `hankel_view_run_for_gold_qualification_status` | Award Pool × Sales | 保存所有候选资格状态。 |
| `hankel_view_run_for_gold_leaderboard` | Award Pool × qualified Sales | 综合奖排行榜。 |
| `hankel_view_run_for_gold_segment_qualification_status` | Segment × Sales | 细分奖候选资格状态。 |
| `hankel_view_run_for_gold_segment_leaderboard` | Segment × qualified Sales | 细分奖排行榜。 |
| `hankel_view_run_for_gold_qualification_gap` | Scope × Pool/Segment × Sales | 入围差距。 |
| `hankel_view_run_for_gold_report_reconciliation` | QA check × business key | Raw-derived view 与 golden report 的差异诊断。 |

## 6. 当前发布指标

### 6.1 Customer-confirmed / 可正式回答

| Metric | 来源 view | 计算 |
|---|---|---|
| `hankel_sell_in_nes` | `public.hankel_view_distr_sell_in` | `sum(nes)` |
| `hankel_sell_in_quantity` | `public.hankel_view_distr_sell_in` | `sum(sell_in_quantity)` |
| `hankel_sell_out_value` | `public.hankel_view_distr_sell_out` | `sum(sell_out_value)` |
| `hankel_sell_out_quantity` | `public.hankel_view_distr_sell_out` | `sum(sell_out_quantity)` |
| `hankel_inventory_quantity` | `public.hankel_view_distr_inventory_current` | `sum(inventory_quantity)` |
| `hankel_required_new_order_value` | `public.hankel_view_run_for_gold_sales_summary` | `sum(required_new_order_value)` |
| `hankel_matched_new_order_value` | `public.hankel_view_run_for_gold_sales_summary` | `sum(matched_new_order_value)` |
| `hankel_order_coverage_rate` | `public.hankel_view_run_for_gold_sales_summary` | `matched_new_order_value / validation_won_y1` |

### 6.2 Pending / Reference / Quality

| Metric | 状态 | 说明 |
|---|---|---|
| `hankel_sell_in_gross_margin` | `pending_business_confirmation` | Gross Margin 正式来源列和过滤规则待确认。 |
| `hankel_sell_in_gross_margin_rate` | `pending_business_confirmation` | 依赖 Gross Margin，使用加权比率。 |
| `hankel_sell_in_contribution` | `pending_business_confirmation` | `15*` contribution 字段业务含义待确认。 |
| `hankel_inventory_value` | `written_spec_reference` | Inventory 默认可解释为金额，但当前 River POC 指标是 quantity。 |
| `hankel_new_projects_count` | `written_spec_reference` | Run for Gold 书面规则指标。 |
| `hankel_new_projects_y1` | `written_spec_reference` | Run for Gold 书面规则指标。 |
| `hankel_validated_won_count` | `written_spec_reference` | 去重 Opportunity，不是 match key count。 |
| `hankel_validation_won_y1` | `written_spec_reference` | Check-period Won Y1。 |
| `hankel_competition_won_y1` | `written_spec_reference` | 排名输入，非当前客户确认 POC。 |
| `hankel_final_score` | `written_spec_reference` | 非可加总，必须按 `canonical_sales_name` 分组。 |
| `hankel_segment_final_score` | `written_spec_reference` | 非可加总，必须按 `segment + canonical_sales_name` 分组。 |
| `hankel_new_order_gap` | `pending_business_confirmation` | Action Gap，非负缺口。 |
| `hankel_new_order_signed_gap` | `pending_business_confirmation` | Signed Gap，负数代表超额覆盖。 |
| `hankel_match_key_count` | `technical_diagnostic` | 匹配键数量，只能用于诊断。 |
| `hankel_sell_out_excluded_value` | `demo_quality` | 被质量规则排除的金额。 |
| `hankel_sell_out_quality_issue_count` | `demo_quality` | 触发质量规则的分摊行数。 |

## 7. 数据质量洞察报告

建议在独立 Hankel DB 中保存两类质量信息：

1. `quality_report`：一次报告的标题、生成时间、数据范围、摘要。
2. `quality_issue`：每个问题的 priority、dataset、evidence、impact、customer action、examples。

当前已整理的数据质量洞察如下。

### 7.1 River 数据集

| 优先级 | 问题 | 证据 | 影响 | 建议动作 |
|---|---|---|---|---|
| P0 | Sell-out 极端数量/金额 | 7 个分摊行、3 笔基础交易触发 demo guardrail；示例数量 `2,147,483,647`，金额 `343,929,169,743.46`。 | 会污染 Sell-out 排名、趋势和金额规模。 | 请客户确认是真实交易、单位错误、系统溢出还是测试数据。 |
| P0 | 产品单位不统一 | 666 个 product code 存在多个非空单位，例如 `CON / PC / PCS / 支 / KG`。 | 数量、Sell-through、库存周转不能直接跨单位汇总。 | 提供产品基础单位和换算表。 |
| P0 | End Customer 身份碰撞 | 规范化后 25 组名称对应多个非空 End Customer Number，影响 5,248 行。 | 按名称聚合会合并不同编号；按编号聚合可能拆分同一客户。 | 确认终端客户唯一键和主数据治理规则。 |
| P1 | Sales Type 跨事实映射缺失 | 尚无 Sell-in/Sell-out/Inventory Team 到统一 Sales Type 的受治理映射。 | 不能正式做跨事实 Sales Type 对比。 | 提供维护表和冲突处理规则。 |
| P1 | 负数交易含义未闭环 | Sell-in 有效 NES 负数 5,974 行，Sell-out 金额负数 709 行，数量负数 740 行。 | 趋势分析需要决定按原月冲减还是回冲业务发生月。 | 确认退货、贷项、冲销和跨月回补规则。 |
| P1 | 数据截至时点 | Sell-in、Sell-out、Inventory 当前最新数据月为 2026-07。 | 查询 2026-08 不能返回 0 或暗示完整。 | 对外回答必须展示数据截至月。 |

### 7.2 Caren 数据集

| 优先级 | 问题 | 证据 | 影响 | 建议动作 |
|---|---|---|---|---|
| P0 | Sales Name Mapping 覆盖不足 | 2026 年有效 New Order 有 7,803 行未映射，占 38.67%；金额 179,054,447.57，占 44.22%。 | 未映射订单不能进入自动 Won 验证，coverage 可能偏低。 | 补齐 Project/New Order sales name 到 canonical name 的维护表。 |
| P1 | Golden Report 与 raw-derived 结果未完全对齐 | 统一竞赛口径后，Won detail 仍有 356 个 key 只存在一侧，119 个双方 key 金额不一致。 | 不应直接判断 raw 错，可能是快照版本或报表过滤差异。 | 获取 golden report 生成版本、数据时间、映射表版本和取整规则。 |
| P1 | Project 关键字段缺失 | 有效业务行中 Sold-to 缺失 163 行；缺 Sold-to 的行不能形成完整 match key。 | 不能自动参与 Won→New Order 匹配。 | 补齐 Sold-to 或明确排除规则。 |
| P1 | Gap 展示口径待固化 | 底层同时支持 Action Gap 与 Signed Gap。 | 报表、告警、Agent 回答如果混用会产生差异。 | 客户确认默认展示口径。 |

### 7.3 已通过的基础质量检查

- Sell-in 日期无法转换：0 行。
- Sell-out Territory 金额缺失或非数字：0 行。
- Inventory 日期无法转换：0 行。
- Inventory Customer/Product/Combined ID/Unit 缺失：0 行。
- New Order 的 Order Number、Created On、Sold-to、Product 缺失：0 行。
- Sales Mapping 内部一对多冲突：0 组。
- Project/New Order 标识符末尾 Excel `.0` 已标准化，当前残留为 0 行。

## 8. 独立 Hankel DB 建议表

建议使用随本文档配套的 DDL：

```text
metrics-server/scripts/hankel/hankel-standalone-governance-ddl.sql
```

核心表：

| Schema | Table | 用途 |
|---|---|---|
| `hankel_governance` | `datasource_profile` | 保存 datasource 非敏感配置和迁移标识。 |
| `hankel_governance` | `datasource_access_grant` | 保存 Builder scope 和 runtime exact grants。 |
| `hankel_governance` | `semantic_table_meta` | 保存 table/view meta 镜像。 |
| `hankel_governance` | `semantic_metric_meta` | 保存 metric meta 镜像。 |
| `hankel_governance` | `runtime_endpoint_contract` | 保存 Agent/Metrics API 调用边界。 |
| `hankel_quality` | `quality_report` | 保存质量报告头。 |
| `hankel_quality` | `quality_issue` | 保存结构化质量问题。 |
| `hankel_kb` | `knowledge_document` | 保存知识库文档索引、状态、标签和正文。 |

## 9. 推荐迁移顺序

1. 在独立 Hankel PostgreSQL DB 中创建 raw/report 表，导入当前 `hankel_*` 和 `hankel_report_*` 数据。
2. 保持初始 schema/table 名不变，即继续使用 `public.hankel_*`，先避免破坏现有 SQL。
3. 执行 view 脚本：
   - `metrics-server/scripts/hankel/distributor-sell-out-view.sql`
   - `metrics-server/scripts/hankel/distributor-sell-in-inventory-views.sql`
   - `metrics-server/scripts/hankel/run-for-gold-views.sql`
4. 执行 `hankel-standalone-governance-ddl.sql`，建立 governance、quality、kb 表并写入当前资产摘要。
5. 在 Metrics Server master DB 中创建新的 Hankel datasource config，指向独立 Hankel DB。
6. 针对新的 datasource id 重跑 meta 发布脚本：
   - `publish-distributor-sell-out-meta.sh`
   - `publish-distributor-sell-in-inventory-meta.sh`
   - `publish-run-for-gold-meta.sh`
7. 验证：
   - `GET /api/v1/datasources/{newDsId}/meta`
   - `POST /api/v1/metrics/query`
   - 数据质量 report reconciliation view。

## 10. Agent 回答要求

Hankel Agent 回答指标问题时至少包含：

```json
{
  "metric": "指标名称",
  "value": "精确结果",
  "timeRange": "查询范围",
  "groupBy": ["实际分组维度"],
  "filters": ["实际筛选条件"],
  "asOf": "数据截至日期或月份",
  "definitionStatus": "customer_confirmed | written_spec_reference | pending_business_confirmation | demo_quality",
  "qualityNotes": [],
  "source": "已发布语义视图/指标"
}
```

禁止行为：

- 不根据字段名猜测业务含义。
- 不使用未发布 raw 表回答普通业务问题。
- 不把 quality metric 当作业务 KPI。
- 不把 Run for Gold 竞赛口径和通用 Project YTD 口径混用。
- 不把 Inventory 多个月份累加成库存。
- 不把 Sell-in、Sell-out、Inventory 三张事实表明细 join 后做跨事实指标。
