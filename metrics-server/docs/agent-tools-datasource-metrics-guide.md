# Metrics Agent Tools API Guide

本文档说明如何把 Metrics Server 封装成 3 类 Agent 工具：

1. `metrics_datasource_tool`：Datasource 探查工具。
2. `metrics_meta_tool`：Metrics meta 创建和维护工具。
3. `metrics_runtime_tool`：Runtime meta 读取和指标查询工具。

核心原则：

- 租户 Agent 不能看到 datasource 的全量物理表清单。
- Datasource 探查工具必须先读取 `table-grants`，确认建议探查范围；Builder/Admin 级 `/datasources/{dsId}/query` 是 datasource 级只读查询能力。
- `meta/tables` 是发布给 runtime 的语义表，不是 datasource 物理表 inventory。
- `meta/metrics` 是发布给 runtime 的指标定义。
- 普通业务 Agent 只使用 `metrics_runtime_tool`。
- Builder/Admin Agent 可以使用 `metrics_datasource_tool` 和 `metrics_meta_tool`。
- v1 中 Metrics Server 把 `tenantId` 作为弱上下文：优先使用当前 tenant 的 grants；如果该 tenant 没有 grants，则 fallback 到相同 `datasourceId` 下已有的 active grants。真正的隔离前提是 datasourceId 已按客户/租户隔离。

## 1. Common Conventions

所有 Metrics Server runtime API 都建议携带租户 header：

```http
X-Tenant-Id: hankel
```

Agent 侧调用前还需要校验：

- `datasourceId` 必须属于当前租户配置的 `selectedDataSources`。
- Metrics Server 会优先按 `X-Tenant-Id` 取 grants；如果没有 tenant grants，则按 `datasourceId` fallback 到已有 active grants。
- `serverUrl` 应配置到 Metrics Server 的 `/api/v1` 层。
- SQL 只允许 `SELECT` 或 `WITH`。
- 禁止多语句。
- 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`CREATE` 等写入或 DDL。
- 查询行数通过 `maxRows` 或 `limit` 控制。

线上示例 base URL：

```text
https://ada.alphafina.cn/api/metrics/api/v1
```

Agent runtime config 示例：

```json
{
  "tenantId": "hankel",
  "metricsDataSource": {
    "serverKey": "default",
    "datasourceId": 15
  }
}
```

Metrics Server config 示例：

```json
{
  "type": "semantic",
  "serverUrl": "http://metrics-server:5704/api/v1",
  "selectedDataSources": [15],
  "headers": {},
  "apiKey": null
}
```

## 2. Tool Boundary

建议按以下边界实现 3 类工具。

| Tool | Purpose | Permission Boundary |
|---|---|---|
| `metrics_datasource_tool` | 探查 datasource 内的数据结构和样例数据，用于 Builder/Admin 建模 | 必须先读取 `table-grants` 作为建议探查范围；`/query` 需要 datasource 级权限 |
| `metrics_meta_tool` | 创建和维护 runtime 可用的 table meta / metric meta | Builder/Admin 权限，写入 published semantic assets |
| `metrics_runtime_tool` | 普通 Agent 读取 runtime meta 并执行指标查询 | 只能访问已发布 table meta 和 metric meta |

三类对象不要混淆：

| Object | Meaning | Used By |
|---|---|---|
| `table-grants` | 当前 datasource 建议/允许发布给 runtime 的表范围；读接口支持 tenant 优先、datasource fallback | Datasource Tool / Meta Tool |
| `meta/tables` | 当前租户已发布给 runtime 的语义表 | Metrics Meta Tool / Runtime |
| `meta/metrics` | 当前租户已发布给 runtime 的指标 | Metrics Meta Tool / Runtime |

## 3. Tool 1: `metrics_datasource_tool`

### 3.1 Purpose

`metrics_datasource_tool` 用于 datasource 探查，帮助 Builder/Admin Agent 理解当前 datasource 的字段结构、样例数据和数据分布。

这个工具不负责发布 meta，也不负责回答业务指标问题。

### 3.2 HTTP APIs

```http
GET  /api/v1/datasources/{dsId}/table-grants
POST /api/v1/datasources/{dsId}/query
POST /api/v1/datasources/{dsId}/test
GET  /api/v1/datasources/{dsId}/pool
```

### 3.3 Recommended Tool Input

```json
{
  "datasourceId": 15,
  "action": "query",
  "sql": "select count(*) as row_count from hankel_distr_sell_in",
  "params": {},
  "maxRows": 10,
  "debug": true
}
```

`action` 建议支持：

| Action | API |
|---|---|
| `get_grants` | `GET /datasources/{dsId}/table-grants` |
| `query` | `POST /datasources/{dsId}/query` |
| `test_connection` | `POST /datasources/{dsId}/test` |
| `pool_status` | `GET /datasources/{dsId}/pool` |

### 3.4 Get Table Grants

`table-grants` 是 datasource 的治理边界。工具必须先调用这个接口，确认当前 datasource 建议建模和发布哪些 schema/table pattern。

```bash
curl -s \
  -H 'X-Tenant-Id: hankel' \
  'https://ada.alphafina.cn/api/metrics/api/v1/datasources/15/table-grants'
```

返回示例：

```json
[
  {
    "id": 1,
    "tenantId": "hankel",
    "datasourceId": 15,
    "schemaName": "public",
    "tablePattern": "hankel_",
    "patternType": "PREFIX",
    "caseSensitive": false,
    "status": 1
  }
]
```

含义：

```text
datasource 15 的建议建模范围是 public schema 下 hankel_ 前缀的表。
```

v1 读路径的 tenant 规则：

- 如果 `X-Tenant-Id` 对应的 grants 存在，返回该 tenant 的 grants。
- 如果当前 tenant 没有 grants，返回同一 `datasourceId` 下已有的 active grants。
- 这让 Agent 在 header 缺失或 tenant 不一致时仍能按 datasourceId 工作；前提是 datasourceId 已经按客户隔离。
- grant 的 create/update/delete 仍然按当前 tenant 精确处理，不做 fallback。

### 3.5 Run Datasource Query

`/query` 用于 Builder/Admin 在 datasource 范围内执行只读 SQL，例如查询 metadata、抽样、验证计算口径。它不按 `table-grants` 做 runtime 过滤，所以不要暴露给普通业务 Agent。

```bash
curl -s \
  -H 'X-Tenant-Id: hankel' \
  -H 'Content-Type: application/json' \
  -X POST \
  'https://ada.alphafina.cn/api/metrics/api/v1/datasources/15/query' \
  -d '{
    "sql": "select count(*) as row_count from hankel_distr_sell_in",
    "maxRows": 10
  }'
```

字段探查可以通过数据库 metadata SQL 完成，不需要单独 columns API。例如 PostgreSQL：

```json
{
  "sql": "select column_name, data_type from information_schema.columns where table_schema = :schemaName and table_name = :tableName order by ordinal_position",
  "params": {
    "schemaName": "public",
    "tableName": "hankel_distr_sell_in"
  },
  "maxRows": 200
}
```

样例数据探查：

```json
{
  "sql": "select * from hankel_distr_sell_in",
  "maxRows": 20
}
```

数据分布探查：

```json
{
  "sql": "select sales_team, posting_year, count(*) as row_count from hankel_distr_sell_in group by sales_team, posting_year",
  "maxRows": 100
}
```

### 3.6 Security Rules

`metrics_datasource_tool` 必须执行以下规则：

- 先调用 `table-grants`。
- 让探查围绕建议建模范围展开。
- `/query` 需要 datasource 级 Builder/Admin 权限；普通业务 Agent 不应直接使用。
- 不允许返回 datasource 全量物理表清单。
- 不允许告诉普通 Agent 未授权表是否存在。
- runtime `customSql` 或 `/sql/probe` 如果引用未授权表，返回 `TABLE_NOT_GRANTED` 或同类业务错误。

不建议给普通租户 Agent 暴露：

```http
GET /api/v1/datasources/{dsId}/schema/tables
```

如果该接口保留，应只作为 super-admin 内部接口，或者只返回 grant-filtered tables。

## 4. Tool 2: `metrics_meta_tool`

### 4.1 Purpose

`metrics_meta_tool` 用于创建、更新、删除 Metrics Runtime 使用的语义资产。

它维护两类 meta：

- `meta/tables`：runtime 可用语义表。
- `meta/metrics`：runtime 可用指标。

### 4.2 HTTP APIs

Table meta：

```http
GET    /api/v1/datasources/{dsId}/meta/tables
POST   /api/v1/datasources/{dsId}/meta/tables
GET    /api/v1/datasources/{dsId}/meta/tables/{tableKey}
PUT    /api/v1/datasources/{dsId}/meta/tables/{tableKey}
DELETE /api/v1/datasources/{dsId}/meta/tables/{tableKey}
```

Metric meta：

```http
GET    /api/v1/datasources/{dsId}/meta/metrics
POST   /api/v1/datasources/{dsId}/meta/metrics
GET    /api/v1/datasources/{dsId}/meta/metrics/{metricKey}
PUT    /api/v1/datasources/{dsId}/meta/metrics/{metricKey}
DELETE /api/v1/datasources/{dsId}/meta/metrics/{metricKey}
```

### 4.3 Recommended Tool Input

```json
{
  "datasourceId": 15,
  "action": "create_metric",
  "objectKey": "hankel_sell_in_nes",
  "objectType": "metric_detail",
  "status": 1,
  "payload": {}
}
```

`action` 建议支持：

| Action | API |
|---|---|
| `list_tables` | `GET /datasources/{dsId}/meta/tables` |
| `get_table` | `GET /datasources/{dsId}/meta/tables/{tableKey}` |
| `create_table` | `POST /datasources/{dsId}/meta/tables` |
| `update_table` | `PUT /datasources/{dsId}/meta/tables/{tableKey}` |
| `delete_table` | `DELETE /datasources/{dsId}/meta/tables/{tableKey}` |
| `list_metrics` | `GET /datasources/{dsId}/meta/metrics` |
| `get_metric` | `GET /datasources/{dsId}/meta/metrics/{metricKey}` |
| `create_metric` | `POST /datasources/{dsId}/meta/metrics` |
| `update_metric` | `PUT /datasources/{dsId}/meta/metrics/{metricKey}` |
| `delete_metric` | `DELETE /datasources/{dsId}/meta/metrics/{metricKey}` |

### 4.4 Create Table Meta

`meta/tables` 表示把某张表发布为当前租户 runtime 可用的语义表。

```bash
curl -s \
  -H 'X-Tenant-Id: hankel' \
  -H 'Content-Type: application/json' \
  -X POST \
  'https://ada.alphafina.cn/api/metrics/api/v1/datasources/15/meta/tables' \
  -d '{
    "objectType": "table_view_detail",
    "objectKey": "hankel_distr_sell_in",
    "status": 1,
    "payload": {
      "schemaName": "public",
      "tableName": "hankel_distr_sell_in",
      "displayName": "Hankel Distributor Sell In",
      "description": "Distributor sell-in transaction table for Hankel tenant.",
      "columns": [
        {
          "name": "posting_year",
          "label": "Posting Year",
          "type": "number",
          "role": "dimension"
        },
        {
          "name": "sales_team",
          "label": "Sales Team",
          "type": "string",
          "role": "dimension"
        },
        {
          "name": "nes",
          "label": "Net External Sales",
          "type": "number",
          "role": "measure"
        }
      ]
    },
    "accessGrant": {
      "schemaName": "public",
      "tablePattern": "hankel_distr_sell_in",
      "patternType": "EXACT",
      "caseSensitive": false,
      "status": 1
    }
  }'
```

`accessGrant` 可省略。省略时，服务端可以从 `payload.tableName`、`payload.viewName` 或 `objectKey` 派生 `EXACT` grant。

### 4.5 Create Metric Index

建议每个指标都有一个轻量 index meta，用于指标列表和搜索。

```json
{
  "objectType": "metric_index",
  "objectKey": "hankel_sell_in_nes",
  "status": 1,
  "payload": {
    "name": "hankel_sell_in_nes",
    "displayName": "Sell-in NES",
    "description": "Net external sales amount from distributor sell-in data.",
    "category": "sales",
    "sourceTable": "hankel_distr_sell_in"
  }
}
```

### 4.6 Create Metric Detail

新建 DB-backed metric meta 时，优先使用 SQL-free calculation DSL，不建议直接保存 SQL 表达式。

Aggregate metric：

```json
{
  "objectType": "metric_detail",
  "objectKey": "hankel_sell_in_nes",
  "status": 1,
  "payload": {
    "name": "hankel_sell_in_nes",
    "displayName": "Sell-in NES",
    "description": "Net external sales amount from distributor sell-in rows.",
    "sourceTable": "hankel_distr_sell_in",
    "calculation": {
      "type": "aggregate",
      "aggregation": "sum",
      "measure": "nes"
    },
    "dimensions": [
      "posting_year",
      "posting_month",
      "sales_team",
      "region",
      "product_category"
    ]
  }
}
```

Derived ratio metric：

```json
{
  "objectType": "metric_detail",
  "objectKey": "hankel_sell_in_gross_margin_rate",
  "status": 1,
  "payload": {
    "name": "hankel_sell_in_gross_margin_rate",
    "displayName": "Gross Margin Rate",
    "description": "Gross margin divided by sell-in NES.",
    "sourceTable": "hankel_distr_sell_in",
    "calculation": {
      "type": "derived",
      "operator": "ratio",
      "numerator": "hankel_gross_margin",
      "denominator": "hankel_sell_in_nes"
    },
    "dimensions": [
      "posting_year",
      "sales_team",
      "region"
    ]
  }
}
```

### 4.7 Meta Validation Rules

`metrics_meta_tool` 应校验或依赖服务端校验：

- `objectKey` 在同一 datasource 下唯一。
- table meta 引用的物理表应在 datasource `table-grants` 范围内；读路径会先看当前 tenant grants，没有时按 datasourceId fallback。
- metric meta 的 `sourceTable` 必须存在对应 table meta。
- metric meta 的 `dimensions` 必须来自 table meta columns。
- derived metric 依赖的指标必须存在。
- derived metric 依赖的指标应来自同一 `sourceTable` 或服务端明确支持的 join/view。
- 删除 table meta 前应检查是否仍有 metric 依赖。

## 5. Tool 3: `metrics_runtime_tool`

### 5.1 Purpose

`metrics_runtime_tool` 是普通业务 Agent 使用的查询工具。

它负责：

- 读取当前租户可用 runtime meta。
- 执行 semantic metric query。
- 可选执行受控 runtime custom SQL。

它不负责：

- 探查 datasource 全量物理表。
- 创建 meta。
- 管理 table grants。

### 5.2 HTTP APIs

```http
GET  /api/v1/datasources/{dsId}/meta
POST /api/v1/metrics/query
```

### 5.3 Recommended Tool Input

```json
{
  "datasourceId": 15,
  "action": "query",
  "metrics": ["hankel_sell_in_nes", "hankel_sell_in_gross_margin_rate"],
  "groupBy": ["sales_team", "posting_year"],
  "filters": [
    {
      "dimension": "posting_year",
      "operator": "GTE",
      "values": [2024]
    }
  ],
  "orderBy": [
    {
      "field": "hankel_sell_in_nes",
      "direction": "DESC"
    }
  ],
  "limit": 10,
  "debug": true
}
```

`action` 建议支持：

| Action | API |
|---|---|
| `get_meta` | `GET /datasources/{dsId}/meta` |
| `query` | `POST /metrics/query` |

### 5.4 Get Runtime Meta

```bash
curl -s \
  -H 'X-Tenant-Id: hankel' \
  'https://ada.alphafina.cn/api/metrics/api/v1/datasources/15/meta'
```

该接口返回当前租户可见的 runtime meta，例如：

- published tables
- metric index
- metric details
- dimensions
- filters
- semantic model information

### 5.5 Run Semantic Metric Query

```bash
curl -s \
  -H 'X-Tenant-Id: hankel' \
  -H 'Content-Type: application/json' \
  -X POST \
  'https://ada.alphafina.cn/api/metrics/api/v1/metrics/query' \
  -d '{
    "datasourceId": 15,
    "metrics": [
      "hankel_sell_in_nes",
      "hankel_sell_in_gross_margin_rate"
    ],
    "groupBy": [
      "sales_team",
      "posting_year"
    ],
    "filters": [
      {
        "dimension": "posting_year",
        "operator": "GTE",
        "values": [2024]
      }
    ],
    "orderBy": [
      {
        "field": "hankel_sell_in_nes",
        "direction": "DESC"
      }
    ],
    "limit": 10,
    "debug": true
  }'
```

### 5.6 Runtime Custom SQL

`customSql` 是 runtime escape hatch，应优先使用 semantic query。使用时必须限制在已发布 table meta 范围内。

```json
{
  "datasourceId": 15,
  "customSql": "select sales_team, count(*) as row_count from public.hankel_view_distr_sell_in group by sales_team",
  "limit": 20,
  "debug": true
}
```

如果 SQL 引用了未发布或未授权表，例如 raw 表 `hankel_distr_sell_in`
或系统表 `t_datasource_config`，服务端应拒绝。

### 5.7 Runtime Query Rules

- `metrics` 必须是已发布 metric。
- `groupBy` 必须是 metric detail 允许的 dimension。
- `filters` 必须引用已发布 dimension。
- `orderBy` 必须引用输出字段或已发布 metric。
- semantic query 的 SQL 由 Metrics Server 根据 meta 生成。
- `customSql` 必须经过只读校验和 published table 校验。

## 6. End-to-End Flow

完整流程：

```text
1. Admin configures table grants
        |
        v
2. metrics_datasource_tool reads table-grants
        |
        v
3. metrics_datasource_tool probes authorized tables through read-only query
        |
        v
4. metrics_meta_tool publishes table meta and metric meta
        |
        v
5. metrics_runtime_tool reads runtime meta
        |
        v
6. metrics_runtime_tool executes metrics query
```

关键区别：

```text
table-grants
= datasource 探查范围
= access boundary before modeling

meta/tables
= runtime 语义表
= published asset after modeling

meta/metrics
= runtime 指标
= published semantic metric after modeling
```

## 7. Permission Model

### 7.1 Builder/Admin Agent

可使用：

```text
metrics_datasource_tool
metrics_meta_tool
metrics_runtime_tool
```

适用场景：

- datasource 接入测试。
- 授权范围内的数据探查。
- 字段和样例数据分析。
- table meta 创建。
- metric meta 创建。
- runtime 查询验证。

### 7.2 Business Runtime Agent

只使用：

```text
metrics_runtime_tool
```

适用场景：

- 销售分析。
- 财务分析。
- 运营分析。
- 自然语言问数。
- 指标查询和汇总。

不应暴露：

```text
全库物理表清单
未授权表名
未发布表名
datasource credential
raw table-grant administration
```

## 8. Compatibility Notes

必须保持以下老接口兼容：

```http
GET  /api/v1/datasources/{dsId}/meta
POST /api/v1/metrics/query
```

兼容策略：

- 老的 static meta 继续可用。
- 新的 DB-backed meta 与 static meta 合并。
- legacy metric 仍可兼容 `sql_expression`。
- 新建 DB-backed metric 建议使用 `calculation` DSL。
- 不改变 HANA、PostgreSQL、SQL Server 原有 semantic query builder 行为。
- datasource query 不拼接 `LIMIT` 或 `TOP`，使用 JDBC `setMaxRows` 控制行数。

## 9. Acceptance Tests

Datasource Tool：

- 可以读取当前 datasource 的 effective table grants。
- Builder/Admin `/query` 可以做 datasource 级只读探查。
- runtime `customSql` 查询 grant 范围外的表被拒绝。
- DDL/DML 被拒绝。
- 多语句 SQL 被拒绝。
- 不返回 datasource 全量物理表清单。

Metrics Meta Tool：

- 可以发布 table meta。
- 可以发布 aggregate metric。
- 可以发布 derived metric。
- metric 引用不存在 table meta 时失败。
- metric 引用未发布 dimension 时失败。
- 删除 table meta 时能识别 metric 依赖。

Metrics Runtime Tool：

- 可以读取 `/datasources/{dsId}/meta`。
- 可以执行单指标查询。
- 可以执行多指标、多维度查询。
- 未发布 metric 查询失败。
- 未发布 dimension 查询失败。
- runtime `customSql` 查询已发布表成功。
- runtime `customSql` 查询未发布表失败。

Regression：

- `GET /api/v1/datasources/{dsId}/meta` 响应结构保持兼容。
- `POST /api/v1/metrics/query` semantic 查询保持兼容。
- HANA、PostgreSQL、SQL Server 已有查询测试继续通过。

## 10. Hankel Example

Tenant：

```text
hankel
```

Datasource：

```text
15
```

Datasource grant：

```json
{
  "tenantId": "hankel",
  "datasourceId": 15,
  "schemaName": "public",
  "tablePattern": "hankel_",
  "patternType": "PREFIX",
  "caseSensitive": false,
  "status": 1
}
```

Allowed datasource probe:

```json
{
  "sql": "select count(*) as row_count from hankel_distr_sell_in",
  "maxRows": 10
}
```

Rejected datasource probe:

```json
{
  "sql": "select * from t_datasource_config",
  "maxRows": 10
}
```

Runtime metric query:

```json
{
  "datasourceId": 15,
  "metrics": [
    "hankel_sell_in_nes",
    "hankel_sell_in_gross_margin_rate"
  ],
  "groupBy": [
    "sales_team",
    "posting_year"
  ],
  "filters": [
    {
      "dimension": "posting_year",
      "operator": "GTE",
      "values": [2024]
    }
  ],
  "limit": 10,
  "debug": true
}
```
