# Metrics Agent Tools API Guide

本文档说明如何把 Metrics Server 封装成 3 类 Agent 工具：

1. `metrics_datasource_tool`：Datasource 探查工具。
2. `metrics_meta_tool`：Metrics meta 创建和维护工具。
3. `metrics_runtime_tool`：Runtime meta 读取和指标查询工具。

核心原则：

- 可见范围归 `datasourceId` 所有，不归调用方 tenant 所有；tenant header 不是可见范围授权依据。
- Datasource 的 `visibleScopeMode` 决定基础边界：`ALL` 允许连接可访问的业务表，`RESTRICTED` 只允许 active visible scope 规则匹配的表。`RESTRICTED` 且没有 active 规则时不允许访问任何业务表，不能当作 `ALL`。
- Datasource 探查工具先读取 datasource 配置和 `visible-scopes`；Builder/Admin `/datasources/{dsId}/query` 同样受 `ALL/RESTRICTED` 可见范围约束，不是绕过边界的查询入口。
- `meta/tables` 是发布给 runtime 的语义表，不是 datasource 物理表 inventory。
- `meta/metrics` 是发布给 runtime 的指标定义。
- 普通业务 Agent 只使用 `metrics_runtime_tool`。
- Builder/Admin Agent 可以使用 `metrics_datasource_tool` 和 `metrics_meta_tool`。
- 发布、更新或删除 table/metric meta 不得创建、扩展、收缩或删除 visible scopes；发布语义资产与配置可见范围是独立动作。
- `metrics_visible_scope_*` 及旧 `metrics_table_grant_*` 别名属于 Admin 管理工具，不应配置给普通业务 Agent。本工具切片不新增身份认证或角色授权层。

## 1. Common Conventions

Agent 仍使用 `runConfig.tenantId` 解析 Metrics Server 配置，并校验 `selectedDataSources`。这个配置选择过程不变，但不代表 tenant 拥有服务端 visible scope。

旧 runtime 请求可继续携带以下 header 作为兼容上下文，不得把它当作 scope 授权依据：

```http
X-Tenant-Id: hankel
```

Agent 侧调用前还需要校验：

- `datasourceId` 必须属于当前租户配置的 `selectedDataSources`。
- `visible-scopes` 的 list/create/update/delete 不要求、也不发送 `X-Tenant-Id` 或 body `tenantId`；旧工具别名同样如此。Agent 调用 `metricsFetch` 时显式设置 `includeTenantHeader: false`，同时移除配置和请求头中大小写不同的 tenant header，保留原有 API key/Basic auth 配置。
- 不存在 tenant 优先或 datasource fallback 的授权逻辑；可见范围直接由 datasource 配置和规则决定。
- `runConfig.metricsDataSource.serverKey` 仍优先于输入 `serverKey`；读取工具的 datasource 仍先取 `runConfig.metricsDataSource.datasourceId`，再取输入，只有一个 selected datasource 时可省略。管理写入仍要求显式传入 allow-list 内的 `datasourceId`。
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
| `metrics_datasource_tool` | 探查 datasource 内的数据结构和样例数据，用于 Builder/Admin 建模 | 按 datasource `ALL/RESTRICTED` 模式约束 schema inventory 和 `/query`；scope CRUD 是 Admin 管理能力 |
| `metrics_meta_tool` | 创建和维护 runtime 可用的 table meta / metric meta | Builder/Admin 工具，发布语义资产但不能修改 scope |
| `metrics_runtime_tool` | 普通 Agent 读取 runtime meta 并执行指标查询 | 已发布 meta 与 datasource 可见范围的交集 |

三类对象不要混淆：

| Object | Meaning | Used By |
|---|---|---|
| `visible-scopes` | datasource 拥有的基础可见范围规则，配合 `visibleScopeMode` 使用；`table-grants` 为旧兼容名称 | Datasource Tool / Admin |
| `meta/tables` | datasource 已发布给 runtime 的语义表，不扩大基础可见范围 | Metrics Meta Tool / Runtime |
| `meta/metrics` | datasource 已发布给 runtime 的指标，不扩大基础可见范围 | Metrics Meta Tool / Runtime |

## 3. Tool 1: `metrics_datasource_tool`

### 3.1 Purpose

`metrics_datasource_tool` 用于 datasource 探查，帮助 Builder/Admin Agent 理解当前 datasource 的字段结构、样例数据和数据分布。

这个工具不负责发布 meta，也不负责回答业务指标问题。

### 3.2 HTTP APIs

```http
GET  /api/v1/datasources/{dsId}/visible-scopes
GET  /api/v1/datasources/{dsId}/schema/tables
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
| `get_visible_scopes` | `GET /datasources/{dsId}/visible-scopes` |
| `list_tables` | `GET /datasources/{dsId}/schema/tables` |
| `query` | `POST /datasources/{dsId}/query` |
| `test_connection` | `POST /datasources/{dsId}/test` |
| `pool_status` | `GET /datasources/{dsId}/pool` |

### 3.4 Visible Scope Admin Tools

本仓库注册的是以下独立工具；`metrics_datasource_tool` 等名称只是能力分组，不是额外的复合工具。Scope 管理不需要 tenant 输入：

| Tool | Method / Path | Required Input |
|---|---|---|
| `metrics_visible_scope_list` | `GET /datasources/{dsId}/visible-scopes` | `datasourceId` 可按公共规则解析 |
| `metrics_visible_scope_create` | `POST /datasources/{dsId}/visible-scopes` | `datasourceId`, `tablePattern` |
| `metrics_visible_scope_update` | `PUT /datasources/{dsId}/visible-scopes/{scopeId}` | `datasourceId`, `scopeId`, `tablePattern` |
| `metrics_visible_scope_delete` | `DELETE /datasources/{dsId}/visible-scopes/{scopeId}` | `datasourceId`, `scopeId` |

所有工具支持可选 `serverKey`。Create/update 还支持 `schemaName`、`patternType` (`PREFIX` / `EXACT`，默认 `PREFIX`)、`caseSensitive` (默认 `false`) 和 `status` (默认 `1`)。Update 是完整规则更新，不是局部 PATCH。响应中的 `id` 用作后续更新/删除的数字 `scopeId`。

`metrics_table_grant_list/create/update/delete` 保留为兼容别名，调用同一 `visible-scopes` 路径；旧 update/delete 仍接受数字 `grantId`，不要求旧调用方改传 `scopeId`。名称兼容不恢复 tenant 所有权。

先从 `metrics_datasource_list` 的 datasource 配置读取 `visibleScopeMode`，再列出规则。不能只凭规则数量推断 `ALL/RESTRICTED`：

```bash
curl -s \
  'https://ada.alphafina.cn/api/metrics/api/v1/datasources/15/visible-scopes'
```

返回示例：

```json
[
  {
    "id": 1,
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
datasource 15 在 RESTRICTED 模式下允许 public schema 中 hankel_ 前缀的表。
```

工具调用示例：

```json
{
  "datasourceId": 15,
  "scopeId": 1,
  "schemaName": "public",
  "tablePattern": "hankel_",
  "patternType": "PREFIX",
  "caseSensitive": false,
  "status": 1
}
```

以上是 `metrics_visible_scope_update` 输入；使用旧 `metrics_table_grant_update` 时只需把 `scopeId` 换为 `grantId`。Scope CRUD 管理规则，不代替 datasource 配置接口设置 `visibleScopeMode`。

### 3.5 Run Datasource Query

`metrics_datasource_query` 调用 `/query`，用于 Builder/Admin 只读抽样和验证计算口径。它不要求预先发布 table meta，但必须遵守 datasource `ALL/RESTRICTED` 可见范围，不应暴露给普通业务 Agent。

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

结构探查优先使用 `metrics_datasource_table_list` 对应的 `/schema/tables`，结果受可见范围过滤。`RESTRICTED` 下禁止查询 `information_schema` 等系统目录，即使为系统目录配置了规则也不会放行。查询可见表的列信息可以使用已有接口：

```http
GET /api/v1/datasources/15/tables/hankel_distr_sell_in/columns?schemaName=public
```

以下 PostgreSQL metadata SQL 仅适用于 `ALL` 模式：

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

- 先读取 datasource 的 `visibleScopeMode` 和 `visible-scopes`。
- 让探查严格遵守 datasource 可见范围；`RESTRICTED` 无 active 规则时拒绝业务表访问。
- `/query` 是 Builder/Admin 工具能力；普通业务 Agent 不应直接使用。本切片不增加新的身份授权层。
- `RESTRICTED` 不允许返回范围外的表或全库清单；`ALL` 不代表普通业务 Agent 可以使用 Admin inventory 工具。
- 不允许告诉普通 Agent 未授权表是否存在。
- datasource `/query` 及 runtime `customSql` 如果引用范围外的表，必须由服务端拒绝。Agent 的 SQL 只读预检不能代替服务端的表范围校验。
- 建议始终使用带 schema 的表名，例如 `public.hankel_distr_sell_in` 或 `[dbo].[OINV]`。`RESTRICTED` 下没有配置默认 schema 时必须显式限定；SQL Server 的 datasource schema 配置不会改变登录账号的默认 schema，因此受限 SQL Server 查询也必须显式限定。
- SQL 使用受支持的只读语法和函数集合；无法验证的语法、跨数据库名称、表函数及未知函数返回 400。范围外访问返回 403。

不建议给普通租户 Agent 暴露：

```http
GET /api/v1/datasources/{dsId}/schema/tables
```

该接口属于 Builder/Admin 探查能力，必须按 datasource `ALL/RESTRICTED` 模式过滤，不因已发布 meta 或 tenant header 而放宽。

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

`meta/tables` 表示把某张表发布为 datasource runtime 可用的语义表。发布前该物理表必须符合 datasource 可见范围；发布不会修改 scope。

`schemaName` 与已限定的 `tableName` 必须一致。DB table meta 在 runtime index 中保留完整 schema 身份，例如 `"public"."hankel_distr_sell_in"`，避免不同 schema 的同名表被当成同一个发布对象。

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
    }
  }'
```

旧 `accessGrant` 参数仅作为弃用的兼容输入被 Agent 接受并忽略，不发送到发布接口。服务端不得从 `payload.tableName`、`payload.viewName` 或 `objectKey` 自动派生 scope。需要修改可见范围时，由 Admin 单独调用 `metrics_visible_scope_*`；删除 table meta 也不得删除同名 scope。

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
- table meta 引用的物理表必须符合 datasource `ALL/RESTRICTED` 可见范围；不存在 tenant 优先或 fallback 授权路径。
- 创建、更新和删除 table/metric meta 均不能改变 visible scope 规则或模式。
- metric meta 的 `sourceTable` 必须存在对应 table meta。
- metric meta 的 `dimensions` 必须来自 table meta columns。
- derived metric 依赖的指标必须存在。
- derived metric 依赖的指标应来自同一 `sourceTable` 或服务端明确支持的 join/view。
- 删除 table meta 前应检查是否仍有 metric 依赖。

## 5. Tool 3: `metrics_runtime_tool`

### 5.1 Purpose

`metrics_runtime_tool` 是普通业务 Agent 使用的查询工具。

它负责：

- 读取所选 datasource 可用 runtime meta。
- 执行 semantic metric query。
- 可选执行受控 runtime custom SQL。

它不负责：

- 探查 datasource 全量物理表。
- 创建 meta。
- 管理 visible scopes 或修改 datasource 可见范围模式。

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

该接口返回所选 datasource 在基础可见范围内已发布的 runtime meta，例如：

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

`customSql` 是受控 runtime 查询入口，应优先使用 semantic query。使用时必须同时符合已发布 table meta 和 datasource `ALL/RESTRICTED` 可见范围，不能用发布 meta 或 tenant header 绕过 scope。

`metrics_datasource_sql_probe` 保留旧工具名，仍转发到 `POST /metrics/query` 的 `customSql`，不是 Builder/Admin `/datasources/{dsId}/query` 的别名。

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
- semantic query 与 `customSql` 均受 datasource `ALL/RESTRICTED` 可见范围约束。
- `customSql` 必须经过只读校验和 published table 校验；`ALL` 也不跳过 runtime 发布要求。

## 6. End-to-End Flow

完整流程：

```text
1. Admin configures datasource visibleScopeMode and visible scopes
        |
        v
2. metrics_datasource_tool reads datasource mode and visible-scopes
        |
        v
3. metrics_datasource_tool probes authorized tables through read-only query
        |
        v
4. metrics_meta_tool publishes table meta and metric meta without changing scope
        |
        v
5. metrics_runtime_tool reads runtime meta
        |
        v
6. metrics_runtime_tool executes metrics query
```

关键区别：

```text
visibleScopeMode + visible-scopes
= datasource 基础可见范围
= boundary for both modeling and runtime queries

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
visible-scope administration (including legacy table-grant aliases)
```

## 8. Compatibility Notes

- 新管理工具统一调用 `/datasources/{dsId}/visible-scopes`，不携带 tenant 身份。
- 旧 `metrics_table_grant_*` 工具名仍可用；update/delete 的旧 `grantId` 输入仍可用，底层转发到新路径。
- 服务端旧 `/datasources/{dsId}/table-grants` 路径属于兼容接口，其 scope 语义也必须归 datasource 所有，不能恢复 tenant 优先/fallback。
- `selectedDataSources` 和既有服务器/数据源解析逻辑不变。
- Table meta 的 `accessGrant` 不再发送；发布/删除 meta 不再隐式管理 scope。

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

- 新旧 scope 工具都能读取当前 datasource 的 visible scopes，不要求 tenant 输入且不发送 tenant header/body。
- `metrics_visible_scope_update/delete` 使用 `scopeId`；旧别名 update/delete 使用 `grantId`，均指向新 scope 路径。
- Scope 管理请求保留服务器认证配置，并在 fetch 前拒绝 `selectedDataSources` 之外的 datasource。
- Builder/Admin `/query` 可以在 datasource 可见范围内只读探查，不依赖已发布 meta。
- `RESTRICTED` 模式下 datasource `/query` 和 runtime `customSql` 查询范围外的表均被拒绝；无 active 规则不放行。
- DDL/DML 被拒绝。
- 多语句 SQL 被拒绝。
- `RESTRICTED` inventory 不返回范围外的物理表；`ALL` 不绕过只读和 runtime 发布要求。

Metrics Meta Tool：

- 可以发布 table meta。
- 发布/更新 meta 不发送旧 `accessGrant`，不扩展或创建 visible scopes；删除 meta 不移除 scopes。
- 可以发布 aggregate metric。
- 可以发布 derived metric。
- metric meta 可以先保存；其 source table 未发布或不可见时，runtime 查询拒绝执行。
- 查询未发布 dimension 时失败。
- 删除 table meta 保留 metric 定义；对应 metric 的 runtime 查询随后被拒绝。

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

## 10. Local Upgrade and Verification

本地升级会自动执行 `sql/init.sql` 和 `sql/datasource_visible_scope_migration.sql`，需要 master DB 的 DDL 权限。也可由运维先执行这两个脚本，再设置 `metrics.master-schema-init=false`。已有 datasource 有任意历史范围规则时迁移为 `RESTRICTED`，无规则时迁移为 `ALL`；迁移只初始化空模式，重复执行不会覆盖管理员后续选择，也不会重建已删除的 Hankel 规则。新 datasource 默认 `RESTRICTED`。

Visible Scope 和模式修改提交后立即清除本实例对应 datasource 的 runtime meta cache；其他实例通过现有变更轮询检测更新，默认间隔 60 秒。Meta 发布本身继续使用现有轮询失效机制。无需 meta version。

可复用的本地 HTTP 验证脚本是 `metrics-server/scripts/test-visible-scope.mjs`。只允许连接 `127.0.0.1`，使用独立测试数据库，覆盖范围 CRUD、只读探查、meta 发布、多维聚合、缓存收紧/恢复、最后一条规则删除、ALL 模式和旧接口兼容。不要对业务数据库执行该脚本。

```bash
cd metrics-server
METRICS_TEST_PG_URL=jdbc:postgresql://127.0.0.1:55486/scope_smoke \
METRICS_TEST_PG_USER=scope_test ./gradlew test

# 先启动连接到同一独立测试库的本地服务。
METRICS_BASE_URL=http://127.0.0.1:15704 PGPORT=55486 \
PGDATABASE=scope_smoke PGUSER=scope_test node scripts/test-visible-scope.mjs
```

## 11. Hankel Example

Tenant：

```text
hankel
```

Datasource：

```text
15
```

Datasource 模式：`RESTRICTED`。

Datasource visible scope：

```json
{
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
