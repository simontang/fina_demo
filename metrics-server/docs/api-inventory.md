# Metrics Server API 接口清单

**Base URL**（经 Nginx 代理）：`https://demo.alphafina.cn/api/metrics`  
**Base URL**（直连服务）：`http://localhost:5704`

统一响应格式：`{"code":200,"message":"success","data": ...}`，错误时 `code` 非 200。

---

## 一、Data Source 管理（DataSourceController）

| Method | Path | 说明 |
|--------|------|------|
| **GET** | `/api/v1/datasources` | 列出所有 datasource（含未激活） |
| **GET** | `/api/v1/datasources/active` | 仅列出已激活的 datasource |
| **GET** | `/api/v1/datasources/{id}` | 按 id 查询单个 datasource |
| **POST** | `/api/v1/datasources` | 新建 datasource（body 含明文密码，落库前 AES 加密） |
| **PUT** | `/api/v1/datasources/{id}` | 更新 datasource（password 可选，不传则保留原密码） |
| **DELETE** | `/api/v1/datasources/{id}` | 软删除 datasource，并关闭连接池 |

### 状态开关

| Method | Path | 说明 |
|--------|------|------|
| **PATCH** | `/api/v1/datasources/{id}/status` | 设置状态 1=激活 / 0=停用 |
| **POST** | `/api/v1/datasources/{id}/enable` | 激活（等价 status=1） |
| **POST** | `/api/v1/datasources/{id}/disable` | 停用（等价 status=0） |

### 连接测试

| Method | Path | 说明 |
|--------|------|------|
| **POST** | `/api/v1/datasources/test` | 用请求体中的连接信息测试（不落库） |
| **POST** | `/api/v1/datasources/{id}/test` | 用已保存的 datasource 测试连接，返回 `{connected, message, datasourceId}` |

### 连接池

| Method | Path | 说明 |
|--------|------|------|
| **POST** | `/api/v1/datasources/{id}/reload` | 从 DB 重新加载该 datasource 的配置并重建连接池 |
| **GET** | `/api/v1/datasources/{id}/pool` | 查询 HikariCP 连接池状态（totalConnections, activeConnections 等） |

---

## 二、指标发现与查询（MetricsController）

### Agent 发现流程（推荐顺序）

| Method | Path | 说明 |
|--------|------|------|
| **GET** | `/api/v1/datasources/{dsId}/metrics/index` | 指标索引：该 datasource 下所有目录指标及是否已配置 SQL（registered） |
| **GET** | `/api/v1/datasources/{dsId}/metrics/{metricName}/detail` | 单个指标的完整说明（维度、时间、AI 上下文、query_info 等） |
| **GET** | `/api/v1/datasources/{dsId}/meta` | **一次性返回**统一 index、所有指标 detail 和 Table/View detail。静态 classpath meta 与 datasource DB overlay 合并，当前 tenant 的 published table/metric 受 grant 过滤。 |
| **POST** | `/api/v1/metrics/query` | 执行语义查询或自定义 SQL 查询，返回单结果集（data: semanticModel, columns, rows, debug） |

**GET /api/v1/datasources/{dsId}/meta 响应说明**  
`data` 包含三部分：
- `index`：统一索引；`index.metrics` 是指标索引，`index.tables` 是 Table/View 索引。
- `metricsDetails`：每个指标的完整 detail（同 GET .../metrics/{metricName}/detail）。
- `tablesDetails`：Table/View 详情列表，每项含 tableName、docType、mainTable、lineTable、selectSql（可选）、columns（name/label/description/type/example）。

Table/View 数据来自 `meta/view-*.json`（优先）与 `meta/MTC_VW_AI_*.csv`（仅当该表无对应 view JSON 时使用）。

### 指标定义 CRUD（t_metrics_meta）

| Method | Path | 说明 |
|--------|------|------|
| **GET** | `/api/v1/datasources/{dsId}/metrics` | 列出该 datasource 下所有“已配置 SQL”的指标定义 |
| **GET** | `/api/v1/datasources/{dsId}/metrics/{code}` | 按 code 查询一条指标定义（SQL 等） |
| **POST** | `/api/v1/datasources/{dsId}/metrics` | 为该 datasource 新增一条指标定义 |
| **PUT** | `/api/v1/metrics/{id}` | 按主键 id 更新指标定义 |
| **DELETE** | `/api/v1/metrics/{id}` | 按主键 id 删除指标定义 |

---

## 三、POST /api/v1/metrics/query 请求体与响应

**请求体**

语义查询模式：`datasourceId`（必填）、`metrics`（指标名数组）、`groupBy`、`filters`、`orderBy`、`limit`（默认 1000，最大 10000）、`debug`（可选 true，响应中返回 debug 对象）。  

自定义 SQL 模式：`datasourceId`（必填）、`customSql`（可带 `:paramName`）、`params`、`limit`。

**响应 data 结构**（BI Metrics Query API 文档一致）：

- `semanticModel`（string）：命中的语义模型名（如 source.table_view；adhoc 时为 `"adhoc"`）
- `columns`（array）：列元信息，每项 `{ "name": "<列名>", "type": "<数据类型>" }`，与 rows 每行元素一一对应
- `rows`（array）：行数组，每行为与 columns 同序的**值数组**（非对象）
- `debug`（object | null）：仅当请求 `debug: true` 时存在，可含 `sql`、`params` 等

多指标请求时，所有指标须来自同一 source.table_view，接口返回**单结果集**（一条 SQL，columns = 维度 + 各指标名）。

---

## 四、路径汇总（供复制）

```
GET  /api/v1/datasources
GET  /api/v1/datasources/active
GET  /api/v1/datasources/{id}
POST /api/v1/datasources
PUT  /api/v1/datasources/{id}
DELETE /api/v1/datasources/{id}
PATCH  /api/v1/datasources/{id}/status
POST /api/v1/datasources/{id}/enable
POST /api/v1/datasources/{id}/disable
POST /api/v1/datasources/test
POST /api/v1/datasources/{id}/test
POST /api/v1/datasources/{id}/reload
GET  /api/v1/datasources/{id}/pool

GET  /api/v1/datasources/{dsId}/metrics/index
GET  /api/v1/datasources/{dsId}/metrics/{metricName}/detail
GET  /api/v1/datasources/{dsId}/meta
POST /api/v1/metrics/query

GET  /api/v1/datasources/{dsId}/metrics
GET  /api/v1/datasources/{dsId}/metrics/{code}
POST /api/v1/datasources/{dsId}/metrics
PUT  /api/v1/metrics/{id}
DELETE /api/v1/metrics/{id}
```

对外访问时上述路径前加 Base URL，例如：  
`https://demo.alphafina.cn/api/metrics/api/v1/datasources`
