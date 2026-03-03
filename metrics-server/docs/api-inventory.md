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
| **GET** | `/api/v1/datasources/{dsId}/meta` | **一次性返回** index + 所有指标的 detail（合并 index 与 details） |
| **POST** | `/api/v1/metrics/query` | 执行语义查询或自定义 SQL 查询，返回 results[] |

### 指标定义 CRUD（t_metrics_meta）

| Method | Path | 说明 |
|--------|------|------|
| **GET** | `/api/v1/datasources/{dsId}/metrics` | 列出该 datasource 下所有“已配置 SQL”的指标定义 |
| **GET** | `/api/v1/datasources/{dsId}/metrics/{code}` | 按 code 查询一条指标定义（SQL 等） |
| **POST** | `/api/v1/datasources/{dsId}/metrics` | 为该 datasource 新增一条指标定义 |
| **PUT** | `/api/v1/metrics/{id}` | 按主键 id 更新指标定义 |
| **DELETE** | `/api/v1/metrics/{id}` | 按主键 id 删除指标定义 |

---

## 三、POST /api/v1/metrics/query 请求体说明

**语义查询模式**（按目录指标名 + 时间/分组）：

- `datasourceId`（必填）：datasource 主键
- `metrics`：指标名数组，如 `["net_sales_amt","order_amt_tax_inc"]`
- `timeRange`：`{ "start": "2025-01-01", "end": "2025-12-31" }`
- `groupBy`：维度或时间粒度，如 `["DocDate__month"]`、`["customer_group"]`
- `filters`：可选，筛选条件数组
- `orderBy`：可选，排序
- `limit`：可选，每指标最大行数（默认 1000，最大 10000）
- `debug`：可选 true，返回 executed_sqls

**自定义 SQL 模式**（不走目录）：

- `datasourceId`（必填）
- `customSql`：带 `:paramName` 的 SQL
- `params`：参数对象，如 `{"startDate":"2025-01-01"}`

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
