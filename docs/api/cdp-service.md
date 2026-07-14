### 1. 分群定义

**基础路径**: `/api/v1/segment-definitions`

#### 1.1 查询分群定义列表

```http
GET /api/v1/segment-definitions
```

**请求头**:


| 头部            | 说明                    |
| ------------- | --------------------- |
| `X-Tenant-Id` | 租户标识（可选，默认 `default`） |


**响应示例**:

```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "id": 1,
      "tenantId": "default",
      "name": "高价值客户",
      "description": "近30天消费超过5000元的客户",
      "datasourceId": 1,
      "querySql": "SELECT customer_id, SUM(amount) AS total FROM orders ...",
      "status": 1,
      "createdAt": "2026-07-13 10:00:00",
      "updatedAt": "2026-07-13 10:00:00"
    }
  ]
}
```



#### 1.2 查询单个分群定义

```http
GET /api/v1/segment-definitions/{id}
```

**路径参数**:


| 参数   | 类型     | 说明      |
| ---- | ------ | ------- |
| `id` | `Long` | 分群定义 ID |




#### 1.3 创建分群定义

```http
POST /api/v1/segment-definitions
Content-Type: application/json
X-Tenant-Id: default
```

**请求体**:


| 字段             | 类型       | 必填  | 说明                                    |
| -------------- | -------- | --- | ------------------------------------- |
| `name`         | `string` | ✅   | 分群名称                                  |
| `description`  | `string` | —   | 分群描述                                  |
| `datasourceId` | `Long`   | ✅   | 关联数据源 ID（对应 `t_datasource_config.id`） |
| `querySql`     | `string` | ✅   | 分群 SQL（仅允许 SELECT / WITH）             |
| `status`       | `int`    | —   | 状态：1-启用，0-禁用（默认 1）                    |


**SQL 安全规则**:

- 必须以 `SELECT` 或 `WITH` 开头
- 禁止包含分号（单条语句）
- 禁止写操作/DDL 关键字：`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `MERGE`, `CALL`, `GRANT`, `REVOKE`, `VACUUM`, `ANALYZE`

**请求示例**:

```json
{
  "name": "高价值客户",
  "description": "近30天消费超过5000元的客户",
  "datasourceId": 1,
  "querySql": "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at > NOW() - INTERVAL '30 days' GROUP BY customer_id HAVING SUM(amount) > 5000",
  "status": 1
}
```

**响应**: 返回创建成功的 `SegmentDefinitionVO` 对象。

#### 1.4 更新分群定义

```http
PUT /api/v1/segment-definitions/{id}
Content-Type: application/json
```

请求体字段同创建接口。

#### 1.5 删除分群定义

```http
DELETE /api/v1/segment-definitions/{id}
```

逻辑删除（`deleted = 1`），关联的分群数据不会级联删除。

**响应**:

```json
{
  "code": 200,
  "message": "success"
}
```



#### 1.6 执行分群处理

```http
POST /api/v1/segment-definitions/{id}/process
Content-Type: application/json
```

这是 CDP 服务最核心的操作：连接该分群关联的 CDP 数据源，执行 `querySql`，将结果集序列化为 JSON 保存到 `t_segment_data` 表。

**路径参数**:


| 参数   | 类型     | 说明      |
| ---- | ------ | ------- |
| `id` | `Long` | 分群定义 ID |


**请求体** (可选):


| 字段       | 类型                    | 说明                                    |
| -------- | --------------------- | ------------------------------------- |
| `params` | `Map<String, Object>` | SQL 命名参数，用于替换 SQL 中的 `:paramName` 占位符 |


**请求示例**:

```json
{
  "params": {
    "min_amount": 5000,
    "start_date": "2026-06-01"
  }
}
```

对应的 SQL 可使用命名参数：

```sql
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE created_at >= :start_date::date
GROUP BY customer_id
HAVING SUM(amount) > :min_amount
```

**响应**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": 1,
    "tenantId": "default",
    "definitionId": 1,
    "runId": "6a2f41a3-8b7c-4d1e-9f5a-3c8d2e1b0f4a",
    "dataJson": "[{\"customer_id\":\"C001\",\"total\":6200.00},{\"customer_id\":\"C002\",\"total\":5800.00}]",
    "rowCount": 2,
    "createdAt": "2026-07-13 10:05:00",
    "updatedAt": "2026-07-13 10:05:00"
  }
}
```

`runId` 为本次执行的唯一标识（UUID），可关联到具体的分群数据快照。

---



### 2. 分群数据

**基础路径**: `/api/v1/segment-data`

#### 2.1 查询分群数据列表（分页）

```http
GET /api/v1/segment-data?definitionId=1&page=1&pageSize=20
```

**查询参数**:


| 参数             | 类型     | 必填  | 说明          |
| -------------- | ------ | --- | ----------- |
| `definitionId` | `Long` | —   | 按分群定义 ID 过滤 |
| `page`         | `int`  | —   | 页码（默认 1）    |
| `pageSize`     | `int`  | —   | 每页条数（默认 20） |


**响应**:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "id": 1,
        "tenantId": "default",
        "definitionId": 1,
        "runId": "6a2f41a3-8b7c-4d1e-9f5a-3c8d2e1b0f4a",
        "dataJson": "[{\"customer_id\":\"C001\",\"total\":6200.00}]",
        "rowCount": 2,
        "createdAt": "2026-07-13 10:05:00",
        "updatedAt": "2026-07-13 10:05:00"
      }
    ],
    "total": 1,
    "page": 1,
    "pageSize": 20
  }
}
```



#### 2.2 查询单条分群数据

```http
GET /api/v1/segment-data/{id}
```



#### 2.3 手动创建分群数据

```http
POST /api/v1/segment-data
Content-Type: application/json
```

**请求体**:


| 字段             | 类型       | 必填  | 说明               |
| -------------- | -------- | --- | ---------------- |
| `definitionId` | `Long`   | ✅   | 所属分群定义 ID        |
| `runId`        | `string` | —   | 执行标识             |
| `dataJson`     | `string` | ✅   | 分群结果 JSON（字符串格式） |




#### 2.4 更新分群数据

```http
PUT /api/v1/segment-data/{id}
Content-Type: application/json
```

请求体字段同创建接口。

#### 2.5 删除分群数据

```http
DELETE /api/v1/segment-data/{id}
```

逻辑删除。
