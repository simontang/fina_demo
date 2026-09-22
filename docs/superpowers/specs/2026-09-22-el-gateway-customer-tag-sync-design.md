# 网关客户标签同步（tag_definition + customer_tag）设计

- 日期：2026-09-22
- 状态：Draft（设计已定稿，待实现）
- 范围：`el-ai-gateway/`（`src/routes/tasks.ts`、`src/routes/customers.ts`、删除 `src/mock/customerTags.ts`、新增 BO 工具适配器、测试、单文件 `API.md`）+ `elc` store 的两个 BO 对象（`tag_definition` 新建、`customer_tag` 改造）
- 不改动：platform-service（复用已暴露的 `business-objects_*` open-mcp 工具）、任务创建/列表/详情逻辑

## 1. 背景与目标

网关目前用 mock（`src/mock/customerTags.ts`）提供客户标签：`PUT /voice-tagging/:id/tags` 只校验 32-hex `tagId` 是否在 mock 目录里，`GET /customers/:customerId/tags` 读 mock 映射。

目标：改为真实 BO 数据。

- 新增标签主数据对象 `tag_definition`（标签字典）。
- 改造 `customer_tag` 为客户标签明细（`customer_no` + `tag_id`，反范式存标签组/标签名）。
- `PUT` 按"有 id 复用、无 id 新增主数据"处理，并把 id 回写任务标签；随后**对账**该客户 `customer_tag`（按该客户所有任务标签的并集增删，最终一致、无残留）。
- `GET /customers/:customerId/tags` 读 `customer_tag`。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| 标签主数据 | 新建 BO 对象 `tag_definition`（elc store，`deleteMode=hard`） |
| `tag_definition` 列 | `category` / `tag_group` / `tag_name` / `tag_id` / `is_xiaofu` |
| 标签组 = tagKey | `tag_group` 即 tagKey；`tag_id` 是标签值 id（文本 32-hex） |
| 客户标签 | 改造 `customer_tag`：加 `tag_id`，反范式保留 `tag_key`(标签组)/`tag_value`(标签名) |
| `customer_tag` 唯一键 | `(customer_no, tag_id)` |
| `customer_tag` 其他索引 | `tag_id`（按标签反查客户）、`customer_no` |
| 有 `tagId` | 校验 `tag_definition` 存在；不修改主数据；标签组/名以主数据为准（请求里的 `tagValue` 忽略） |
| 无 `tagId` | 先按 `(tag_group="客户画像", tag_name=tagValue)` 查主数据，命中则**复用其 `tag_id`**；否则新增 `tag_definition`：`tag_group="客户画像"`、`tag_name=tagValue`、`tag_id=网关生成32hex`、`category=null`、`is_xiaofu=null` |
| 对账 | `PUT` 后自动重算该客户：`listTasks` 并集 → `customer_tag` 增/删 |
| 汇总行字段 | `source="voice"`、`confidence=null`、`tagged_at`=该标签在各任务中的最新时间（无则 now） |
| 任务标签 | `result` 存 `[{ tagId, tagKey, tagValue }]` |
| 删除 | 对账删除"并集外"的 `customer_tag` 行（对象为 hard，物理删除） |
| 删除 mock | 删除 `src/mock/customerTags.ts` |

## 3. 数据模型（elc store）

### 3.1 新建 `tag_definition`（标签主数据）
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `category` | string | 否 | 类别（如 进阶标签/产品系列/其他产品） |
| `tag_group` | string | 是 | 标签组 = tagKey |
| `tag_name` | string | 是 | 标签名 |
| `tag_id` | string | 是 | 标签值 id（32-hex 文本），唯一索引 `uk_tag_id` |
| `is_xiaofu` | boolean | 否 | 小肤标签 |
| `deleteMode` | — | — | `hard` |

### 3.2 改造 `customer_tag`（客户标签明细，反范式）
| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `customer_no` | string | 是 | 客户编号（= `customerId`） |
| `tag_key` | string | 是 | 标签组（冗余自 `tag_definition`） |
| `tag_value` | string | 是 | 标签名（冗余自 `tag_definition`） |
| `tag_id` | string | 是 | `tag_definition.tag_id` |
| `source` | string | 否 | 来源（voice/manual） |
| `confidence` | decimal | 否 | 置信度 |
| `tagged_at` | datetime | 否 | 打标时间 |
| `deleteMode` | — | — | `hard` |

索引：唯一 `(customer_no, tag_id)`；普通 `tag_id`、`customer_no`。

> 迁移：重建对象（表内为空），旧唯一键 `(customer_no, tag_key)` 移除。

## 4. 网关接口

### 4.1 `PUT /api/v1/voice-tagging/:id/tags`
请求：`{ "tags": [ { "tagId"?: "…", "tagValue": "…" } ] }`（`tags` 必须为数组；`tagValue` 必填文本）

算法：
1. `getTask(id)` → 从 `metadata` 取 `baId`、`customerId`（缺任一 → `400`）。
2. 逐个标签：
   - **有 `tagId`**：`business-objects_get_record("tag_definition", tagId)`；
     - 不存在 → `400 BAD_REQUEST`（`Unknown tagId`）；
     - 取主数据的 `tag_group`/`tag_name`（请求里的 `tagValue` 忽略）。
   - **无 `tagId`**：先 `query_records("tag_definition", filters:[{field:"tag_group",op:"eq",value:"客户画像"},{field:"tag_name",op:"eq",value:tagValue}])`；命中则复用其 `tag_id`，否则 `create_record("tag_definition", { tag_group:"客户画像", tag_name:tagValue, tag_id:<32hex>, category:null, is_xiaofu:null })` → 得到 `tag_id`。
3. `updateResult(id, JSON.stringify(tagsOut))`，`tagsOut = [{ tagId, tagKey, tagValue }]`。
4. **对账**该客户 `customer_tag`：
   - `listTasks({ ownerId, baId, customerId })` → 汇总所有任务 `result` 里的标签并集（按 `tagId` 去重；兼容旧形状 `{tagId,name,dimension}`，`tagValue` 回退到 `name`）。
   - 对并集内每个 `tag_id`：查 `tag_definition` 取 `tag_group`/`tag_name`；`customer_tag` 不存在则 create（`source="voice"`、`confidence=null`、`tagged_at`=该标签所在任务里最新的 `updatedAt`（无则 `createdAt`，再无不则 now）），已存在则不动。
   - 删除该客户 `customer_tag` 中 `source="voice"` 且 `tag_id` 不在并集里的行（`delete_records`，`confirm:true`）；`source="manual"` 的行保留。
5. 返回更新后的任务详情（`tags` 为 `tagsOut`，附最新 activity）。

### 4.2 `GET /api/v1/customers/:customerId/tags`
`business-objects_query_records("customer_tag", filters:[{field:"customer_no",op:"eq",value:customerId}])`，返回：
```json
{ "customerId": "…", "total": N,
  "tags": [ { "tagId":"…", "tagKey":"…", "tagValue":"…", "source":"voice", "confidence":null, "taggedAt":"…" } ] }
```
无标签 → `200` + 空数组（不返回 404）。

### 4.3 任务 `tags` 形状
任务详情/列表的 `tags` 来自任务 `result`，元素为 `{ tagId, tagKey, tagValue }`。

## 5. 网关内部改动

- 新增 `src/upstream/boTools.ts`：基于现有 `McpCaller` 的薄适配器，封装
  `getRecord(objectKey,id)`、`createRecord(objectKey,data)`、`deleteRecords(objectKey,ids)`、`queryRecords(objectKey,filters)`，
  统一解析 `business-objects_*` 返回（`get_record` 返回 `{...record}`；`query_records` 返回 `{rows,total}`；`delete_records` 返回 `{deleted,ids}`）。
- `src/routes/tasks.ts`：`PUT` 用 `boTools` + `tag_definition` 完成上述算法；`TaskRouteDeps` 增加 `boTools`。
- `src/routes/customers.ts`：改读 `customer_tag`；依赖 `boTools`。
- 删除 `src/mock/customerTags.ts` 及其 import。
- `src/server.ts`：装配 `boTools`（复用 `createMcpClient`）。

## 6. 错误处理

- 任务不存在 → `404`（沿用现有 `getTask`）。
- `tags` 非数组 / 元素缺 `tagValue`（且无有效 `tagId`）→ `400 BAD_REQUEST`。
- `tagId` 在 `tag_definition` 查不到 → `400 BAD_REQUEST`。
- BO/MCP 调用失败 → `502 UPSTREAM_ERROR`（沿用 `McpCaller` 的 `MCP_ERROR`；必要时包成 502）。
- 对账为尽力而为：对账失败不回滚已更新的任务标签（最终一致），记录日志并返回任务详情；下次 PUT 会再次对账。

## 7. 测试

- `PUT`：有 `tagId` 命中主数据 → 不调用 create，回写任务 `result`；`tagId` 不存在 → `400`；无 `tagId` → 调用 `tag_definition` create（`tag_group=客户画像`）并回写新 id。
- 对账：并集新增/删除正确（用假 `boTools` 断言 `delete_records` 收到差集 id）。
- `GET /customers/:id/tags`：映射 `customer_tag` 行 → 响应字段；空 → `200` 空数组。
- 移除 mock 后原 `tasksRoute`/`customers` 测试改为基于假 `boTools`。
- `API.md`：更新标签相关章节（字段、示例、错误码）——单文件、无外链、无内部实现措辞。

## 8. 不做（YAGNI）

- `tag_definition` 的增删改管理接口（本次只由网关按需新增）。
- 定时/后台对账任务（对账在 PUT 时同步触发）。
- 标签合并/去重、`tag_definition` 变更后回刷历史 `customer_tag`（可后续按需加）。
- 并发写同一客户的强一致（接受最终一致、last-writer-wins）。
