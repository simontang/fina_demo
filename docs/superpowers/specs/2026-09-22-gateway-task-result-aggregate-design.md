# 任务结果聚合（transcript / tags / like）接口设计

- 日期：2026-09-22
- 状态：Draft（设计已定稿，待实现）
- 范围：`el-ai-gateway/`（任务 `result` 结构、`GET /voice-tagging/:id`、`GET /voice-tagging`、`PUT /voice-tagging/:id/tags`、新增 `PUT /voice-tagging/:id/like`、测试、`API.md`）
- 前置（不在本仓库代码内）：打标 agent 改为按本结构写 `result`；BA 标签主数据导入 `tag_definition`

## 1. 背景与目标

任务结果目前分散：
- 网关 `PUT tags` 把 `result` 覆盖成 `tags` 数组；
- 打标 agent 把**语音原文**写进 activity 的 markdown（不是结构化字段）；
- 点赞（用户反馈）尚无落点。

目标：让任务 `result` 成为**唯一的任务结果聚合**，结构清晰、字段职责分明，接口对外把三个字段定义清楚：

- `transcript`：语音原文（由打标 agent 写入）；
- `tags`：`{tagId, tagKey, tagValue, evidence?}` 列表（agent 首次写入，网关编辑时 read-modify-write）；
- `like`：用户点赞反馈，**两态 `true` / `null`**（默认 `null`）。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| `result` 类型 | JSON 文本，内容为**对象**（非数组） |
| `result` 字段 | `transcript`（string）、`tags`（数组）、`like`（`true`\|`null`） |
| 标签元素 | `{ tagId, tagKey, tagValue, evidence? }`（`tagKey`=标签组，`tagValue`=标签名） |
| `transcript` 写入方 | 打标 agent（首次写 `result` 时一并写入） |
| `tags` 写入方 | agent 首次写；网关 `PUT tags` 用 read-modify-write 替换 |
| `like` 写入方 | 默认 `null`（agent 可写 `null` 或省略）；用户经 `PUT like` 修改 |
| `like` 取值 | 仅 `true` 或 `null`（`false`/其他 → `400`） |
| 编辑保留 | 网关编辑 `tags`/`like` 时保留其余字段（RMW） |
| `evidence` 保留 | 换标签时按 `tagId` 从旧 `result.tags` 继承 `evidence` |
| 旧任务兼容 | `result` 为数组 → 视为只有 `tags`；`transcript=null`、`like=null` |
| 列表是否含原文 | `GET /voice-tagging` 列表**含 `tags`/`like`，不含 `transcript`**（避免列表过大）；原文只在详情返回 |

## 3. `result` 数据结构（interface）

```jsonc
{
  "transcript": "……完整语音原文……",          // string；无则 null
  "tags": [
    {
      "tagId": "9ce355bfacca49c4a9e9322a9317c196",
      "tagKey": "concerns",                    // 标签组
      "tagValue": "抗老/紧致",                 // 标签名
      "evidence": "很喜欢用黑钻光灿面霜"        // string，可选
    }
  ],
  "like": true                                 // true | null
}
```

- 字段缺省：`transcript` 缺省视为 `null`；`tags` 缺省视为 `[]`；`like` 缺省视为 `null`。

## 4. 接口

### 4.1 `GET /voice-tagging/:taskId`（任务详情）

**响应 `200`**：

```json
{
  "taskId": "b3ca978f-3832-4a2c-959b-40fa48c43352",
  "fileId": "2ccf6fef88b64a16b62fe491a8f7a132",
  "status": "completed",
  "createdAt": "2026-09-15T05:18:00Z",
  "title": "Voice tagging: 2ccf6fef88b64a16b62fe491a8f7a132",
  "transcript": "……完整语音原文……",
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "tagKey": "concerns", "tagValue": "抗老/紧致", "evidence": "很喜欢用黑钻光灿面霜" }
  ],
  "like": true
}
```

- `transcript`：string | `null`。
- `tags`：数组；元素 `{tagId, tagKey, tagValue, evidence?}`（读取时兼容旧形状 `{tagId, name, dimension}`，映射 `name→tagValue`、`dimension→tagKey`）。
- `like`：`true` | `null`。
- 任务不存在 → `404 NOT_FOUND`。

### 4.2 `GET /voice-tagging?baId=&customerId=`（任务列表）

每个 `tasks[]` 元素：

```json
{
  "taskId": "…",
  "fileId": "…",
  "status": "completed",
  "createdAt": "2026-09-15T05:18:00Z",
  "tags": [ { "tagId": "…", "tagKey": "…", "tagValue": "…", "evidence": "…" } ],
  "like": true
}
```

- **不含 `transcript`**（列表不返回原文；需要原文用详情接口）。

### 4.3 `PUT /voice-tagging/:taskId/tags`（修改任务标签）

**请求**：

```json
{ "tags": [ { "tagId": "9ce355bfacca49c4a9e9322a9317c196" },
            { "tagValue": "敏感肌" } ] }
```

- 每个元素 `{ "tagId"?, "tagValue"? }`：
  - 带 `tagId`：必须在 `tag_definition` 存在，标签组/名以主数据为准（`tagValue` 忽略）；不存在 → `400`。
  - 不带 `tagId`：按 `tagValue` 新建（标签组 `客户画像`、类别 `自定义标签`，同名复用）；生成 `tagId`。
- **整体覆盖** `result.tags`；`transcript`/`like` 保留；`evidence` 按 `tagId` 从旧值继承。

**响应 `200`**：任务详情（同 §4.1，含 `tags`/`transcript`/`like`）+ `activity`。

- `tags` 非数组 / 元素既无 `tagId` 又无 `tagValue` / `tagId` 不存在 → `400 BAD_REQUEST`；任务不存在 → `404`。

### 4.4 `PUT /voice-tagging/:taskId/like`（点赞 / 取消）

**请求**：

```json
{ "like": true }
```

- `true` = 点赞；`null` = 取消 / 未点赞。
- 仅接受 `true` 或 `null`；其他值（如 `false`、`"x"`）→ `400 BAD_REQUEST`。
- 只更新 `result.like`，`transcript`/`tags` 保留；追加一条 activity（`action: "updated"`）。

**响应 `200`**：任务详情（同 §4.1）+ `activity`。

- `like` 字段缺失 → `400`；任务不存在 → `404`。

## 5. 写入与合并规则

- `result` 是 JSON 文本。读写统一：解析为对象 → 修改目标字段 → 序列化写回。
- **agent 首次写入**：`{ transcript, tags:[{tagId,tagKey,tagValue,evidence}], like: null }`（`like` 可省略）。
- **网关写入**（`PUT tags` / `PUT like`）：
  - 读当前 `result`（兼容数组→`{tags}`）；
  - 改 `tags` 或 `like`；
  - 写回整个对象，未涉及字段原样保留。
- **兼容**：`result` 为空/`null`/数组/非法 JSON 时，按缺省处理（`tags=[]`，`transcript=null`，`like=null`），不报错。
- `GET` 只读，不改 `result`。

## 6. 前置依赖（本次需一并确认/处理）

1. **agent 改配置**：`voice-tagging-agent` 完成后把 `result` 写成 §3 结构（含 `transcript` 与 `{tagId,tagKey,tagValue,evidence}` 标签）。这是**平台侧 agent 提示词**的改动。
2. **主数据导入**：agent 产出的 `tagId` 需存在于 `tag_definition`，否则用户再次 PUT 会 `400`。需把 BA 标签主数据（`类别/标签组/标签/标签id/小肤标签`）导入 `tag_definition`。

## 7. 错误处理

- `like` 非 `true`/`null` → `400 BAD_REQUEST`。
- `tags` 相关校验同 §4.3。
- 上游/B0 故障 → `502 UPSTREAM_ERROR`；任务不存在 → `404`。

## 8. 测试

- 详情：`result` 对象 → 顶层 `transcript`/`tags`/`like` 正确；`result` 为旧数组 → `tags` 正确、`transcript`/`like` 为 `null`；`result` 为空/非法 → 全部缺省。
- 列表：含 `tags`/`like`，不含 `transcript`。
- `PUT tags`：RMW 保留 `transcript`/`like`；`evidence` 按 `tagId` 继承；覆盖未提交的标签。
- `PUT like`：`true` → `like:true`；`null` → `like:null`；`false`/缺字段 → `400`；保留 `tags`/`transcript`。
- 旧形状兼容与错误分支。

## 9. 不做（YAGNI）

- `like` 的第三种状态（取消与未点赞合并为 `null`）。
- 点赞计数/多用户（当前是任务级单值）。
- 列表返回 `transcript`（只在详情）。
- `transcript` 由网关解析 activity（改为 agent 直写 `result`）。
