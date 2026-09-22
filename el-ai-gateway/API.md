# el-ai-gateway API 文档（客户语音打标 + 文件 / 任务 / 标签管理）

雅思兰黛 AI 项目 · 业务 API 网关：**客户语音转写打标** 及配套的 **文件、任务、客户标签管理接口**。

- **Base URL（线上）**：`https://ada.alphafina.cn/api/el-ai-gateway`
- **协议**：HTTPS，JSON / multipart
- **版本**：v1

---

## 适用对象

本文面向**对接开发同学**（业务应用 / 前端 / 集成方）：说明如何调用网关接口、每个业务/界面需要先调什么后调什么、每步拿到什么、字段怎么衔接。内部实现细节不在此展开。

## 业务场景

本服务提供三类能力：

**1. 语音文件打标（核心）**
把一段**客户语音**（导购/客服与客户的沟通录音）自动处理成**结构化客户画像标签**：

- 上传语音文件，拿到 `uuid`；
- 触发处理：**语音转文本** → 按客户画像**多维度打标签**（肤质、诉求、感兴趣产品、购买意向、服务机会、自定义标签…），每条标签都带原文 **evidence**；
- 业务可**查询任务状态与结果**（转写全文、打标结果），并可**修正任务标签**。

**2. 业务 / 界面管理接口**
支撑"客户详情"等界面的读取与管理：

- **文件管理**：上传、按 BA + 客户查询文件、获取音频播放链接；
- **任务管理**：发起打标、按 BA + 客户列任务、任务详情、时间线、修正标签；
- **客户标签**：查询客户画像标签（由内部系统更新，外部只读）。

> 当前转写为 mock（结果里 `mock:true`），接口契约与真实链路一致。

## 文档结构

- **第一部分 · 场景与界面（怎么用）**：从 [§1](#1-快速开始鉴权与通用约定) 起，按界面说明调用顺序。
- **第二部分 · 接口参考（查字段）**：从 [§7 文件](#7-文件) 起，按业务域给出每个接口的参数、响应与错误。
- **附录**：从 [§10 错误码](#10-错误码) 起。

> 建议阅读顺序：先看第一部分里对应的界面，再回第二部分查字段。

---

# 第一部分 · 场景与界面（怎么用）

## 1. 快速开始（鉴权与通用约定）

### 1.1 鉴权

所有业务接口都需要在请求头带平台签发的 API Key：

```
Authorization: Bearer <API_KEY>
```

- 缺失或错误 → `401` `{"code":"UNAUTHORIZED","message":"Missing or invalid API key"}`
- Key 映射到一个租户（tenant），用于数据隔离与任务归属。当前测试租户为 `estee_lauder`。
- API Key 由平台**单独下发**，**不在本文档中**；调用时按上面的请求头携带。
- 测试期间：**直接使用密钥访问**（在请求头携带 `Authorization: Bearer <API_KEY>`）。

### 1.3 跨域（CORS）

网关默认开启 CORS，**允许任意来源**（`Access-Control-Allow-Origin: *`），前端可直接联调。生产如需收紧，由后端配置来源白名单，前端无需改动。

> 说明：允许 `*` 来源时，浏览器**不会**携带 Cookie 凭证；本 API 使用 `Authorization` 请求头鉴权，不受影响。

### 1.2 通用约定

- 响应均为 JSON。错误统一信封：
  ```json
  { "code": "UNAUTHORIZED", "message": "Missing or invalid API key" }
  ```
  上游错误会带 `upstream`：
  ```json
  { "code": "UPSTREAM_ERROR", "message": "...", "upstream": { "status": 502, "code": "..." } }
  ```
- **路径拼接**：下文所有接口路径都**相对 Base URL**，最终 URL = `Base + 路径`（Base 末尾无斜杠）。例：`https://ada.alphafina.cn/api/el-ai-gateway/files`。
- **时间字段**：均为 ISO 8601（如 `2026-09-15T06:13:00Z`）；不同接口精度可能是秒或微秒，解析请容错。
- **发起是异步的**：调用立即返回，系统在后台处理并写入任务时间线。前端可**每 3–5 秒轮询一次** [查询任务状态](#83-查询任务状态)，直到 `status` 进入**终态**（`completed` / `failed` / `cancelled`）后再停止。
- 当前转写为 **mock**（处理结果里 `mock: true`）。
- 结果读取方式：转写与打标结果**全部通过查询接口获取**（无 webhook / 无回调）。

## 2. 核心概念：两层标签（客户标签 vs 任务标签）

网关里的"标签"有**两个层级**，接口不通用，对接时务必区分：

| 层级 | 归属对象 | 语义 | 接口 |
|---|---|---|---|
| **客户标签** | 客户（`customerId`） | 该客户当前的业务标签（由系统内部更新），即"客户画像" | `GET /customers/:customerId/tags`（**仅查询**） |
| **任务标签** | 打标任务（`taskId`） | **某一次**语音打标算出的标签 | 读 `GET /voice-tagging/:taskId`；写 `PUT /voice-tagging/:taskId/tags`（**整体覆盖，非追加**） |

- **任务标签**是"某条语音的结论"；**客户标签**是"客户维度的汇总画像"。两者不要混用。
- **客户标签由系统内部更新**，外部（本网关对接方）**只能查询、不能写入**——这是**设计如此**，不是待补接口：
  - 要"补充 / 修正**某次打标**结果" → 改**任务标签**，用 `PUT /voice-tagging/:taskId/tags`；
  - 客户维度的标签更新由**内部系统**完成，外部不提供写接口。
- `PUT /voice-tagging/:taskId/tags` 是**覆盖式**：传入的数组会**替换该任务原有的全部标签**，不是追加。若要在原基础上加，请先 `GET /voice-tagging/:taskId` 读出 `tags`，合并后再整体 PUT。

> 修改标签是任务上**唯一的写操作**：会覆盖标签结果并自动在任务时间线追加一条记录。

## 3. 场景：客户详情（主界面）

**界面结构**：顾客选择器 / 会员卡 / 标签信息 / 后台标签 / 语音手记 FAB。

**区域 → 接口**：

| 区域 | 数据 / 动作 | 接口 | 层级 |
|---|---|---|---|
| 会员卡（偏干肤质 · 关注修护） | 肤质/诉求等画像标签 | [`GET /customers/:customerId/tags`](#91-查询客户业务标签) | 客户 |
| 标签信息（现有标签 / 自定义标签） | 客户标签列表 | [`GET /customers/:customerId/tags`](#91-查询客户业务标签) | 客户 |
| 后台标签（`#参与会员节`）+ 会员编号 / 臻钻 | 只读，来自**原客户系统** | 非本网关 | 客户 |
| 顾客选择器（林女士 ▾） | 切换客户 | 客户基础资料来自原客户系统；选中后拿到 `customerId`，再调上面的标签接口 | 客户 |
| 语音手记 FAB | 进入录音 / 任务 | 见 [§6 语音手记 / 任务详情](#6-场景语音手记--任务详情) | 任务 |

**调用顺序**：

1. 进入页面 → `GET /customers/:customerId/tags` 渲染会员卡画像与标签信息；
2. 切换顾客 → 重新取得 `customerId` 后重复第 1 步；
3. 会员编号 / 臻钻 / 后台标签来自**原客户系统**，网关不提供。

## 4. 场景：全部记录

展示该 BA 在某客户名下的语音任务时间线（标签 & 修改记录）。

| 动作 | 接口 |
|---|---|
| 列出该 BA + 客户名下的语音任务 | [`GET /voice-tagging?baId=&customerId=`](#82-查询任务列表) |
| 展开某任务的转写 / 打标 / 修改记录时间线 | [`GET /voice-tagging/:taskId/activities`](#84-查询任务时间线) |

**调用顺序**：列表页 `GET /voice-tagging?baId=&customerId=` 拿到 `tasks[]`（含 `taskId`）→ 点开某条任务时用其 `taskId` 调 `GET /voice-tagging/:taskId/activities`。

## 5. 场景：补充客户标签

| 动作 | 接口 | 说明 |
|---|---|---|
| 回显客户现有标签 | [`GET /customers/:customerId/tags`](#91-查询客户业务标签) | **客户级、只读** |
| 补充 / 更新客户标签 | 无（内部系统更新） | 客户标签由**系统内部更新**，外部**只能查询** |
| 若是修正"某条语音"的标签 | `GET /voice-tagging/:taskId` 读取 → 合并 → [`PUT /voice-tagging/:taskId/tags`](#85-修改任务标签) | **任务级、覆盖式**，不是客户级追加 |

> ⚠️ 详见 [§2 核心概念](#2-核心概念两层标签客户标签-vs-任务标签)：**客户标签只读**；可写的是**任务标签**，且为**覆盖**。

> 前端注意：**"补充 / 更新客户标签"当前没有写接口**（由内部系统同步）。该入口应做**只读展示或隐藏**，不要等待网关联调写入接口。

## 6. 场景：语音手记 / 任务详情

上传录音 → 查看转写与打标结果、修正标签。

| 步骤 | 接口 |
|---|---|
| 录音上传 | [`POST /files?baId=&customerId=&fileCategory=raw&usage=voice-tagging`](#71-上传文件) → 得 `uuid` |
| 触发转写 + 打标 | [`POST /voice-tagging`](#81-发起打标任务) `{uuid, baId, customerId}` → 得 `taskId` |
| 手记列表 / 音频播放 | [`GET /voice-tagging?baId=&customerId=`](#82-查询任务列表)、[`GET /files/:uuid/url`](#73-获取文件播放下载链接) |
| 查看转写与打标结果 | [`GET /voice-tagging/:taskId`](#83-查询任务状态) |
| 查看时间线 | [`GET /voice-tagging/:taskId/activities`](#84-查询任务时间线) |
| 修正**本次打标**标签（覆盖） | [`PUT /voice-tagging/:taskId/tags`](#85-修改任务标签) |

**时序**：上传 → 发起（立即返回 `taskId`）→ 后台转写 + 打标 → 轮询状态可看到 `tags` 与 activity → 修正标签再写入任务时间线。

---

# 第二部分 · 接口参考（查字段）

## 7. 文件

### 7.1 上传文件

上传语音文件，返回文件 `uuid`（后续发起任务用）。

```
POST /files?path=<dir>&fileName=<name>
Content-Type: multipart/form-data
```

| 参数 | 位置 | 必填 | 说明 |
|---|---|---|---|
| `file` | form-data | 是 | 语音文件 |
| `path` | query | 否 | 逻辑目录，如 `voice-tagging` |
| `fileName` | query | 否 | 显示文件名；缺省用上传文件名 |
| `baId` | query | 是 | 业务员（BA）id（并入 `meta.baId`） |
| `customerId` | query | 是 | 关联客户 id（并入 `meta.customerId`） |
| `fileCategory` | query | 否 | 文件分类，如 `raw` |
| `usage` | query | 否 | 用途，如 `voice-tagging` |
| `meta` | query | 否 | 自定义元数据，**URL 编码的 JSON 对象**，如 `{"src":"wms"}` |

- `meta` 会与 `baId`/`customerId` 合并（后两者覆盖同名键），随文件一起保存，可在文件 `GET` 元数据里取回。
- **`baId`、`customerId` 必填**；缺任一个 → `400 BAD_REQUEST`。
- `meta` 非法 JSON 或非对象 → `400 BAD_REQUEST`。
- **前端一般只传 `baId` / `customerId` 即可**，无需自己拼 `meta`；仅在需要附加业务字段（门店、来源等）时才传，且要 `encodeURIComponent(JSON.stringify({...}))`。
- 响应里的 `meta` 是**JSON 字符串**（不是对象），如需读取请自行 `JSON.parse`。
- 文件大小上限 **50MB**，超出 → `413 PAYLOAD_TOO_LARGE`。
- 语音格式建议 `audio/wav` / `audio/mpeg` / `audio/mp4`（服务端不强制校验 MIME）。

示例：

```bash
curl -X POST "https://ada.alphafina.cn/api/el-ai-gateway/files?path=voice-tagging\
&baId=u_1001&customerId=cus_8899&fileCategory=raw&usage=voice-tagging\
&meta=%7B%22store%22%3A%22XA001%22%7D" \
  -H "Authorization: Bearer <API_KEY>" \
  -F "file=@clip.wav;type=audio/wav"
```

**响应 `200`**（文件回执）：

```json
{
  "uuid": "471c20082b524316accc1b23cba8a4de",
  "fullPath": "voice-tagging/domain.wav",
  "path": "voice-tagging",
  "filename": "domain.wav",
  "version": 1,
  "sha256": "…",
  "md5": "…",
  "size": 34,
  "mime": "audio/wav",
  "fileCategory": "raw",
  "usage": "voice-tagging",
  "meta": "{\"store\":\"XA001\",\"baId\":\"u_1001\",\"customerId\":\"cus_8899\"}",
  "status": "active",
  "createdBy": "api",
  "createdAt": "2026-09-15T06:13:00.000000",
  "deduplicated": false
}
```

> **衔接**：本接口返回的 `uuid` 是下一步（[发起打标任务](#81-发起打标任务)）的入参。

### 7.2 查询文件

按业务员与客户查询已上传的文件（**两个参数都必填**）。

```
GET /files?baId=<id>&customerId=<id>&path=&q=&recursive=&page=&size=
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `baId` | 是 | 业务员 id |
| `customerId` | 是 | 客户 id |
| `path` | 否 | 目录范围 |
| `q` | 否 | 文件名子串 |
| `recursive` | 否 | 是否含子目录（默认 false） |
| `page` / `size` | 否 | 分页（`page` 1-based；`size` 默认 20，最大 1000） |

**响应 `200`**：

```json
{
  "path": "voice-tagging",
  "recursive": false,
  "query": null,
  "directories": [],
  "files": [
    {
      "uuid": "471c20082b524316accc1b23cba8a4de",
      "fullPath": "voice-tagging/domain.wav",
      "filename": "domain.wav",
      "size": 34,
      "mime": "audio/wav",
      "meta": "{\"baId\":\"u_1001\",\"customerId\":\"cus_8899\"}",
      "status": "active",
      "createdAt": "2026-09-15T06:13:00.000000"
    }
  ],
  "page": 1,
  "size": 20,
  "total": 1,
  "totalPages": 1
}
```

- 过滤条件是 `baId` 与 `customerId` 的**同时精确匹配**。
- 缺 `baId` 或 `customerId` → `400 BAD_REQUEST`。

### 7.3 获取文件播放/下载链接

获取某个文件的限时预签名 URL，**可直接用于 `<audio>` 播放**（云存储直链，不经过网关）。

```
GET /files/:uuid/url?ttlSeconds=
```

| 参数 | 位置 | 必填 | 说明 |
|---|---|---|---|
| `uuid` | path | 是 | 文件 uuid（32 hex），即任务里的 `fileId` |
| `ttlSeconds` | query | 否 | 链接有效期（秒），缺省由服务端决定 |

**响应 `200`**：

```json
{
  "uuid": "471c20082b524316accc1b23cba8a4de",
  "url": "https://finademo.tos-s3-cn-beijing.volces.com/estee_lauder/471c2008…?X-Amz-Signature=…",
  "kind": "presigned",
  "expiresInSeconds": 600
}
```

播放示例：

```html
<audio controls src="https://…（响应里的 url）"></audio>
```

- `kind`：`presigned`（云存储直链）或 `direct`（服务转发）。
- `uuid` 非 32 位 hex → `400 BAD_REQUEST`。

## 8. 语音打标任务

### 8.1 发起打标任务

用上一步的 `uuid` 创建一个打标任务；系统在后台执行**语音转写 + 客户画像打标签**。

```
POST /voice-tagging
Content-Type: application/json
```

```json
{
  "uuid": "471c20082b524316accc1b23cba8a4de",
  "baId": "ba_001",
  "customerId": "cus_8899",
  "title": "可选，任务名；缺省 Voice tagging: <uuid>",
  "description": "可选",
  "assistantId": "可选，覆盖服务端默认配置"
}
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `uuid` | 是 | 上一步上传返回的文件 uuid（缺省可用服务端配置的 `VOICE_TAGGING_FILE_UUID`） |
| `baId` | 是 | 业务员 id；写入任务元数据，供 [任务列表](#82-查询任务列表) 过滤 |
| `customerId` | 是 | 客户 id；写入任务元数据，供 [任务列表](#82-查询任务列表) 过滤 |
| `title` / `description` | 否 | 任务名 / 描述 |
| `assistantId` | 否 | 覆盖服务端默认配置 |

**响应 `200`**：

```json
{
  "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
  "status": "in_progress",
  "file": {
    "uuid": "471c20082b524316accc1b23cba8a4de",
    "url": "https://finademo.tos-s3-cn-beijing.volces.com/estee_lauder/471c2008…?X-Amz-Signature=…"
  },
  "agent": { "dispatched": true }
}
```

- `file.url` 为该文件的限时预签名下载地址。
- `agent.dispatched=true` 表示任务已受理、后台处理中（派发失败不影响本响应）。

> **衔接**：前置是上一步的 `uuid`。返回的 `taskId` 供 [查询状态](#83-查询任务状态) / [修改标签](#85-修改任务标签) 使用。

### 8.2 查询任务列表

查询某业务员在某客户下的任务列表，含**文件 id、任务 id、状态、打出的标签**。

```
GET /voice-tagging?baId=<id>&customerId=<id>
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `baId` | 是 | 业务员 id |
| `customerId` | 是 | 客户 id |

**响应 `200`**：

```json
{
  "baId": "ba_001",
  "customerId": "cus_8899",
  "total": 2,
  "tasks": [
    {
      "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
      "fileId": "471c20082b524316accc1b23cba8a4de",
      "status": "completed",
      "createdAt": "2026-09-15T06:13:00Z",
      "tags": [
        {
          "tagId": "9ce355bfacca49c4a9e9322a9317c196",
          "name": "抗老/紧致",
          "dimension": "concerns"
        }
      ]
    }
  ]
}
```

- 缺 `baId` 或 `customerId` → `400 BAD_REQUEST`。
- `tasks[].status` 枚举同 [§8.3 查询任务状态](#83-查询任务状态)。
- 说明：按任务的 `baId` + `customerId` **元数据精确过滤**（由 [发起打标任务](#81-发起打标任务) 创建时写入）；无匹配时返回空列表（`total:0`）。

### 8.3 查询任务状态

```
GET /voice-tagging/:taskId
```

**响应 `200`**：

```json
{
  "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
  "fileId": "471c20082b524316accc1b23cba8a4de",
  "status": "completed",
  "createdAt": "2026-09-15T06:13:00Z",
  "title": "Voice tagging: 471c20082b524316accc1b23cba8a4de",
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "name": "抗老/紧致", "dimension": "concerns" }
  ]
}
```

- `status`：`pending | in_progress | review | failed | interrupted | completed | cancelled`
- `fileId`：本次任务关联的文件 uuid（用于播放 / 追问）。
- `tags`：该任务已生成的标签（`tagId` 32 位 hex / `name` / `dimension`）
- 任务不存在 → `404 NOT_FOUND`

> **衔接**：用发起接口返回的 `taskId` 查询；`activities` 会随转写/打标与修改标签而增长。

### 8.4 查询任务时间线

```
GET /voice-tagging/:taskId/activities
```

```json
{ "taskId": "…", "total": 1, "activities": [ { "id": "…", "action": "updated", "markdown": "", "createdAt": "…" } ] }
```

- `activities[].markdown`：转写全文、打标结果等正文（如有）。
- `activities[].createdAt`：记录时间（ISO 8601）。
- 任务不存在 → `404 NOT_FOUND`。

### 8.5 修改任务标签

整体**覆盖**某**任务**的标签集合，并记录一条 activity。

> ⚠️ 这是**任务级、覆盖式**接口，只影响该 `taskId` 这一条打标任务的标签；
> 它**不会**修改客户画像，也**不是**在客户维度"追加"标签（客户标签见 [§9](#9-客户标签)，由系统内部更新，外部仅查询）。

```
PUT /voice-tagging/:taskId/tags
Content-Type: application/json
```

```json
{ "tags": ["9ce355bfacca49c4a9e9322a9317c196", "4a1b2c3d5e6f47089a0b1c2d3e4f5061"] }
```

> `tags` 也可写成 `[{ "tagId": "…" }, …]`；元素必须是目录中已存在的 32-hex tagId。

**响应 `200`**（更新后的任务 + 本次记录的 activity）：

```json
{
  "taskId": "b3ca978f-3832-4a2c-959b-40fa48c43352",
  "fileId": "2ccf6fef88b64a16b62fe491a8f7a132",
  "status": "in_progress",
  "createdAt": "2026-09-15T05:18:00Z",
  "title": "Voice tagging: 2ccf6fef88b64a16b62fe491a8f7a132",
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "name": "抗老/紧致", "dimension": "concerns" },
    { "tagId": "4a1b2c3d5e6f47089a0b1c2d3e4f5061", "name": "黑钻光灿面霜", "dimension": "interested_products" }
  ],
  "activity": {
    "id": "…",
    "action": "updated",
    "markdown": "",
    "createdAt": "2026-09-15T07:20:00Z"
  }
}
```

- 未知 `tagId` / `tags` 非数组 / tagId 非 32-hex → `400 BAD_REQUEST`
- 任务不存在 → `404 NOT_FOUND`
- 说明：该接口会**整体替换**本任务已生成的标签；同时自动在该任务时间线上追加一条记录（`action: updated`）。

## 9. 客户标签

### 9.1 查询客户业务标签

查询某客户的全部业务标签（**标签名称 + 标签 uuid**）。

> 这是**客户级、仅查询**接口：返回该客户当前的业务标签汇总（客户画像）。**客户标签由系统内部更新，外部不能通过本接口写入/追加**；它**不是**某条语音任务的结果（见 [§2 核心概念](#2-核心概念两层标签客户标签-vs-任务标签)）。

```
GET /customers/:customerId/tags
```

**响应 `200`**：

```json
{
  "customerId": "cus_8899",
  "total": 3,
  "tags": [
    {
      "tagId": "9ce355bfacca49c4a9e9322a9317c196",
      "name": "抗老/紧致",
      "dimension": "concerns",
      "evidence": "很喜欢用黑钻光灿面霜"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `customerId` | 客户 id |
| `total` | 标签数量 |
| `tags[].tagId` | 标签 uuid（32 位十六进制，如 `9ce355bfacca49c4a9e9322a9317c196`） |
| `tags[].name` | 标签名称 |
| `tags[].dimension` | 维度：`concerns` / `interested_products` / `purchase_intent` / `price_sensitivity` / `service_opportunities` / `custom_tags` 等 |
| `tags[].evidence` | 依据原文（可选） |

- 客户不存在 → `404 NOT_FOUND`。
- 说明：当前为 **mock 数据**（示例客户 `cus_8899`、`cus_1001`）；接入真实标签存储后接口契约不变。

---

# 附录

## 10. 错误码

| HTTP | code | 说明 |
|---|---|---|
| 401 | `UNAUTHORIZED` | 缺少或错误的 API Key |
| 400 | `BAD_REQUEST` | 参数非法（缺 `uuid` / `baId` / `customerId` / `tags` 非法等） |
| 404 | `NOT_FOUND` | 任务 / 客户不存在 |
| 413 | `PAYLOAD_TOO_LARGE` | 上传超限 |
| 502 | `UPSTREAM_ERROR` | 上游服务错误 |
| 504 | `UPSTREAM_TIMEOUT` | 上游超时 |
| 500 | `INTERNAL_ERROR` | 其他 |

## 11. 端到端示例

```bash
BASE=https://ada.alphafina.cn/api/el-ai-gateway
KEY=<API_KEY>

# 1) 上传（baId / customerId 必填）
UUID=$(curl -s -X POST "$BASE/files?path=voice-tagging&fileName=clip.wav&baId=ba_001&customerId=cus_8899" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav;type=audio/wav" | jq -r .uuid)

# 2) 发起（baId / customerId 必填）
TASK=$(curl -s -X POST "$BASE/voice-tagging" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"uuid\":\"$UUID\",\"baId\":\"ba_001\",\"customerId\":\"cus_8899\"}" | jq -r .taskId)

# 3) 查询状态（tags；转写/打标正文在 activities[].markdown）
curl -s "$BASE/voice-tagging/$TASK" -H "Authorization: Bearer $KEY" | jq

# 4) 修正标签（整体覆盖）
curl -s -X PUT "$BASE/voice-tagging/$TASK/tags" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"tags":["9ce355bfacca49c4a9e9322a9317c196"]}' | jq

# 5) 客户标签（客户详情界面）
curl -s "$BASE/customers/cus_8899/tags" -H "Authorization: Bearer $KEY" | jq
```

**时序**：上传 → 发起（立即返回 `taskId`）→ 后台转写+打标 → 轮询状态可看到 `tags` 与 activity → 修正标签再写入任务时间线。
