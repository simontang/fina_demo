# el-ai-gateway API 文档（客户语音打标 + 文件 / 任务 / 标签管理）

雅思兰黛 AI 项目 · 业务 API 网关：**客户语音转写打标** 及配套的 **文件、任务、客户标签管理接口**。

- **Base URL（线上）**：`https://ada.alphafina.cn/api/el-ai-gateway`
- **协议**：HTTPS，JSON / multipart
- **版本**：v1.5

---

## 修订日志

### v1.5（2026-09-23）

- **文档移除** `POST /files`（单独上传）与 `POST /voice-tagging`（按 uuid 发起）；统一使用合并接口 `POST /voice-tagging/upload`（[§8.1](#81-发起打标任务)）。接口仍保留兼容，仅不再对外提供文档。
- **变更** Webhook 事件（[§12](#12-webhook-事件jobcompleted)）：`eventType=job.completed`，`payload.type=customer_tag.updated`；**只带 `customerId`、不带标签数据**，收到后调查询接口取最新标签。

### v1.4（2026-09-23）

- **新增** `POST /voice-tagging/upload`：**合并"上传 + 发起"**为一个 multipart 请求（[§8.1](#81-发起打标任务)）。
- **新增** `durationSec`（录音时长，秒）：`POST /voice-tagging` 与 `POST /voice-tagging/upload` **必填**；`POST /files` 可选写入文件 meta；任务详情/列表返回 `durationSec`。
- **新增** 固定播放地址 `GET /voice-tagging/:taskId/audio`（[§8.3.1](#831-播放录音固定地址)）：现取预签名并**代理音频流**（支持 `Range`/`206`/`HEAD`），任务详情/列表返回 `audioUrl`。
- **变更** 任务详情/列表新增 `durationSec` 与 `audioUrl` 字段。

### v1.3（2026-09-23）

- **文档**：新增 §12 Webhook 事件说明（当时为 `job.completed`；v1.5 已改为 `customer_tag.updated`）。
- 说明：事件仅作**通知**用途，最终结果仍以查询接口为准。

### v1.2（2026-09-23）

- **任务结果聚合**：任务 `result` 统一为 `{ transcript, tags, like }`。
  - `GET /voice-tagging/:taskId` 返回 `transcript` / `tags` / `like`；`tags` 元素为 `{ tagId, tagKey, tagValue, evidence? }`。
  - `GET /voice-tagging` 列表返回 `tasks[].tags` / `tasks[].like`（**不含 `transcript`**）。
- **新增** `PUT /voice-tagging/:taskId/like`：点赞 `true` / 取消（或未点赞）`null`。
- **变更** `PUT /voice-tagging/:taskId/tags`：
  - 由"整体 replace `result`"改为**只覆盖 `tags`**，保留 `transcript` / `like`；每个标签按 `tagId` 继承原有 `evidence`。
  - 请求体只取 `tags`；请求中若带 `transcript` / `like` 会被忽略。
  - 标签元素：`{ tagId }`（引用已存在标签）或 `{ tagValue }`（新增标签）。
- **新增** `DELETE /voice-tagging/:taskId`：删除任务（不可恢复）。
- **变更** `GET /customers/:customerId/tags`：返回字段为 `tagId` / `tagKey` / `tagValue` / `source` / `confidence` / `taggedAt`；客户无标签时返回 `200` + 空数组（不再 `404`）。
- **移除** 任务时间线接口 `GET /voice-tagging/:taskId/activities`（暂不提供，后续按需再加）。
- **说明**：任务标签字段由 `name` / `dimension` 调整为 `tagKey`（标签组） / `tagValue`（标签名）。
- **接入建议**：接口由**应用后台**调用（API Key 为租户级，无法识别具体终端用户）；浏览器直连仅用于开发联调；异步结果建议**每 60 秒轮询**任务详情直到终态。

---

## 适用对象

本文面向**对接开发同学**（业务应用 / 前端 / 集成方）：说明如何调用网关接口、每个业务/界面需要先调什么后调什么、每步拿到什么、字段怎么衔接。内部实现细节不在此展开。

## 业务场景

本服务提供三类能力：

**1. 语音文件打标（核心）**
把一段**客户语音**（导购/客服与客户的沟通录音）自动处理成**结构化客户画像标签**：

- 上传录音文件并发起打标任务（一步完成）；
- 触发处理：**语音转文本** → 按客户画像**多维度打标签**（肤质、诉求、感兴趣产品、购买意向、服务机会、自定义标签…），每条标签都带原文 **evidence**；
- 业务可**查询任务状态与结果**（转写全文、打标结果），并可**修正任务标签**。

**2. 业务 / 界面管理接口**
支撑"客户详情"等界面的读取与管理：

- **文件管理**：上传、按 BA + 客户查询文件、获取音频播放链接；
- **任务管理**：发起打标、按 BA + 客户列任务、任务详情（含原文/标签/点赞）、修正标签；
- **客户标签**：查询客户画像标签（由任务标签自动汇总，外部只读）。

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

#### 调用方建议：由应用后台调用，而非前端直连

- 该 API Key 是**应用级密钥**，只对应到**租户**，**无法识别具体的终端用户**（不区分登录人、不区分 BA/店员）。
- 因此架构上建议：**由你们应用的后端（服务端 / BFF）持有并调用本接口**，前端请求经你们后端转发；不要把 Key 下发到浏览器或客户端。
  - 在你们后端按登录用户做鉴权、审计与限流；
  - 避免密钥泄露——Key 是租户粒度，一旦外泄即等于整个租户的读写能力。
- 网关默认开启 CORS 只为**联调便利**，并不代表推荐生产环境由浏览器直连。

### 1.3 跨域（CORS）

网关默认开启 CORS，**允许任意来源**（`Access-Control-Allow-Origin: *`）。浏览器**直连只为开发/联调阶段的便利**，不代表生产用法——生产请按 [§调用方建议](#调用方建议由应用后台调用而非前端直连) 由你们后端调用。生产如需收紧，由后端配置来源白名单，前端无需改动。

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
- **发起是异步的**：调用立即返回，系统在后台处理并把结果写入任务。前端可**每 60 秒轮询一次** [查询任务详情](#83-查询任务详情)，直到 `status` 进入**终态**（`completed` / `failed` / `cancelled`）后再停止。
- 当前转写为 **mock**（处理结果里 `mock: true`）。
- 结果读取方式：**以查询接口为准**；客户标签汇总更新后会投递 `job.completed` 事件（`payload.type=customer_tag.updated`，见 [§12](#12-webhook-事件jobcompleted)），可作通知，收到后请调查询接口取最新标签。

## 2. 核心概念：两层标签（客户标签 vs 任务标签）

网关里的"标签"有**两个层级**，接口不通用，对接时务必区分：

| 层级 | 归属对象 | 语义 | 接口 |
|---|---|---|---|
| **客户标签** | 客户（`customerId`） | 该客户当前的业务标签，由该客户的任务标签自动汇总（最终一致） | `GET /customers/:customerId/tags`（**仅查询**） |
| **任务标签** | 打标任务（`taskId`） | **某一次**语音打标算出的标签 | 读 `GET /voice-tagging/:taskId`；写 `PUT /voice-tagging/:taskId/tags`（**整体覆盖，非追加**） |

- 任务标签是"某条语音的结论"；客户标签是该客户所有任务标签的汇总（最终一致）。两者不要混用。
- 客户标签由**任务标签汇总**而来：编辑某任务的标签（`PUT /voice-tagging/:taskId/tags`）后，系统会重算该客户的标签汇总。外部**没有客户级写接口**，只能查询。
  - 要"补充 / 修正**某次打标**结果" → 改**任务标签**，用 `PUT /voice-tagging/:taskId/tags`；
  - 客户维度的标签由任务标签自动汇总，不需要（也不提供）单独的客户级写接口。
- `PUT /voice-tagging/:taskId/tags` 是**覆盖式**：传入的数组会**替换该任务原有的全部标签**，不是追加。若要在原基础上加，请先 `GET /voice-tagging/:taskId` 读出 `tags`，合并后再整体 PUT。

> 任务上的写操作有两个：**修改标签**（覆盖式，`PUT .../tags`）与**点赞/取消**（`PUT .../like`）。

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

展示该 BA 在某客户名下的语音任务列表（标签 & 修改）。

| 动作 | 接口 |
|---|---|
| 列出该 BA + 客户名下的语音任务 | [`GET /voice-tagging?baId=&customerId=`](#82-查询任务列表) |
| 查看某任务的转写 / 打标结果 | [`GET /voice-tagging/:taskId`](#83-查询任务详情) |
| 删除某条任务 | [`DELETE /voice-tagging/:taskId`](#85-删除任务) |

**调用顺序**：列表页 `GET /voice-tagging?baId=&customerId=` 拿到 `tasks[]`（含 `taskId`）→ 点开某条任务时用其 `taskId` 调 `GET /voice-tagging/:taskId`。

## 5. 场景：补充客户标签

| 动作 | 接口 | 说明 |
|---|---|---|
| 回显客户现有标签 | [`GET /customers/:customerId/tags`](#91-查询客户业务标签) | **客户级、只读** |
| 补充 / 更新客户标签 | 无（由任务标签自动汇总） | 编辑任务标签后，客户汇总会自动更新 |
| 若是修正"某条语音"的标签 | `GET /voice-tagging/:taskId` 读取 → 合并 → [`PUT /voice-tagging/:taskId/tags`](#84-修改任务标签) | **任务级、覆盖式**，不是客户级追加 |

> ⚠️ 详见 [§2 核心概念](#2-核心概念两层标签客户标签-vs-任务标签)：**客户标签只读**；可写的是**任务标签**，且为**覆盖**。

> 前端注意：**"补充 / 更新客户标签"当前没有写接口**（由内部系统同步）。该入口应做**只读展示或隐藏**，不要等待网关联调写入接口。

## 6. 场景：语音手记 / 任务详情

上传录音 → 查看转写与打标结果、修正标签。

| 步骤 | 接口 |
|---|---|
| 上传录音并发起转写+打标 | [`POST /voice-tagging/upload`](#81-发起打标任务)（multipart：`file` + `baId`/`customerId`/`durationSec`）→ 得 `taskId` |
| 手记列表 / 音频播放 | [`GET /voice-tagging?baId=&customerId=`](#82-查询任务列表)、固定播放地址 `GET /voice-tagging/:taskId/audio`（见 [§8.3.1](#831-播放录音固定地址)） |
| 查看转写与打标结果 | [`GET /voice-tagging/:taskId`](#83-查询任务详情) |
| 修正**本次打标**标签（覆盖） | [`PUT /voice-tagging/:taskId/tags`](#84-修改任务标签) |
| 点赞 / 取消点赞 | [`PUT /voice-tagging/:taskId/like`](#86-任务点赞) |

**时序**：上传并发起（立即返回 `taskId`）→ 后台转写 + 打标 → 轮询详情可看到 `tags` / `transcript` → 修正标签或点赞。

---

# 第二部分 · 接口参考（查字段）

## 7. 文件

### 7.1 查询文件

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

### 7.2 获取文件播放/下载链接

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

上传录音并创建打标任务：**一个 multipart 请求**同时完成"上传 + 发起"，无需先上传拿 `uuid`。

```
POST /voice-tagging/upload            # multipart/form-data
```

**Query 参数**

| 参数 | 必填 | 说明 |
|---|---|---|
| `baId` | 是 | 业务员 id |
| `customerId` | 是 | 客户 id |
| `durationSec` | 是 | 录音时长（秒，正数） |
| `title` / `description` | 否 | 任务名 / 描述 |
| `assistantId` | 否 | 覆盖服务端默认配置 |
| `path` / `fileName` / `fileCategory` / `usage` | 否 | 透传给文件上传 |

**multipart**：文件字段名 `file`（必填）。

**响应 `200`**：
```json
{
  "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
  "status": "in_progress",
  "file": { "uuid": "471c20082b524316accc1b23cba8a4de", "url": "https://…（预签名）" },
  "agent": { "dispatched": true }
}
```

- 缺 `baId` / `customerId` / `durationSec` → `400 BAD_REQUEST`；缺 `file` → `400`。

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
      "durationSec": 12.5,
      "audioUrl": "/voice-tagging/c3915a5a-85ed-4e31-a09e-492b3c11e938/audio",
      "tags": [
        {
          "tagId": "9ce355bfacca49c4a9e9322a9317c196",
          "tagKey": "concerns",
          "tagValue": "抗老/紧致",
          "evidence": "很喜欢用黑钻光灿面霜"
        }
      ],
      "like": true
    }
  ]
}
```

- 缺 `baId` 或 `customerId` → `400 BAD_REQUEST`。
- `tasks[].status` 枚举同 [§8.3 查询任务详情](#83-查询任务详情)。
- `tasks[].durationSec` / `tasks[].audioUrl` / `tasks[].tags` / `tasks[].like` 同 [§8.3](#83-查询任务详情)；**列表不含 `transcript`**（原文请用任务详情接口获取）。
- 说明：按任务的 `baId` + `customerId` **元数据精确过滤**（由 [发起打标任务](#81-发起打标任务) 创建时写入）；无匹配时返回空列表（`total:0`）。

### 8.3 查询任务详情

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
  "durationSec": 12.5,
  "audioUrl": "/voice-tagging/c3915a5a-85ed-4e31-a09e-492b3c11e938/audio",
  "transcript": "……完整语音原文……",
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "tagKey": "concerns", "tagValue": "抗老/紧致", "evidence": "很喜欢用黑钻光灿面霜" }
  ],
  "like": true
}
```

- `status`：`pending | in_progress | review | failed | interrupted | completed | cancelled`
- `fileId`：本次任务关联的文件 uuid。
- `durationSec`：录音时长（秒，number，无则 `null`）。
- `audioUrl`：**固定播放地址**（相对 Base 的路径），见 [§8.3.1 播放录音](#831-播放录音固定地址)。
- `transcript`：语音原文（string，无则 `null`）。
- `tags`：该任务已生成的标签，元素 `{ tagId, tagKey（标签组）, tagValue（标签名）, evidence?（原文依据） }`。
- `like`：用户点赞反馈，`true` 或 `null`（从未点赞 / 已取消均为 `null`）。
- 任务不存在 → `404 NOT_FOUND`

> **衔接**：用发起接口返回的 `taskId` 查询转写/标签结果。

### 8.3.1 播放录音（固定地址）

`audioUrl` 是一个**固定不变**的播放地址：每次访问时由服务端**现取预签名并转发音频流**，因此**不受 TOS 链接过期影响**，客户端始终用同一个 URL。

```
GET /voice-tagging/:taskId/audio          # 需 Authorization（同其他接口）
```

- 请求头可带 `Range: bytes=…`，返回 `206` + `Content-Range`（支持拖动进度）；支持 `HEAD`。
- 响应为音频流（`audio/*`）。
- 鉴权：`Authorization: Bearer <API_KEY>`（与其它接口一致）。因 `<audio>` / 小程序 `innerAudioContext` 不能带请求头，**由你们后端携带 Key 调用本接口、再把流转发给前端**。
- 任务不存在或无文件 → `404 NOT_FOUND`；上游取流失败 → `502 UPSTREAM_ERROR`。

### 8.4 修改任务标签

整体**覆盖**某**任务**的标签集合。

> ⚠️ 这是**任务级、覆盖式**接口，只影响该 `taskId` 这一条打标任务的标签；
> 它**不会**直接写客户画像，而是在提交后由系统按该客户的任务标签**汇总刷新**（见 [§9](#9-客户标签)）。

```
PUT /voice-tagging/:taskId/tags
Content-Type: application/json
```

**请求体**

```json
{
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196" },
    { "tagValue": "敏感肌" }
  ]
}
```

`tags` 是**该任务提交后的完整标签列表**（整体覆盖）。每个元素是 `{ "tagId"?, "tagValue"? }`，按**有没有 `tagId`** 分两种含义：

| 元素写法 | 含义 | 系统行为 |
|---|---|---|
| **带 `tagId`** | 选中一个**已存在**的标签 | 按 `tagId` 到**标签主数据**查询：查到 → 直接采用其标签组/标签名（元素里的 `tagValue` 会被忽略，可省略）；查不到 → `400 BAD_REQUEST`。**不会新建、也不会修改主数据**。 |
| **不带 `tagId`**（只给 `tagValue`） | **新增**一个标签 | 以 `tagValue` 作为标签名；标签组固定为 `客户画像`，类别固定为 `自定义标签`。先按（标签组 + 标签名）查重：已存在 → 复用其 `tagId`；不存在 → 新建并生成一个 `tagId`。 |

- 两种写法最终都会得到 `tagId`，并**回写**到任务标签；响应里的 `tags` 就是回写后的结果（含 `tagId`）。
- 同一元素同时给了 `tagId` 和 `tagValue` 时，**以 `tagId` 为准**。
- 元素既没有 `tagId`、`tagValue` 也为空 → `400 BAD_REQUEST`。

**示例：只选已存在的标签**

```json
{ "tags": [ { "tagId": "9ce355bfacca49c4a9e9322a9317c196" } ] }
```

**示例：新增一个标签（无需先有 id）**

```json
{ "tags": [ { "tagValue": "敏感肌" } ] }
```
（无需再单独提交 `tagValue` 对应的 id；提交后响应会返回它的 `tagId`。）

**响应 `200`**（更新后的任务详情）：

```json
{
  "taskId": "b3ca978f-3832-4a2c-959b-40fa48c43352",
  "fileId": "2ccf6fef88b64a16b62fe491a8f7a132",
  "status": "in_progress",
  "createdAt": "2026-09-15T05:18:00Z",
  "title": "Voice tagging: 2ccf6fef88b64a16b62fe491a8f7a132",
  "transcript": "……完整语音原文……",
  "tags": [
    { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "tagKey": "concerns", "tagValue": "抗老/紧致" },
    { "tagId": "4a1b2c3d5e6f47089a0b1c2d3e4f5061", "tagKey": "interested_products", "tagValue": "黑钻光灿面霜" }
  ],
  "like": true
}
```

- `tags` 非数组 / 元素同时缺 `tagId` 与 `tagValue` / `tagId` 不存在 → `400 BAD_REQUEST`
- 任务不存在 → `404 NOT_FOUND`
- 只替换 `tags`；`transcript` 与 `like` 保持不变；每个标签按 `tagId` 保留原有的 `evidence`。
- 请求体**只取 `tags`**；若请求里还带了 `transcript` / `like`，会被**忽略**（原文由系统/agent 维护，点赞见 [§8.6](#86-任务点赞)）。
- 说明：该接口会**整体替换**本任务已生成的标签，并**重算该客户的标签汇总**（最终一致）。

### 8.5 删除任务

删除一条打标任务（**不可恢复**）；删除后系统会重算该客户的标签汇总。

```
DELETE /voice-tagging/:taskId
```

> 该接口**无请求体**（不要带 `Content-Type: application/json` 的空 body）。

**响应 `200`**：

```json
{ "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938", "deleted": true }
```

- 任务不存在 → `404 NOT_FOUND`
- 说明：删除后，该任务贡献的标签会从该客户汇总（见 [§9 客户标签](#9-客户标签)）中移除（最终一致）。

### 8.6 任务点赞

对任务结果点赞 / 取消点赞（用户反馈）。

```
PUT /voice-tagging/:taskId/like
Content-Type: application/json
```

```json
{ "like": true }
```

| 值 | 含义 |
|---|---|
| `true` | 点赞 |
| `null` | 取消点赞 / 未点赞 |

**响应 `200`**：更新后的任务详情（同 [§8.3](#83-查询任务详情)，含 `transcript` / `tags` / `like`）。

- `like` 缺失或非 `true`/`null`（如 `false`）→ `400 BAD_REQUEST`
- 任务不存在 → `404 NOT_FOUND`
- 说明：只更新 `like`，`tags` / `transcript` 保持不变。

## 9. 客户标签

### 9.1 查询客户业务标签

查询某客户的全部业务标签（**标签组 + 标签名 + 标签 id**）。

> 这是**客户级、仅查询**接口：返回该客户当前的业务标签汇总（客户画像）。**客户标签由任务标签自动汇总，外部不能通过本接口写入/追加**；它不是某条语音任务的结果（见 [§2 核心概念](#2-核心概念两层标签客户标签-vs-任务标签)）。

```
GET /customers/:customerId/tags
```

**响应 `200`**：

```json
{
  "customerId": "cus_8899",
  "total": 1,
  "tags": [
    {
      "tagId": "9ce355bfacca49c4a9e9322a9317c196",
      "tagKey": "concerns",
      "tagValue": "抗老/紧致",
      "source": "voice",
      "confidence": null,
      "taggedAt": "2026-09-15T06:00:00Z"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `customerId` | 客户 id |
| `total` | 标签数量 |
| `tags[].tagId` | 标签值 id（32 位十六进制，如 `9ce355bfacca49c4a9e9322a9317c196`） |
| `tags[].tagKey` | 标签组 |
| `tags[].tagValue` | 标签名 |
| `tags[].source` | 来源：`voice`（语音打标汇总）/ `manual` |
| `tags[].confidence` | 置信度（可空） |
| `tags[].taggedAt` | 打标时间（ISO 8601） |

- 客户没有标签时返回 `200` + 空数组（`total: 0`）。
- 数据来源为标签主数据与任务标签的汇总，**最终一致**（刚编辑完任务标签后可能有短暂延迟）。

---

# 附录

## 10. 错误码

| HTTP | code | 说明 |
|---|---|---|
| 401 | `UNAUTHORIZED` | 缺少或错误的 API Key |
| 400 | `BAD_REQUEST` | 参数非法（缺 `uuid` / `baId` / `customerId`；`tags` 非法、`tagId` 不存在、`like` 非 `true`/`null` 等） |
| 404 | `NOT_FOUND` | 任务不存在 |
| 413 | `PAYLOAD_TOO_LARGE` | 上传超限 |
| 502 | `UPSTREAM_ERROR` | 上游服务错误 |
| 504 | `UPSTREAM_TIMEOUT` | 上游超时 |
| 500 | `INTERNAL_ERROR` | 其他 |

## 11. 端到端示例

```bash
BASE=https://ada.alphafina.cn/api/el-ai-gateway
KEY=<API_KEY>

# 1) 上传 + 发起（一步；baId / customerId / durationSec 必填）
TASK=$(curl -s -X POST "$BASE/voice-tagging/upload?baId=ba_001&customerId=cus_8899&durationSec=12&path=voice-tagging&fileName=clip.wav" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav;type=audio/wav" | jq -r .taskId)

# 2) 查询任务详情（transcript / tags / like）
curl -s "$BASE/voice-tagging/$TASK" -H "Authorization: Bearer $KEY" | jq

# 3) 修正标签（整体覆盖）
curl -s -X PUT "$BASE/voice-tagging/$TASK/tags" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"tags":[{"tagId":"9ce355bfacca49c4a9e9322a9317c196"}]}' | jq

# 4) 客户标签（客户详情界面）
curl -s "$BASE/customers/cus_8899/tags" -H "Authorization: Bearer $KEY" | jq
```

**时序**：上传并发起（立即返回 `taskId`）→ 后台转写+打标 → 轮询任务详情可看到 `transcript` / `tags` → 修正标签或点赞。

## 12. Webhook 事件（`job.completed`）

某客户的**标签汇总被更新**后（编辑任务标签、删除任务等触发汇总重算），平台会向**已注册的接收端**投递 webhook 事件。

- 事件类型（`eventType`）：**`job.completed`**。
- body 的 **`payload.type`** 标明业务类型：**`customer_tag.updated`**（客户标签已更新）。
- **不携带标签数据**，只带**客户 id**（`customerId`）；收到后请调 [§9.1 查询客户业务标签](#91-查询客户业务标签) 取最新标签。
- 投递语义：Standard Webhooks 签名（`webhook-*` 头，部分实现用别名 `svix-*`）、**at-least-once**、失败自动重试。
- 订阅时用 `filterTypes: ["job.completed"]`。

事件 body：

```json
{
  "eventType": "job.completed",
  "payload": {
    "type": "customer_tag.updated",
    "customerId": "cus_8899"
  }
}
```

| 字段 | 说明 |
|---|---|
| `eventType` | 固定 `job.completed` |
| `payload.type` | 固定 `customer_tag.updated`（客户标签已更新） |
| `payload.customerId` | 标签发生变化的客户 id |
