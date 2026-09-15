# el-ai-gateway API 文档（语音文件打标签）

雅思兰黛 AI 项目 · 语音文件转写与客户画像打标签业务 API。

- **Base URL（线上）**：`https://ada.alphafina.cn/api/el-ai-gateway`
- **协议**：HTTPS，JSON / multipart
- **版本**：v1

---

## 目录

1. [鉴权](#1-鉴权)
2. [通用约定](#2-通用约定)
3. [上传文件](#3-上传文件)
4. [发起打标任务](#4-发起打标任务)
5. [查询任务状态](#5-查询任务状态)
6. [提交反馈](#6-提交反馈)
7. [Webhook 回调（2 次）](#7-webhook-回调2-次)
8. [错误码](#8-错误码)
9. [端到端示例](#9-端到端示例)

---

## 1. 鉴权

所有业务接口都需要在请求头带平台签发的 API Key：

```
Authorization: Bearer <API_KEY>
```

- 缺失或错误 → `401` `{"code":"UNAUTHORIZED","message":"Missing or invalid API key"}`
- Key 映射到一个租户（tenant），该租户决定文件的归属（platform-service `X-Tenant-Id`）与任务所有者（task `ownerId`）。当前测试租户为 `estee_lauder`。

## 2. 通用约定

- 响应均为 JSON。错误统一信封：
  ```json
  { "code": "UNAUTHORIZED", "message": "Missing or invalid API key" }
  ```
  上游错误会带 `upstream`：
  ```json
  { "code": "UPSTREAM_ERROR", "message": "...", "upstream": { "status": 502, "code": "..." } }
  ```
- **发起/反馈是异步的**：调用立即返回，agent 在后台处理并写入任务 activity。请轮询状态接口，或等待 [Webhook 回调](#7-webhook-回调2-次)。
- 当前转写为 **mock**（处理结果里 `mock: true`）。

## 3. 上传文件

把语音文件上传到统一文件服务，返回文件 `uuid`（后续发起任务用）。

```
POST /files?path=<dir>&fileName=<name>
Content-Type: multipart/form-data
```

| 参数 | 位置 | 必填 | 说明 |
|---|---|---|---|
| `file` | form-data | 是 | 语音文件 |
| `path` | query | 否 | 逻辑目录，如 `voice-tagging` |
| `fileName` | query | 否 | 显示文件名；缺省用上传文件名 |

**响应 `200`**（平台文件 receipt）：

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
  "status": "active",
  "createdBy": "api",
  "createdAt": "2026-09-15T06:13:00.000000",
  "deduplicated": false
}
```

## 4. 发起打标任务

用上一步的 `uuid` 换取可访问 URL，创建任务并派发 agent 做**语音转写 + 客户画像打标签**。

```
POST /voice-tagging
Content-Type: application/json
```

```json
{
  "uuid": "471c20082b524316accc1b23cba8a4de",
  "title": "可选，任务名；缺省 Voice tagging: <uuid>",
  "description": "可选",
  "assistantId": "可选，覆盖服务端默认 agent"
}
```

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
- `agent.dispatched=true` 表示消息已派发给 agent（后台执行；派发失败不影响本响应，服务端记日志）。

## 5. 查询任务状态

```
GET /voice-tagging/:taskId
```

**响应 `200`**：

```json
{
  "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
  "status": "in_progress",
  "title": "Voice tagging: 471c20082b524316accc1b23cba8a4de",
  "result": "",
  "activities": [
    {
      "id": "…",
      "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
      "tenantId": "estee_lauder",
      "action": "activity",
      "actor": "agent:voice-tagging-agent",
      "detail": { "markdown": "## 客户画像标签（基于转写文本，MOCK）\n…" },
      "createdAt": "2026-09-15T06:15:21.000Z"
    }
  ]
}
```

- `status`：`pending | in_progress | review | failed | interrupted | completed | cancelled`
- `activities`：任务时间线。转写/打标结果、反馈都由 agent 以 Markdown 追加到这里（`detail.markdown`）。最新在前。
- 任务不存在 → `404` `{"code":"NOT_FOUND","message":"Task '…' not found or inaccessible"}`

## 6. 提交反馈

把客户/业务反馈追加到任务 activity 时间线（由 agent 以自身身份写入 `add_activity`）。

```
POST /voice-tagging/:taskId/feedback
Content-Type: application/json
```

```json
{
  "content": "客户反馈：purchase_intent 应为「低」，并补充回访建议。",
  "summary": "可选，反馈摘要",
  "assistantId": "可选，覆盖服务端默认 agent"
}
```

**响应 `200`**：

```json
{
  "taskId": "c3915a5a-85ed-4e31-a09e-492b3c11e938",
  "forwarded": true,
  "agent": { "dispatched": true }
}
```

随后（约 10–30s）该任务的 `activities` 会多出一条 `## 客户反馈…`。`content` 为空/缺失 → `400 BAD_REQUEST`。

## 7. Webhook 回调（2 次）

任务处理过程中，agent 平台会向已注册的 **delivery destination** 发起 **2 次** 回调（转写完成、打标完成各一次）。

- **投递方式**：platform-service / Svix（Standard Webhooks）。请求头含 `svix-id`、`svix-timestamp`、`svix-signature`，用注册 destination 时返回的 `whsec_…` 验签。
- **事件类型（Svix `eventType`）**：`job.completed`（两次相同）；**阶段由 body 的 `event` 字段区分**。
- **注册回调地址**：见 platform-service webhooks API（`POST /api/webhooks/destinations`）。

### 7.1 `voice.transcribed`（转写完成）

```json
{"event":"voice.transcribed","stage_status":"success","task_id":"c3915a5a-85ed-4e31-a09e-492b3c11e938","file_id":"471c20082b524316accc1b23cba8a4de","local_path":"/project/audio/471c20082b524316accc1b23cba8a4de","text":"王女士，陕西西安人，老公是上海人，女儿的话呢，现在是在大学里面读大三，学的是和舞台设计有关的。他们是想让女儿去考研的，但是在择校这方面呢是有点小分歧。最近呢也是为了这个考研的那个择校问题。呃，也是有点纠结。然后本人的话呢是很喜欢用我们的呃黑钻光灿面霜，然后之前因为是用过花精粹的面霜，呃，黑钻光灿面霜在天气比较呃，热的时候他会选择，然后冬天的话呢，他还是选择花精粹的面霜比较多一点。呃，之前的话购买的频率是比较高，但最近的话购买频率比较低。呃，一方面是因为他父亲在年头的时候呢过世了，然后他本人呢一直沉浸在这个悲伤中，呃，一直在调整自己，有参加一些呃，公益活动也参加一些，呃，庙里的一些义工的活动。他的日常的话呢，也会发在抖音上面。那抖音上我一直可以看到他有发一些最近的生活。呃，最近的话呢，他因为也是在调理他的，做一些复健。他的腿和他的肩颈一直是有点小问题，所以一直在做调理。那在抖音上呢，我也会一直帮他点赞和问他最近的生活如何。呃，她的话呢，本身呢是一个很有气质的一位女性。如果是参加一些呃酒店的活动，都会是精心打扮的。对我们品牌呢也是比较认可的。而且他的妹妹呢也是用我们雅诗兰黛的，不过用的是我们小棕瓶系列。呃，一直会委托他买，呃，让我帮他去快递给他妹妹。嗯，对我们品牌的认可度还是很高的。","mock":true,"language":"zh-CN","download_ok":true}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `event` | string | 固定 `voice.transcribed` |
| `stage_status` | string | `success` / `failed` |
| `task_id` | string(uuid) | 任务 id |
| `file_id` | string(32hex) | 文件 uuid |
| `local_path` | string | 转写时的沙盒本地路径 |
| `text` | string | 转写全文 |
| `mock` | boolean | 是否 mock 转写 |
| `language` | string | 语言，如 `zh-CN` |
| `download_ok` | boolean | 音频是否下载成功 |

### 7.2 `voice.tagged`（打标完成）

```json
{"event":"voice.tagged","stage_status":"success","task_id":"c3915a5a-85ed-4e31-a09e-492b3c11e938","file_id":"471c20082b524316accc1b23cba8a4de","mock":true,"summary":"王女士为西安人，雅诗兰黛品牌高认可度客户，偏好黑钻光灿面霜与花精粹面霜，冬季更倾向花精粹。近期因父亲过世情绪低落、购买频率下降，正通过公益与义工活动调整状态，并有腿与肩颈的复健调理需求。女儿在读大三（舞台设计相关）正面临考研择校分歧，可作为情感关怀与回访切入点。","tags":{"skin_type":[],"concerns":[{"tag":"抗老/紧致","evidence":"很喜欢用我们的呃黑钻光灿面霜，然后之前因为是用过花精粹的面霜"},{"tag":"保湿","evidence":"冬天的话呢，他还是选择花精粹的面霜比较多一点"}],"interested_products":[{"tag":"黑钻光灿面霜","evidence":"很喜欢用我们的呃黑钻光灿面霜"},{"tag":"花精粹面霜","evidence":"之前因为是用过花精粹的面霜"},{"tag":"小棕瓶系列","evidence":"他的妹妹呢也是用我们雅诗兰黛的，不过用的是我们小棕瓶系列"}],"purchase_intent":[{"tag":"中","evidence":"对我们品牌的认可度还是很高的；之前的话购买的频率是比较高，但最近的话购买频率比较低"}],"price_sensitivity":[{"tag":"中","evidence":"原文无直接价格表述，仅从高端产品线（黑钻光灿面霜）使用推断"}],"competitor_mentions":[],"service_opportunities":[{"tag":"情感关怀回访","evidence":"他父亲在年头的时候呢过世了，然后他本人呢一直沉浸在这个悲伤中"},{"tag":"健康关怀（腿、肩颈复健调理）","evidence":"他的腿和他的肩颈一直是有点小问题，所以一直在做调理"},{"tag":"抖音互动维护","evidence":"他的日常的话呢，也会发在抖音上面；我也会一直帮他点赞和问他最近的生活如何"},{"tag":"妹妹小棕瓶代购/快递服务","evidence":"一直会委托他买，呃，让我帮他去快递给他妹妹"}],"custom_tags":[{"tag":"西安人","evidence":"王女士，陕西西安人"},{"tag":"老公上海人","evidence":"老公是上海人"},{"tag":"女儿大三（舞台设计相关）","evidence":"女儿的话呢，现在是在大学里面读大三，学的是和舞台设计有关的"},{"tag":"关注女儿考研择校","evidence":"他们是想让女儿去考研的，但是在择校这方面呢是有点小分歧"},{"tag":"雅诗兰黛高认可度","evidence":"对我们品牌呢也是比较认可的；对我们品牌的认可度还是很高的"},{"tag":"抖音活跃","evidence":"他的日常的话呢，也会发在抖音上面"},{"tag":"有气质/注重打扮","evidence":"本身呢是一个很有气质的一位女性。如果是参加一些呃酒店的活动，都会是精心打扮的"},{"tag":"公益/义工活动参与","evidence":"有参加一些呃，公益活动也参加一些，呃，庙里的一些义工的活动"}]}}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `event` | string | 固定 `voice.tagged` |
| `stage_status` | string | `success` / `failed` |
| `task_id` / `file_id` | string | 任务 id / 文件 uuid |
| `mock` | boolean | 是否 mock |
| `summary` | string | 客户画像摘要 |
| `tags` | object | 多维标签，见下 |

`tags` 各维度均为 `{ "tag": string, "evidence": string }[]`：

| 维度 | 说明 |
|---|---|
| `skin_type` | 肤质 |
| `concerns` | 护肤诉求 |
| `interested_products` | 感兴趣产品 |
| `purchase_intent` | 购买意向 |
| `price_sensitivity` | 价格敏感度 |
| `competitor_mentions` | 竞品提及 |
| `service_opportunities` | 服务/回访机会 |
| `custom_tags` | 自定义标签（地域、家庭、兴趣等） |

## 8. 错误码

| HTTP | code | 说明 |
|---|---|---|
| 401 | `UNAUTHORIZED` | 缺少或错误的 API Key |
| 400 | `BAD_REQUEST` | 参数非法（缺 `uuid` / `content` 为空等） |
| 404 | `NOT_FOUND` | 任务不存在 |
| 413 | `PAYLOAD_TOO_LARGE` | 上传超限 |
| 502 | `MCP_ERROR` / `UPSTREAM_ERROR` | 上游（MCP / 文件服务）错误 |
| 504 | `UPSTREAM_TIMEOUT` | 上游超时 |
| 500 | `INTERNAL_ERROR` | 其他 |

## 9. 端到端示例

```bash
BASE=https://ada.alphafina.cn/api/el-ai-gateway
KEY=<API_KEY>

# 1) 上传
UUID=$(curl -s -X POST "$BASE/files?path=voice-tagging&fileName=clip.wav" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@clip.wav;type=audio/wav" | jq -r .uuid)

# 2) 发起
TASK=$(curl -s -X POST "$BASE/voice-tagging" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"uuid\":\"$UUID\"}" | jq -r .taskId)

# 3) 查询状态（转写/打标结果在 activities[].detail.markdown）
curl -s "$BASE/voice-tagging/$TASK" -H "Authorization: Bearer $KEY" | jq

# 4) 反馈
curl -s -X POST "$BASE/voice-tagging/$TASK/feedback" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"content":"客户反馈：purchase_intent 应为低。","summary":"客户修正"}' | jq
```

**时序**：上传 → 发起（立即返回 `taskId`）→ agent 后台转写+打标（期间发 2 次 webhook：`voice.transcribed`、`voice.tagged`）→ 轮询状态可看到 activity → 反馈再次派发 agent 追加 activity。
