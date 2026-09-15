# Webhook 回调接入文档（语音打标签）

本文说明**如何准备并注册回调地址、以及如何实现接收端**，用于接收语音打标任务的 2 次回调事件。

- **回调来源**：任务处理过程中平台向已注册的 destination 投递。
- **事件**：每个任务处理期间发 **2 次**——转写完成、打标完成。两次的**事件类型都是 `job.completed`**，**阶段由 body 的 `event` 字段区分**（`voice.transcribed` / `voice.tagged`）。
- **投递语义**：Standard Webhooks 签名、at-least-once，失败自动重试。

事件 body 的完整格式见 [API.md §7](./API.md#7-webhook-回调2-次)。

---

## 1. 准备一个回调地址

- 一个**公网可达**的 URL（生产建议 HTTPS），能接收 `POST` + JSON。
- 收到后**尽快返回 `2xx`**（建议 <5s），把业务处理异步化；返回非 2xx 会被判为失败并触发重试。
- 建议按 `webhook-id`（即 messageId）做**幂等去重**（at-least-once 可能重复投递）。

## 2. 注册回调地址

调用平台 Webhook 管理接口，把上一步的 URL 注册为 destination。

```
POST https://ada.alphafina.cn/api/webhooks/destinations
X-Tenant-Id: estee_lauder
Content-Type: application/json
```

```json
{
  "url": "https://your-app.example.com/hooks/voice-tagging",
  "filterTypes": ["job.completed"],
  "description": "voice tagging callbacks"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `url` | 是 | 回调地址（http/https） |
| `filterTypes` | 否 | 只接收指定事件类型；本场景填 `["job.completed"]`。留空则接收全部 |
| `channels` | 否 | 频道过滤；本场景留空 |
| `description` | 否 | 备注 |

**响应 `200`**：

```json
{
  "endpointId": "ep_3JLY4WVEN7I6nv9SQYFSeViLFRf",
  "url": "https://your-app.example.com/hooks/voice-tagging",
  "filterTypes": ["job.completed"],
  "channels": [],
  "secret": "whsec_2OxENpeHlS8GZRep/Kh9yyt6WFs9td/t"
}
```

- `endpointId`：该 destination 的标识，用于查询/删除。
- **`secret`（`whsec_…`）：验签密钥，只在注册时返回一次，务必保存**；之后无法再获取。

> 注册需带 `X-Tenant-Id`（本项目为 `estee_lauder`）。`url` 必须公网可达，否则投递失败。

### 管理接口

| 操作 | 请求 |
|---|---|
| 列出 destinations | `GET /api/webhooks/destinations`（`X-Tenant-Id`） |
| 删除 destination | `DELETE /api/webhooks/destinations/{endpointId}` |
| 最近事件 | `GET /api/webhooks/messages?limit=20` |
| 某事件的投递状态 | `GET /api/webhooks/messages/{messageId}/attempts` |

## 3. 实现接收端

平台发给你的是标准 Webhook 请求：

```
POST /hooks/voice-tagging
webhook-id: msg_3JLuyEjvp3lU3QTSMwhehtlnMjw
webhook-timestamp: 1789316892
webhook-signature: v1,IBSl8td+sPuCKa2qk3/nitf92oObm9RlsroTqEcuHzY=
Content-Type: application/json

{ "event": "voice.transcribed", "stage_status": "success", "task_id": "…", "file_id": "…", ... }
```

> 部分实现会使用别名字段：`svix-id` / `svix-timestamp` / `svix-signature`（含义一致）。

### 3.1 验签（必须）

用注册时拿到的 `whsec_…` 密钥校验 `webhook-signature`：

1. 取 `signedContent = "{webhook-id}.{webhook-timestamp}.{原始请求体}"`（**用原始 body 字节，不要先 parse 再 stringify**）。
2. `key = base64decode(secret 去掉 `whsec_` 前缀后的部分)`。
3. `expected = base64(HMAC-SHA256(key, signedContent))`。
4. 与请求头 `webhook-signature` 中每个 `v1,<sig>` 逐项**常量时间比较**；任一匹配即通过。
5. 建议校验 `webhook-timestamp` 距当前时间不超过若干分钟，防重放。

**Node 示例**

```js
import crypto from "node:crypto";

export function verifyWebhook(secret, headers, rawBody /* Buffer|string */) {
  const id = headers["webhook-id"];
  const ts = headers["webhook-timestamp"];
  const sigHeader = headers["webhook-signature"] ?? "";
  const key = Buffer.from(secret.replace(/^whsec_/, ""), "base64");
  const expected = crypto
    .createHmac("sha256", key)
    .update(`${id}.${ts}.${rawBody}`)
    .digest("base64");
  const a = Buffer.from(expected);
  return sigHeader.split(" ").some((part) => {
    const [version, sig] = part.split(",");
    if (version !== "v1" || !sig) return false;
    const b = Buffer.from(sig);
    return a.length === b.length && crypto.timingSafeEqual(a, b);
  });
}
```

**Python 示例**

```python
import base64, hmac, hashlib

def verify_webhook(secret: str, headers: dict, raw_body: bytes) -> bool:
    key = base64.b64decode(secret[len("whsec_"):])
    signed = f'{headers["webhook-id"]}.{headers["webhook-timestamp"]}.'.encode() + raw_body
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    for part in headers.get("webhook-signature", "").split():
        version, _, sig = part.partition(",")
        if version == "v1" and hmac.compare_digest(sig, expected):
            return True
    return False
```

> 收原始 body：Express 用 `express.raw({ type: "application/json" })`；Fastify 关闭该路由的 JSON parse 或用 `addContentTypeParser` 保留原始串；Flask 用 `request.get_data()`。

### 3.2 处理与响应

```
验签通过 → 解析 body → 按 event 分发（voice.transcribed / voice.tagged）
        → 入业务队列/异步处理
        → 立即返回 200 {"ok":true}
```

- 幂等：用 `webhook-id` 去重（同一事件可能重复投递）。
- 失败重试：非 2xx 会按平台重试计划自动重投，直至成功或超出上限。

## 4. 回调事件格式

两次回调 body 结构见 [API.md §7](./API.md#7-webhook-回调2-次)。要点：

### `voice.transcribed`（转写完成）

| 字段 | 说明 |
|---|---|
| `event` | `voice.transcribed` |
| `stage_status` | `success` / `failed` |
| `task_id` / `file_id` | 任务 id / 文件 uuid |
| `text` | 转写全文 |
| `mock` / `language` / `download_ok` | 是否 mock / 语言 / 音频是否下载成功 |

### `voice.tagged`（打标完成）

| 字段 | 说明 |
|---|---|
| `event` | `voice.tagged` |
| `stage_status` | `success` / `failed` |
| `task_id` / `file_id` | 任务 id / 文件 uuid |
| `summary` | 客户画像摘要 |
| `tags` | 多维标签 `{tag, evidence}[]`：`skin_type` / `concerns` / `interested_products` / `purchase_intent` / `price_sensitivity` / `competitor_mentions` / `service_opportunities` / `custom_tags` |

## 5. 自测与排查

1. **先验通路**：注册一个临时接收地址（如 `https://webhook.site/<你的地址>`），发起一次打标任务，观察是否收到 2 条请求。
2. **查投递状态**：
   ```bash
   curl "https://ada.alphafina.cn/api/webhooks/messages?limit=5" -H "X-Tenant-Id: estee_lauder"
   curl "https://ada.alphafina.cn/api/webhooks/messages/msg_xxx/attempts" -H "X-Tenant-Id: estee_lauder"
   ```
   `attempts` 会给出每个 destination 的 `status`（`success`/`pending`/`fail`）及重试时间 `nextAttempt`。
3. **常见问题**：
   - **收不到**：`filterTypes` 未包含 `job.completed`；或 `url` 非公网可达；或用了 `channels` 过滤但发布未带对应 channel。
   - **验签失败**：密钥错、未使用**原始 body**、或比较时未去 `whsec_` 前缀做 base64 解码。
   - **重复收到**：正常（at-least-once），用 `webhook-id` 幂等处理。
   - **只收到 1 条**：转写/打标两个阶段各一条；若处理中断可能只有前者，`stage_status` 会反映失败。

## 6. 端到端串起来

```
业务应用          网关(el-ai-gateway)        平台              你的回调地址
   │ 上传/发起 ───────►│
   │                    │ 处理 ───────────────►│
   │                    │                     │ voice.transcribed ─►│ 验签+入库
   │                    │                     │ voice.tagged ──────►│ 验签+入库
   │ 查状态(可选) ─────►│◄── 任务时间线 ───────│
```

回调里的 `task_id` / `file_id` 分别对应发起返回的 `taskId` 与上传返回的 `uuid`，用于把回调关联到你的业务单据。
