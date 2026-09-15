# Webhook Agent 使用手册

本文面向 Agent / MCP 工具 / 网关工具实现者，说明如何正确使用
`platform-service` 的 webhook 能力。该服务背后使用自托管 Svix，但对
Agent 暴露的是 tenant-scoped facade。

## 1. 核心模型

Agent 必须按 Svix 模型理解 webhook，不要使用旧的 `topic/data` 叫法作为主协议。

| 概念 | API 字段 | 含义 |
|---|---|---|
| Tenant | `X-Tenant-Id` | 第一层硬隔离边界；每个 tenant 映射到一个 Svix Application |
| Destination / Endpoint | `endpointId`, `url` | 一个真实 callback URL；每个 endpoint 有自己的 signing secret |
| Event type | `eventType` | 事件类型，例如 `job.completed` |
| Endpoint event filter | `filterTypes` | endpoint 订阅哪些 eventType |
| Message payload | `payload` | 实际投递给 callback URL 的 JSON body |
| Channel | `channels` | Svix 路由/分组标签，用于定向或分组投递 |

投递命中规则可以理解为：

```text
same tenant/application
AND message.eventType matches endpoint.filterTypes
AND message.channels matches endpoint.channels
```

如果 endpoint 没有配置 `channels`，它是 catch-all channel 行为：只要
`eventType/filterTypes` 匹配，就可能收到所有 channel 的消息。

## 2. Agent 必须遵守的边界

1. Tenant 必须来自运行上下文，例如 `runConfig.tenantId`、认证主体或网关注入。
   不要相信普通用户输入里的 tenant。
2. Agent 只能使用 canonical 字段：
   - 创建 destination：`url`, `filterTypes`, `channels`, `description`
   - 发布 event：`eventType`, `payload`, `channels`
3. 不要使用 `topic`, `topics`, `data`。服务端短期兼容旧字段，但 Agent 文档和工具
   schema 必须使用新字段。
4. 不要使用 `endpointIds` 发布。服务端会返回 `400`。Svix 原生 publish 不支持
   direct endpoint id targeting。
5. 如果用户要求“发给某个 endpointId”，Agent 应解释：当前支持的是 channel
   targeting；严格 endpoint targeting 需要提前给该 endpoint 配置 private channel，
   例如 `endpoint.ep_xxx`。
6. 不要创建没有 `channels` 的 catch-all endpoint，除非用户明确要接收该 eventType
   下所有分组的事件。

## 3. 推荐工具

### `webhooks.register_destination`

用途：注册一个 callback endpoint。

HTTP：

```http
POST /api/v1/webhooks/destinations
X-Tenant-Id: <tenant>
Content-Type: application/json
```

请求：

```json
{
  "url": "https://client.example.com/webhook",
  "filterTypes": ["job.completed", "gate.passed"],
  "channels": ["client_a"],
  "description": "Client A production callback"
}
```

响应：

```json
{
  "endpointId": "ep_xxx",
  "secret": "whsec_xxx",
  "url": "https://client.example.com/webhook",
  "filterTypes": ["job.completed", "gate.passed"],
  "channels": ["client_a"]
}
```

Agent 注意事项：

- `secret` 只会在创建时返回一次，必须提示调用方安全保存。
- `filterTypes` 为空表示接收所有 eventType；通常不建议 Agent 默认这样做。
- `channels` 为空表示不按 channel 过滤；这会形成 catch-all endpoint。

### `webhooks.list_destinations`

用途：列出当前 tenant 的 callback endpoints。

HTTP：

```http
GET /api/v1/webhooks/destinations
X-Tenant-Id: <tenant>
```

响应示例：

```json
[
  {
    "endpointId": "ep_xxx",
    "url": "https://client.example.com/webhook",
    "filterTypes": ["job.completed"],
    "channels": ["client_a"],
    "disabled": false
  }
]
```

### `webhooks.delete_destination`

用途：删除一个 endpoint。

HTTP：

```http
DELETE /api/v1/webhooks/destinations/{endpointId}
X-Tenant-Id: <tenant>
```

响应：

```json
{"deleted": "ep_xxx"}
```

### `webhooks.publish_event`

用途：发布事件。Agent 必须使用 `eventType/payload/channels`。

HTTP：

```http
POST /api/v1/webhooks/publish
X-Tenant-Id: <tenant>
Content-Type: application/json
```

请求：

```json
{
  "eventType": "job.completed",
  "channels": ["client_a"],
  "payload": {
    "jobId": "J001",
    "status": "done"
  }
}
```

响应：

```json
{
  "messageId": "msg_xxx",
  "eventType": "job.completed",
  "channels": ["client_a"]
}
```

Agent 注意事项：

- `channels` 可选。省略时是 eventType-only broadcast。
- 如果用户说“只发给 client_a”，应使用 `channels: ["client_a"]`，并确保目标 endpoint
  已经配置了 `channels: ["client_a"]`。
- 不要传 `endpointIds`。服务端会拒绝，避免误群发。

### `webhooks.list_recent_events`

用途：查看当前 tenant 最近消息。

HTTP：

```http
GET /api/v1/webhooks/messages?limit=20
X-Tenant-Id: <tenant>
```

响应：

```json
[
  {
    "messageId": "msg_xxx",
    "eventType": "job.completed",
    "timestamp": "2026-09-15T02:01:43Z"
  }
]
```

### `webhooks.get_delivery_status`

用途：查看某条 message 对各 endpoint 的投递状态。

HTTP：

```http
GET /api/v1/webhooks/messages/{messageId}/attempts
X-Tenant-Id: <tenant>
```

响应：

```json
[
  {
    "endpointId": "ep_xxx",
    "url": "https://client.example.com/webhook",
    "status": "success"
  }
]
```

## 4. Channel 设计建议

`channels` 是业务路由标签，不是 endpoint id。

推荐用法：

| 场景 | channel 示例 |
|---|---|
| 按客户定向 | `client_a` |
| 按项目定向 | `project_123` |
| 按环境区分 | `prod`, `staging` |
| 按业务模块分组 | `finance`, `ops` |
| 同客户多接收方 | `client_a.ops`, `client_a.audit` |

如果确实需要 endpoint 级定向，可使用 private channel 模式，但当前平台不会自动维护：

```text
endpointId = ep_abc
private channel = endpoint.ep_abc
```

只有在 endpoint 显式配置了 `channels: ["endpoint.ep_abc"]` 后，publish 带
`channels: ["endpoint.ep_abc"]` 才能近似 endpoint targeting。

## 5. 常见流程

### 5.1 给 client_a 定向发送 job.completed

1. 创建 endpoint：

```json
{
  "url": "https://client-a.example.com/webhook",
  "filterTypes": ["job.completed"],
  "channels": ["client_a"]
}
```

2. 发布事件：

```json
{
  "eventType": "job.completed",
  "channels": ["client_a"],
  "payload": {"jobId": "J001"}
}
```

只有同时匹配 `job.completed` 和 `client_a` 的 endpoint 会收到。

### 5.2 client_a 收 job.completed，client_b 收 job.started

Endpoint A：

```json
{
  "url": "https://client-a.example.com/webhook",
  "filterTypes": ["job.completed"],
  "channels": ["client_a"]
}
```

Endpoint B：

```json
{
  "url": "https://client-b.example.com/webhook",
  "filterTypes": ["job.started"],
  "channels": ["client_b"]
}
```

发布给 A：

```json
{
  "eventType": "job.completed",
  "channels": ["client_a"],
  "payload": {"jobId": "J001"}
}
```

发布给 B：

```json
{
  "eventType": "job.started",
  "channels": ["client_b"],
  "payload": {"jobId": "J002"}
}
```

## 6. 字段校验

`channels` 遵循 Svix OpenAPI 约束：

- 最多 10 个
- 每个最多 128 字符
- 允许字符：`a-zA-Z0-9-_.:+`
- 空字符串会被忽略
- 重复值会被去重

## 7. 兼容字段与禁用字段

服务端短期兼容旧字段，但 Agent 不应继续使用：

| 旧字段 | 新字段 |
|---|---|
| `topic` | `eventType` |
| `topics` | `filterTypes` |
| `data` | `payload` |

禁用字段：

| 字段 | 原因 |
|---|---|
| `endpointIds` | Svix publish 不支持 endpoint id 直投；静默忽略会造成误群发 |

如果请求同时传新旧字段且值不一致，例如 `eventType=job.completed` 和
`topic=job.started`，服务端返回 `400`。

## 8. 错误处理

| 情况 | 状态码 | Agent 行为 |
|---|---:|---|
| 缺少 tenant | 400 | 检查 runConfig/auth context |
| 缺少 `eventType` | 400 | 补齐事件类型 |
| channel 格式不合法 | 400 | 按 §6 修正 |
| 传入 `endpointIds` | 400 | 改用 channels |
| endpoint/message 不存在 | 404 | 告知资源不存在或 tenant 不匹配 |
| Svix 后端不可用 | 502 | 稍后重试或升级给运维 |

## 9. 当前限制

- 目前没有 `PATCH /destinations/{endpointId}`。如果 endpoint 的 `filterTypes` 或
  `channels` 变化，只能删除后重建；这会换 `endpointId` 和 signing secret。
- 当前不自动维护 `endpointId -> private channel` 映射。
- 当前 `eventType` 是懒注册，拼写错误也可能创建新 EventType。Agent 应使用固定枚举或业务配置来减少拼写错误。

