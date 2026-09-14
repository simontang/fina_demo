# platform-service 插件化设计

- 日期：2026-09-14
- 状态：Draft（设计已定稿，待实现）
- 范围：`agent/src/agents/platform_service/`（新增）+ `agent/src/routes/`（新增代理）+ `agent/src/controllers/fileController.ts`（改造）+ `ai_web/`（新增管理页）+ `platform-service/`（服务侧改动）
- 依赖（core，另见 agentic `2026-09-14-connection-discover-tenant-context-design.md`）：`connection.discover/test` 需要租户上下文

## 1. 背景与目标

platform-service（Spring Boot，5707）提供两域能力：

- **文件**：租户级持久资源库。uuid（32 hex）寻址，逻辑路径仅元数据；内容去重、版本化；离散存储 `{tenant}/{uuid}`；下载走预签名/直链。
- **Webhook**：对自托管 Svix 的租户化 facade（destinations / publish / messages / attempts）。

本设计把这两域封装为 Axiom Lattice 插件，暴露给平台 agent 与 Open/MCP 面。

**关键区分**：平台已有沙盒文件工具（`sandbox_files_*`，domain `sandbox`，项目级临时工作区）。本设计的文件域是**统一持久资源库**，从命名到语义与之区分。

**领域定位**：platform-service 是平台基座（跨产品/租户共享）。但本次实现放在 fina_demo/agent（应用层，与 `sap_b1` 同款）。core 只提供插件开发所需的基座能力（见 §5），不内置本插件。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| 覆盖范围 | files + webhooks |
| 打包 | 两个 Plugin：`storage`、`webhooks`（独立 MCP grant 域） |
| 插件位置 | `agent/src/agents/platform_service/`（应用层） |
| 投递方 | **只能由 Svix/platform-service 投递**，插件永不直接 POST 到目标 |
| agent Connection | **连的是 platform-service**（`baseUrl + apiKey`），不是 destination |
| destination 归属 | platform-service/Svix 资源 |
| 目标选择 | connection 的 `selectedEntities`（在连接 UI 里 discover + 选中） |
| 租户来源 | 会话租户 `runConfig.tenantId`（非连接固定） |
| 签名密钥 `whsec_` | `manage_webhook` 管理域（独立 grant domain）的 register 工具在工具结果中返回 `whsec`（**已批准例外**，见 §6.5）；`webhooks` 运行时域保持 publish/只读，不接触 secret。secret 不写入 Connection |
| 破坏性操作 | 仅 `storage_delete`；`openExpose.destructive` 标注 + 入参 `confirm: true` 双重门。webhooks 四个工具均非破坏性 |
| 实现方式 | 纯 Plugin（不走 ToolLattice；插件工具不支持 needUserApprove） |
| 文件域命名 | domain `storage`，工具 `storage_*` |
| MCP 暴露 | 除 `storage_upload` 外全部上 Open 面（见 §7.4） |

## 3. 架构与目录

```
agent/src/agents/
  index.ts                      // += import "./platform_service"
  platform_service/
    index.ts                    // 副作用注册：import "./storage/plugin"; import "./webhooks/plugin"
    client.ts                   // 共享：连接解析 + 租户提取 + fetch + 错误映射
    storage/
      plugin.ts                 // Plugin type="storage"
      executors.ts              // 纯执行器（可单测）
      __tests__/storage.test.ts
    webhooks/
      plugin.ts                 // Plugin type="webhooks"
      executors.ts
      __tests__/webhooks.test.ts
    __tests__/client.test.ts
    __tests__/registration.test.ts

agent/src/routes/platformServiceProxy.ts    // 反向代理（仿 cdpProxy.ts）
agent/src/controllers/fileController.ts     // 前端上传改造（§8.1）
ai_web/                                     // webhook 管理页（§9）
platform-service/                           // 定向发布 + secret reveal（§10）
```

`meta.type` 既是插件 id，也是 Connection Store type，也是 Open grant domain。
工具名 = `{domain}_{action}`：`storage_list`、`webhooks_publish_event` …；与内置 `sandbox_files_*` 不冲突。

## 4. 连接与租户解析

两个插件共用 `client.ts`。

连接字段（`PluginConnectionFieldSchema`）：

| key | type | widget | required | default | 说明 |
|---|---|---|---|---|---|
| baseUrl | string | input | 是 | `process.env.PLATFORM_SERVICE_URL` | 服务根地址 |
| apiKey | password | password | 否 | `process.env.FILE_SERVICE_API_KEY ?? ""` | 作为 `X-Api-Key`，留空则不发送 |

`configSchema` 使用标准连接选择器：`connections: string[]`（widget `connectionSelect`；`connectionType` 由前端从 `meta.type` 自动推导，**不写入 configSchema**）、`connectAll?: boolean`。

`connection.test`：`GET {baseUrl}/actuator/health`，返回 `{ ok, message }`。
`connection.discover`（webhooks 才有）：`GET {baseUrl}/api/v1/webhooks/destinations`，带 `X-Tenant-Id`（来自 discover 的租户上下文，见 §5），映射为资源列表。

`client.ts` 职责：

```ts
// 连接解析：build 期 rawConfig._resolvedConnections[0].config
//           或 invoke 期 exeConfig.configurable.runConfig._resolvedConnections[0].config
//           → 回退 process.env
resolveConnection(rawConfig, exeConfig): { baseUrl: string; apiKey?: string }

// 租户：仅从会话取，缺失即硬失败（生产安全，不做默认租户兜底）
tenantFromExeConfig(exeConfig): string

// 网关代理用：从 Fastify request 解析（经网关注入的租户）
tenantFromRequest(request): string

// 统一请求：注入 X-Tenant-Id / X-Api-Key，JSON 解析，错误映射
request(opts): Promise<unknown>
```

## 5. 依赖：core 基座能力（必须在 core 先做）

`connection.discover` / `connection.test` 当前只收到连接配置，没有租户：

- `packages/protocols/src/PluginProtocol.ts`：`discover` / `test` 增加可选第二参数 `context?: { tenantId?: string }`。
- `packages/gateway/src/controllers/connections.ts:146,178`：调用时传入 `{ tenantId: getTenant(request) }`。
- 向后兼容（旧调用如 `sandboxPluginsMiddleware.ts:142` 不受影响）。

没有这一条，webhooks 的 `discover` 无法列出本租户 destination，`selectedEntities` scope 模型不成立。

## 6. webhooks 插件

### 6.1 连接
`type = "webhooks"`。连接 = platform-service（§4）。**不是 destination。**

### 6.2 资源发现与 scope
- `discover` 列出租户已注册的 destination：资源项 `{ id: endpointId, name: url, description: topics.join(", ") }`。**不返回 secret。**
- 在连接配置 UI 里选中资源 → 存为连接配置的 `selectedEntities`（标准字段，参考 `SemanticMetricsV2Client.ts:132`）。
- 运行时从 `_resolvedConnections[0].config.selectedEntities` 读取 scope。
- 注册/删除 destination 由 `manage_webhook` 管理域（独立 grant domain）的 register/delete 工具负责；`webhooks` 运行时域只做发布与查询。

### 6.3 工具目录

| 工具 | 映射 | scope 规则 | MCP 注解 |
|---|---|---|---|
| `webhooks_list_destinations` | `GET /api/v1/webhooks/destinations` | 只列 scope 内（若有 selectedEntities） | readOnly |
| `webhooks_publish_event` | `POST /api/v1/webhooks/publish`（带 `endpointIds`） | 目标 = `args.endpointIds` ∩ `selectedEntities`；无入参则 `selectedEntities`；`selectedEntities` 为空则 topic 默认扇出 | 写（非破坏） |
| `webhooks_list_recent_events` | `GET /api/v1/webhooks/messages?limit=` | — | readOnly |
| `webhooks_get_delivery_status` | `GET /api/v1/webhooks/messages/{messageId}/attempts` | — | readOnly |

- `webhooks_publish_event` 入参：
  - `topic`：`z.enum([...5 个契约 topic])`（`import.completed` / `gate.passed` / `decision.captured` / `job.completed` / `run.published`）。**用 enum 校验，防懒注册导致的 typo 静默成功。**
  - `data`：`z.record(z.unknown())`。
  - `endpointIds?`：`z.array(z.string()).optional()`，**只能在 scope 内收窄，不能扩大**。
- 工具结果**不包含 secret**（publish 只回 `{messageId, topic}`）。

### 6.4 定向发布依赖
`POST /publish` 必须支持 `endpointIds[]`（服务侧改动，§10）。若 Svix 不支持直接按 endpoint 发，用 **channel 过滤**（每 destination 一个 channel，publish 带 `channels`）。**实现前先验证。**

### 6.5 secret 生命周期
`whsec_` 只属于接收方，且**不写入 Connection**。经批准例外：`manage_webhook`（独立管理域）的 `register` 工具会在工具结果中返回 `whsec`，因此会进入对话与审计历史；该工具仅应授权给管理员，结果不得记录/转发。`webhooks` 运行时域及其工具结果**永不包含 secret**。

## 7. storage 插件

### 7.1 连接
`type = "storage"`。连接 = platform-service（§4）。

### 7.2 工具目录

| 工具 | 映射 | MCP 注解 |
|---|---|---|
| `storage_upload` | 沙盒路径 → 字节流 → `PUT /api/v1/files/{新uuid}` | **不上 Open** |
| `storage_list` | `GET /api/v1/files`（path, q, recursive, fileCategory, usage, from, to, page, size） | readOnly |
| `storage_get_metadata` | `GET /api/v1/files/{uuid}` | readOnly |
| `storage_get_download_url` | `POST /api/v1/files/presign`（uuid, ttlSeconds） | readOnly |
| `storage_delete` | `DELETE /api/v1/files/{uuid}` | destructive |

- `storage_get_download_url` 返回的预签名 URL 是限时凭据，会在工具描述里注明"链接即凭据"。
- `storage_delete` 是**软删单个版本**（`API.md §8`），描述里不得写成"删除文件"；入参 `confirm: true` 才执行。

### 7.3 storage_upload 细节
- 入参：`sandboxPath`（必填）、`logicalPath`、`fileName`（默认取 basename）、`fileCategory`、`usage`、`meta`。
- 沙盒解析复用 core 范式：读 `exe_config.configurable.runConfig` 的 `assistant_id/thread_id/tenantId/workspaceId/projectId`，`getSandBoxManager().getSandboxFromConfig(...)`。
- `sandbox.file.downloadFile({ file })` → Buffer（`SandboxInstance.ts:53`）。
- 大小上限：默认 **50MB**（与 nginx `client_max_body_size 50m` 对齐），可配 `STORAGE_TOOL_MAX_UPLOAD_BYTES`；超限返回明确错误，避免撑爆 agent 内存。
- uuid 客户端生成：`crypto.randomUUID().replace(/-/g,"")`；`Content-Type` 按扩展名推断，缺省 `application/octet-stream`。
- 服务端仍按 内容 sha + 逻辑路径 去重，重复上传不新建对象。
- 返回 `FileReceipt`（uuid/fullPath/sha256/size/version/deduplicated…）。

### 7.4 为什么 `storage_upload` 不上 Open
它的入参是会话沙盒路径。MCP 路径合成 runConfig 时只注入 `tenantId + _resolvedConnections`，不带 `assistant_id/thread_id/workspaceId/projectId`（`pluginExposeAdapter.ts:70-77`），外部 MCP 客户端也没有沙盒。只在 agent 会话内有意义。与 `GATEWAY-INTEGRATION.md §11.3`「上传走 REST，MCP 不承载字节」一致。

## 8. 两条上传路径

### 8.1 前端用户上传（HTTP 集成，非 LLM）
现状：前端 `POST /api/files/upload` → BFF `fileController.uploadFile` 写本地 `UPLOAD_DIR`，返回 `{id, originalName, size, mimetype}`。

改造：
- 复用 `client.ts`，把入站 multipart 流转发到 platform-service（`PUT /api/v1/files/{新uuid}` 原始流，Node 20 fetch + `duplex:"half"`，不落地、不缓冲）。
- 租户来自请求租户（经网关/代理注入，§11）。
- 响应兼容：`{ success, id: <uuid>, originalName, size, mimetype }`（`id` 即文件 uuid）。
- `FILE_UPLOAD_BACKEND=platform|local`（默认 `platform`）便于灰度/回滚。
- `uploadMultipleFiles` 同样改造。

**前置审计（必须）**：`id` 语义从"文件名"变为"uuid"。现有 `getUploadedFiles`/`deleteFile` 仍按 `UPLOAD_DIR` 工作（`fileController.ts:170,201`），新 uuid 不在其中。切换前必须全量审计所有 `files[].id` 的读者，确认无本地回读依赖。`getUploadedFiles`/`deleteFile` 重构不在本次范围。

### 8.2 agent 沙盒上传（LLM 工具）
`storage_upload`：agent 在沙盒里产出文件，LLM 只传路径，字节由工具搬运，永不进 LLM 上下文。

## 9. webhook 管理（agent 管理域 `manage_webhook`；控制台 `ai_web/`）

- 管理动作作为独立插件域 `manage_webhook` 暴露：`manage_webhook_list_destinations`、`manage_webhook_register_destination`、`manage_webhook_delete_destination`。与运行时 `webhooks` 域分离，便于按域授权。
- 映射：列出 `GET /api/v1/webhooks/destinations`、注册 `POST /api/v1/webhooks/destinations`、删除 `DELETE /api/v1/webhooks/destinations/{endpointId}`（删除须 `confirm: true`）。
- 注册成功 → 工具结果返回 `whsec_`（已批准例外，见 §2/§6.5）：密钥会进入对话与审计历史，仅应授权给管理员。
- Axiom 侧**不持久化明文 secret**；不写入 Connection Store（见 §2：destination 不是 Connection）。
- 可选后续：控制台页面（`ai_web/`）复用 `EntityListView` 类组件走 platform-service 路由，以及 rotate secret、attempts 视图。

## 10. platform-service 服务侧改动

1. **定向发布**：`POST /api/v1/webhooks/publish` 接受可选 `endpointIds: string[]`；facade 映射到 Svix 定向发送（端点过滤或 channel）。**实现前先验证 Svix OSS 能力。**
2. **可选 secret 取回 / 轮换**：`GET /api/v1/webhooks/destinations/{endpointId}/secret`（reveal）、rotate。平台侧 `SvixServerClient` 已能取 endpoint secret（创建时调用过）。
3. 现有鉴权/租户模型不变：只信网关设置的 `X-Tenant-Id`。

## 11. 网关代理与安全收口

新增 `agent/src/routes/platformServiceProxy.ts`（仿 `cdpProxy.ts`），注册进 `agent/src/gateway.ts:62`：

- 路由：`/api/webhooks/*` → `{PLATFORM_SERVICE_URL}/api/v1/webhooks/*`；如需文件管理再开 `/api/filesvc/*`。
- **覆盖/清除入站 `X-Tenant-Id`**（绝不让客户端自声明租户），注入服务凭据 `X-Api-Key`，设置 `X-User-Id`（可选审计）。
- 不透传客户端提供的 `X-Api-Key`。
- 下载/流式响应保持流式，不整体缓冲。
- nginx：把 `/api/webhooks/*` 从直连 5707 改为指向 agent 网关，并在 location 内 `proxy_set_header X-Tenant-Id "";`。

## 12. 错误处理

- 透传 platform-service 错误体 `{code,message}`，附状态码；统一为工具文本结果，不抛裸异常。
- 连接不可达：错误消息带 `baseUrl`，提示检查连接配置。
- 缺 `X-Tenant-Id`：明确报错，绝不使用默认租户兜底。
- `apiKey` 永不写入日志或返回体。
- `storage_delete` 无 `confirm: true` 时返回"需先获得用户明确确认"的拒绝。
- 上传超限 / 沙盒路径不存在：明确错误，不重试。
- Zod schema 校验失败：返回结构化错误文本（不因校验失败让整轮崩溃）。

## 13. 测试策略（jest + mock fetch，无网络）

- `client`：连接解析优先级（build `_resolvedConnections` vs invoke `_resolvedConnections` vs env）、租户提取、请求头注入、错误映射、apiKey 不泄漏。
- 执行器：每个工具的 method/path/query/body/headers；`storage_upload` mock 沙盒 `downloadFile`；`storage_delete` 无 confirm 拒绝。
- **scope 不变量**：`webhooks_publish_event` 的 `endpointIds` 必须是 `selectedEntities` 子集，越界被拒。
- **openExpose 不变量**：`openExpose` 中每个名字必须存在于 middleware 实际工具名集合（防声明漂移）。
- 注册冒烟：import `platform_service/index` 后恰好注册 `storage`、`webhooks` 两个插件。
- 代理/控制器：流透传 + 响应映射 + 租户头覆盖（伪造 `X-Tenant-Id` 必须无效）。

## 14. 不在本次范围（YAGNI）

- 原始字节下载工具（走 `storage_get_download_url`）。
- 入站 webhook 接收端。
- OpenAPI 代码生成工具定义。
- 破坏性操作的框架级审批中间件（插件工具不支持 needUserApprove，后续评估）。
- `connection.discover` 资源面板以外的资源管理抽象。
- 本地 `getUploadedFiles` / `deleteFile` 路由重构。
- 文件上传以外的 POST `/upload` multipart 路径改造（沙盒上传走 PUT）。

## 15. 开放问题 / 风险

1. **Svix 定向发送能力**（最高风险）：OSS svix-server 能否按 endpoint 定向；不行则用 channel。**必须先验证。**
2. **`id` 语义变更**：见 §8.1，切换前必须审计所有 `files[].id` 消费者。
3. **两个 Connection type**：storage 与 webhooks 各需一份指向同一 platform-service 的连接配置；用 env 默认值缓解，UI 会重复。
4. **MCP scope**：Open/MCP 路径的 `_resolvedConnections` 是否携带 `selectedEntities` 需验证；否则 MCP 侧 scope 失效（需在 OpenCredentialService 侧补齐）。
5. **沙盒默认隔离**：`getSandboxFromConfig` 省略 `vmIsolation` 时的默认策略需实现期验证。
6. **流式上传兼容**：确认 Node ≥20 且 `duplex:"half"` 可用；否则退化为 POST multipart。
7. **命名**：`storage` 作为 core/连接 type 较泛；本次在应用层注册，风险低；若未来进 core 建议改 `platform-files`。
