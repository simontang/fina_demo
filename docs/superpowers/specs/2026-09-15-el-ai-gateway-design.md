# el-ai-gateway（雅思兰黛 AI 项目 API 网关）设计

- 日期：2026-09-15
- 状态：Draft（设计已定稿，待实现）
- 范围：新增独立项目 `el-ai-gateway/`（源码 + 测试 + Dockerfile + `.env.example` + README）。不改动 `docker-compose*.yml`、nginx。
- 依赖（现有服务，只调用不改动）：
  - `platform-service`(5707) files 域：上传 + presign。
  - `agent`(5702) A2A 端点：触发语音处理 agent。
  - `agent`(5702) `/open/mcp`：agent 平台的 MCP 工具面（任务/activity 等）。

## 1. 背景与目标

雅思兰黛 AI 项目要把一个**语音文件打标签**能力以 API 形式暴露给一个应用程序。业务链路：

1. 应用上传语音文件；
2. 应用发起任务：网关把文件换成一个可访问 URL，交给 agent 平台上的语音 agent 做**语音转文本**并对文本**打标签**；
3. 处理过程中由 **agent 平台**发起 **2 次 webhook 回调**（网关不实现投递）；
4. 应用可查询任务状态、更新任务反馈（写入任务的 activity 时间线）。

网关是**面向使用场景的粗粒度业务 API 层**，负责编排 platform-service、A2A 与 agent 平台的 MCP 工具，自身尽量无状态。

## 2. 范围

**做**
- 新建独立服务 `el-ai-gateway`（Node 20 + Fastify 5 + TypeScript）。
- 4 个 REST 接口 + 入站 API Key 鉴权。
- 出站适配器：platform-service files、A2A JSON-RPC、agent 平台 MCP 客户端。
- 单测 + 手动 smoke 脚本 + Dockerfile + README + `.env.example`。

**不做（首版）**
- 不改 nginx / compose，不接公网反向代理（独立项目，端口直连）。
- **不实现 webhook 投递**：2 次回调由 agent 平台通过工具完成。
- 不复用平台 `a2a_` key 做入站校验（入站用环境变量静态 key；`a2a_` key 仅用于出站）。
- 不做本地任务存储：任务/activity 都存在 agent 平台（MCP `task_manage_task`）。
- 不暴露 MCP server（网关是 MCP **客户端**）。
- 不接入 prediction_app / metrics-server / b1s / cdp-service / document_service。

## 3. 决策总览

| 项 | 决策 |
|---|---|
| 定位 | 独立边缘业务 API 网关（语音打标场景的 4 个语义接口） |
| 技术栈 | Node 20 + Fastify 5 + TypeScript；pnpm + tsup；Vitest |
| 目录/服务名 | `el-ai-gateway` |
| 端口 | `5708`（不进 compose） |
| 对外前缀 | `/api/v1` |
| 入站鉴权 | 环境变量静态 API Key（`Authorization: Bearer`），key → tenant 映射来自配置 |
| 文件后端 | `platform-service`(5707) files：upload 原样返回；presign 拿可访问 URL |
| 任务/activity | agent 平台 MCP 工具 `task_manage_task`（创建一个 user-owned 任务，状态与 activity 都落在这里） |
| A2A 上游 | `agent`(5702) `/api/a2a/agents/{assistantId}/jsonrpc`，Bearer `a2a_` key；负责触发**转写 + 打标** |
| webhook | 由 agent 平台在任务处理中触发 2 次；**网关不实现** |
| MCP 上游 | `agent`(5702) `/open/mcp`（Streamable HTTP），Bearer `a2a_` key；网关是客户端 |
| 状态存储 | 网关无状态；任务状态即 MCP 任务状态 |
| 错误 | 统一信封 `{code,message,upstream?}` |

## 4. 架构与目录

```
el-ai-gateway/
  src/
    index.ts            # 入口：加载配置 → 启动
    server.ts           # 构建 Fastify 实例、挂中间件与路由
    config.ts           # zod 校验环境变量
    auth.ts             # 入站 Bearer key → principal { tenantId, keyLabel }
    routes/
      files.ts          # POST /api/v1/files
      tasks.ts          # POST /api/v1/tasks, GET /api/v1/tasks/:id,
                        # POST /api/v1/tasks/:id/feedback
    upstream/
      platformFiles.ts  # platform-service files 客户端（upload, presign）
      a2a.ts            # A2A JSON-RPC 客户端（message/send, tasks/get）
      mcp.ts            # MCP 客户端（initialize 会话 + tools/call）
      taskTools.ts      # task_manage_task 的薄封装（create/get/addActivity）
    lib/
      errors.ts         # 错误信封与映射
      http.ts           # 带超时的 fetch 封装
  test/                 # Vitest 单测（按模块）
  scripts/
    smoke.sh            # 打真实上游的手动验收脚本
    mcp-probe.ts        # 连接 MCP server 打印 tools/list
  Dockerfile
  .env.example
  package.json
  README.md
```

分层原则：路由只做参数校验与编排；每个上游一个适配器，适配器可独立单测（注入 fetch/transport）。`taskTools.ts` 把 MCP 的 `task_manage_task` 动作收敛成领域方法（`createTask` / `getTask` / `addActivity`），路由不直接拼 MCP 参数。

## 5. 对外 API 规格

全部需要 `Authorization: Bearer <key>`。

### 5.1 `POST /api/v1/files`

- 请求：`multipart/form-data`，字段 `file`（必填）、`path?`、`fileName?`。
- 行为：转 platform-service `POST /api/v1/files/upload`，**原样返回平台结果**（不做二次加工）。
- 响应 `200`：平台 upload receipt，例如
  ```json
  { "uuid": "06c1097c09694b0d94f49f1a36e84123",
    "fullPath": "ops/2026-09/stock.csv", "path": "ops/2026-09", "filename": "stock.csv",
    "version": 1, "sha256": "6310...", "md5": "a65f...", "size": 37, "mime": "text/csv",
    "status": "active", "createdAt": "...", "deduplicated": false }
  ```
- 保持 multipart 流式转发，不整体缓冲；超过 `GATEWAY_MAX_UPLOAD_BYTES` 返回 413。

### 5.2 `POST /api/v1/tasks`

- 请求 `application/json`：
  ```json
  { "uuid": "<上传返回的 uuid>", "title": "可选", "description": "可选" }
  ```
- 行为：
  1. `platformFiles.presign(uuid)` → 可访问 URL（失败则终止）；
  2. MCP `task_manage_task {action:"create", title, description?, status:"in_progress", metadata:{uuid,url}}` → `taskId`；
  3. A2A `message/send` 调 `A2A_VOICE_TAGGING_ASSISTANT_ID` 对应 agent，消息含 URL（默认模板，可被 `A2A_MESSAGE_TEMPLATE` 覆盖），让 agent **转写 + 打标**；
  4. 返回。
- 响应 `200`：
  ```json
  { "taskId": "task_...", "status": "in_progress", "file": { "uuid": "...", "url": "https://..." } }
  ```
- `title` 缺省用 `Voice tagging: {filename 或 uuid}`。

### 5.3 `GET /api/v1/tasks/:id`

- 行为：MCP `task_manage_task {action:"get", id}`。
- 响应 `200`：
  ```json
  { "taskId": "...", "status": "pending|in_progress|review|failed|interrupted|completed|cancelled",
    "title": "...", "result": "可选", "activities": [ { "id": "...", "action": "...", "content": "...", "createdAt": "..." } ] }
  ```
- 任务不存在 → 404 `NOT_FOUND`。

### 5.4 `POST /api/v1/tasks/:id/feedback`

- 请求 `application/json`：`{ "content": "Markdown 反馈内容", "summary": "可选" }`
- 行为：MCP `task_manage_task {action:"add_activity", id, content, summary?}`。
- 响应 `200`：`{ "taskId": "...", "added": true }`
- 任务不存在 → 404 `NOT_FOUND`；`content` 为空/缺失 → 400 `BAD_REQUEST`。

## 6. 鉴权与租户

- `GATEWAY_API_KEYS="key1:tenant1,key2:tenant2"`：逗号分隔，每项 `key:tenant`。
- 中间件解析 `Authorization: Bearer <key>`，查表得 principal `{ tenantId, keyLabel }`；缺失/不匹配 → 401 `UNAUTHORIZED`。
- platform-service 调用注入 `X-Tenant-Id: {tenantId}`。
- A2A / MCP 的租户由其各自的 `a2a_` key 自身携带，网关不额外注入。
- `AUTH_DISABLED=true`（仅本地 dev）跳过校验并使用 `AUTH_DEV_TENANT`。

## 7. 出站适配器

### 7.1 `upstream/platformFiles.ts`

- `upload({ tenantId, fileStream, filename, mime, path?, fileName? })`
  → `POST {PLATFORM_FILES_BASE_URL}/api/v1/files/upload`；头 `X-Tenant-Id`，若配置 `FILE_SERVICE_API_KEY` 则加 `X-Api-Key`。原样返回响应的 JSON。
- `presign({ tenantId, uuid, ttlSeconds? })`
  → `POST {PLATFORM_FILES_BASE_URL}/api/v1/files/presign`，JSON `{uuid,ttlSeconds?}` → `{url, kind, expiresInSeconds}`。

### 7.2 `upstream/a2a.ts`

- `sendTask({ assistantId, text })`
  → `POST {A2A_BASE_URL}/api/a2a/agents/{assistantId}/jsonrpc`
  头：`Authorization: Bearer {A2A_API_KEY}`、`Content-Type: application/json`。
  body：
  ```json
  { "jsonrpc": "2.0", "id": "<uuid>",
    "method": "message/send",
    "params": { "message": { "role": "user", "messageId": "<uuid>",
      "parts": [ { "kind": "text", "text": "<text>" } ] } } }
  ```
  从 result 提取 `id`（taskId）与 `status.state`。
- `getTask({ assistantId, taskId })`：JSON-RPC `tasks/get`，params `{ "id": taskId }`（备用）。
- 首版用非流式 `message/send`；若上游返回 SSE，适配器做一次聚合读取。
- 触发是“尽力而为”：A2A 调用失败不回滚已创建的 MCP 任务，错误上报但 `taskId` 仍返回（见 §9）。

### 7.3 `upstream/mcp.ts` + `upstream/taskTools.ts`

- 使用 `@modelcontextprotocol/sdk` 的 `Client` + `StreamableHTTPClientTransport` 连接 `MCP_SERVER_URL`，头 `Authorization: Bearer {MCP_API_KEY}`。
- 会话：`initialize`（保存服务端返回的 `Mcp-Session-Id`）→ `notifications/initialized` → `tools/call`；连接复用，失败重连。
- `taskTools.ts` 收敛动作：
  - `createTask({ title, description?, status?:"in_progress", metadata })`；
  - `getTask({ id })`；
  - `addActivity({ id, content, summary? })`。
- MCP 工具名：`task_manage_task`（core 内置）。字段以实测 schema 为准（`action/id/title/description/status/metadata/content/summary`）。

## 8. 配置（`.env.example`）

```
PORT=5708
GATEWAY_API_KEYS=dev_key:tenant_demo
AUTH_DISABLED=false
AUTH_DEV_TENANT=tenant_demo

PLATFORM_FILES_BASE_URL=http://127.0.0.1:5707
FILE_SERVICE_API_KEY=
GATEWAY_MAX_UPLOAD_BYTES=52428800

A2A_BASE_URL=http://127.0.0.1:5702
A2A_API_KEY=a2a_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
A2A_VOICE_TAGGING_ASSISTANT_ID=
A2A_MESSAGE_TEMPLATE=

MCP_SERVER_URL=http://127.0.0.1:5702/open/mcp
MCP_API_KEY=a2a_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

UPSTREAM_TIMEOUT_MS=30000
```

`config.ts` 用 zod 校验；缺必填项启动即失败并打印明确信息。

## 9. 错误处理

统一信封：
```json
{ "code": "UPSTREAM_ERROR", "message": "...", "upstream": { "status": 401, "code": "API_KEY_INVALID" } }
```

| 场景 | 状态 | code |
|---|---|---|
| 入站 key 缺失/错误 | 401 | `UNAUTHORIZED` |
| 参数非法（缺 file/uuid、feedback content 为空等） | 400 | `BAD_REQUEST` |
| 上传超限 | 413 | `PAYLOAD_TOO_LARGE` |
| presign 失败 / 上游业务错误 | 502 | 透传上游 code（如 `TENANT_REQUIRED`、`NOT_FOUND`） |
| MCP 工具调用失败 | 502 | `MCP_ERROR` |
| A2A 触发失败（任务已建） | 502 | `A2A_ERROR`（响应仍含 `taskId`） |
| 上游超时 | 504 | `UPSTREAM_TIMEOUT` |
| 任务不存在 | 404 | `NOT_FOUND` |
| 其它未预期错误 | 500 | `INTERNAL_ERROR` |

## 10. 测试

Vitest。注入 mock 的 fetch / MCP transport：

- `config`：必填缺失、格式非法、合法解析。
- `auth`：合法 key → principal；缺失/错误 → 401；`AUTH_DISABLED` 行为。
- `platformFiles`：upload 透传、presign 成功/失败映射。
- `a2a`：`message/send` 参数与响应解析；错误映射。
- `mcp`/`taskTools`：mock transport 下会话握手、`tools/call` 参数与结果解析。
- 路由级：`app.inject()` 覆盖 4 个端点的成功与失败（含 A2A 失败但 taskId 返回）。
- `scripts/smoke.sh`：对真实上游手动跑 上传→发起→查状态→写反馈。

## 11. 交付物与运行

- 源码 + 测试；`pnpm dev|build|start|test|lint|typecheck|mcp:probe`。
- `Dockerfile`：node:20-alpine 多阶段构建，暴露 5708。
- `README.md`：环境变量表、启动方式、4 个 curl 示例、MCP 探查步骤。
- `.env.example`。

## 12. 待办 / 开放项（不阻塞骨架）

1. **语音打标 agent 的 assistantId**：`A2A_VOICE_TAGGING_ASSISTANT_ID` 取值。
2. **谁把任务置为 completed**：由 agent 平台在完成后更新该任务，还是网关轮询 A2A 后 `set_status`；首版按“agent 平台更新、网关只读”实现，若不符再加网关侧 `set_status`。
3. **endpoint 4 反馈字段语义**：默认 Markdown `content`；如应用需要结构化反馈（评分/修正标签）再扩展。
4. **2 次 webhook 的事件类型与 payload**：由 agent 平台侧工具决定（可用枚举 `import.completed/gate.passed/decision.captured/job.completed/run.published`），网关不关心。
5. 后续是否把入站鉴权换成平台签发的 `a2a_` key 校验（复用 `lattice_a2a_api_keys`）。
