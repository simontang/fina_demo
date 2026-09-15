# el-ai-gateway（雅思兰黛 AI 项目 API 网关）设计

- 日期：2026-09-15
- 状态：Draft（设计已定稿，待实现）
- 范围：新增独立项目 `el-ai-gateway/`（源码 + 测试 + Dockerfile + `.env.example` + README）。不改动 `docker-compose*.yml`、nginx。
- 依赖（现有服务，只调用不改动）：
  - `platform-service`(5707) files 域：上传 + presign。
  - `agent`(5702) `POST /api/runs`：以消息方式后台触发语音处理 agent（需登录会话 token）。
  - `agent`(5702) `/open/mcp`：agent 平台的 MCP 工具面（任务/activity 等）。

## 1. 背景与目标

雅思兰黛 AI 项目要把一个**语音文件打标签**能力以 API 形式暴露给一个应用程序。业务链路：

1. 应用上传语音文件；
2. 应用发起任务：网关把文件换成一个可访问 URL，交给 agent 平台上的语音 agent 做**语音转文本**并对文本**打标签**；
3. 处理过程中由 **agent 平台**发起 **2 次 webhook 回调**（网关不实现投递）；
4. 应用可查询任务状态、更新任务反馈（写入任务的 activity 时间线）。

网关是**面向使用场景的粗粒度业务 API 层**，负责编排 platform-service、agent 运行接口（`/api/runs`）与 agent 平台的 MCP 工具，自身尽量无状态。

## 2. 范围

**做**
- 新建独立服务 `el-ai-gateway`（Node 20 + Fastify 5 + TypeScript）。
- 4 个 REST 接口 + 入站 API Key 鉴权。
- 出站适配器：platform-service files、agent runs（登录 + background run）、agent 平台 MCP 客户端。
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
| 任务/activity | agent 平台 MCP 工具 `task_manage_task`：网关只做 **create**（带显式 `ownerId`）和 **get**；`add_activity` / `set_status` 由 agent（有身份）完成 |
| agent 触发 | `agent`(5702) `POST /api/runs`（`background:true`）：网关用 `AGENT_LOGIN_EMAIL/PASSWORD` 登录拿会话 token（缓存），带 `x-tenant-id/x-workspace-id/x-project-id` 头，消息只带 `taskId`；负责**转写 + 打标** |
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
      tasks.ts          # POST /api/v1/voice-tagging, GET /api/v1/voice-tagging/:id,
                        # POST /api/v1/voice-tagging/:id/feedback
    upstream/
      platformFiles.ts  # platform-service files 客户端（upload, presign）
      agentRuns.ts      # agent /api/runs 客户端（登录 + background run）
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

- 请求：`multipart/form-data`，字段 `file`（必填）；`path?`、`fileName?` 走 query string（避免 multipart 字段顺序问题）。
- 行为：读 `file` 流；网关生成 32-hex uuid；以**原始字节流**转 platform-service `PUT /api/v1/files/{uuid}`（避免在网关整体缓冲），**原样返回平台结果**（不做二次加工）。
- 响应 `200`：平台 upload receipt，例如
  ```json
  { "uuid": "06c1097c09694b0d94f49f1a36e84123",
    "fullPath": "ops/2026-09/stock.csv", "path": "ops/2026-09", "filename": "stock.csv",
    "version": 1, "sha256": "6310...", "md5": "a65f...", "size": 37, "mime": "text/csv",
    "status": "active", "createdAt": "...", "deduplicated": false }
  ```
- 保持 multipart 流式转发，不整体缓冲；超过 `GATEWAY_MAX_UPLOAD_BYTES` 返回 413。

### 5.2 `POST /api/v1/voice-tagging`

- 请求 `application/json`：
  ```json
  { "uuid": "可选，缺省用 VOICE_TAGGING_FILE_UUID", "title": "可选", "description": "可选", "assistantId": "可选，覆盖 VOICE_TAGGING_ASSISTANT_ID" }
  ```
- 行为：
  1. `uuid = body.uuid ?? VOICE_TAGGING_FILE_UUID`；两者都没有 → 400；
  2. `platformFiles.presign(uuid)` → 可访问 URL（失败则终止）；
  3. MCP `task_manage_task {action:"create", title, description?, status:"in_progress", ownerType:"user", ownerId:<入站 tenantId>, metadata:{uuid,url}}` → `taskId`；
  4. `POST {AGENT_RUNS_URL}`（`background:true`，Bearer 登录会话 token，头 `x-tenant-id/x-workspace-id/x-project-id`）触发 `VOICE_TAGGING_ASSISTANT_ID` 对应 agent，**消息只带 `taskId`**；采用 **fire-and-forget**：登录后异步派发即返回，失败只记日志（任务状态/activity 以 MCP 任务为准，agent 通过读取任务 metadata 拿到文件 uuid/url 并回写 activity）；
  5. 返回。
- 响应 `200`：
  ```json
  { "taskId": "task_...", "status": "in_progress", "file": { "uuid": "...", "url": "https://..." }, "agent": { "dispatched": true } }
  ```
- `title` 缺省用 `Voice tagging: {uuid}`。

### 5.3 `GET /api/v1/voice-tagging/:id`

- 行为：MCP `task_manage_task {action:"get", id}`。
- 响应 `200`：
  ```json
  { "taskId": "...", "status": "pending|in_progress|review|failed|interrupted|completed|cancelled",
    "title": "...", "result": "可选", "activities": [ { "id": "...", "action": "...", "content": "...", "createdAt": "..." } ] }
  ```
- 任务不存在 → 404 `NOT_FOUND`。

### 5.4 `POST /api/v1/voice-tagging/:id/feedback`

- 请求 `application/json`：`{ "content": "Markdown 反馈内容", "summary": "可选", "assistantId": "可选" }`
- 行为：网关**不直接写 activity**（MCP 路径缺少运行时身份，`add_activity` 会返回 `MISSING_ACTOR_IDENTITY`）；改为把反馈通过 **`POST {AGENT_RUNS_URL}`（`background:true`）** 派发给语音 agent，由 agent 以自身身份调用 `add_activity` / `set_status` 写入任务 activity。
- 响应 `200`：`{ "taskId": "...", "forwarded": true, "agent": { "dispatched": true } }`
- 任务不存在由 agent 侧感知；`content` 为空/缺失 → 400 `BAD_REQUEST`。

## 6. 鉴权与租户

- `GATEWAY_API_KEYS="key1:tenant1,key2:tenant2"`：逗号分隔，每项 `key:tenant`。
- 中间件解析 `Authorization: Bearer <key>`，查表得 principal `{ tenantId, keyLabel }`；缺失/不匹配 → 401 `UNAUTHORIZED`。
- platform-service 调用注入 `X-Tenant-Id: {tenantId}`。
- agent runs 的租户由 `AGENT_TENANT_ID/WORKSPACE_ID/PROJECT_ID` 头注入；MCP 的租户由 `a2a_` key 自身携带，网关不额外注入。
- `AUTH_DISABLED=true`（仅本地 dev）跳过校验并使用 `AUTH_DEV_TENANT`。

## 7. 出站适配器

### 7.1 `upstream/platformFiles.ts`

- `upload({ tenantId, body, uuid, filename, mime, path?, fileName? })`
  → `PUT {PLATFORM_FILES_URL}/{uuid}`，body 为原始字节流（`duplex: "half"`）；头 `X-Tenant-Id`、`Content-Type: mime`、`X-File-Path`/`X-File-Name`（有则带），若配置 `FILE_SERVICE_API_KEY` 则加 `X-Api-Key`。原样返回响应的 JSON。
- `presign({ tenantId, uuid, ttlSeconds? })`
  → `POST {PLATFORM_FILES_URL}/presign`，JSON `{uuid,ttlSeconds?}` → `{url, kind, expiresInSeconds}`。
- `PLATFORM_FILES_URL` 为**含路径前缀的完整 files 基址**：本地 `http://127.0.0.1:5707/api/v1/files`；线上经 nginx 为 `https://ada.alphafina.cn/api/filesvc`（线上当前直接信任调用方 `X-Tenant-Id`，实测上传/预签名/下载全链路通过）。

### 7.2 `upstream/agentRuns.ts`

- `login()`
  → `POST {AGENT_AUTH_URL}`，JSON `{email, password}`（`AGENT_LOGIN_EMAIL/PASSWORD`）→ `{data:{token}}`；token 为 agent 控制台的签名会话 token（默认 24h）。解码 payload 的 `exp` 缓存，临近过期（<60s）自动重登。
- `startRun({ assistantId, threadId, text, taskId, timeoutMs? })`
  → `POST {AGENT_RUNS_URL}`（默认 `…/api/runs`）
  头：`Authorization: Bearer <session token>`、`x-tenant-id`、`x-workspace-id`、`x-project-id`、`Content-Type`。
  body：
  ```json
  { "assistant_id": "<assistantId>", "thread_id": "<uuid>",
    "message": "<text>", "background": true,
    "custom_run_config": { "taskId": "<taskId>" } }
  ```
  返回 `202 { success, messageId, queued }`（`background:true` 时不等待 agent 跑完）。
- 收到 401 时清缓存重登并重试一次。
- 为什么不用 A2A：A2A `message/send` 走 standard 路由的 governed task 生命周期，要求 agent 在运行中更新该 A2A 任务；而 Open/MCP 路径无运行时身份、scope 也对不上，导致任务 403/失败。`/api/runs` 是纯消息派发，配 `x-tenant-id/x-workspace-id/x-project-id` 与登录身份（`user_id`）后，agent 能直接 `manage_task` 到网关建的同一个任务上。
- 触发是“尽力而为”：失败不回滚已创建的 MCP 任务，只记日志、`taskId` 仍返回（见 §9）。

### 7.3 `upstream/mcp.ts` + `upstream/taskTools.ts`

- 使用 `@modelcontextprotocol/sdk` 的 `Client` + `StreamableHTTPClientTransport` 连接 `MCP_SERVER_URL`，头 `Authorization: Bearer {MCP_API_KEY}`。
- 会话：`initialize`（保存服务端返回的 `Mcp-Session-Id`）→ `notifications/initialized` → `tools/call`；连接复用，失败重连。
- `taskTools.ts` 收敛动作：
  - `createTask({ title, description?, status?:"in_progress", ownerId, metadata })` → `{ taskId }`；
  - `getTask({ id })`。
- MCP 工具名：`task_manage_task`（core 内置）。入参以实测为准，响应为 `{success, data:{...}}`；`create` 必须带 `ownerType:"user"` + `ownerId`（Open MCP 路径无运行时身份，缺 `ownerId` 会因 `owner_id` NOT NULL 失败）。`add_activity` / `set_status` 不在网关侧调用（见 §5.4）。

## 8. 配置（`.env.example`）

```
PORT=5708
GATEWAY_API_KEYS=dev_key:estee_lauder
AUTH_DISABLED=false
AUTH_DEV_TENANT=estee_lauder

PLATFORM_FILES_URL=http://127.0.0.1:5707/api/v1/files
FILE_SERVICE_API_KEY=
GATEWAY_MAX_UPLOAD_BYTES=52428800

AGENT_RUNS_URL=http://127.0.0.1:5702/api/runs
AGENT_AUTH_URL=http://127.0.0.1:5702/api/auth/login
AGENT_LOGIN_EMAIL=simon@fina.com
AGENT_LOGIN_PASSWORD=
AGENT_TENANT_ID=estee_lauder
AGENT_WORKSPACE_ID=default-workspace
AGENT_PROJECT_ID=default
VOICE_TAGGING_ASSISTANT_ID=voice-tagging-agent
VOICE_TAGGING_FILE_UUID=2ccf6fef88b64a16b62fe491a8f7a132
VOICE_TAGGING_MESSAGE_TEMPLATE=
AGENT_TRIGGER_TIMEOUT_MS=600000

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
| agent 触发/反馈派发失败（fire-and-forget） | 不返回 | 仅记录日志（`[voice-tagging] agent run dispatch failed ...`），响应始终 200 + `taskId` |
| 上游超时 | 504 | `UPSTREAM_TIMEOUT` |
| 任务不存在 | 404 | `NOT_FOUND` |
| 其它未预期错误 | 500 | `INTERNAL_ERROR` |

## 10. 测试

Vitest。注入 mock 的 fetch / MCP transport：

- `config`：必填缺失、格式非法、合法解析。
- `auth`：合法 key → principal；缺失/错误 → 401；`AUTH_DISABLED` 行为。
- `platformFiles`：upload 透传、presign 成功/失败映射。
- `agentRuns`：登录 → background run 的请求（头/body）、token 缓存、401 重登重试。
- `mcp`/`taskTools`：mock transport 下会话握手、`tools/call` 参数与结果解析。
- 路由级：`app.inject()` 覆盖 4 个端点的成功与失败（含派发失败但 taskId 返回）。
- `scripts/smoke.sh`：对真实上游手动跑 上传→发起→查状态→写反馈。

## 11. 交付物与运行

- 源码 + 测试；`pnpm dev|build|start|test|lint|typecheck|mcp:probe`。
- `Dockerfile`：node:20-alpine 多阶段构建，暴露 5708。
- `README.md`：环境变量表、启动方式、4 个 curl 示例、MCP 探查步骤。
- `.env.example`。

## 12. 待办 / 开放项（不阻塞骨架）

1. **语音打标 agent 的 assistantId**：`VOICE_TAGGING_ASSISTANT_ID` 取值（当前 `voice-tagging-agent`）。
6. **agent 登录凭据的运维**：当前用 `AGENT_LOGIN_EMAIL/PASSWORD` 登录拿 24h 会话 token（缓存、401 重登）。生产建议改用专用服务账号或改由平台签发长期 token；密码不要进仓库。
2. **平台 task 工具限制（已实测）**：Open MCP 路径无运行时身份 → `add_activity` 报 `MISSING_ACTOR_IDENTITY`、`set_status` 报 `TASK_STATUS_UNSUPPORTED`。因此网关只 create/get；agent 侧负责 activity 与状态流转。若平台后续为 Open MCP 注入身份，可把 §5.4 改回直接 `add_activity`。
3. **endpoint 4 反馈字段语义**：当前把 `content` 作为 Markdown 反馈转发给 agent；如应用需要结构化反馈（评分/修正标签）再扩展。
4. **2 次 webhook 的事件类型与 payload**：由 agent 平台侧工具决定（可用枚举 `import.completed/gate.passed/decision.captured/job.completed/run.published`），网关不关心。
5. 后续是否把入站鉴权换成平台签发的 `a2a_` key 校验（复用 `lattice_a2a_api_keys`）。
