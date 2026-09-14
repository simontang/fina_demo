# el-ai-gateway（雅思兰黛 AI 项目 API 网关）设计

- 日期：2026-09-15
- 状态：Draft（设计已定稿，待实现）
- 范围：新增独立项目 `el-ai-gateway/`（源码 + 测试 + Dockerfile + `.env.example` + README）。不改动 `docker-compose*.yml`、nginx。
- 依赖（现有服务，只调用不改动）：`platform-service`(5707) files 域、`agent`(5702) A2A 端点、外部 MCP server（地址由环境变量提供）。

## 1. 背景与目标

雅思兰黛 AI 项目需要一个**面向使用场景的业务 API 网关**：外部调用方拿平台签发的凭据，调用少量粗粒度语义接口，由网关在内部编排现有服务与协议。

首版只做 3 个语义接口：

1. **上传文件 → 得到可访问 URL**
2. **通过 A2A 触发 agent 任务 → 得到一个任务**
3. **查询任务状态**

目标是先把一条完整链路（文件 → 触发 → 查状态）跑通，后续再扩。

## 2. 范围

**做**
- 新建独立服务 `el-ai-gateway`（Node 20 + Fastify 5 + TypeScript）。
- 3 个 REST 语义接口 + 入站 API Key 鉴权。
- 出站适配器：platform-service files、A2A JSON-RPC、MCP 客户端。
- 单测 + 手动 smoke 脚本 + Dockerfile + README + `.env.example`。

**不做（首版）**
- 不改 nginx / compose，不接公网反向代理（独立项目，端口直连）。
- 不复用平台 `a2a_` key 做入站校验（入站用环境变量静态 key；`a2a_` key 仅用于出站）。
- 不建数据库、不做任务登记/幂等/历史（任务状态直接透传上游 A2A）。
- 不暴露 MCP server（网关是 MCP **客户端**）。
- 不接入 prediction_app / metrics-server / b1s / cdp-service / document_service（后续扩展）。

## 3. 决策总览

| 项 | 决策 |
|---|---|
| 定位 | 独立边缘业务 API 网关（面向使用场景的粗粒度接口） |
| 技术栈 | Node 20 + Fastify 5 + TypeScript；pnpm + tsup；Vitest |
| 目录/服务名 | `el-ai-gateway` |
| 端口 | `5708`（仅服务自身，不进 compose） |
| 对外前缀 | `/api/v1` |
| 入站鉴权 | 环境变量静态 API Key（`Authorization: Bearer`），key → tenant 映射来自配置 |
| 租户 | 入站 key 解析出 `tenantId`；出站注入 `X-Tenant-Id`（platform-service 必需） |
| 文件后端 | `platform-service`(5707) files：upload 拿 uuid，再 presign 拿 URL |
| A2A 上游 | 本仓库 `agent`(5702) `/api/a2a/agents/{assistantId}/jsonrpc`，Bearer `a2a_` key |
| A2A 方法 | `message/send`（非流式触发）、`tasks/get`（查状态） |
| MCP 上游 | 外部 MCP server，地址/凭据由环境变量提供；网关用 `@modelcontextprotocol/sdk` 作为客户端 |
| MCP 使用点 | 在 3 个语义接口内部按需调用；**具体 tool 名/入参待探活后填** |
| 状态存储 | 无（无状态透传） |
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
      tasks.ts          # POST /api/v1/tasks, GET /api/v1/tasks/:taskId
    upstream/
      platformFiles.ts  # platform-service files 客户端
      a2a.ts            # A2A JSON-RPC 客户端
      mcp.ts            # MCP 客户端（tools/list, tools/call）
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

分层原则：路由只做参数校验与编排；每个上游一个适配器，适配器可独立单测（注入 fetch/transport）。

## 5. 对外 API 规格

全部需要 `Authorization: Bearer <key>`。

### 5.1 `POST /api/v1/files`

- 请求：`multipart/form-data`，字段 `file`（必填）、`path?`、`fileName?`。
- 流程：`platformFiles.upload()` → 拿 `uuid` → `platformFiles.presign()` → 拿 `url`。
- 响应 `200`：
  ```json
  {
    "uuid": "06c1097c09694b0d94f49f1a36e84123",
    "filename": "stock.csv",
    "size": 37,
    "mime": "text/csv",
    "path": "ops/2026-09",
    "url": "https://...",
    "kind": "presigned",
    "expiresInSeconds": 600
  }
  ```
- 保持 multipart 流式转发，不整体缓冲；超过 `GATEWAY_MAX_UPLOAD_BYTES` 返回 413。

### 5.2 `POST /api/v1/tasks`

- 请求 `application/json`：
  ```json
  { "assistantId": "optional", "text": "触发内容" }
  ```
  `assistantId` 缺省用 `A2A_DEFAULT_ASSISTANT_ID`。
- 流程：A2A JSON-RPC `message/send`。
- 响应 `200`：`{ "taskId": "...", "status": "submitted|working|...", "raw": {...} }`

### 5.3 `GET /api/v1/tasks/:taskId`

- 流程：A2A JSON-RPC `tasks/get`，params `{ "id": "<taskId>" }`（可选 `historyLength`）。
- 响应 `200`：`{ "taskId": "...", "status": "...", "artifacts": [...], "raw": {...} }`
- 上游找不到任务 → 404 `NOT_FOUND`。

## 6. 鉴权与租户

- `GATEWAY_API_KEYS="key1:tenant1,key2:tenant2"`：逗号分隔，每项 `key:tenant`。
- 中间件解析 `Authorization: Bearer <key>`，查表得 principal `{ tenantId, keyLabel }`；缺失/不匹配 → 401 `UNAUTHORIZED`。
- 出站把 `tenantId` 注入 `X-Tenant-Id`（platform-service 必需）。
- A2A / MCP 的租户由其各自的 `a2a_` key 自身携带，网关不额外注入租户。
- `AUTH_DISABLED=true`（仅本地 dev）跳过校验并使用 `AUTH_DEV_TENANT` 作为租户。

## 7. 出站适配器

### 7.1 `upstream/platformFiles.ts`

- `upload({ tenantId, fileStream, filename, mime, path?, fileName? })`
  → `POST {PLATFORM_FILES_BASE_URL}/api/v1/files/upload`
  头：`X-Tenant-Id: {tenantId}`，若配置 `FILE_SERVICE_API_KEY` 则加 `X-Api-Key`。
- `presign({ tenantId, uuid, ttlSeconds? })`
  → `POST {PLATFORM_FILES_BASE_URL}/api/v1/files/presign`，JSON `{uuid,ttlSeconds?}`。
- 上游错误信封 `{code,message}` 原样映射到网关错误（保留 code，如 `TENANT_REQUIRED`）。

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
  从 result 提取 `id`（taskId）与 `status.state`。若上游直接返回终态 Message（无 task）而非 Task，则按 502 `UPSTREAM_ERROR` 处理并在 `message` 中说明（首版只支持 Task 语义）。
- `getTask({ assistantId, taskId })`
  → JSON-RPC `tasks/get`，params `{ "id": taskId }`。
- MVP 用非流式 `message/send`；如上游返回 SSE 流，适配器做一次聚合读取。

### 7.3 `upstream/mcp.ts`

- `@modelcontextprotocol/sdk` 的 `Client` + `StreamableHTTPClientTransport` 连接 `MCP_SERVER_URL`（例如 agent 的 `/open/mcp`），头 `Authorization: Bearer {MCP_API_KEY}`。
- `listTools()` / `callTool(name, args)`；单会话懒连接，失败重连。
- **具体调用点与 tool 名待定**：先用 `pnpm mcp:probe`（`scripts/mcp-probe.ts`）连上去打印 `tools/list`，再决定在哪个语义接口里调哪个 tool，并把调用接到路由编排中。

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
A2A_DEFAULT_ASSISTANT_ID=

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
| 参数非法（缺 file、assistantId 非法等） | 400 | `BAD_REQUEST` |
| 上传超限 | 413 | `PAYLOAD_TOO_LARGE` |
| 上游业务错误 | 502 | 透传上游 code（如 `TENANT_REQUIRED`） |
| 上游超时 | 504 | `UPSTREAM_TIMEOUT` |
| 任务不存在 | 404 | `NOT_FOUND` |
| 其它未预期错误 | 500 | `INTERNAL_ERROR` |

## 10. 测试

Vitest。注入 mock 的 fetch / MCP transport：

- `config`：必填缺失、格式非法、合法解析。
- `auth`：合法 key → principal；缺失/错误 → 401；`AUTH_DISABLED` 行为。
- `platformFiles`：upload+presign 成功链路；上游错误映射。
- `a2a`：`message/send`、`tasks/get` 参数与响应解析；错误映射。
- `mcp`：wrapper 在 mock transport 下的 `tools/list` / `tools/call`。
- 路由级：`app.inject()` 覆盖 3 个端点的成功与失败。
- `scripts/smoke.sh`：对真实上游手动跑上传→触发→查状态。

## 11. 交付物与运行

- 源码 + 测试；`pnpm dev|build|start|test|lint|typecheck|mcp:probe`。
- `Dockerfile`：node:20-alpine 多阶段构建，暴露 5708。
- `README.md`：环境变量表、启动方式、3 个 curl 示例、MCP 探查步骤。
- `.env.example`。

## 12. 待办 / 开放项（不阻塞首版骨架）

1. **MCP 具体 tool 名与入参**：探活 `MCP_SERVER_URL` 后确定，并接入相应语义接口。
2. **`A2A_DEFAULT_ASSISTANT_ID`**：默认触发哪个 agent（待定）。
3. **租户映射实际值**：`GATEWAY_API_KEYS` 与 platform-service 的 `X-Tenant-Id` 取值需对齐真实租户。
4. 后续是否把入站鉴权换成平台签发的 `a2a_` key 校验（复用 `lattice_a2a_api_keys`）。
