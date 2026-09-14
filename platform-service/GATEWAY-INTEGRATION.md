# platform-service → API Gateway 交接文档

- 交接对象：平台 api/mcp gateway 的开发者
- 本文范围：platform-service 已上线的能力、网关需要实现什么、切换步骤与验收方法
- 相关文档：[API.md](./API.md)（文件接口全量 request/response）、[WEBHOOK-API.md](./WEBHOOK-API.md)（webhook 接口）、[DEPLOY.md](./DEPLOY.md)（部署与踩坑）

---

## 1. 当前线上状态（已可用，可直接联调）

platform-service 已部署到生产并全链路验证通过。

| 项 | 值 |
|---|---|
| 生产域名 | `https://ada.alphafina.cn` |
| 服务器 | `ssh deploy@ada.alphafina.cn`（应用目录 `/app/fina_demo`；docker 操作需 `sudo`） |
| 服务监听 | `127.0.0.1:5707`（**仅本机**，公网不可直达） |
| 容器 | `fina_demo-platform-service-1`、`fina_demo-svix-server-1` |
| 镜像 | `ghcr.io/409zhangshu/fina-demo-platform-service:latest`（CI 构建）、`ghcr.io/409zhangshu/svix-server:latest`（CI 镜像转发） |
| 依赖 | RDS（`postgres` 库存文件元数据、`svix` 库存 webhook 状态）、TOS 对象存储、`document-redis`（svix 队列 db 3） |

**当前 nginx 路由**（`/etc/nginx/sites-enabled/default`，80 与 443 两个 server 块各一份）：

| 公网前缀 | 转发到 |
|---|---|
| `/api/filesvc/*` | `http://127.0.0.1:5707/api/v1/files/*` |
| `/api/webhooks/*` | `http://127.0.0.1:5707/api/v1/webhooks/*` |
| `/portal/*` | `http://127.0.0.1:5707/portal/*` |

**已验证通过**（全新租户，公网实测）：上传 → 列表 → 下载 → 下载链接（`kind=presigned`，指向 TOS）→ 建投递目标 → 发布事件 → Portal 200 → 未匹配路径 404。

## 2. 当前安全状态（网关接入要解决的核心问题）

**这三个前缀现在完全绕过网关，且不校验任何凭据**：

- nginx 直连 `127.0.0.1:5707`，**没有经过 agent 网关（5702）**，网关也没有这两个前缀的路由；
- nginx **不清洗**入站的 `X-Tenant-Id` → **调用方自己声明租户即可读写该租户数据**，包括 `POST /upload`、`PUT /{uuid}` 这类写操作；
- `FILE_SERVICE_API_KEY` 当前**未设置**（有意为之：网关未接入前设了会 401 掉所有直连调用）；
- `/portal/` 公网可达且**自身无登录**，能增删投递目标（跨租户控制面）。

这是**测试阶段的有意选择**，但是网关接入时第一批要收口的事项。

## 3. platform-service 的设计定位（网关需理解的前提）

- 它**不对租户做认证**：只读 `X-Tenant-Id`，据此做行级隔离（MyBatis-Plus 租户拦截器为每条 SQL 追加 `tenant_id`）；
- 它**无法区分**"网关设置的租户头"与"客户端伪造的租户头"——**保证该头可信是网关的职责**；
- 它只应作为**内部服务**存在。nginx 不清洗头部的前提下，直连即等于无鉴权。

## 4. 网关需要实现的路由与头契约

### 4.1 路由映射（注意上游路径带 `files` / `webhooks` 段）

| 公网前缀 | 网关转发到 | 备注 |
|---|---|---|
| `/api/filesvc/{rest}` | `http://platform-service:5707/api/v1/files/{rest}` | `/api/files/*` 已被网关现有 BFF 占用，故文件服务用 `filesvc` |
| `/api/webhooks/{rest}` | `http://platform-service:5707/api/v1/webhooks/{rest}` | 服务也接受 `/api/webhooks/{rest}` 别名（portal 页面走这个） |
| `/portal/*`（建议不对外） | `http://platform-service:5707/portal/*` | 跨租户运维界面，见 §7 |

Compose 网络内服务名为 `platform-service`（容器名 `fina_demo-platform-service-1`）。
**列表接口的 URL 是 `/api/filesvc/?path=…`（带尾斜杠）**——服务已同时映射 `""` 与 `"/"` 两种形态，代理无需重写。

### 4.2 请求头契约

| 头 | 谁设置 | 说明 |
|---|---|---|
| **`X-Tenant-Id`** | **网关（必须设置或清除入站值）** | 租户标识；缺省时服务返回 `400 TENANT_REQUIRED` |
| `X-Api-Key` | 网关（内部令牌） | 服务配置 `FILE_SERVICE_API_KEY` 后校验；**客户端不应持有**。当前未启用，见 §6 |
| `X-User-Id` | 网关（可选） | 写入文件的 `createdBy`，用于审计 |
| `Content-Type` / `Accept` | 透传 | 上传为 `multipart/form-data`（**不要解析或重编码 multipart 请求体**） |

**网关必须做的两件事**：

1. **清除或覆写**入站的 `X-Tenant-Id`（绝不能让客户端自声明租户）；
2. 不透传客户端提供的 `X-Api-Key`，改为注入网关自己的内部令牌（§6 启用后）。

### 4.3 响应契约

- 成功：`200` + JSON；下载是**流式**响应（`Content-Disposition`、`ETag`=sha256、`X-File-Version`），**代理需保持流式，不要整体缓冲**；
- 错误：统一信封 `{"code": "...", "message": "..."}`：

| 状态 | code | 语义 |
|---|---|---|
| 400 | `TENANT_REQUIRED` | 缺 `X-Tenant-Id`——通常意味着**网关漏设** |
| 400 | `BAD_REQUEST` | 参数错误（uuid 格式、日期、路径非法） |
| 401 | `API_KEY_INVALID` | 内部令牌不匹配 |
| 404 | `NOT_FOUND` | 对象不存在，**或属于其它租户**（不泄露存在性，勿当路由错误排查） |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | upload 未使用 multipart |
| 502 | `WEBHOOK_BACKEND_ERROR` | webhook 后端（svix-server）不可用——**不代表本服务挂了**，探针仍 UP |

健康探针：`GET http://platform-service:5707/actuator/health` → `{"status":"UP"}`。

## 5. 切换步骤（nginx 直连 → 经网关）

按顺序执行，每步可独立回滚：

```bash
# 0) 备份（必须放在 sites-enabled 之外的目录！nginx 会加载该目录下所有文件，
#    备份留在里面会导致 "duplicate default server" 而无法 reload）
sudo mkdir -p /etc/nginx/backups
sudo cp /etc/nginx/sites-enabled/default /etc/nginx/backups/default.$(date +%s)

# 1) 网关侧实现 §4.1 路由 + §4.2 头处理并部署

# 2) 网关自测（不切流量）：直接打网关，带凭据但不带 X-Tenant-Id，期望成功；
#    带伪造的 X-Tenant-Id，期望被网关覆写

# 3) 切 nginx：把 /api/filesvc/ 与 /api/webhooks/ 的 proxy_pass 指向网关(5702)

# 4) 双保险：在两个 location 内清除入站租户头
#    proxy_set_header X-Tenant-Id "";

# 5) sudo nginx -t && sudo nginx -s reload

# 6) 按 §8 验收
```

**关于第 4 步的顺序**：清除入站租户头必须在网关开始设置它之后才生效；否则在网关就位前就切，会让公网调用因无人设置租户而全部 400。

## 6. 内部令牌（纵深防御，建议启用）

服务已支持：设置 `FILE_SERVICE_API_KEY` 后，只应答携带匹配 `X-Api-Key` 的调用方。**不需要改代码或重新构建**：

```bash
echo "FILE_SERVICE_API_KEY=$(openssl rand -hex 24)" | sudo tee -a /app/fina_demo/.env
cd /app/fina_demo && sudo docker compose up -d platform-service
```

**启用前提**：网关已开始注入该令牌，否则所有调用立刻 401。令牌值按密钥管理流程下发，不要写进仓库。

## 7. Portal 的处理

`/portal/` 是 webhook 的运维界面（投递目标、事件、投递状态），页面调用 `/api/webhooks/*`（服务同时挂载该别名，直连与经代理都能用）。

- 它是**跨租户控制面**（运维用途），建议**不暴露公网**，或放在网关运维通道后单独鉴权；
- 若必须公网可访问：至少加 Basic Auth；
- 它需要一种"可访问任意租户"的凭据——对应网关中的**管理/运维角色**，请一并设计。

## 8. 验收清单（切换后逐项确认）

```bash
B=https://ada.alphafina.cn

# 1) 服务健康（绕过网关/nginx 直连容器）
curl -s http://127.0.0.1:5707/actuator/health          # {"status":"UP"}

# 2) 经网关上传：凭据由网关校验，租户由网关注入（客户端不传 X-Tenant-Id）
curl -X POST "$B/api/filesvc/upload" -H "Authorization: Bearer <网关令牌>" \
  -F "file=@x.csv;type=text/csv" -F "path=ops" -F "fileName=x.csv"

# 3) 伪造租户头必须无效（结果应属于网关解析出的租户，而非 hankel）
curl -H "X-Tenant-Id: hankel" "$B/api/filesvc/?path=" -H "Authorization: Bearer <网关令牌>"

# 4) 未认证必须 401/400，而不是返回数据
curl "$B/api/filesvc/?path="

# 5) 列表与下载
curl "$B/api/filesvc/?path=ops&recursive=true" -H "Authorization: Bearer <网关令牌>"
curl -I "$B/api/filesvc/<uuid>/download" -H "Authorization: Bearer <网关令牌>"

# 6) webhook 全链路
curl -X POST "$B/api/webhooks/destinations" -H "Authorization: Bearer <网关令牌>" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://your-receiver/hook","topics":["job.completed"]}'
curl -X POST "$B/api/webhooks/publish" -H "Authorization: Bearer <网关令牌>" \
  -H "Content-Type: application/json" -d '{"topic":"job.completed","data":{"jobId":"1"}}'

# 7) 未匹配路径 404、错误信封一致
curl -o /dev/null -w "%{http_code}\n" "$B/api/filesvc/nope/x/y"   # 404
```

## 9. 待网关侧确认的问题

1. **租户来源**：网关从凭据（API key / JWT claim）解析租户的机制是什么？需要服务侧配合什么（额外头、令牌格式）？
2. **取值空间**：`X-Tenant-Id` 是否就是 Tenant Management 的 tenant id？命名是否兼容 `[A-Za-z0-9._-]`？
3. **运维凭据**：Portal 需要的跨租户能力，网关用哪种角色表达？
4. **是否暴露为 MCP/A2A 工具**：若需要，工具名与参数形态？服务侧可提供 OpenAPI 描述。
5. **限流与体积**：上传体积上限、超时策略由网关还是服务负责？（nginx 现为 `client_max_body_size 50m`，服务 multipart 上限 512MB）
6. **切换方式**：是否需要灰度（例如先切 `/api/webhooks/` 再切 `/api/filesvc/`）？

## 10. 服务侧承诺与边界

- 只信任网关设置的租户头，**不实现租户认证、不解析凭据**；
- 只监听 `127.0.0.1:5707`，容器不暴露宿主机以外的端口；
- 数据隔离由行级拦截器保证（跨租户访问返回 404，不泄露存在性）；
- 接口契约以 `API.md` / `WEBHOOK-API.md` 为准，变更走本仓库 CI 与文档流程。
