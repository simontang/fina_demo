# File Service 与 Webhook Service 落地记录

- 日期：2026-09-13
- 状态：M1（File Service）已实现并本机冒烟全绿；M2（Webhook Service）已实现+脚本交付，Outpost 本机不可运行（镜像 Linux-only、本机 Docker Hub 拉取受限），待部署目标验证
- 关联：[review-briefing §11 平台基座](./review-briefing.md)、[dbt-dynamic-deploy](./dbt-dynamic-deploy.md)

## 0. 选型决策

### Webhook：Hookdeck Outpost v1.3.0（Apache-2.0）

调研对比（2026-09-13 核实）：

| 项目 | License | 多租户 | 结论 |
|---|---|---|---|
| **Hookdeck Outpost** | Apache-2.0 | tenant→destinations 原生 | **选定**：license 干净、tenant 模型与平台图一致、每租户 Portal、自带 MCP server |
| Svix | MIT（自 AGPL 重授权） | Application 模型 | 强备选；开源服务为托管版精简但 API 兼容 |
| Convoy | Elastic License v2 | organizations/projects | 出局：社区版限 1 用户/1 组织/2 项目 |
| Hook0 | SSPL v1 | projects | 备选；SSPL 弱、签名能力未明确 |

v1.3.0 部署形态（比旧文档简单）：`hookdeck/outpost:v1.3.0` 镜像按 `SERVICE=api/delivery/log` 拆容器 + `migrate apply` 一次性任务，**只需 Redis**（复用 document-redis，db 2），无需 RabbitMQ/独立 PG。

### File Service：标准 Spring Boot（用户指定，不用 Grails）

Spring Boot 3.2.3 / Java 17 / Gradle KTS / mybatis-plus / Lombok，完全照 metrics-server 模板；S3 访问用 AWS SDK v2（path-style，兼容 MinIO 与 TOS）。

## 1. File Service 设计（对齐 fina-ai file 服务 + bjy/fuli 透明租户）

### fina-ai file 服务对齐点

| fina-ai file（~/gitlab/file，Grails） | 本服务 |
|---|---|
| `MultiTenant<FileInfo>` + `currentTenant.get()` | `X-Tenant-Id` header + tenant_id 列（等价、更显式） |
| `download(id)` @Deprecated → `download2ByPath` 演进 | **path 为一等寻址**，id/uuid 为兼容别名 |
| `fileMd5` 指纹 | md5 + sha256 双指纹（sha256 进 storage key） |
| 下载对非 HDFS 文件补 UTF-8 BOM | `bom=true` 选项（CSV/文本类文件，已有 BOM 则不重复加） |
| fileName/fileCategory/usage/meta/uuid 字段 | 原样保留 |
| HdfsAttachment.hdfsPath 存真实路径 | 物理存储 key = `{tenant}/{dir}/{filename}@{sha256前8}` |

### 透明多租户三件套（bjy_crm_ai CONTEXT_INTERCEPTOR 模式 + 查询侧补全）

1. `TenantContextHolder`（ThreadLocal，仿 bjy ContextHolder）
2. `TenantContextInterceptor`：忽略大小写提取 `X-Tenant-Id`（+可选 `X-Api-Key` 校验、`X-User-Id` 上下文），请求后清理；`FILE_SERVICE_DEFAULT_TENANT` 提供 fuli 式 dev 默认租户（生产留空强制 400）
3. MyBatis-Plus `TenantLineInnerInterceptor`：**所有 SQL 自动追加/填充 tenant_id**（bjy 只做 insert 填充，这里把查询隔离也透明化）+ `AuditMetaHandler` 自动填 created_by/时间

业务代码（service/mapper/controller）零租户逻辑；跨租户隔离由冒烟用例直接验证（同路径两租户互不可见）。

### 无 folder + 不可变版本化（用户约束）

- **无 folders 表、无 parent_id 树**：目录只是 path 前缀；`GET /api/v1/files?prefix=` 返回对象 + 下一层目录聚合。并发问题的根源（目录行的创建竞争）不存在。列表性能未来不足再异步物化目录索引。
- **同路径重复上传 = 追加 version+1 新行**，物理 key 带 sha256 前缀后缀（`{tenant}/{dir}/{filename}@{sha8}`），旧行永不覆盖；内容相同则直接去重返回 `deduplicated=true` 不建新行。

### API（内部 `/api/v1`，nginx `/api/filesvc/`）

见 `MICROSERVICES_PORTS_AND_ROUTES.md` §8。注意公网前缀是 `/api/filesvc/`——`/api/files/*` 已被 agent BFF 占用。

### 冒烟证据（2026-09-13 本机）

`file-service/scripts/smoke.sh` 11 项全绿，存储端为 TOS 真实 S3 兼容端点（本机 Docker Hub 拉取受限，MinIO 以 TOS 替代验证；compose 内仍默认 document-minio）：

幂等去重、同路径 v2 追加且 v1 原样可下载、path 寻址下载、BOM 前缀、伪目录列表、**跨租户同路径互不可见**、按 path/id 回执、软删后 404。

过程中修的两个问题（已固化）：timestamptz→LocalDateTime 驱动不兼容（DDL 改 `timestamp`）；macOS bash 3.2 空数组 + `set -u`（安全展开惯用法）。

## 2. Webhook Service 设计

### 事件契约

| topic | 发布方 | 载荷要点 |
|---|---|---|
| `import.completed` | M1 import 动词/loader | file receipt、snapshot 引用 |
| `gate.passed` | PM agent 状态机 | gate 名、决策 id 列表 |
| `decision.captured` | 确认 app 回传 | decision_id、topic、status |
| `job.completed` | factory-runner | job id、run_results 摘要 |
| `run.published` | Publisher | meta diff 摘要、版本 |

投递头遵循 Standard Webhooks（`webhook-id`/`webhook-timestamp`/`webhook-signature`，HMAC-SHA256 base64，`whsec_` 密钥按 destination 配置），at-least-once + 自动重试。

### 交付物

- compose：`webhook-migrate`（一次性）+ `webhook-api`（5708→3333，nginx `/api/webhooks/`）+ `webhook-delivery` + `webhook-log`；Redis 复用 document-redis（db 2）；secrets 走 `WEBHOOK_*` env（demo 有默认值，生产必须覆盖）
- `scripts/provision-tenant.sh`：开租户 → 建 destination（topics + whsec 签名密钥）→ 打 Portal 链接（后续飞书通知"点此查看投递状态"）
- `publish.py`：发布薄封装（factory-runner/PM agent 复用）
- `scripts/mock-receiver.py`：标准验签收端；`scripts/smoke.sh`：开租户→发布→收端验签全链路

### 待办（部署目标上完成）

1. `docker compose up webhook-migrate webhook-api webhook-delivery webhook-log`（本机拉不到镜像）
2. 跑 `webhook-service/scripts/smoke.sh` 验证投递+验签
3. nginx reload 后公网路由验证

## 3. 与平台基座的关系

两个服务按"内部插件"约定挂载：REST + Bearer/共享密钥 v1 认证；租户经 header 传播。平台 Tenant Management/API Key 签发体系建成后，`FILE_SERVICE_API_KEY` / `WEBHOOK_SERVICE_API_KEY` 的静态共享密钥应替换为签发的 key（接口不变，仅换凭据来源）。
