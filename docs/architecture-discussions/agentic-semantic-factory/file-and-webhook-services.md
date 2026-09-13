# File Service 与 Webhook Service 落地记录

- 日期：2026-09-13
- 状态：两模块已合并为 **platform-service**（Spring Boot，5707）；files 模块本机冒烟 11/11 全绿；**webhooks 模块本机全链路冒烟通过**（2026-09-13：facade 建目标→发布→svix-server 投递→Standard Webhooks 验签→messages 可查，svix-server 经 1panel 镜像源拉取）
- 决策记录：Webhook 后端 **选型 Svix**（2026-09-13 与业务方评审后确定，评估过程见 §4）；**两服务合并为一个 Java 可复用组件**（同日确认，架构守则见 §5）
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
| HdfsAttachment.hdfsPath（物理与逻辑解耦，UUID 目录） | **离散存储 + 两级 hex 扇出**：物理 key 与逻辑路径完全无关 = `{tenant}/{uuid[0:2]}/{uuid[2:4]}/{uuid}`（每租户 65,536 桶均匀填充，规避单前缀海量对象的枚举退化与 FS 类后端目录爆炸）；逻辑路径/改名/版本纯粹是 DB 元数据，1 行 = 1 对象 |

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

> ⚠️ 本节为 Outpost 方案时期的历史记录，交付物与 compose 已被 §5 的 Svix 合并架构取代；事件契约表仍然有效。

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

两个服务按"内部插件"约定挂载：REST + Bearer/共享密钥 v1 认证；租户经 header 传播。平台 Tenant Management/API Key 签发体系建成后，`FILE_SERVICE_API_KEY` 等静态共享密钥应替换为签发的 key（接口不变，仅换凭据来源）。

## 4. Webhook 后端选型决策：Svix（2026-09-13）

### 评估比对结论（与业务方评审后确定）

两候选均为合格选择，差异在三个维度（评估时的事实基础，全部来自当日官方源核实）：

| 维度 | Hookdeck Outpost v1.3.0 | **Svix 开源版（选定）** |
|---|---|---|
| License | Apache-2.0 | MIT |
| 运行依赖 | 仅 Redis（migrate+api+delivery+log 四容器） | PostgreSQL 必须 + Redis 可选（单容器） |
| 多租户模型 | tenants→destinations 一级资源 | applications（≈每租户）→endpoints |
| 客户门户 | **内置** per-tenant Portal（JWT 直达链接） | 开源版 API-only，门户需 embed 组件自建 |
| 重试/DLQ | 指数退避；DLQ/SSRF 文档未明确 | 成熟重试调度；自管 DLQ+redrive；**默认 SSRF 防护** |
| 开源 vs 托管 | 托管跑同一代码库 | 官方明说开源版精简（未列清单） |
| 成熟度 | 1.1k stars | 品类开创者、Standard Webhooks 规范制定者 |

**定选 Svix 的理由**（业务方拍板）：品类标准制定者的原厂实现、更完整的投递可观测（DLQ/attempts API）、默认 SSRF 防护（收端 URL 由客户配置的场景必要）、MIT。接受的两个代价：门户需自建（先用 attempts API 顶，接飞书时上 embed 组件）、开源精简版行为差异需实测。

**保留的对比记录**：Outpost 曾为首选（license 干净、tenant 模型贴合、Portal 内置），切换成本分析成立——集成面收敛在 `SvixServerClient` 一个类里，未来再评估 Outpost 只需重写该类 + compose 块。

### 实测验收（2026-09-13 本机完成 ✅）

`webhook-smoke.sh` 全绿：建目标（facade 返回 whsec）→ 验签收端 → 发布 → 30s 内投递到达 → `svix-*` 签名 HMAC 验证通过 → facade messages 可查。svix-server 镜像经 `docker.1panel.live` 镜像源拉取（Docker Hub 在本网络不可达）。

**实测发现的 OSS svix-server 差异**（已固化进适配层，选型评估里"精简版未列清单"的具体化）：

1. 创建端点的响应**不含**签名密钥，需另调 `GET .../endpoint/{id}/secret/`；
2. **没有 `GET /api/v1/app/uid/{uid}` 路由**——幂等创建改为"POST 优先，409 时列表按 uid 解析"+ 进程内缓存；
3. 列表路由不接受 `{limit}` 路径段，用 `?limit=` query 参数；
4. 投递签名头是 `svix-id/svix-timestamp/svix-signature`（Standard Webhooks 的前 SKU 名），收端需兼容两套前缀。

## 5. 合并架构：platform-service（2026-09-13 定稿）

**决策**：file 与 webhook 两个能力合并为一个 Java 可复用组件，而非两个独立服务。

```
[factory agents / scripts]
   │  X-Tenant-Id + X-Api-Key
   ▼
platform-service :5707（nginx /api/filesvc/ 与 /api/webhooks/）
   ├─ files 模块    /api/v1/files/*      → MinIO/TOS + file_objects（原样迁移，冒烟全绿）
   └─ webhooks 模块 /api/v1/webhooks/*   → SvixServerClient → svix-server 容器（不对外暴露）
svix-server：SVIX_DB_DSN=document-postgres/svix，SVIX_REDIS_DSN=document-redis/3，自动建目标 whsec 签名
```

**三条架构守则**：

1. **Svix 本体独立容器**——Java 里只有 HTTP 适配层（`SvixServerClient` 一个类知道 Svix 存在），换产品=重写该类+compose 块；
2. **facade 是唯一调用面**——业务方只见 `X-Tenant-Id` + 工厂 topic + facade API；租户=Svix Application（uid=tenantId）、topic=EventType（发布时懒注册）、destination=Endpoint（svix 生成 whsec）；
3. **模块边界即拆分线**——`com.fina.platform.files/...` 逻辑分离（现包结构：entity/mapper/service 与 webhooks/ 平行），将来要拆沿包切。

**facade API**：

- `POST /api/v1/webhooks/destinations` {url, topics[], description?} → {endpointId, secret(whsec), topics}
- `GET /api/v1/webhooks/destinations` / `DELETE /api/v1/webhooks/destinations/{endpointId}`
- `POST /api/v1/webhooks/publish` {topic, data} → {messageId, topic}
- `GET /api/v1/webhooks/messages?limit=` / `GET /api/v1/webhooks/messages/{messageId}/attempts`

**Portal**：`http://localhost:5707/portal` —— 自包含静态单页（原生 JS，零构建链），输入租户 ID 即可管理目标（增删）、浏览事件、逐消息查看投递尝试与 HTTP 状态。Svix 官方 Portal 是 `@svix/react` 托管生态组件，对自托管兼容未验证且需要 React 工具链；自建轻量版与"Java 可复用组件"定位一致，`@svix/react` 留作 ai_web（React）集成时的升级路径。门户是租户无关的静态壳，其 API 调用按请求携带租户头；生产外露时应在 nginx 加 `/portal/` 路由并叠加平台鉴权。

**实现细节**：svix-server 鉴权 token 由 `SvixTokenService` 用共享 `SVIX_JWT_SECRET` 现场铸造 HS256 JWT（sub=orgId，10 年期，demo 够用；生产接平台签发后收紧）；EventType 发布时懒注册；`SVIX_WHITELIST_SUBNETS` demo 默认放行私网段（host.docker.internal 收端需要），生产留空保持 SSRF 严格。

**合并的代价与兜底**：两个能力的发布节奏被绑在一起（可接受——都是平台基座、同一团队）；存储带宽与投递吞吐互相影响（demo 规模无关；未来拆分沿模块线）。
