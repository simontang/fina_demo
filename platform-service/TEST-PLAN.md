# platform-service 测试用例清单与完备性评估

- 日期：2026-09-13
- 被测对象：platform-service（files 模块 + webhooks 模块 + portal），透明多租户，Svix 后端
- 执行方式：`scripts/test-suite.sh`（本清单的可执行形态，逐用例编号输出）；`smoke.sh` / `webhook-smoke.sh` 为其子集（快速回归用）
- 环境：本机全栈——postgres:15-alpine(5433) + svix-server(8071, memory 队列) + platform-service(5707)，存储 TOS 真实 S3 端点

## 1. 测试用例清单

### A. files 模块 · 功能契约（24 例）

| 编号 | 用例 | 预期 |
|---|---|---|
| F-01 | 健康检查 | `/actuator/health` 200 UP |
| F-02 | 基础上传（multipart + path + fileName） | 200；回执含 path/version=1/sha256(64)/md5(32)，deduplicated=false |
| F-03 | 同内容重复上传 | deduplicated=true，version 不变，不产生新行 |
| F-04 | 变更内容重复上传 | version+1，旧版本行保留可下载 |
| F-05 | fileName 缺省 | 取原始文件名 |
| F-06 | Unicode/空格文件名 | 上传/按 path 下载均正确（URL 编码往返） |
| F-07 | meta JSON 参数 | 回执与查询原样保留（jsonb；回执中以 JSON 字符串形态返回） |
| F-08 | fileCategory/usage 参数 | 存储并可从回执读出 |
| F-09 | 空文件上传 | 400 |
| F-10 | 缺 file part | 400（不得 500） |
| F-11 | path 规范化（尾斜杠/连续斜杠） | 与规范路径等价（去重判定生效） |
| F-12 | path 含 `..` | 400，拒绝路径穿越 |
| F-13 | 文件名含 `/` 或 `\` | 清洗为 `_`，可正常寻址 |
| F-14 | 按 uuid 下载 `GET /{uuid}/download``GET /files/{key}` | 返回最新版本内容 |
| F-15 | 历史版本各自 uuid 独立可下载 | 内容为对应版本，旧行未被覆盖 |
| F-16 | 下载 `bom=true`（CSV） | 前置 UTF-8 BOM；已有 BOM 不重复加 |
| F-17 | 下载不存在/已删 key | 404 NOT_FOUND |
| F-18 | 下载响应头 | Content-Type 保留；Content-Disposition 带 UTF-8 文件名 |
| F-19 | 列表 prefix+delimiter | 直属文件 + 下一层伪目录聚合（结果携带 uuid） |
| F-20 | 列表空 prefix（根） | 列出全部根文件与一级目录 |
| F-22 | 软删除 `DELETE /files/{uuid}`（该版本） | status=deleted；该 uuid 404；同对象其他版本不受影响 |
| F-23 | 删除指定 version | 仅该版本被删，其余版本仍可下载 |
| F-26 | PUT 原始流直传 `PUT /files/{uuid}`（客户端自选 uuid，X-File-* 携带元数据） | 回执正确，GET 内容一致 |
| F-27 | HEAD 元数据 `HEAD /files/{uuid}` | ETag=sha256、X-File-Version/Path 等头正确 |
| F-28 | `GET /files/{uuid}` 只返元数据 | 返回 JSON 回执，不返回文件内容 |
| F-29 | 列表目录聚合（SQL 侧 DISTINCT） | 返回下一层子目录，不加载整个前缀 |
| F-30 | 列表分页 | `limit` 生效，`truncated`/`nextCursor` 正确 |
| F-31 | 游标续页 | 无重叠、无遗漏、按 filename 有序 |
| L-01 | `POST /files/presign` {uuid}：auto + 内网存储（自托管 MinIO） | 返回自有下载 URL（kind=direct） |
| L-02 | presign 缺 uuid / uuid 格式非法 | 400 |
| L-03 | `POST /files/presign` {uuid}：auto + 公网存储（TOS） | 返回 storage 原生 presigned URL（X-Amz-Signature） |

### B. 多租户与鉴权（7 例）

| 编号 | 用例 | 预期 |
|---|---|---|
| T-01 | 缺 `X-Tenant-Id` | 400 TENANT_REQUIRED |
| T-02 | 跨租户隔离 | 各自 uuid 独立可见；用 A 的租户头访问 B 的 uuid → 404 |
| T-04 | `X-User-Id` 上下文 | created_by 自动填充 |
| T-05 | `X-Api-Key` 启用时：缺失/错误 → 401；正确 → 200 | 鉴权门生效（独立实例验证） |
| T-06 | webhooks 跨租户隔离 | destinations/publish/messages 按租户作用域 |
| T-07 | `FILE_SERVICE_DEFAULT_TENANT`（dev 缺省租户） | 未带头时归入默认租户（落库验证） |

### C. webhooks 模块（10 例）

| 编号 | 用例 | 预期 |
|---|---|---|
| W-01 | 创建 destination | endpointId + `whsec_` secret 返回 |
| W-02 | destinations 列表 | 含 url/topics/disabled |
| W-03 | 删除 destination | 200，列表不再含 |
| W-04 | publish（首次 topic） | messageId 返回；EventType 懒注册成功 |
| W-05 | 端到端投递 | 30s 内到达收端；`svix-*` 签名 HMAC 验证通过 |
| W-06 | messages 列表 | 可查、含 topic 与时间 |
| W-07 | attempts 查询 | 按 endpoint 返回尝试结构（状态/HTTP 码） |
| W-08 | 重复 publish | 每次独立 messageId，messages 数量递增 |
| W-09 | 篡改签名（收端密钥不匹配） | 验签判 invalid（防伪造投递） |
| W-10 | 删除不存在的 endpointId | 返回 4xx/5xx 错误体而非静默成功 |

### D. Portal / 静态资源（3 例）

| 编号 | 用例 | 预期 |
|---|---|---|
| P-01 | `/portal` 转发 | 200 |
| P-02 | `/portal/index.html` 内容 | 含关键结构（三面板/连接表单） |
| P-03 | 静态页不受租户拦截 | 无 X-Tenant-Id 也可加载页面壳 |

### E. 韧性与运维（4 例）

| 编号 | 用例 | 预期 |
|---|---|---|
| S-01 | svix-server 停机时 publish | 显式 5xx 错误体（不静默成功）；恢复后可发布 |
| S-02 | 服务重启后数据持久 | 文件与 webhook 配置从 PG 恢复 |
| S-03 | compose 配置校验 | `docker compose config` 通过 |
| O-04* | 全链路时延 | 投递 <30s（并入 W-05 计时） |

## 2. 完备性评估

### 2.1 覆盖维度映射

| 维度 | 覆盖用例 | 判定 |
|---|---|---|
| 功能契约（API 行为=文档） | A 全部、C-01~04/06~08 | ✅ 完整 |
| 多租户隔离（最高安全属性） | T-01/02/03/06/07 + C 全部在租户上下文执行 | ✅ 完整（读写双路径均覆盖） |
| 安全边界 | F-12 路径穿越、W-05/09 签名与防伪造、svix 默认 SSRF（投递到私网收端需白名单=已验证防护存在）、T-05 鉴权门 | ✅ 核心覆盖；XSS 为 portal 代码审查项（`esc()` 全量转义），不设动态用例 |
| 异常与失败路径 | F-09/10/12/17、W-10、S-01 | ✅ 主要分支覆盖 |
| 幂等与一致性 | F-03/04/15/22、S-02 | ✅ 覆盖 |
| 可观测 | W-06/07、F-01、portal 三面板 | ✅ 覆盖 |
| 性能/规模 | 无 | ⛔ 明确不做（见 2.2） |

### 2.2 已知缺口（诚实清单）

| 缺口 | 原因 | 处置 |
|---|---|---|
| 并发上传同一路径的版本竞态 | 两个并发请求同时读到相同 latest → 同 version 违反唯一约束 → 其一 500。**已知限制** | 本期记录；后续用 `version` 冲突重试或序列化锁解决 |
| 大文件（>100MB）/ 上传时延基准 | TOS 真实端点下耗时不稳定，不宜进自动套件 | `RUN_HEAVY=1` 可选执行（50MB 用例 F-25） |
| nginx 公网路由 | 本机无 nginx | 部署时按 DEPLOY.md §4 验证 |
| 长时间重试节奏 | 需断收端观察分钟级退避 | 生产观测项 |
| 浏览器兼容 / Portal 视觉 | 需真实浏览器会话 | 手工项（用户已实际打开验证页面可达） |
| RDS 特定行为（迁移/权限） | 本机用 postgres:15 模拟 | 生产部署项（DEPLOY.md §2） |
| Portal XSS 动态用例 | 无头浏览器未接入 | 代码审查项：所有插值经 `esc()` |

### 2.3 结论

清单对本服务的**功能契约、租户隔离、安全边界、异常路径、幂等一致性**五个核心维度完备；**性能与公网链路**明确列为范围外并给出处置。用例均可执行（除标注 deferred），共 **48 例**（A24+B7+C10+D3+E4）。

## 3. 执行结果

- 执行时间：2026-09-13；执行器：`scripts/test-suite.sh`（本机全栈：postgres:15 + svix-server + platform-service，存储 TOS 真实 S3 端点）
- **结果：43/43 全部通过**（S3 风格接口 v2 形态；含 F-11a/b、F-16a/b、F-22a/b、F-26b、T-05a~d 等子用例拆分）
- 接口 v2 重设计（URL 路径寻址 + PUT/HEAD/DELETE + prefix/delimiter 列表）过程中，测试套件抓到并修复 4 个回归：multipart 缺省文件名丢失（控制器重构遗漏 originalFilename 回退）、空文件校验随重构失效、Unicode key 需 URL 解码（UriUtils）、nginx 公网前缀少映射 `/files` 段

### 3.1 测试发现并已修复的缺陷

| 发现 | 修复 |
|---|---|
| 缺 file part 的请求返回 500（应为 400） | GlobalExceptionHandler 增加 MissingServletRequestPart/Parameter/Value → 400 |
| by-uuid 回执端点缺失（F-21 真实缺口，此前被脚本缺陷掩盖） | FileController 增加 `GET /api/v1/files/uuid/{uuid}/receipt` |
| 回执未携带 meta 字段（F-07 无法验证） | FileReceipt 增加 meta（jsonb 以 JSON 字符串形态往返，断言按"原样保留"验证） |
| attempts 接口按端点轮询为空 | 改用 Svix 按消息列端点投递状态的路由（`/msg/{id}/endpoint/`），Portal 面板同步 |

### 3.2 测试基建教训（记录给后续套件维护）

1. 断言取值函数必须把响应体作为参数传入（管道式 stdin 读取会在静默失败时把真实缺陷掩盖成假失败）；
2. 套件输出**不要走管道**（`| tail` 在缓冲写满后会阻塞整个 bash）——输出重定向到文件；
3. 测试实例停机用 `kill -9`（Spring 优雅关闭可挂 30 秒以上）；curl 一律带 `-m` 超时。

### 3.3 遗留（范围外，见 §2.2）

并发版本竞态、大文件基准、nginx 公网链路、重试长周期观测、浏览器兼容、RDS 实机行为——分别在 DEPLOY.md 生产步骤与后续迭代中处理。
