# platform-service ↔ API Gateway 对接契约

面向：实现平台 api/mcp gateway 的同事。
本文只描述 **platform-service 这一侧期望的输入**；网关如何认证、如何解析租户由网关负责，本服务不实现。

## 1. 定位

platform-service 是**网关后面的内部服务**，不是面向公网的入口。

- 它对租户**不做认证**：只读取网关设置的 `X-Tenant-Id`，并据此做行级隔离（MyBatis-Plus 租户拦截器会给每条 SQL 追加 `tenant_id`）。
- 它**无法区分**"网关设置的租户头"和"客户端伪造的租户头"——所以**入站请求的租户头必须由网关设置或清除**。
- 当前定位（仅测试环境）：nginx 直连 `127.0.0.1:5707`，故意保留宽松行为，便于联调。

## 2. 路由映射

网关需要把两个前缀转到该服务：

| 公网前缀（建议） | 网关转发到 | 说明 |
| --- | --- | --- |
| `/api/filesvc/*` | `http://platform-service:5707/api/v1/files/*` | 注意上游路径含 `files` 段 |
| `/api/webhooks/*` | `http://platform-service:5707/api/v1/webhooks/*` | 上游路径含 `webhooks` 段 |

> `/api/files/*` 已被 agent 网关的 BFF 占用，所以文件服务用 `filesvc` 前缀，避免冲突。

其余路径：`/actuator/health`（存活探针，可直连）、`/portal`（内部运维页，建议不对外）。

## 3. 请求头契约

| 头 | 谁设置 | 说明 |
| --- | --- | --- |
| **`X-Tenant-Id`** | **网关**（必须设置或清除入站值） | 租户标识，`[A-Za-z0-9._-]` 短字符串。缺省时服务返回 `400 TENANT_REQUIRED` |
| `X-Api-Key` | 网关（内部令牌，可选） | 当服务配置了 `FILE_SERVICE_API_KEY` 时校验。用于"只有网关能调用本服务"的纵深防御；客户端不应持有该值 |
| `X-User-Id` | 网关（可选） | 会写入文件的 `createdBy` 字段，用于审计 |
| `Content-Type` / `Accept` | 透传 | 上传为 `multipart/form-data`；原始流上传为调用方给定的类型 |

**网关必须做的两件事**（否则隔离形同虚设）：

1. **清除或覆写**入站请求里的 `X-Tenant-Id`——绝不能让客户端自己声明租户；
2. 不要透传客户端提供的 `X-Api-Key`，改用网关自己的内部令牌。

## 4. 响应契约

- **成功**：`200` + JSON（文件元数据 / 操作结果）。下载接口是流式响应，带 `Content-Disposition`、`ETag`(=sha256)、`X-File-Version` 等头。
- **错误**：统一信封 `{"code": "...", "message": "..."}`，HTTP 状态语义化：

| 状态 | code | 场景 |
| --- | --- | --- |
| 400 | `TENANT_REQUIRED` | 缺 `X-Tenant-Id`（网关漏设） |
| 400 | `BAD_REQUEST` | 参数错误（uuid 格式、日期、路径非法等） |
| 401 | `API_KEY_INVALID` | 内部令牌不匹配 |
| 404 | `NOT_FOUND` | 对象不存在，**或属于其它租户**（不泄露存在性） |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | upload 未用 multipart |
| 502 | `WEBHOOK_BACKEND_ERROR` | webhook 后端（svix-server）不可用 |

**502 需要网关注意**：webhook 模块依赖同栈的 svix-server，svix 不可用时 publish 返回 502；这不代表 platform-service 挂了，探针仍为 UP。

## 5. 现状与待办（按优先级）

1. **现在**：nginx 把上述两个前缀直接转到 `127.0.0.1:5707`，且**不清洗** `X-Tenant-Id` → 公网可自声明租户读写数据。**仅可用于测试环境**；
2. **接入网关后**：nginx 改为转到网关；网关认证后设置租户头；nginx 同时清除入站 `X-Tenant-Id`（双保险）；
3. **纵深防御**：prod compose 配置 `FILE_SERVICE_API_KEY`，只允许持令牌者调用；令牌由网关持有并注入；
4. **Portal（`/portal`）**：由本服务托管，页面调用同源 `/api/v1/webhooks/*`。它需要跨租户运维能力，接入网关后应走网关的运维通道，**不要暴露到公网**。

## 6. 本地/测试联调

```bash
# 直连（当前测试用法）：租户头由调用方自填
curl -H "X-Tenant-Id: hankel" http://localhost:5707/api/v1/files?path=

# 模拟网关：设置租户 + 内部令牌（服务需以 FILE_SERVICE_API_KEY=xxx 启动）
curl -H "X-Tenant-Id: hankel" -H "X-Api-Key: xxx" http://localhost:5707/api/v1/files?path=
```

接口细节见 [`API.md`](./API.md) 与 [`WEBHOOK-API.md`](./WEBHOOK-API.md)。
