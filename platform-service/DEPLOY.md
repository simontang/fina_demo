# platform-service 部署手册（files + webhooks + portal）

- 适用环境：214/38 生产演示服务器（docker compose + nginx，镜像走火山引擎仓库）
- 组成：`platform-service`（Java，5707）+ `svix-server`（Svix，仅内网）+ `platform-db-init`（dev compose 才有；生产按 §3 手工初始化）
- 门户：部署即含，`http://<host>:5707/portal`（端口只绑 127.0.0.1，外网访问见 §6）

## 1. 镜像准备（由 CI 构建，本地不要构建）

镜像由 **GitHub Actions 构建并推送**（`.github/workflows/ci.yml`）：

| 镜像 | 产出 | 说明 |
|---|---|---|
| `ghcr.io/409zhangshu/fina-demo-platform-service:latest` | `build-and-push-platform-service` job | 推送 `main` 且 `platform-service/**` 有变更时触发 |
| `ghcr.io/409zhangshu/svix-server:latest` | `mirror-svix-server` job | 第三方镜像镜像到 ghcr，固定 amd64；也可手动 `workflow_dispatch` 触发 |

**为什么必须由 CI 构建（踩过的坑）**：本机是 Apple Silicon，本地构建/拉取的镜像是 **linux/arm64**，
而服务器是 **x86_64**，容器启动即 `exec format error`（`exec /opt/java/openjdk/bin/java: exec format error`）。
CI runner 是 amd64，产物天然匹配；`build-push-action` 也显式声明了 `platforms: linux/amd64`。

**不要**再用本地 `docker build` / `docker push` 发布这两个镜像。

服务器拉取镜像无需额外操作：root 的 docker 已登录 `ghcr.io` 与火山仓库。

## 2. 服务器侧准备

```bash
cd /app/fina_demo && git pull   # 或同步 docker-compose.prod.yml / nginx-fina-demo.conf
```

服务器 `.env` 需要补充（已有 SPRING_DATASOURCE_* 则复用）：

```bash
SPRING_DATASOURCE_PASSWORD=<RDS 密码>        # prod compose 已要求
SVIX_JWT_SECRET=<随机长串>                    # openssl rand -base64 32
FILE_SERVICE_API_KEY=<随机串>                 # 建议生产启用机器鉴权
# 可选：FILE_OBJECT_STORAGE_BUCKET（默认 finademo）、SVIX_ORG_ID
```

RDS 上一次性建库（只需 **`svix`** 一个：file_objects 表由 platform-service
启动时 Flyway 自动建在 `postgres` 库，与 metrics-server / document_service
同库同模式；svix-server 的表为通用名，保留独立库作失败域隔离）：

```bash
psql "host=pgm-uf615169n98t95tflo.pg.rds.aliyuncs.com user=postgres_fuli dbname=postgres" \
  -c "CREATE DATABASE svix"
```

## 3. 启动

```bash
SPRING_DATASOURCE_PASSWORD=... SVIX_JWT_SECRET=... \
  docker compose -f docker-compose.prod.yml up -d platform-service svix-server
```

svix-server 首次启动自动跑数据库迁移（RDS 的 `svix` 库）。

## 4. nginx

`nginx-fina-demo.conf` 已含版本化路由（80 与 443 两个 server 块都要有）：

- `/api/v1/files/` → `127.0.0.1:5707/api/v1/files/`
- `/api/v1/webhooks/` → `127.0.0.1:5707/api/v1/webhooks/`
- `/api/filesvc/`、`/api/webhooks/` 可作为旧调用方兼容入口保留，不作为新接口文档入口。

同步配置并 reload：

```bash
scp nginx-fina-demo.conf root@14.103.67.214:/etc/nginx/conf.d/fina-demo.conf
ssh root@14.103.67.214 nginx -t && ssh root@14.103.67.214 nginx -s reload
```

## 5. 验证

```bash
# 在服务器上
curl -s http://127.0.0.1:5707/actuator/health           # {"status":"UP"}
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5707/portal   # 200

# 公网（本机）
curl -s "https://demo.alphafina.cn/api/v1/files/?path=" -H "X-Tenant-Id: smoke"
curl -s "https://demo.alphafina.cn/api/v1/webhooks/messages?limit=5" -H "X-Tenant-Id: smoke"

# webhook 全链路（可选，需一个可达的收端 URL；服务器→公网收端）
PLATFORM_SERVICE_URL=http://127.0.0.1:5707 bash platform-service/scripts/webhook-smoke.sh
```

## 6. Portal 的暴露方式

Portal 由 platform-service 直接托管（`/portal`，自包含静态页），部署服务即部署门户。nginx 已配置好 `location /portal/`（80 与 443 两个 server 块各一份），转发到 `127.0.0.1:5707/portal/`。

访问方式：

| 方式 | 做法 | 说明 |
|---|---|---|
| 公网/内网经 nginx | `http://<host>/portal/` | 页面调用 `/api/webhooks/*`（同一 nginx 已转发到本服务），因此经代理可用 |
| 直连服务 | `http://127.0.0.1:5707/portal` | 需 ssh 隧道；页面同样可用（服务同时挂载 `/api/v1/webhooks` 与 `/api/webhooks`） |

**安全提醒**：Portal 是控制面（可增删投递目标），且它本身不做登录。经 nginx 暴露时请叠加认证（Basic Auth 或接入网关的运维通道）；等 api/mcp 网关接入后，建议改走网关统一鉴权，详见 [GATEWAY-INTEGRATION.md](./GATEWAY-INTEGRATION.md) §5。

## 7. 首次部署踩过的坑（已固化，勿回退）

| 现象 | 根因 | 修复 |
|---|---|---|
| 容器 `exec /opt/java/openjdk/bin/java: exec format error` | 本地 Apple Silicon 构建的是 arm64 镜像，服务器是 x86_64 | 镜像一律由 CI 构建（已加 `platforms: linux/amd64`） |
| Flyway `Migration V2 failed: relation "file_objects" does not exist` | `baseline-version: 1` 让已存在的库被基线化为 v1，跳过 V1 直接跑 V2 | `baseline-version: 0`（V1 幂等，存量库可安全重跑） |
| svix-server 启动即退出：`invalid type: found string "" ... WHITELIST_SUBNETS` | 该变量未设置时渲染成空字符串，svix 要求 JSON 数组 | 默认值改为 `[]`（严格 SSRF 防护） |
| svix 相关接口全 404 / 首次发布新 eventType 失败 | `ensureEventType` 依赖 `RestClientException` 捕获，被统一状态处理改成 `ApiException` 后失效 | 改为捕获 `ApiException(404)`；新增 W-15 回归用例 |
| 列表接口经 nginx 500 | `/api/v1/files/`（尾斜杠）未映射，落到静态资源解析 | 集合端点同时映射 `""` 与 `"/"`；未匹配路径统一返回 404 |
| 部署后 nginx 起不来：`duplicate default server` | 备份文件放在了 `sites-enabled/`，nginx 会加载该目录下所有文件 | 备份移到 `/etc/nginx/backups/` |

## 8. 常见问题

1. **svix-server 起不来：`SVIX_WHITELIST_SUBNETS` invalid** —— 该变量必须是 JSON 数组：`'["10.0.0.0/8"]'`，逗号串会拒绝启动。
2. **投递到内网收端被拦** —— Svix 默认 SSRF 防护阻断私网地址；仅 demo 需要 `SVIX_WHITELIST_SUBNETS` 放行，生产留空。
3. **签名头是 `svix-*` 不是 `webhook-*`** —— 收端验签需兼容两族前缀（`scripts/mock-receiver.py` 已兼容）。
4. **file_objects 表缺失报错** —— §2 的 DDL 未执行；`platform-db-init` 容器仅在 dev compose 存在。
5. **TOS 上传 403** —— `OBJECT_STORAGE_ACCESS_KEY/SECRET_KEY` 未传入（生产 compose 默认为空，从 .env 注入）。
