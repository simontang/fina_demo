# platform-service 部署手册（files + webhooks + portal）

- 适用环境：214/38 生产演示服务器（docker compose + nginx，镜像走火山引擎仓库）
- 组成：`platform-service`（Java，5707）+ `svix-server`（Svix，仅内网）+ `platform-db-init`（dev compose 才有；生产按 §3 手工初始化）
- 门户：部署即含，`http://<host>:5707/portal`（端口只绑 127.0.0.1，外网访问见 §6）

## 1. 镜像准备

**✅ 2026-09-13 已完成**：两个镜像已推送至火山仓库，服务器可直接拉取：

- `finai-cn-shanghai.cr.volces.com/default/fina-demo-platform-service:latest`（sha256:963bab3f…）
- `finai-cn-shanghai.cr.volces.com/default/svix-server:latest`（sha256:6414372a…）

后续更新版本时重复以下步骤：

```bash
# platform-service：宿主机先构建 jar（网络受限环境下不要用容器内 gradle 构建，
# Dockerfile 的多阶段构建需要访问 services.gradle.org / maven central）
cd platform-service && ./gradlew bootJar --no-daemon -q

# 受限网络：用预构建 jar 打运行时镜像（跳过容器内构建阶段）
docker build -f - -t finai-cn-shanghai.cr.volces.com/default/fina-demo-platform-service:latest platform-service <<'DOCKERFILE'
FROM eclipse-temurin:17-jre-jammy
WORKDIR /app
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
COPY build/libs/platform-service.jar app.jar
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app
USER appuser
EXPOSE 5707
ENTRYPOINT ["java", "-Xms256m", "-Xmx512m", "-Djava.security.egd=file:/dev/./urandom", "-jar", "app.jar"]
DOCKERFILE

# svix-server：基础镜像经镜像源拉取后改 tag
docker pull docker.1panel.live/svix/svix-server:latest
docker tag docker.1panel.live/svix/svix-server:latest finai-cn-shanghai.cr.volces.com/default/svix-server:latest

# 推送
docker push finai-cn-shanghai.cr.volces.com/default/fina-demo-platform-service:latest
docker push finai-cn-shanghai.cr.volces.com/default/svix-server:latest
```

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

`nginx-fina-demo.conf` 已含两条路由（80 与 443 两个 server 块都要有）：

- `/api/filesvc/` → `127.0.0.1:5707/api/v1/`（文件；公网前缀避开 agent BFF 的 `/api/files/*`）
- `/api/webhooks/` → `127.0.0.1:5707/api/v1/webhooks/`

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
curl -s "https://demo.alphafina.cn/api/filesvc/api/v1/files?prefix=" -H "X-Tenant-Id: smoke"
curl -s "https://demo.alphafina.cn/api/webhooks/messages?limit=5" -H "X-Tenant-Id: smoke"

# webhook 全链路（可选，需一个可达的收端 URL；服务器→公网收端）
PLATFORM_SERVICE_URL=http://127.0.0.1:5707 bash platform-service/scripts/webhook-smoke.sh
```

## 6. Portal 的暴露方式

Portal 由 platform-service 直接托管（`/portal`，自包含静态页），部署服务即部署门户。三种暴露方式按需选：

| 方式 | 做法 | 适用 |
|---|---|---|
| 内部使用（默认） | `ssh -L 5707:127.0.0.1:5707 root@<host>` 后访问 `http://localhost:5707/portal` | 最安全，日常运维 |
| 公网只读/受控 | nginx 加 `location /portal/ { proxy_pass http://127.0.0.1:5707/portal/; }` + Basic Auth 或平台鉴权 | 给客户演示 |
| 完全内网 | 不加任何路由，仅 127.0.0.1 | 生产建议 |

注意：Portal 是**控制面**（可增删投递目标），公网暴露必须叠认证；当前版本门户本身无登录（与整个 demo 栈一致）。

## 7. 常见问题

1. **svix-server 起不来：`SVIX_WHITELIST_SUBNETS` invalid** —— 该变量必须是 JSON 数组：`'["10.0.0.0/8"]'`，逗号串会拒绝启动。
2. **投递到内网收端被拦** —— Svix 默认 SSRF 防护阻断私网地址；仅 demo 需要 `SVIX_WHITELIST_SUBNETS` 放行，生产留空。
3. **签名头是 `svix-*` 不是 `webhook-*`** —— 收端验签需兼容两族前缀（`scripts/mock-receiver.py` 已兼容）。
4. **file_objects 表缺失报错** —— §2 的 DDL 未执行；`platform-db-init` 容器仅在 dev compose 存在。
5. **TOS 上传 403** —— `OBJECT_STORAGE_ACCESS_KEY/SECRET_KEY` 未传入（生产 compose 默认为空，从 .env 注入）。
