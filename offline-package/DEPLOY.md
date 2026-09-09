# Fina Demo 离线部署指南

## 架构说明

离线包包含 **3 个镜像**（Web 前端、Agent、PostgreSQL）。

**沙盒（Microsandbox）为独立部署组件**，不包含在离线包内。
需要目标机器上已运行 Microsandbox 服务（默认 `http://127.0.0.1:4002`），
Agent 通过 `microsandbox-remote` 协议调用它执行代码。

## 环境要求

- Linux x86_64（Ubuntu 20.04+ / CentOS 7+）
- Docker Engine 24+ 和 Docker Compose V2
- 可用磁盘空间 ≥ 10GB
- **独立部署的 Microsandbox 沙盒服务**（默认监听 `4002` 端口，API Key 与 Agent 配置一致）

安装 Docker（如未安装）：

```bash
curl -fsSL https://get.docker.com | bash
```

## 部署步骤

### 1. 解压

```bash
tar -xzf fina-offline-package-20260606.tar.gz
cd offline-package
```

### 2. 加载镜像

```bash
./load-images.sh
```

### 3. 启动服务

```bash
./start.sh
```

脚本会提示输入两项配置：
- **LLM Base URL** — 大模型 API 地址
- **API Key** — 大模型 API 密钥

输入后服务自动启动并后台运行。

### 4. 确认沙盒配置

Agent 默认通过以下配置连接沙盒（已写入 `.env`，可在启动后修改）：

```bash
SANDBOX_PROVIDER_TYPE=microsandbox-remote
MICROSANDBOX_SERVICE_BASE_URL=http://127.0.0.1:4002
MICROSANDBOX_API_KEY=qwertyuiop1234567890
MICROSANDBOX_MEMORY=1024
```

> ⚠️ **注意**：Microsandbox 服务如果运行在**同一台宿主机**上，
> Docker 容器内的 `127.0.0.1` 指向容器自身，无法访问宿主机端口。
> 请将 `MICROSANDBOX_SERVICE_BASE_URL` 改为宿主机 IP，例如：
> `http://172.17.0.1:4002` 或 `http://<内网IP>:4002`。
> 修改后重启服务：`docker compose -f docker-compose.offline.yml restart`

### 5. 初始化数据（仅首次运行）

```bash
./init-data.sh
```

初始化完成后会创建默认账户。

### 6. 访问

浏览器打开 `http://<服务器IP>:5701`

默认账户：`admin@fina.ai` / `admin`

## 配置 Nginx（可选）

如果希望通过 80 端口访问，复制 `nginx.conf` 到服务器：

```bash
cp nginx.conf /etc/nginx/conf.d/fina.conf
nginx -t && nginx -s reload
```

之后直接 `http://<服务器IP>` 访问，无需端口号。

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| Web 管理端 | 5701 | 离线包内 |
| Agent API | 5702 | 离线包内 |
| PostgreSQL | 5432 | 离线包内 |
| Microsandbox | 4002 | **独立部署**，不在离线包内 |

## 常用命令

```bash
# 查看日志
docker compose -f docker-compose.offline.yml logs -f

# 查看运行状态
docker compose -f docker-compose.offline.yml ps

# 停止服务
docker compose -f docker-compose.offline.yml down

# 重启服务
docker compose -f docker-compose.offline.yml restart
```

## 修改配置

如需修改大模型或沙盒配置，编辑 `.env` 文件后重启服务：

```bash
docker compose -f docker-compose.offline.yml restart
```
