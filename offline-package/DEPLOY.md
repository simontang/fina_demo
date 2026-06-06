# Fina Demo 离线部署指南

## 环境要求

- Linux x86_64（Ubuntu 20.04+ / CentOS 7+）
- Docker Engine 24+ 和 Docker Compose V2
- 可用磁盘空间 ≥ 10GB

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

### 4. 初始化数据（仅首次运行）

```bash
./init-data.sh
```

初始化完成后会创建默认账户。

### 5. 访问

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

| 服务 | 端口 |
|------|------|
| Web 管理端 | 5701 |
| Agent API | 5702 |
| Sandbox | 8080 |
| PostgreSQL | 5432 |

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

如需修改大模型配置，编辑 `.env` 文件后重启服务：

```bash
docker compose -f docker-compose.offline.yml restart
```
