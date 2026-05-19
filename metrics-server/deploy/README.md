# metrics-server 部署说明

## 发布到服务器

使用仓库**根目录**的 `docker-transfer.sh`，metrics-server 对应**服务编号 4**：

```bash
cd /path/to/fina_demo
./docker-transfer.sh 4
```

该脚本会从 GitHub Container Registry (ghcr.io) 拉取 `fina-demo-metrics-server:latest`，推送到火山引擎镜像仓库，并在服务器上执行部署。需事先将 metrics-server 镜像构建并推送到 ghcr.io（如通过 CI 或本地 `docker build` + `docker push`）。

## 首次部署或重启后容器反复重启时

容器内以非 root 用户 `appuser` (UID 101) 运行，挂载的日志目录必须在主机上对该 UID 可写，否则会因 **Permission denied** 无法写 `/app/logs/*.log` 而启动失败。

在服务器上执行（需 sudo）：

```bash
sudo mkdir -p /app/fina_demo/logs/metrics-server
sudo chown -R 101:101 /app/fina_demo/logs/metrics-server
```

然后重启服务：

```bash
cd /app/fina_demo && docker-compose restart metrics_server
```

## 镜像与 Compose

- 镜像：`finai-cn-shanghai.cr.volces.com/default/fina-demo-metrics-server:latest`
- 端口：5704
- 日志卷：`./logs/metrics-server:/app/logs`
