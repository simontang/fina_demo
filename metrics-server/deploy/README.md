# metrics-server 部署说明

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
