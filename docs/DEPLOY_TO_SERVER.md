# 发版到服务器

## 流程概览

1. **CI**（GitHub Actions）：推送到 `main` 后自动构建并推送镜像到 **GitHub Container Registry**（ghcr.io）。
2. **docker-transfer.sh**：从 ghcr.io 拉取镜像 → 推送到 **火山引擎镜像仓库** → 可选部署到服务器。

服务器：`deploy@14.103.152.204`，路径：`/app/fina_demo`。

---

## 方式一：完整发版（推荐）

在项目根目录执行：

```bash
# 1. 设置凭证（不要提交到 Git）
export GITHUB_TOKEN="你的 GitHub PAT"
export VOLCANO_USERNAME="火山引擎镜像仓库用户名"
export VOLCANO_PASSWORD="火山引擎镜像仓库密码"

# 2. 可选：使用生产环境变量文件
# 有 .env.prod 时会自动使用；或明确指定：
# export ENV_FILE=.env.prod

# 3. 传输全部三个服务镜像并部署到服务器
./docker-transfer.sh
```

脚本会依次：
- 从 ghcr.io 拉取 `fina-demo-agent`、`fina-demo-prediction-app`、`fina-demo-ai-web` 的 `latest`
- 打标签并推送到 `finai-cn-shanghai.cr.volces.com/default/...`
- 上传 `docker-compose.prod.yml`（作为服务器上的 `docker-compose.yml`）和 `.env`
- 在服务器上执行 `docker-compose pull` 和 `docker-compose up -d`

---

## 方式二：只发部分服务

```bash
# 仅 agent 和 ai_web（1=agent, 2=prediction_app, 3=ai_web）
./docker-transfer.sh 1 3
```

---

## 方式三：只部署、不传镜像

镜像已在火山引擎仓库更新时（例如刚跑完方式一），只更新服务器上的编排并重启：

```bash
./docker-transfer.sh --deploy-only
```

---

## 方式四：只传镜像、不部署

```bash
./docker-transfer.sh --no-deploy
```

---

## 发版前检查

- [ ] 代码已合并到 `main`，且 GitHub Actions CI 已成功（镜像在 ghcr.io 可用）。
- [ ] 本地已配置 `GITHUB_TOKEN`、`VOLCANO_USERNAME`、`VOLCANO_PASSWORD`。
- [ ] 生产配置使用 `.env.prod` 或通过 `ENV_FILE` 指定，避免误用本地 `.env`。

---

## 常见问题

- **拉取 ghcr.io 失败**：确认 CI 已跑完，且 `GITHUB_TOKEN` 有 `read:packages` 权限。
- **推送火山引擎失败**：检查 `VOLCANO_USERNAME` / `VOLCANO_PASSWORD` 及仓库权限。
- **服务器部署失败**：检查 SSH `deploy@14.103.152.204` 是否可达，以及服务器上 Docker 与 `docker-compose` 是否正常。

环境变量与多项目配置详见 [ENV_FILE_GUIDE.md](../ENV_FILE_GUIDE.md)。
