# Fina Demo

本仓库用于 Fina Demo 项目，包含后端 Fastify 网关与前端 React（Refine）管理界面，提供 Agent 能力与文件上传。

## 项目职责与功能规划

- 后端：提供模型调用、任务编排、文件上传与数据处理能力，对外暴露统一 API
- 前端：提供管理界面与操作入口，支持任务配置、上传与结果查看
- 目标：以最小可用闭环展示端到端 Agent 能力，保持可扩展性与可维护性

## 如何启动 & 开发

### 后端 `agent/`

1. 安装依赖

```bash
cd agent
pnpm install
```

2. 配置环境（创建 `agent/.env`）

```bash
PORT=6203                  # 可选，默认 6203
VOLCENGINE_API_KEY2=...    # 必填：模型 API Key
UPLOAD_DIR=./uploads       # 可选，默认 agent/uploads
# DATABASE_URL=postgres://...  # 可选，如需启用 Postgres checkpoint
```

3. 开发模式

```bash
pnpm dev
```

### 前端 `ai_web/`

1. 安装依赖

```bash
cd ai_web
pnpm install
```

2. 配置环境（创建 `ai_web/.env.local`）

```bash
VITE_API_URL=http://localhost:6203   # 若后端有前缀如 /bff，请一并写入
```

3. 开发模式

```bash
pnpm dev
```
