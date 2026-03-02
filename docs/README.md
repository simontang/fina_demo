# Fina Demo 项目文档

本目录为 Fina Demo 的文档索引与说明。根目录下部分文档仍保留在原位，此处统一列出并分类。

## 文档索引

### 开发与架构

| 文档 | 说明 |
|------|------|
| [发版到服务器](./DEPLOY_TO_SERVER.md) | 如何将新版本发到生产服务器（镜像传输与部署） |
| [架构图](./ARCHITECTURE.md) | 系统总览、服务与请求流、Data Agent 内部结构（含 Mermaid 图） |
| [开发 Data Agent 指南](./DEVELOPING_A_DATA_AGENT.md) | 如何在本项目中开发一个 Data Agent（业务数据分析智能体） |
| [Data Agent 提示词整理](./DATA_AGENT_PROMPTS.md) | Data Agent 主/子代理及全部技能的提示词汇总 |
| [Deep Research 提示词整理](./DEEP_RESEARCH_PROMPTS.md) | Deep Research Agent 主/子代理的提示词汇总 |
| [微服务、端口与路由](../MICROSERVICES_PORTS_AND_ROUTES.md) | 服务划分、端口、API 命名空间与 URL 设计 |
| [环境变量文件管理](../ENV_FILE_GUIDE.md) | 开发/生产环境变量、部署时 .env 优先级与安全 |

### 功能与规格

| 文档 | 说明 |
|------|------|
| [数据集管理功能规格](../DATASET_MANAGEMENT_SPEC.md) | 数据集列表、详情、预览、列统计等前端与 API 规格 |

### Agent 相关（agent 子目录）

| 文档 | 说明 |
|------|------|
| [agent/API_ROUTES_QUICK_REFERENCE.md](../agent/API_ROUTES_QUICK_REFERENCE.md) | Agent 服务 API 路由速查 |
| [agent/API_ROUTES.md](../agent/API_ROUTES.md) | Agent 路由详细说明 |
| [agent/API_ROUTES_REFACTORED.md](../agent/API_ROUTES_REFACTORED.md) | 重构后的路由设计 |
| [agent/REFACTOR_REMOVE_BFF_PROPOSAL.md](../agent/REFACTOR_REMOVE_BFF_PROPOSAL.md) | 移除 BFF 的改造提案 |
| [agent/skills/*/SKILL.md](../agent/skills/) | 各技能（Skill）的 Markdown 说明，如 sql-query、analyst、notebook-report 等 |

### 前端与预测服务

| 文档 | 说明 |
|------|------|
| [ai_web/STYLE_GUIDE.md](../ai_web/STYLE_GUIDE.md) | 前端样式与 UI 规范 |
| [prediction_app/README.md](../prediction_app/README.md) | Python 预测服务说明 |
| [prediction_app/README_STARTUP.md](../prediction_app/README_STARTUP.md) | 预测服务启动与脚本说明 |
| [prediction_app/scripts/README.md](../prediction_app/scripts/README.md) | 数据导入等脚本说明 |

## 项目结构速览

```
fina_demo/
├── agent/           # Node.js 网关 + Lattice Agent 编排（含 Data Agent）
├── ai_web/           # React (Refine) 管理端 UI
├── prediction_app/   # Python (FastAPI) 数据集与预测 API
├── docs/             # 本目录，文档索引与开发指南
├── docker-compose*.yml
├── docker-transfer.sh
├── README.md         # 项目总览与快速开始
├── ENV_FILE_GUIDE.md
├── MICROSERVICES_PORTS_AND_ROUTES.md
└── DATASET_MANAGEMENT_SPEC.md
```

快速开始与运行说明见根目录 [README.md](../README.md)。开发 Data Agent 见 [开发 Data Agent 指南](./DEVELOPING_A_DATA_AGENT.md)。
