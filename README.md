# Fina Demo

Demo repository that combines:
- **agent/**：Node.js (Fastify) 网关与 Agent 编排（含 Data Agent、Research Agent、Voice Agent 等）
- **prediction_app/**：Python (FastAPI) 数据集与预测 API（销售预测、RFM、库存等）
- **ai_web/**：React (Refine) 管理端 UI
- **metrics-server/**：Java (Spring Boot) 指标与语义查询服务
- **b1s/**：Java (Spring Boot) SAP B1 Service Layer 代理与 B1 指标服务
- **cdp-service/**：Java (Spring Boot) CDP 分群定义、分群数据与 SQL processing 服务
- **document_service/**：Python (FastAPI + Celery) 文档解析统一 IO 服务，上传文件后异步分发到 Datalab、MinerU、TextIn、Qwen OCR、PaddleOCR 等远程 engine，并输出 Markdown/JSON。
- **file-service/**：Java (Spring Boot) 多租户文件服务：路径寻址、不可变版本、内容去重（端口 5707，nginx `/api/filesvc/`）。
- **webhook-service/**：Hookdeck Outpost 出站 Webhook 投递：多租户、Standard Webhooks 签名、重试与 Portal（端口 5708，nginx `/api/webhooks/`）。

功能亮点：
- **Data Agent**：业务数据分析智能体，支持自然语言查数、多步分析与报告生成；详见 [开发 Data Agent 指南](docs/DEVELOPING_A_DATA_AGENT.md)。
- **Voice Agent**：通过 `agent` 的 `/api/rtc/*` 与 Volcengine RTC 语音对话，UI 在 `/admin/agents/voice/rtc`。

## 文档

- **[文档索引](docs/README.md)**：项目文档列表与分类。
- **[开发 Data Agent](docs/DEVELOPING_A_DATA_AGENT.md)**：如何开发/扩展 Data Agent（后端注册、技能、子代理、前端对接）。
- [微服务与路由](MICROSERVICES_PORTS_AND_ROUTES.md)、[环境变量与部署](ENV_FILE_GUIDE.md)、[数据集管理规格](DATASET_MANAGEMENT_SPEC.md) 等见文档索引。

## Quickstart (Dev)

### 1) Start the Python API (`prediction_app/`, port 8000)

```bash
cd prediction_app
./start_api.sh
```

### 2) Start the gateway (`agent/`, port 6203)

```bash
cd agent
pnpm install
pnpm dev
```

Environment (`agent/.env`):
```bash
PORT=6203
VOLCENGINE_API_KEY2=...
UPLOAD_DIR=./uploads
VOLCENGINE_APP_ID=...
VOLCENGINE_APP_KEY=...
VOLC_ACCESSKEY=...
VOLC_SECRETKEY=...
```

### 3) Start the admin UI (`ai_web/`, port 5173)

```bash
cd ai_web
pnpm install
pnpm dev
```

Notes:
- Vite proxies:
  - `/api/v1/*` -> `http://localhost:6203` (gateway, reverse-proxies to Python)
  - `/api/*` -> `http://localhost:6203` (gateway)

## Docker (Compose)

Bring up the full demo stack (CSV + model assets are loaded from disk; no DB required):

```bash
docker compose up --build
```

Notes:
- `raw_data/` and `models/` are baked into the `prediction_app` image during build (no host bind mounts needed).
- If you update CSVs or model files locally, re-run `docker compose up --build`.

Services:
- Fina Admin UI: http://localhost:5701/admin/
- Evario Admin UI: http://localhost:5711/admin/
- Agent gateway: http://localhost:5702
- Python API (debug): http://localhost:5703
- Metrics server: http://localhost:5704
- B1S service: http://localhost:5705
- CDP service: http://localhost:5706
- Document service: http://localhost:5710/docs
- File service: http://localhost:5707/actuator/health
- Webhook service (Outpost): http://localhost:5708
- Document API via nginx: `/api/documents/v1/*`

Optional (AI insights / explanations):
- Export `VOLCENGINE_API_KEY2` before running compose, or create a local `.env` file (not committed) at repo root:

```bash
export VOLCENGINE_API_KEY2=...
docker compose up --build
```

Document service uses `document_service/.env` for engine credentials, object
storage, Redis, and database settings. Create it from
`document_service/.env.example` before starting the full stack if it is missing.
