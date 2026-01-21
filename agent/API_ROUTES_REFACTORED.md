# Agent API Routes 清单（重构后 - 移除 BFF Pattern）

本文档展示了移除 BFF pattern 后的新路由结构。

---

## 路由总览表

| 分类 | 方法 | 路径 | 控制器 | 说明 |
|------|------|------|--------|------|
| **API Info** | GET | `/api/` | - | API 信息和端点列表 |
| **Health** | GET | `/api/health` | - | 健康检查 |
| **LatticeGateway** | ALL | `/api/runs/*` | - | Agent 运行管理（框架提供） |
| **Python Proxy** | ALL | `/api/v1` | - | 代理到 Python FastAPI |
| **Python Proxy** | ALL | `/api/v1/*` | - | 代理到 Python FastAPI |
| **RTC** | POST | `/api/rtc/voice_chat` | `rtcController.voiceChatEventCallback` | 语音聊天事件回调 |
| **RTC** | POST | `/api/rtc/proxyFetch` | `rtcController.rtcProxyFetch` | RTC 统一代理端点 |
| **RTC** | POST | `/api/rtc/update-trigger` | `rtcController.updateTrigger` | 更新触发器 |
| **Agent** | GET | `/api/agents` | `agentController.getAgentList` | 获取 Agent 列表 |
| **Agent** | GET | `/api/agents/:id` | `agentController.getAgent` | 获取单个 Agent |
| **File** | POST | `/api/files/upload` | `fileController.uploadFile` | 上传单个文件 |
| **File** | POST | `/api/files/upload-multiple` | `fileController.uploadMultipleFiles` | 上传多个文件 |
| **File** | GET | `/api/files` | `fileController.getUploadedFiles` | 获取文件列表 |
| **File** | DELETE | `/api/files/:filename` | `fileController.deleteFile` | 删除文件 |
| **Dataset** | GET | `/api/datasets` | `getDatasetsList` | 获取数据集列表 |
| **Dataset** | GET | `/api/datasets/:id` | `getDatasetDetail` | 获取数据集详情 |
| **Dataset** | GET | `/api/datasets/:id/preview` | `previewDataset` | 预览数据集（分页） |

---

## 路由分类说明

### 1. API 信息端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/` | 返回 API 信息和所有可用端点 |

**响应示例：**
```json
{
  "name": "Research Data Agent API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/api/health",
    "agents": "/api/agents",
    "files": "/api/files",
    "upload": "/api/files/upload",
    "uploadMultiple": "/api/files/upload-multiple",
    "datasets": "/api/datasets",
    "rtc": "/api/rtc",
    "runs": "/api/runs"
  }
}
```

### 2. 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 返回服务健康状态 |

**响应示例：**
```json
{
  "status": "ok",
  "timestamp": "2026-01-21T12:00:00.000Z"
}
```

### 3. LatticeGateway 框架路由

这些路由由 `@axiom-lattice/gateway` 框架提供，用于 Agent 运行和管理。

| 方法 | 路径 | 说明 |
|------|------|------|
| `ALL` | `/api/runs/*` | Agent 运行相关操作 |

**说明：** 这些路由由框架自动管理，用于 Agent 的创建、运行、状态查询等。

### 4. Python 服务代理

| 方法 | 路径 | 说明 |
|------|------|------|
| `ALL` | `/api/v1` | 代理到 Python FastAPI 服务 |
| `ALL` | `/api/v1/*` | 代理到 Python FastAPI 服务的所有子路径 |

**目标服务：** 
- 环境变量：`PYTHON_API_URL` 或 `PREDICTION_API_URL` 或 `PY_API_URL`
- 默认：`http://localhost:5703`

**示例路径：**
- `/api/v1/datasets` → Python 服务的数据集端点
- `/api/v1/models` → Python 服务的模型端点
- `/api/v1/datasets/{id}/rfm` → RFM 分析端点

### 5. RTC / 语音聊天路由

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/rtc/voice_chat` | `voiceChatEventCallback` | Volcengine 语音聊天事件回调 |
| `POST` | `/api/rtc/proxyFetch` | `rtcProxyFetch` | RTC 统一代理端点 |
| `POST` | `/api/rtc/update-trigger` | `updateTrigger` | 更新触发器 |

**RTC ProxyFetch Actions：**
- `StartVoiceChat` - 启动语音聊天
- `StopVoiceChat` - 停止语音聊天
- `UpdateVoiceChat` - 更新语音聊天配置
- `GenerateRtcToken` - 生成 RTC 访问令牌

### 6. Agent 管理路由

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/agents` | `getAgentList` | 获取所有可用的 Agent 列表 |
| `GET` | `/api/agents/:id` | `getAgent` | 获取单个 Agent 详情 |

### 7. 文件管理路由

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/files/upload` | `uploadFile` | 上传单个文件（最大 50MB） |
| `POST` | `/api/files/upload-multiple` | `uploadMultipleFiles` | 上传多个文件（最多 10 个） |
| `GET` | `/api/files` | `getUploadedFiles` | 获取已上传文件列表 |
| `DELETE` | `/api/files/:filename` | `deleteFile` | 删除指定文件 |

### 8. 数据集管理路由

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/datasets` | `getDatasetsList` | 获取数据集列表 |
| `GET` | `/api/datasets/:id` | `getDatasetDetail` | 获取数据集详情 |
| `GET` | `/api/datasets/:id/preview` | `previewDataset` | 预览数据集数据（分页） |

**查询参数（preview）：**
- `page` - 页码（默认：1）
- `pageSize` - 每页大小（默认：20）

---

## 路径对比（重构前后）

| 功能 | 重构前 | 重构后 |
|------|--------|--------|
| API 信息 | `/bff/` 或 `/api/bff/` | `/api/` |
| 健康检查 | 无 | `/api/health` |
| Agent 列表 | `/bff/agents` 或 `/api/bff/agents` | `/api/agents` |
| Agent 详情 | `/bff/agents/:id` 或 `/api/bff/agents/:id` | `/api/agents/:id` |
| 文件上传 | `/bff/files/upload` 或 `/api/bff/files/upload` | `/api/files/upload` |
| 文件列表 | `/bff/files` 或 `/api/bff/files` | `/api/files` |
| 数据集列表 | `/bff/datasets` 或 `/api/bff/datasets` | `/api/datasets` |
| 数据集详情 | `/bff/datasets/:id` 或 `/api/bff/datasets/:id` | `/api/datasets/:id` |
| Python 代理 | `/api/v1/*` | `/api/v1/*` ✅（不变） |
| RTC 路由 | `/api/rtc/*` | `/api/rtc/*` ✅（不变） |
| LatticeGateway | `/api/runs/*` | `/api/runs/*` ✅（不变） |

---

## 控制器文件映射

| 控制器文件 | 功能 | 路由文件 |
|-----------|------|---------|
| `src/controllers/agentController.ts` | Agent 管理 | `src/gateway.ts` |
| `src/controllers/fileController.ts` | 文件上传管理 | `src/gateway.ts` |
| `src/controllers/rtcController.ts` | RTC/语音聊天 | `src/routes/rtc.ts` |
| - | 数据集管理 | `src/routes/datasets.ts` |
| - | Python 代理 | `src/routes/pythonProxy.ts` |

---

## 环境变量

### Python 服务代理
- `PYTHON_API_URL` - Python API 服务地址（默认：`http://localhost:5703`）

### RTC / 语音聊天
- `VOLCENGINE_APP_ID` - Volcengine 应用 ID（必需）
- `VOLCENGINE_APP_KEY` - Volcengine 应用密钥（必需）
- `VOLC_ACCESSKEY` - Volcengine Access Key（必需）
- `VOLC_SECRETKEY` - Volcengine Secret Key（必需）
- `VOLC_SPEECH_APP_ID` - Volcengine 语音应用 ID（必需）
- `COZEBOT_BOT_ID` - Coze Bot ID（可选）
- `COZEBOT_APIKEY` - Coze API Key（可选）

### 文件上传
- `UPLOAD_DIR` - 文件上传目录（默认：`agent/uploads`）

### 数据库连接
- `DB_HOST` - 数据库主机
- `DB_PORT` - 数据库端口（默认：5432）
- `DB_NAME` - 数据库名称
- `DB_USER` - 数据库用户
- `DB_PASSWORD` - 数据库密码

---

## 端口配置

默认端口：**5702**（开发环境）或 **5702**（生产环境）

---

## 注意事项

1. **统一路径前缀：** 所有业务 API 统一使用 `/api/*` 前缀，不再有 `/bff/*` 变体。

2. **LatticeGateway 路由：** 框架提供的路由（`/api/runs/*`）保持不变，由框架自动管理。

3. **文件上传限制：**
   - 单个文件最大：50MB
   - 每次请求最多：10 个文件
   - 文件名会自动添加时间戳以确保唯一性

4. **数据集配置：** 数据集配置从 `prediction_app/config/datasets.json` 读取。

5. **错误处理：** 所有路由都包含错误处理，返回统一的错误格式：
   ```json
   {
     "success": false,
     "error": "Error message",
     "message": "Error message"
   }
   ```

---

## 更新日志

- 2026-01-21: 重构版本 - 移除所有 BFF pattern，统一使用 `/api/*` 前缀
