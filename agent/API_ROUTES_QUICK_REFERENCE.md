# Agent API Routes 快速参考

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

## 控制器文件映射

| 控制器文件 | 功能 | 路由文件 |
|-----------|------|---------|
| `src/controllers/agentController.ts` | Agent 管理 | `src/gateway.ts` |
| `src/controllers/fileController.ts` | 文件上传管理 | `src/gateway.ts` |
| `src/controllers/rtcController.ts` | RTC/语音聊天 | `src/routes/rtc.ts` |
| - | 数据集管理 | `src/routes/datasets.ts` |
| - | Python 代理 | `src/routes/pythonProxy.ts` |

## RTC ProxyFetch Actions

| Action | 说明 | 必需参数 |
|--------|------|---------|
| `StartVoiceChat` | 启动语音聊天 | `roomId`, `userId`, `taskId` |
| `StopVoiceChat` | 停止语音聊天 | `roomId`, `taskId` |
| `UpdateVoiceChat` | 更新语音聊天 | `roomId`, `taskId`, `updateFields` |
| `GenerateRtcToken` | 生成 RTC 令牌 | `roomId`, `userId` |

## 数据集预览查询参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | number | 1 | 页码 |
| `pageSize` | number | 20 | 每页大小 |

## 文件上传限制

- 单个文件最大：50MB
- 每次请求最多：10 个文件
- 上传目录：`UPLOAD_DIR` 环境变量或 `agent/uploads`

## 端口

默认端口：**5702**（开发环境）或 **5702**（生产环境）

## 完整文档

详细文档请参考：[API_ROUTES.md](./API_ROUTES.md)
