# Agent API Routes 清单

本文档整理了 Agent 服务中所有的 Controller 和 API URL 清单。

## 目录结构

```
agent/
├── src/
│   ├── controllers/          # 控制器层
│   │   ├── agentController.ts    # Agent 管理控制器
│   │   ├── fileController.ts     # 文件上传控制器
│   │   └── rtcController.ts      # RTC/语音聊天控制器
│   ├── routes/               # 路由定义
│   │   ├── datasets.ts          # 数据集路由
│   │   ├── pythonProxy.ts       # Python 服务代理路由
│   │   └── rtc.ts               # RTC 路由
│   └── gateway.ts           # 网关入口，注册所有路由
```

---

## 路由分类

### 1. LatticeGateway 框架路由

这些路由由 `@axiom-lattice/gateway` 框架提供，用于 Agent 运行和管理。

**路径前缀：**
- `/api/runs/*` - Agent 运行相关

**说明：** 这些路由由框架自动管理，用于 Agent 的创建、运行、状态查询等。

---

### 2. Python 服务代理路由

**文件：** `src/routes/pythonProxy.ts`  
**控制器：** 无（直接代理）

#### 2.1 代理所有 `/api/v1/*` 请求

| 方法 | 路径 | 说明 |
|------|------|------|
| `ALL` | `/api/v1` | 代理到 Python FastAPI 服务 |
| `ALL` | `/api/v1/*` | 代理到 Python FastAPI 服务的所有子路径 |

**目标服务：** 
- 环境变量：`PYTHON_API_URL` 或 `PREDICTION_API_URL` 或 `PY_API_URL`
- 默认：`http://localhost:5703`

**功能：** 将前端对 `/api/v1/*` 的请求反向代理到 Python 服务，保持 Python 端点的完整性。

---

### 3. RTC / 语音聊天路由

**文件：** `src/routes/rtc.ts`  
**控制器：** `src/controllers/rtcController.ts`

#### 3.1 语音聊天事件回调

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/rtc/voice_chat` | `voiceChatEventCallback` | Volcengine 语音聊天事件回调（主要用于日志记录） |

#### 3.2 RTC 代理请求

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/rtc/proxyFetch` | `rtcProxyFetch` | 统一的代理端点，前端使用 |

**请求体格式：**
```json
{
  "action": "StartVoiceChat" | "StopVoiceChat" | "UpdateVoiceChat" | "GenerateRtcToken",
  "params": {
    // 根据不同的 action 有不同的参数
  }
}
```

**支持的 Actions：**
- `StartVoiceChat` - 启动语音聊天
- `StopVoiceChat` - 停止语音聊天
- `UpdateVoiceChat` - 更新语音聊天配置
- `GenerateRtcToken` - 生成 RTC 访问令牌

#### 3.3 更新触发器

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/rtc/update-trigger` | `updateTrigger` | 可选的插件钩子（SendRoomUnicast） |

---

### 4. API 信息端点

**文件：** `src/gateway.ts`

#### 4.1 API 根路径信息

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/` | 返回 API 信息和端点列表 |

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

#### 4.2 健康检查

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

---

### 5. Agent 管理路由

**文件：** `src/gateway.ts`  
**控制器：** `src/controllers/agentController.ts`

#### 5.1 获取 Agent 列表

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/agents` | `getAgentList` | 获取所有可用的 Agent 列表 |

**响应格式：**
```json
{
  "success": true,
  "message": "Successfully retrieved agent list",
  "data": {
    "records": [AgentConfig[]],
    "total": number
  }
}
```

#### 5.2 获取单个 Agent

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/agents/:id` | `getAgent` | 根据 ID 获取单个 Agent 详情 |

**路径参数：**
- `id` - Agent 的唯一标识符

**响应格式：**
```json
{
  "success": true,
  "message": "Successfully retrieved agent",
  "data": AgentConfig
}
```

---

### 6. 文件管理路由

**文件：** `src/gateway.ts`  
**控制器：** `src/controllers/fileController.ts`

#### 6.1 上传单个文件

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/files/upload` | `uploadFile` | 上传单个文件 |

**请求：** `multipart/form-data`  
**限制：**
- 最大文件大小：50MB
- 上传目录：`UPLOAD_DIR` 环境变量或 `agent/uploads`

**响应格式：**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "id": "unique_filename",
  "originalName": "original_filename.ext",
  "size": 1024,
  "mimetype": "application/pdf"
}
```

#### 6.2 上传多个文件

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `POST` | `/api/files/upload-multiple` | `uploadMultipleFiles` | 上传多个文件 |

**请求：** `multipart/form-data`  
**限制：**
- 最大文件数：10 个
- 每个文件最大大小：50MB

**响应格式：**
```json
{
  "success": true,
  "message": "2 file(s) uploaded successfully",
  "files": [
    {
      "id": "unique_filename1",
      "originalName": "file1.ext",
      "size": 1024,
      "mimetype": "application/pdf"
    }
  ],
  "total": 2
}
```

#### 6.3 获取已上传文件列表

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/files` | `getUploadedFiles` | 获取所有已上传的文件列表 |

**响应格式：**
```json
{
  "success": true,
  "message": "Successfully retrieved uploaded files",
  "files": ["file1.ext", "file2.ext"],
  "total": 2
}
```

#### 6.4 删除文件

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `DELETE` | `/api/files/:filename` | `deleteFile` | 删除指定的文件 |

**路径参数：**
- `filename` - 要删除的文件名

**响应格式：**
```json
{
  "success": true,
  "message": "File deleted successfully"
}
```

---

### 7. 数据集管理路由

**文件：** `src/routes/datasets.ts`  
**控制器：** 内联在路由文件中

#### 7.1 获取数据集列表

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/datasets` | `getDatasetsList` | 获取所有数据集列表 |

**响应格式：**
```json
{
  "success": true,
  "data": [
    {
      "id": "sales_data",
      "name": "Sales Dataset",
      "description": "...",
      "table_name": "sales_data",
      "type": "sales",
      "row_count": 1000,
      "created_at": "2026-01-18T00:00:00Z",
      "updated_at": "2026-01-18T00:00:00Z",
      "tags": ["sales", "historical"]
    }
  ],
  "total": 1
}
```

#### 7.2 获取数据集详情

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/datasets/:id` | `getDatasetDetail` | 获取指定数据集的详细信息 |

**路径参数：**
- `id` - 数据集的唯一标识符

**响应格式：**
```json
{
  "success": true,
  "data": {
    "id": "sales_data",
    "name": "Sales Dataset",
    "description": "...",
    "table_name": "sales_data",
    "type": "sales",
    "row_count": 1000,
    "column_count": 8,
    "created_at": "2026-01-18T00:00:00Z",
    "updated_at": "2026-01-18T00:00:00Z",
    "tags": ["sales", "historical"],
    "time_range": {
      "min": "2020-01-01",
      "max": "2023-12-31"
    },
    "columns": [
      {
        "name": "invoice_date",
        "type": "date",
        "stats": {
          "columnName": "invoice_date",
          "dataType": "date",
          "nullCount": 0
        }
      }
    ]
  }
}
```

#### 7.3 预览数据集数据

| 方法 | 路径 | 控制器方法 | 说明 |
|------|------|-----------|------|
| `GET` | `/api/datasets/:id/preview` | `previewDataset` | 分页预览数据集数据 |

**路径参数：**
- `id` - 数据集的唯一标识符

**查询参数：**
- `page` - 页码（默认：1）
- `pageSize` - 每页大小（默认：20）

**响应格式：**
```json
{
  "success": true,
  "data": {
    "records": [
      {
        "id": 1,
        "invoice_no": "INV001",
        "stock_code": "SKU001",
        ...
      }
    ],
    "total": 1000,
    "page": 1,
    "pageSize": 20,
    "totalPages": 50
  }
}
```

---

## 路由注册顺序

在 `src/gateway.ts` 的 `registerRoutes` 函数中，路由按以下顺序注册：

1. **Python Proxy 路由** (`/api/v1/*`)
2. **RTC 路由** (`/api/rtc/*`)
3. **LatticeGateway 路由**（根路径）
4. **LatticeGateway 路由**（`/bff` 前缀）
5. **LatticeGateway 路由**（`/api/bff` 前缀）
6. **自定义 BFF 路由**（`/bff` 前缀）
7. **自定义 BFF 路由**（`/api/bff` 前缀）

---

## 环境变量

### Python 服务代理
- `PYTHON_API_URL` - Python API 服务地址（默认：`http://localhost:5703`）
- `PREDICTION_API_URL` - 同 PYTHON_API_URL
- `PY_API_URL` - 同 PYTHON_API_URL

### RTC / 语音聊天
- `VOLCENGINE_APP_ID` - Volcengine 应用 ID（必需）
- `VOLCENGINE_APP_KEY` - Volcengine 应用密钥（必需）
- `VOLC_ACCESSKEY` - Volcengine Access Key（必需）
- `VOLC_SECRETKEY` - Volcengine Secret Key（必需）
- `VOLC_SPEECH_APP_ID` - Volcengine 语音应用 ID（必需）
- `COZEBOT_BOT_ID` - Coze Bot ID（可选，用于 CozeBot 模式）
- `COZEBOT_APIKEY` - Coze API Key（可选，用于 CozeBot 模式）

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

默认端口：**5702**

可通过 `startServer(port)` 函数参数或环境变量配置。

---

## 注意事项

1. **统一路径前缀：** 所有业务 API 统一使用 `/api/*` 前缀，不再有 `/bff/*` 变体。

2. **LatticeGateway 路由：** 框架提供的路由（`/api/runs/*`）保持不变，由框架自动管理。

3. **文件上传限制：**
   - 单个文件最大：50MB
   - 每次请求最多：10 个文件
   - 文件名会自动添加时间戳以确保唯一性

4. **数据集配置：** 数据集配置从 `prediction_app/config/datasets.json` 读取，系统会尝试多个可能的路径。

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

- 2026-01-21: 初始版本，整理所有路由和控制器
