# 移除 BFF Pattern 重构提案

## 目标

移除所有 `/bff/*` 和 `/api/bff/*` 路径前缀，将所有 API 路由统一注册到 `/api/*` 下，简化路由结构并提高一致性。

---

## 当前状态分析

### 当前路由结构

1. **LatticeGateway 框架路由**
   - `/api/runs/*` - Agent 运行管理（必需，框架要求）
   - `/bff/api/runs/*` - BFF 前缀版本（冗余）
   - `/api/bff/api/runs/*` - API/BFF 前缀版本（冗余）

2. **Python 服务代理**
   - `/api/v1/*` - 代理到 Python FastAPI ✅（保持不变）

3. **RTC / 语音聊天**
   - `/api/rtc/*` - RTC 相关路由 ✅（保持不变）

4. **自定义业务路由（当前使用 BFF 前缀）**
   - `/bff/agents` → 应改为 `/api/agents`
   - `/bff/files/*` → 应改为 `/api/files/*`
   - `/bff/datasets/*` → 应改为 `/api/datasets/*`
   - `/bff/` (根信息) → 应改为 `/api/` 或 `/api/health`

### 前端使用情况

1. **直接使用 `/api/v1/*`** - 无需修改
   - `fetch("/api/v1/datasets")`
   - `fetch("/api/v1/models/available")`
   - `fetch("/api/rtc/proxyFetch")`

2. **使用 BFF 路径** - 需要修改
   - `normalizeBffBaseUrl()` 函数会添加 `/bff` 后缀
   - 主要用于 LatticeGateway SDK 的 `baseURL` 配置

3. **硬编码路径** - 需要修改
   - `http://localhost:6203` (部分页面)

### Nginx 配置

当前 nginx 配置：
```nginx
location /api/bff/ {
    proxy_pass http://127.0.0.1:6203/bff/;
}
```

需要改为：
```nginx
location /api/ {
    proxy_pass http://127.0.0.1:6203/api/;
}
```

---

## 重构方案

### 1. Agent 端路由重构

#### 1.1 修改 `src/gateway.ts`

**当前代码：**
```typescript
function registerCustomBffRoutes(app: FastifyInstance): void {
  app.get("/", async () => { ... });
  app.get("/agents", getAgentList);
  app.get("/agents/:id", getAgent);
  app.post("/files/upload", uploadFile);
  // ...
  registerDatasetRoutes(app);
}

export const registerRoutes = (app: FastifyInstance): void => {
  registerPythonProxyRoutes(app);
  registerRtcRoutes(app);
  
  // LatticeGateway routes at root
  LatticeGateway.registerLatticeRoutes(app);
  
  // BFF prefixes (冗余)
  app.register(async (bffApp) => {
    LatticeGateway.registerLatticeRoutes(bffApp);
  }, { prefix: "/bff" });
  
  app.register(async (apiBffApp) => {
    LatticeGateway.registerLatticeRoutes(apiBffApp);
  }, { prefix: "/api/bff" });
  
  // Custom BFF routes (需要移除)
  app.register(async (agentApp) => registerCustomBffRoutes(agentApp), { prefix: "/bff" });
  app.register(async (agentApp) => registerCustomBffRoutes(agentApp), { prefix: "/api/bff" });
};
```

**重构后代码：**
```typescript
function registerApiRoutes(app: FastifyInstance): void {
  // API 根路径信息
  app.get("/", async () => {
    return {
      name: "Research Data Agent API",
      version: "1.0.0",
      status: "running",
      endpoints: {
        health: "/api/health",
        agents: "/api/agents",
        files: "/api/files",
        upload: "/api/files/upload",
        uploadMultiple: "/api/files/upload-multiple",
        datasets: "/api/datasets",
        rtc: "/api/rtc",
        runs: "/api/runs",
      },
    };
  });

  // Health check
  app.get("/health", async () => {
    return { status: "ok", timestamp: new Date().toISOString() };
  });

  // Agent management endpoints
  app.get("/agents", getAgentList);
  app.get("/agents/:id", getAgent);

  // File upload endpoints
  app.post("/files/upload", uploadFile);
  app.post("/files/upload-multiple", uploadMultipleFiles);
  app.get("/files", getUploadedFiles);
  app.delete("/files/:filename", deleteFile);

  // Dataset management endpoints
  registerDatasetRoutes(app);
}

export const registerRoutes = (app: FastifyInstance): void => {
  // Python prediction service reverse-proxy
  registerPythonProxyRoutes(app);

  // RTC / Voice Chat APIs
  registerRtcRoutes(app);

  // Register LatticeGateway routes at root path (required for sub-agent calls)
  LatticeGateway.registerLatticeRoutes(app);

  // Register custom API routes under /api prefix
  app.register(async (apiApp) => registerApiRoutes(apiApp), { prefix: "/api" });
};
```

#### 1.2 路由注册顺序

新的注册顺序：
1. Python Proxy (`/api/v1/*`)
2. RTC Routes (`/api/rtc/*`)
3. LatticeGateway Routes (`/api/runs/*`) - 框架必需
4. Custom API Routes (`/api/*`) - 业务路由

---

### 2. 前端代码修改

#### 2.1 修改 `ai_web/src/components/chating/index.tsx`

**当前代码：**
```typescript
function normalizeBffBaseUrl(raw?: string): string {
  const v = String(raw || "").trim().replace(/\/$/, "");
  if (!v) return "http://localhost:6203/bff";
  if (v.endsWith("/bff")) return v;
  return `${v}/bff`;
}

const apiUrl = normalizeBffBaseUrl(import.meta.env.VITE_API_URL as string | undefined);
```

**重构后代码：**
```typescript
function normalizeApiBaseUrl(raw?: string): string {
  const v = String(raw || "").trim().replace(/\/$/, "");
  if (!v) return "http://localhost:6203/api";
  if (v.endsWith("/api")) return v;
  return `${v}/api`;
}

const apiUrl = normalizeApiBaseUrl(import.meta.env.VITE_API_URL as string | undefined);
```

#### 2.2 修改其他硬编码路径

**需要修改的文件：**
- `ai_web/src/pages/agents/overview/index.tsx`
- `ai_web/src/pages/assets/skills/index.tsx`

**修改方式：**
```typescript
// 从
const apiUrl = "http://localhost:6203";

// 改为
const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:6203/api";
```

---

### 3. Nginx 配置修改

#### 3.1 修改 `nginx-fina-demo.conf`

**当前配置：**
```nginx
location /api/bff/ {
    proxy_pass http://127.0.0.1:6203/bff/;
    # ...
}

location /api/ {
    proxy_pass http://127.0.0.1:6203/api/;
    # ...
}
```

**重构后配置：**
```nginx
# 统一使用 /api/* 代理
location /api/ {
    proxy_pass http://127.0.0.1:6203/api/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # Support streaming responses / SSE if needed
    proxy_buffering off;
    
    # Increase timeout for long-running requests
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
}
```

**移除：**
- `/api/bff/` location 块（不再需要）

---

### 4. 环境变量更新

#### 4.1 Docker Compose

**当前：**
```yaml
environment:
  - VITE_API_URL=/api
```

**保持不变** - 前端仍然使用 `/api` 作为基础路径

#### 4.2 开发环境

**Vite 配置** (`ai_web/vite.config.ts`) - 通常无需修改，因为代理规则是 `/api/*` → agent

---

## 新的路由结构

### 完整路由清单（重构后）

| 分类 | 方法 | 路径 | 说明 |
|------|------|------|------|
| **API Info** | GET | `/api/` | API 信息和端点列表 |
| **Health** | GET | `/api/health` | 健康检查 |
| **LatticeGateway** | ALL | `/api/runs/*` | Agent 运行管理（框架提供） |
| **Python Proxy** | ALL | `/api/v1/*` | 代理到 Python FastAPI |
| **RTC** | POST | `/api/rtc/voice_chat` | 语音聊天事件回调 |
| **RTC** | POST | `/api/rtc/proxyFetch` | RTC 统一代理端点 |
| **RTC** | POST | `/api/rtc/update-trigger` | 更新触发器 |
| **Agent** | GET | `/api/agents` | 获取 Agent 列表 |
| **Agent** | GET | `/api/agents/:id` | 获取单个 Agent |
| **File** | POST | `/api/files/upload` | 上传单个文件 |
| **File** | POST | `/api/files/upload-multiple` | 上传多个文件 |
| **File** | GET | `/api/files` | 获取文件列表 |
| **File** | DELETE | `/api/files/:filename` | 删除文件 |
| **Dataset** | GET | `/api/datasets` | 获取数据集列表 |
| **Dataset** | GET | `/api/datasets/:id` | 获取数据集详情 |
| **Dataset** | GET | `/api/datasets/:id/preview` | 预览数据集（分页） |

---

## 迁移步骤

### Phase 1: Agent 端修改（向后兼容）

1. ✅ 添加新的 `/api/*` 路由
2. ✅ 保留旧的 `/bff/*` 路由（临时，用于过渡）
3. ✅ 更新根路径信息端点

### Phase 2: 前端修改

1. ✅ 修改 `normalizeBffBaseUrl` → `normalizeApiBaseUrl`
2. ✅ 更新硬编码的 API URL
3. ✅ 测试所有功能

### Phase 3: Nginx 配置更新

1. ✅ 更新 nginx 配置
2. ✅ 测试代理功能

### Phase 4: 清理（可选）

1. ⚠️ 移除旧的 `/bff/*` 路由注册
2. ⚠️ 更新文档

---

## 向后兼容性考虑

### 选项 A: 渐进式迁移（推荐）

**优点：**
- 平滑过渡，不影响现有功能
- 可以逐步测试和验证

**实现：**
- 同时保留 `/bff/*` 和 `/api/*` 路由
- 前端逐步迁移到新路径
- 确认无问题后移除旧路径

### 选项 B: 一次性迁移

**优点：**
- 代码更简洁，无冗余
- 立即统一路径结构

**风险：**
- 需要同时更新所有相关代码
- 测试覆盖要求高

**建议：** 采用选项 A（渐进式迁移）

---

## 测试清单

### Agent 端测试

- [ ] `/api/` 返回正确的端点信息
- [ ] `/api/health` 返回健康状态
- [ ] `/api/agents` 返回 Agent 列表
- [ ] `/api/agents/:id` 返回单个 Agent
- [ ] `/api/files/upload` 可以上传文件
- [ ] `/api/files` 可以列出文件
- [ ] `/api/datasets` 返回数据集列表
- [ ] `/api/datasets/:id` 返回数据集详情
- [ ] `/api/datasets/:id/preview` 可以预览数据
- [ ] `/api/runs/*` LatticeGateway 路由正常工作
- [ ] `/api/v1/*` Python 代理正常工作
- [ ] `/api/rtc/*` RTC 路由正常工作

### 前端测试

- [ ] Agent 列表页面正常加载
- [ ] Agent 详情页面正常显示
- [ ] 文件上传功能正常
- [ ] 数据集列表正常显示
- [ ] 数据集详情正常显示
- [ ] 聊天功能正常（LatticeGateway）
- [ ] RTC 语音聊天正常

### 集成测试

- [ ] Nginx 代理正常工作
- [ ] Docker Compose 环境正常
- [ ] 生产环境配置正确

---

## 风险评估

### 低风险

- ✅ Python 代理路由 (`/api/v1/*`) - 无需修改
- ✅ RTC 路由 (`/api/rtc/*`) - 无需修改
- ✅ LatticeGateway 路由 (`/api/runs/*`) - 框架管理，无需修改

### 中风险

- ⚠️ Agent 管理路由 - 需要前端配合修改
- ⚠️ 文件管理路由 - 需要前端配合修改
- ⚠️ 数据集管理路由 - 需要前端配合修改

### 缓解措施

1. 采用渐进式迁移，保留旧路径一段时间
2. 充分测试所有功能
3. 更新文档和 API 清单
4. 通知团队成员路径变更

---

## 预期收益

1. **简化路由结构** - 统一使用 `/api/*` 前缀，更清晰
2. **减少冗余代码** - 移除重复的路由注册
3. **提高一致性** - 所有 API 遵循相同的命名规范
4. **降低维护成本** - 更少的路径变体，更容易维护
5. **改善可读性** - 代码和配置更简洁

---

## 实施时间线

### Week 1: 准备和开发
- Day 1-2: Agent 端路由重构
- Day 3-4: 前端代码修改
- Day 5: 本地测试

### Week 2: 测试和部署
- Day 1-2: 集成测试
- Day 3: Nginx 配置更新
- Day 4-5: 生产环境部署和验证

### Week 3: 清理（可选）
- Day 1-2: 移除旧路径
- Day 3: 文档更新
- Day 4-5: 最终验证

---

## 相关文件清单

### Agent 端需要修改的文件

1. `agent/src/gateway.ts` - 主要路由注册逻辑
2. `agent/API_ROUTES.md` - API 文档（更新）
3. `agent/API_ROUTES_QUICK_REFERENCE.md` - 快速参考（更新）

### 前端需要修改的文件

1. `ai_web/src/components/chating/index.tsx` - BFF URL 标准化函数
2. `ai_web/src/pages/agents/overview/index.tsx` - 硬编码 URL
3. `ai_web/src/pages/assets/skills/index.tsx` - 硬编码 URL

### 配置需要修改的文件

1. `nginx-fina-demo.conf` - Nginx 代理配置
2. `docker-compose.yml` - 环境变量（如需要）
3. `docker-compose.prod.yml` - 生产环境配置（如需要）

---

## 总结

这个重构提案旨在简化路由结构，移除冗余的 BFF 前缀，统一使用 `/api/*` 路径。通过渐进式迁移策略，可以确保平滑过渡，最小化对现有功能的影响。

**关键决策点：**
1. ✅ 统一使用 `/api/*` 前缀
2. ✅ 保留 LatticeGateway 的 `/api/runs/*`（框架要求）
3. ✅ 采用渐进式迁移策略
4. ✅ 更新所有相关文档

**下一步行动：**
1. 评审此提案
2. 确认迁移策略
3. 开始实施 Phase 1
