# 如何开发一个 Data Agent

本文说明 Fina Demo 中 **Data Agent**（业务数据分析智能体）的架构与开发方式，包括后端注册、技能（Skill）、子代理（Sub-Agent）以及前端的对接方式。

## 1. Data Agent 是什么

Data Agent 是一个**业务数据分析智能体**，主要能力包括：

- **自然语言转 SQL**：理解业务问题，生成并执行 SQL，返回结果。
- **按需深度**：
  - **模式 A（简单查询）**：用户只要“查数”（如“上个月销售额”），直接委派给 SQL 子代理，返回数据即可。
  - **模式 B（深度分析）**：用户要求“分析原因、趋势、洞察、预测”时，走完整分析流程：任务规划 → 多步查询 → 结果解读 → 报告/可视化。
- **子代理协作**：
  - **sql-builder-agent**：负责库表探索、SQL 生成/校验/执行，并写回 schema 文档（如 `/db_schema.md`）。
  - **data-analysis-agent**：负责对查询结果做业务解读、发现与数据缺口评估（仅在深度分析时使用）。
- **技能（Skills）**：通过加载 `analysis-methodology`、`analyst`、`sql-query`、`notebook-report`、`data-visualization` 等技能，指导分析步骤与报告生成。

技术上，Data Agent 基于 **@axiom-lattice/core** 的 `registerAgentLattices` 注册为主 Agent，并配置子代理与 SQL 相关工具（如 `list_tables_sql`、`info_sql`、`query_sql` 等）。

---

## 2. 架构与代码位置

### 2.1 后端（agent 服务）

| 内容 | 路径 |
|------|------|
| Data Agent 定义与注册 | `agent/src/agents/data_agent/index.ts` |
| Agent 入口（挂载 data_agent） | `agent/src/agents/index.ts` |
| 技能定义（TS + SKILL.md） | `agent/skills/*.ts` 与 `agent/skills/*/SKILL.md` |
| 网关与路由 | `agent/src/gateway.ts` |

### 2.2 前端（ai_web）

| 内容 | 路径 |
|------|------|
| Data Agent 聊天页 | `ai_web/src/pages/agents/data/index.tsx` |
| 侧栏/路由中的「Data Agent」 | `ai_web/src/App.tsx`、`ai_web/src/components/custom-sider-wrapper/index.tsx` 等 |
| API 基地址 | `ai_web/src/getBaseAPIPath.ts` |

### 2.3 数据与 API

- **数据库**：Data Agent 使用 Postgres，通过 `DATABASE_URL`（或等价配置）连接；在 `data_agent/index.ts` 末尾通过 `initializeDataAgentDatabase("fulidb", { ... })` 注册。
- **预测/数据集 API**：由 `prediction_app` 提供（如 `/api/v1/datasets`），经 agent 反向代理，与 Data Agent 的 SQL 能力互补，不直接写在 Data Agent 内。

---

## 3. 开发一个 Data Agent 的步骤

下面按「从零新增一个 Data Agent」的流程说明（本项目的 Data Agent 已存在，可作为参考实现）。

### 步骤 1：在后端定义并注册 Agent

在 `agent/src/agents/` 下新建目录，例如 `my_data_agent/`，并创建 `index.ts`。

**3.1.1 依赖与类型**

```ts
import {
  registerAgentLattices,
  AgentType,
  AgentConfig,
  sqlDatabaseManager,
  DatabaseConfig,
} from "@axiom-lattice/core";
import z from "zod";
```

**3.1.2 主 Agent 的 System Prompt**

在文件中写一个字符串，定义角色、工作模式（简单查询 vs 深度分析）、何时调用子代理、何时写 `/db_schema.md`、使用哪些技能等。可参考现有 `data_agent/index.ts` 中的 `dataAgentPrompt`。

**3.1.3 子代理（如 SQL 执行、结果分析）**

- 为每个子代理写独立的 `prompt`（如 `sqlBuilderPrompt`、`dataAnalysisPrompt`）。
- 子代理也作为 `AgentConfig` 配置，`type: AgentType.DEEP_AGENT`，并挂上相应 `tools`（如 `list_tables_sql`、`info_sql`、`query_checker_sql`、`query_sql`）。

**3.1.4 配置数组并注册**

```ts
const my_agents: AgentConfig[] = [
  {
    key: "my_data_agent",
    name: "My Data Agent",
    description: "简短描述，会出现在 API/列表中",
    type: AgentType.DEEP_AGENT,
    tools: ["list_tables_sql", "info_sql"],
    prompt: myDataAgentPrompt,
    subAgents: ["sql-builder-agent", "data-analysis-agent"],
    skillCategories: ["analysis", "sql"],
    schema: z.object({}),
    runConfig: { databaseKey: "fulidb" },
  },
  { key: "sql-builder-agent", ... },
  { key: "data-analysis-agent", ... },
];

registerAgentLattices(my_agents);
```

**3.1.5 数据库连接**

- 使用 `sqlDatabaseManager.registerDatabase(key, config)` 注册库（或复用已有的 `initializeDataAgentDatabase` 封装）。
- 在 Agent 的 `runConfig.databaseKey` 中指定该 key，以便 SQL 工具访问。

**3.1.6 挂到 Agent 入口**

在 `agent/src/agents/index.ts` 中增加：

```ts
import "./my_data_agent";
```

保存后，LatticeGateway 会加载该模块，新 Agent 即可通过 `/api` 下的 Lattice 接口被调用。

---

### 步骤 2：配置技能（Skills，可选但推荐）

Data Agent 的深度分析依赖技能来规范“如何拆题、如何查数、如何写报告”。

**2.1 技能目录结构**

每个技能通常包含：

- `agent/skills/<skill-name>/SKILL.md`：Markdown 说明（名称、描述、类别、子技能等）。
- `agent/skills/<skill-name>.ts`：导出一个对象，包含 `name`、`description`、`prompt`（与 SKILL 内容对应）。

**2.2 在 Agent 中引用技能**

在主 Agent 的 `skillCategories` 中写上技能类别（如 `"analysis"`、`"sql"`），与技能 YAML frontmatter 中的 `metadata.category` 对应。主 Agent 的 prompt 里通过「加载 `sql-query` 技能」等自然语言指导模型使用这些技能。

**2.3 参考技能**

- **sql-query**：如何探索 schema、如何向 sql-builder-agent 提需求、如何校验结果。
- **analyst**：端到端分析流程（问题记录、方法论、待办、迭代分析、报告）。
- **analysis-methodology**：问题拆解与方法论。
- **notebook-report**：报告结构与输出格式。
- **data-visualization**：图表与 ECharts 配置。

新增 Data Agent 时，可复用这些技能，或在 `agent/skills/` 下新增自己的 SKILL.md 与 .ts。

---

### 步骤 3：前端接入聊天页

**3.1 添加页面组件**

在 `ai_web/src/pages/agents/` 下新建目录（如 `my-data/`），新建 `index.tsx`：

```tsx
import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { getBaseAPIPath } from "../../../getBaseAPIPath";
import { TOKEN_KEY } from "../../../authProvider";

export const MyDataAgentList = () => {
  return (
    <div style={{ height: "calc(-112px + 100vh)", width: "100%" }}>
      <LatticeChatShell
        initialConfig={{
          baseURL: getBaseAPIPath(),
          apiKey: localStorage.getItem(TOKEN_KEY) || "",
          transport: "sse",
          assistantId: "my_data_agent",  // 与后端 Agent 的 key 一致
          showSideMenu: false,
        }}
      />
    </div>
  );
};
```

**3.2 路由与侧栏**

- 在 `App.tsx` 中为 `/admin/agents/my-data` 配置路由，并渲染 `MyDataAgentList`。
- 在 `custom-sider-wrapper`（或当前使用的 Sider）中增加「My Data Agent」菜单项，指向该路由。

完成后，用户从侧栏进入该页即可与新建的 Data Agent 对话。

---

### 步骤 4：环境与运行

**4.1 后端环境变量（agent）**

- **DATABASE_URL**：Postgres 连接串，供 Data Agent 的 SQL 工具使用。
- 其他 Lattice/模型相关变量按现有 agent 文档配置。

**4.2 本地联调**

1. 启动 Postgres（或使用现有库）。
2. 启动 `prediction_app`（如需要数据集 API）。
3. 启动 `agent`：`cd agent && pnpm dev`。
4. 启动 `ai_web`：`cd ai_web && pnpm dev`。
5. 打开对应路由（如 `/admin/agents/data` 或你新建的 `/admin/agents/my-data`），验证对话与 SQL 执行。

**4.3 部署**

- 使用 `docker-transfer.sh` 部署时，确保 agent 容器内配置了 `DATABASE_URL`（或生产库连接）。
- 若使用 docker-compose，在 `agent` 服务的 `environment` 中配置相同变量。

---

## 4. 现有 Data Agent 配置摘要（参考）

本仓库已实现的 Data Agent 关键配置如下（详见 `agent/src/agents/data_agent/index.ts`）：

| 项目 | 值 |
|------|-----|
| 主 Agent key | `data_agent` |
| 类型 | `AgentType.DEEP_AGENT` |
| 工具 | `list_tables_sql`, `info_sql` |
| 子代理 | `sql-builder-agent`, `data-analysis-agent` |
| 技能类别 | `analysis`, `sql` |
| 数据库 key | `fulidb`（通过 `initializeDataAgentDatabase("fulidb", ...)` 注册） |

前端聊天页使用 `assistantId: "data_agent"`，与上述 key 一致。

---

## 5. 相关文档

- [文档索引](./README.md)
- [微服务、端口与路由](../MICROSERVICES_PORTS_AND_ROUTES.md)：agent、ai_web、prediction_app 的端口与 `/api` 设计
- [环境变量与部署](../ENV_FILE_GUIDE.md)：.env 与生产部署
- Agent 路由与 BFF：[agent/API_ROUTES_QUICK_REFERENCE.md](../agent/API_ROUTES_QUICK_REFERENCE.md)
- 技能说明：`agent/skills/*/SKILL.md`（如 [sql-query](../agent/skills/sql-query/SKILL.md)、[analyst](../agent/skills/analyst/SKILL.md)）
