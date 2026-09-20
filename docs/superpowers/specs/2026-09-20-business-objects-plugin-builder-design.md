# 统一 Business Objects 插件 + BO Builder agent 设计

- 日期：2026-09-20
- 状态：Draft（设计已定稿，待实现）
- 范围：`agent/src/agents/platform_service/business_objects/`（合并改造）+ `agent/src/agents/platform_service/index.ts`（导入）+ `agent/src/agents/platform_service/__tests__/`（测试）
- 不改动：`platform-service/`（Java BO Runtime 接口不变）、`executors.ts` 执行器逻辑、`client.ts` / `connection.ts`

## 1. 背景与目标

现状：BO 能力被拆成两个插件——`business-object-schema`（管理 store / grant / object 定义）与 `business-object-records`（记录 CRUDQ），二者各自声明连接、各自 `discover` 打同一个 `/api/v1/bo/objects`，配置与发现重复。

本设计将其合并为**一个插件 `business-objects`**，并借助 `Plugin.agents`（core `ensurePluginAgentsForTenant`，`index.mjs:28985`）贡献一个 **`business-objects-builder`** agent：以对话方式交互式构建 BO 表（store / object / 字段 / 索引），并用记录查询验证建表结果。

参照先例：core 内置 `semantic-metrics` 插件（`index.mjs:40242`）——单插件 + `category: "data"` + `capabilityBundleEligible` + `skills` + `agents.<builder>`（DEEP_AGENT，middleware 引用插件自身 + skill + task + ask_user + filesystem）。本设计照此骨架实现。

**关键区分**：本插件是"建模 / 运行时数据"能力，属 `data` 分类；`openExpose` 只上只读工具，写操作（建表、造测试数据）仅在 agent 会话内经中间件暴露。

## 2. 决策总览

| 项 | 决策 |
|---|---|
| 覆盖范围 | schema 管理 + record CRUDQ，合并为一个插件 |
| 旧插件 | 直接替换：删除 `schemaPlugin.ts` / `recordsPlugin.ts`，连接类型由 `business-object-schema` / `business-object-records` 变为 `business-objects`（需重新配置一次连接） |
| 插件 type | `business-objects`（同时是 Connection Store type 与 Open grant domain） |
| category | `data` |
| capabilityBundleEligible | `true` |
| 工具集 | 15 个（10 schema + 5 record），命名沿用现有 |
| MCP 暴露面 | 仅只读 7 个（`list_stores` / `test_store` / `list_store_grants` / `list_objects` / `get_object` / `query_records` / `get_record`） |
| Builder agent | key `business-objects-builder`，`type: DEEP_AGENT` |
| Builder 中间件 | `business-objects`(connectAll) + `skill` + `task` + `ask_user_to_clarify` + `filesystem` |
| 技能 | `business-objects-modeling`（详细规范版，前缀满足 `{pluginType}-`） |
| 连接 | 复用 `platformServiceConnection`（baseUrl / apiKey / boConnectionKey + test） |
| 租户身份 | 不发送 `X-Tenant-Id`；BO 作用域由 `X-BO-Connection-Key` 在 platform-service 侧解析 |
| 破坏性操作 | `delete_object` / `delete_record` 需入参 `confirm: true`；`openExpose` 不暴露 |
| 范围裁剪 | 通过 middleware 配置 `allowedTools` 实现（如只读 agent 只给查询工具） |

## 3. 架构与目录

```
agent/src/agents/platform_service/
  index.ts                         // 改为 import "./business_objects/plugin"
  business_objects/
    plugin.ts                      // 新增：合并后的 Plugin（meta + connection + skills + agents + middleware）
    prompt.ts                      // 新增：business-objects-builder 提示词
    skill.ts                       // 新增：business-objects-modeling 技能内容
    executors.ts                   // 不变：全部 schema + record 执行器（含 selectedEntities 校验）
    __tests__/business_objects.test.ts   // 不变：执行器测试
  __tests__/
    barrel.test.ts                 // 更新：期望类型
    registration.test.ts           // 更新：单插件断言 + 新增 meta/agent 不变量
```

删除：`business_objects/schemaPlugin.ts`、`business_objects/recordsPlugin.ts`。

## 4. 连接

```ts
connection: {
  ...platformServiceConnection,          // baseUrl / apiKey / boConnectionKey + test
  discover: async (config) => {          // 取自原 recordsPlugin
    const conn = connectionFromConfig(config);
    const rows = await request<Array<{ objectKey: string; displayName?: string; storeKey?: string }>>({
      conn, method: "GET", path: "/api/v1/bo/objects",
      headers: { "X-BO-Connection-Key": boConnectionKey(conn) },
    });
    return rows.map((row) => ({
      id: row.objectKey,
      name: row.displayName || row.objectKey,
      description: row.storeKey ? `store: ${row.storeKey}` : undefined,
    }));
  },
}
```

- `connection.test` 沿用 `platformServiceConnection.test`（`GET /actuator/health`）。
- `discover` 返回 objectKey 实体，供连接 UI 选择，写入 `selectedEntities`；执行器用 `requireObjectAllowed` 做范围收窄。
- 不发送 `X-Tenant-Id`（`client.ts` 仅在传入 `tenantId` 时注入，BO 调用不传）。

## 5. 插件 meta 与工具目录

```ts
meta: {
  type: "business-objects",
  name: "Business Objects",
  description: "Business Object stores, grants, object definitions and record CRUDQ. PERMISSION MODEL — query-only agents enable this middleware with allowedTools set to the read tools; schema/record writes are for the built-in 'business-objects-builder' agent.",
  version: "1.0.0",
  category: "data",
  capabilityBundleEligible: true,
  tools: [ ...15... ],
  openExpose: [ ...7 只读... ],
  configSchema: { /* connections + connectAll，标准连接选择器 */ },
  defaultConfig: { connections: [], connectAll: false },
}
```

工具目录（middleware 内 15 个）：

| 组 | 工具 | 映射 | 写 |
|---|---|---|---|
| store | `list_stores` | `GET /api/v1/bo/stores` | 否 |
| store | `create_store` | `POST /api/v1/bo/stores` | 是 |
| store | `test_store` | `POST /api/v1/bo/stores/{storeKey}/test` | 否（`SELECT 1` 连通性探测） |
| store | `list_store_grants` | `GET /api/v1/bo/stores/{storeKey}/grants` | 否 |
| store | `grant_store` | `POST /api/v1/bo/stores/{storeKey}/grants` | 是 |
| object | `list_objects` | `GET /api/v1/bo/objects` | 否 |
| object | `get_object` | `GET /api/v1/bo/objects/{objectKey}` | 否 |
| object | `create_object` | `POST /api/v1/bo/objects` | 是 |
| object | `update_object` | `PUT /api/v1/bo/objects/{objectKey}` | 是（v1 仅增量加列） |
| object | `delete_object` | `DELETE /api/v1/bo/objects/{objectKey}` | 是（需 confirm） |
| record | `query_records` | `POST /api/v1/bo/objects/{objectKey}/records/query` | 否 |
| record | `get_record` | `GET /api/v1/bo/objects/{objectKey}/records/{id}` | 否 |
| record | `create_record` | `POST /api/v1/bo/objects/{objectKey}/records` | 是 |
| record | `update_record` | `PATCH /api/v1/bo/objects/{objectKey}/records/{id}` | 是 |
| record | `delete_record` | `DELETE /api/v1/bo/objects/{objectKey}/records/{id}` | 是（需 confirm） |

Zod schema 沿用现有 `schemaPlugin.ts` / `recordsPlugin.ts` 定义，合并到单一 `schemas` 常量。

## 6. MCP 暴露面（openExpose）

只读 7 个，全部 `readOnly: true`：

```
list_stores, test_store, list_store_grants, list_objects, get_object, query_records, get_record
```

`test_store` 虽为 POST，但只执行 `SELECT 1` 连通性探测（`BusinessObjectService.java:150`），无数据副作用，按只读处理。

不暴露：`create_store` / `grant_store` / `create_object` / `update_object` / `delete_object` / `create_record` / `update_record` / `delete_record`。写能力仅存在于 agent 会话中间件内。

不变量：`openExpose` 中每个名字必须存在于 middleware 实际工具名集合（测试断言）。

## 7. 中间件

单个 `createMiddleware({ name: "BusinessObjects", tools: [...15] })`，工厂签名 `(rawConfig) => ...`，与现有插件一致。逻辑分组靠工具命名前缀（`*_store` / `*_object` / `*_record`），协议不做 per-tool group。

范围裁剪：middleware 配置支持 `allowedTools`（`AgentLatticeProtocol.ts:136`）。只读消费方配置示例：

```json
{ "type": "business-objects", "config": { "connections": [], "connectAll": true,
  "allowedTools": ["list_stores","test_store","list_store_grants","list_objects","get_object","query_records","get_record"] } }
```

## 8. 技能 `business-objects-modeling`（详细规范版）

`skills: { "business-objects-modeling": { version, content, resources? } }`，名称满足 `business-objects-` 前缀（core 校验，`index.mjs:11657`）。

技能内容章节（详细规范版）：

1. **概念模型**：store（一个 store = 一个 PostgreSQL 库 + schema）、store grant（granteeKey + can_read/write/manage）、object 定义（objectKey → 物理表）、record。
2. **命名规范**：`storeKey` / `objectKey` / `field.key` / `index.name` 一律 `^[a-z][a-z0-9_]{0,62}$`；避免保留字；object 名用业务单数（`customer`），表名由服务端派生。
3. **store 选择与创建**：先 `list_stores` + `test_store` 确认；无合适 store 再 `create_store`（仅接受 `jdbc:postgresql:`，默认 schema `public`，创建时自动建 `tenant` 全权 grant）；同一业务域复用同一 store，不重复建库。
4. **字段类型映射**：`string`(≤maxLength) / `text` / `integer` / `long` / `decimal`(precision,scale) / `boolean` / `date` / `datetime` / `json`；`required`、`maxLength`、`precision`/`scale` 的使用场景；何时用 `json` 而非拆列。
5. **索引设计**：`indexes[].fields` 顺序、`unique` 语义、命名；高频过滤/排序字段建索引，避免过度索引。
6. **v1 演进约束**：`update_object` 仅支持增量加列，**不支持删列 / 改类型**；需要此类变更时新建 object 或加新列迁移。
7. **权限与 grant**：默认 `tenant` grant；跨 grantee 授权用 `grant_store`；`can_manage` 的边界。
8. **安全与确认**：`delete_object` / `delete_record` 必须先取得用户明确确认再传 `confirm: true`；软删语义。
9. **标准工作流**：
   1. 澄清业务实体与关键字段（必要时 `ask_user_to_clarify`）
   2. `list_stores` / `test_store` 选定 store
   3. `list_objects` / `get_object` 检查是否已存在
   4. `create_object`（或 `update_object` 加列）
   5. `get_object` 复核定义
   6. 验证：`create_record` 造样本 → `query_records` / `get_record` 核对 → `delete_record` 清理
   7. 用 `task` 记录进度与结论
10. **验收清单**：命名合规、类型/长度合理、索引覆盖查询、定义与预期一致、测试数据已清理。

技能内容写入 `skill.ts` 的 `content`（markdown）；如需附表/样例，放 `resources`（安全相对路径）。

## 9. Builder agent `business-objects-builder`

```ts
agents: {
  "business-objects-builder": {
    key: "business-objects-builder",
    name: "Business Objects Builder",
    description: "Interactively design and build Business Object stores, object definitions, fields and indexes; verify with record queries.",
    type: AgentType.DEEP_AGENT,
    prompt: BUSINESS_OBJECTS_BUILDER_PROMPT,
    middleware: [
      { id: "business-objects", type: "business-objects", name: "Business Objects",
        description: "Manage stores/objects and run record CRUDQ for verification",
        enabled: true, config: { connections: [], connectAll: true } },
      { id: "skill", type: "skill", name: "Skill",
        description: "Load the business-objects-modeling policy",
        enabled: true, config: { readAll: false, skills: ["business-objects-modeling", "task-definition"] } },
      { id: "task", type: "task", name: "Task",
        description: "Persistent TaskItems as the planning surface", enabled: true, config: {} },
      { id: "ask_user_to_clarify", type: "ask_user_to_clarify", name: "Ask User",
        description: "Confirm modeling decisions before writes", enabled: true, config: {} },
      { id: "filesystem", type: "filesystem", name: "Filesystem",
        description: "Read user-provided data dictionaries or sample data", enabled: true, config: {} },
    ],
  },
}
```

`prompt.ts`：首要动作是加载 `business-objects-modeling` 技能（对齐 `SEMANTIC_METRICS_BUILDER_PROMPT`），随后按技能工作流执行；写操作前经 `ask_user_to_clarify` 确认。

## 10. 数据流与权限

```
Builder agent / 其他 agent
  → business-objects middleware（可选 allowedTools 裁剪）
  → executors（requireObjectAllowed 校验 selectedEntities）
  → client.request（X-BO-Connection-Key，不带 X-Tenant-Id）
  → platform-service /api/v1/bo/*
```

- 作用域：连接 `selectedEntities` 非空时，objectKey 必须命中，否则拒绝（现有 `executors.ts:85`）。
- 租户：不发送 `X-Tenant-Id`；BO 作用域由 `X-BO-Connection-Key` 在服务端解析为 grant。

## 11. 错误处理

- 沿用 `errorResult`：`{ ok: false, status?, code, message }`，工具不抛裸异常。
- 删除类无 `confirm: true` → 返回 `CONFIRM_REQUIRED` 文本（`executors.ts:193,245`）。
- 缺 `boConnectionKey` → `boConnectionKey()` 抛错并转 `errorResult`。
- Zod 校验失败 → 结构化错误文本。

## 12. 测试策略（jest + mock fetch，无网络）

更新：
- `barrel.test.ts`：注册类型期望 `["business-objects", "storage", "webhooks"]`。
- `registration.test.ts`：
  - 注册单个 `business-objects` 插件，`meta.type === "business-objects"`、`category === "data"`。
  - middleware 恰好 15 个工具（列表断言）。
  - `openExpose` 7 个名字，全部 `readOnly: true`，且都在工具名集合内（不变量）。
  - `agents["business-objects-builder"]` 存在，`type === DEEP_AGENT`，middleware 类型集合 = `["business-objects","skill","task","ask_user_to_clarify","filesystem"]`。
  - `skills` 的 key 以 `business-objects-` 开头。
  - `discover` 映射 objectKey 实体，带 `X-BO-Connection-Key`，无 `X-Tenant-Id`。
- `business_objects.test.ts`：保持不变（执行器层）。

保留的既有不变量：openExpose ⊆ 实际工具名；scope 越界被拒；删除需 confirm。

## 13. 迁移 / 兼容

- 破坏性：连接类型由两个变为一个。已有配置 `business-object-schema` / `business-object-records` 的连接不再被读取，需在 `business-objects` 类型下重新配置（baseUrl / apiKey / boConnectionKey）。用 env 默认值（`PLATFORM_SERVICE_URL` / `FILE_SERVICE_API_KEY`）可缓解 baseUrl/apiKey，但 `boConnectionKey` 必须显式配置。
- 工具名不变，调用方 prompt / allowedTools 无需改名。
- MCP grant domain 由 `business-object-schema` / `business-object-records` 变为 `business-objects`，需在 Open 授权侧同步。

## 14. 不在本次范围（YAGNI）

- per-tool 分组协议字段（协议不支持，靠命名）。
- 修改 platform-service Java 侧 BO Runtime。
- record 聚合 / 关联查询等进阶能力。
- 独立的管理域（`manage_bo`）拆分；本插件用 `allowedTools` 替代。
- 迁移旧连接配置的自动化脚本。

## 15. 开放问题 / 风险

1. **连接配置迁移**：两个旧连接类型废弃后，已部署环境需手工重建 `business-objects` 连接；`boConnectionKey` 无 env 兜底。
2. **Open grant domain 变更**：需确认 Open 授权侧按 `meta.type` 的 grant 是否有存量数据需迁移。
3. **DEEP_AGENT 依赖**：`business-objects-builder` 为 DEEP_AGENT；需确认该 agent 在目标部署的模型/工具预算下可用。
4. **技能前缀校验**：`business-objects-modeling` 必须与插件 type `business-objects` 前缀匹配（已满足）。
5. **openExpose 的 `test_store` 语义**：虽标注 readOnly，实际是 POST；若 Open 侧对 POST + readOnlyHint 有额外约束需复核。
