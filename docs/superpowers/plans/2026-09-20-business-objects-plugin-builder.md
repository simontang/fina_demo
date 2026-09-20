# Unified Business Objects Plugin + BO Builder Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the `business-object-schema` and `business-object-records` agent plugins into a single `business-objects` plugin that also ships a `business-objects-builder` agent (with a modeling skill) for interactively building Business Object tables.

**Architecture:** One `Plugin` object under `agent/src/agents/platform_service/business_objects/plugin.ts` owning the connection (shared `platformServiceConnection` + BO `discover`), the 15 store/object/record tools, the `business-objects-modeling` skill, and the `business-objects-builder` DEEP_AGENT. The agent is contributed through `Plugin.agents` (registered per tenant by core's `ensurePluginAgentsForTenant`). `openExpose` is read-only. Existing `executors.ts` is reused; only a one-line type fix is needed.

**Tech Stack:** TypeScript, Jest (ts-jest), Zod, `langchain` `createMiddleware`/`tool`, `@axiom-lattice/core` `PluginRegistry`, `@axiom-lattice/protocols` types.

**Spec:** `docs/superpowers/specs/2026-09-20-business-objects-plugin-builder-design.md`

**Workdir for all commands:** `agent/` (run `cd agent` once, or prefix commands). Tests are run with `npx jest`.

---

## File Structure

```
agent/src/agents/platform_service/
  index.ts                              # MODIFY: import "./business_objects/plugin"
  business_objects/
    prompt.ts                           # CREATE: BUSINESS_OBJECTS_BUILDER_PROMPT
    skill.ts                            # CREATE: BUSINESS_OBJECTS_MODELING_SKILL
    plugin.ts                           # CREATE: unified Plugin
    executors.ts                        # MODIFY: one-line type fix on QueryRecordsInput
    schemaPlugin.ts                     # DELETE
    recordsPlugin.ts                    # DELETE
    __tests__/builder_assets.test.ts    # CREATE
    __tests__/business_objects.test.ts  # UNCHANGED
  __tests__/
    barrel.test.ts                      # MODIFY: expected plugin types
    registration.test.ts                # MODIFY: business-objects assertions
```

---

## Task 1: Builder prompt + modeling skill modules

**Files:**
- Create: `agent/src/agents/platform_service/business_objects/prompt.ts`
- Create: `agent/src/agents/platform_service/business_objects/skill.ts`
- Test: `agent/src/agents/platform_service/__tests__/builder_assets.test.ts`

- [ ] **Step 1: Write the failing test**

Create `agent/src/agents/platform_service/__tests__/builder_assets.test.ts`:

```ts
import { BUSINESS_OBJECTS_BUILDER_PROMPT } from "../business_objects/prompt";
import { BUSINESS_OBJECTS_MODELING_SKILL } from "../business_objects/skill";

describe("business objects builder assets", () => {
  it("skill declares a version and all modeling sections", () => {
    expect(BUSINESS_OBJECTS_MODELING_SKILL.version).toBe("1.0.0");
    const content = BUSINESS_OBJECTS_MODELING_SKILL.content;
    expect(content).toContain("name: business-objects-modeling");
    for (const heading of [
      "## 1. 概念模型",
      "## 2. 命名规范",
      "## 3. store 选择与创建",
      "## 4. 字段类型映射",
      "## 5. 索引设计",
      "## 6. v1 演进约束",
      "## 7. 权限与 grant",
      "## 8. 安全与确认",
      "## 9. 标准工作流",
      "## 10. 验收清单",
    ]) {
      expect(content).toContain(heading);
    }
  });

  it("builder prompt requires loading the modeling skill first", () => {
    expect(BUSINESS_OBJECTS_BUILDER_PROMPT).toContain("CRITICAL FIRST ACTION");
    expect(BUSINESS_OBJECTS_BUILDER_PROMPT).toMatch(/skill_name:\s*"business-objects-modeling"/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/agents/platform_service/__tests__/builder_assets.test.ts`
Expected: FAIL — `Cannot find module '../business_objects/prompt'`.

- [ ] **Step 3: Create the prompt module**

Create `agent/src/agents/platform_service/business_objects/prompt.ts`:

```ts
export const BUSINESS_OBJECTS_BUILDER_PROMPT = `You are the Business Objects Builder.

CRITICAL FIRST ACTION: Before any response or other action, call the \`skill\` tool with skill_name: "business-objects-modeling" to load the modeling policy and follow it. Never announce the skill load. If it fails, retry once, then stop and explicitly report that the required Business Objects modeling skill could not be loaded.

You design and build Business Object stores, object definitions, fields and indexes through the Business Objects tools, then verify each result with record queries.

Operating rules:
- Inspect before you create: list stores and objects, and test store connectivity before writing.
- Confirm modeling decisions with the user before any write (create_store, grant_store, create_object, update_object, delete_object).
- Never pass storeKey to record tools: objectKey resolves the store.
- Deleting an object or a record requires explicit user confirmation, then pass confirm: true.
- Keep test data out of production: create a sample record to verify, then delete it.
- Track progress as tasks and record the final object definition in the task description.`;
```

- [ ] **Step 4: Create the skill module**

Create `agent/src/agents/platform_service/business_objects/skill.ts`:

```ts
import type { PluginSkillDefinition } from "@axiom-lattice/protocols";

const CONTENT = `---
name: business-objects-modeling
description: Builder policy for modeling Business Object stores, object definitions, fields and indexes; covers naming, store selection, field types, indexes, v1 evolution limits, grants, confirmations, and the standard build/verify workflow.
---

# Business Objects Modeling

You are modeling business data as Business Objects. A store is one PostgreSQL database plus schema; an object definition maps an objectKey to a physical table; records are rows. This policy is mandatory for every build action.

## 1. 概念模型
- **Store**：一个 store 对应一个 PostgreSQL 库 + schema（默认 public）。创建 store 时平台自动写入一条 granteeKey=tenant、can_read/write/manage=true 的默认 grant。
- **Store grant**：granteeKey + can_read / can_write / can_manage，决定该连接键能读/写/管理哪些 store。
- **Object 定义**：objectKey → 物理表，含 fields 与 indexes；由 platform-service 同步生成 DDL。
- **Record**：object 下的一行数据，按 id 读写；删除为软删。

## 2. 命名规范
- storeKey / objectKey / field.key / index.name 一律匹配 ^[a-z][a-z0-9_]{0,62}$。
- 使用业务单数英文名词：customer、order、invoice。
- 避免 SQL 保留字；不要用大写、连字符、前导数字。

## 3. store 选择与创建
- 先 list_stores，再用 test_store 确认连通。
- 无合适 store 才 create_store：仅接受 jdbc:postgresql: URL；默认 schema public。
- 同一业务域复用同一 store，不要为每个 object 新建库。
- 创建后 list_store_grants 确认默认 tenant grant 存在。

## 4. 字段类型映射
- string：短文本，配 maxLength（常见 ≤255）。
- text：长文本，无长度上限。
- integer / long：整数；大计数用 long。
- decimal：金额/比率，必须给 precision 与 scale。
- boolean：真假。
- date / datetime：日期 / 时间戳。
- json：结构不固定的扩展字段；能用列表达就不要用 json。
- required：业务上必填的字段设为 true。
- 每个 field 建议写 description，说明业务含义。

## 5. 索引设计
- indexes[].fields 顺序即索引列顺序，把等值过滤列放前面。
- unique: true 用于业务唯一键。
- 高频过滤/排序列建索引；低基数或写多读少的列不要建。
- index.name 可省略，由服务端派生。

## 6. v1 演进约束
- update_object 只支持增量加列，**不支持删列或改类型**。
- 需要删列/改类型时：新建 object，或加新列后由数据侧迁移，不原地改。
- 加列用 update_object，fields 传完整定义。

## 7. 权限与 grant
- 默认 tenant grant 覆盖读写管理；跨 grantee 授权用 grant_store。
- can_manage 允许改定义；can_write 允许写记录；can_read 只读。
- 收窄权限用 grant_store 覆盖，不要新建 store。

## 8. 安全与确认
- delete_object / delete_record 必须先取得用户明确确认，再传 confirm: true。
- 删除为软删，但不得在未确认时执行。
- 不要把生产数据当测试数据；测试记录用完删除。

## 9. 标准工作流
1. 澄清业务实体与关键字段；不确定时用 ask_user_to_clarify。
2. list_stores / test_store 选定 store。
3. list_objects / get_object 检查是否已存在。
4. create_object（或 update_object 加列）。
5. get_object 复核定义。
6. 验证：create_record 造样本 → query_records / get_record 核对 → delete_record 清理。
7. 用 task 记录进度与最终定义。

## 10. 验收清单
- [ ] 命名全部符合 ^[a-z][a-z0-9_]{0,62}$。
- [ ] 字段类型/长度/精度合理，required 正确。
- [ ] 索引覆盖主要查询，无过度索引。
- [ ] get_object 结果与预期定义一致。
- [ ] 测试记录已清理。
- [ ] task 描述记录了最终 object 定义。`;

export const BUSINESS_OBJECTS_MODELING_SKILL: PluginSkillDefinition = {
  version: "1.0.0",
  content: CONTENT,
};
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx jest src/agents/platform_service/__tests__/builder_assets.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add agent/src/agents/platform_service/business_objects/prompt.ts \
        agent/src/agents/platform_service/business_objects/skill.ts \
        agent/src/agents/platform_service/__tests__/builder_assets.test.ts
git commit -m "feat(agent): add business-objects builder prompt and modeling skill"
```

---

## Task 2: Unified `business-objects` plugin definition

**Files:**
- Create: `agent/src/agents/platform_service/business_objects/plugin.ts`
- Modify: `agent/src/agents/platform_service/business_objects/executors.ts:36-42`
- Modify: `agent/src/agents/platform_service/__tests__/registration.test.ts`

- [ ] **Step 1: Fix the `QueryRecordsInput` filter type (pre-existing TS error)**

In `agent/src/agents/platform_service/business_objects/executors.ts`, change the `QueryRecordsInput.filters` element type so `value` is optional (Zod's `z.unknown()` infers an optional key):

```ts
export interface QueryRecordsInput {
  objectKey: string;
  filters?: Array<{ field: string; op?: string; value?: unknown }>;
  sort?: Array<{ field: string; direction?: "asc" | "desc" }>;
  page?: number;
  pageSize?: number;
}
```

- [ ] **Step 2: Rewrite the business-objects tests in `registration.test.ts`**

At the top of `agent/src/agents/platform_service/__tests__/registration.test.ts`, replace the two old plugin imports (lines 11-12):

```ts
import { businessObjectRecordsPlugin } from "../business_objects/recordsPlugin";
import { businessObjectSchemaPlugin } from "../business_objects/schemaPlugin";
```

with:

```ts
import { AgentType } from "@axiom-lattice/protocols";
import { businessObjectPlugin } from "../business_objects/plugin";
```

Then replace the whole `describe("business object plugins", ...)` block (from `describe("business object plugins", () => {` through its closing `});` just before the end of file) with:

```ts
describe("business objects plugin", () => {
  it("registers one business-objects plugin in the data category", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(businessObjectPlugin);
    expect(businessObjectPlugin.meta.type).toBe("business-objects");
    expect(businessObjectPlugin.meta.category).toBe("data");
  });

  it("provides the full store/object/record tool set", async () => {
    const mw = await businessObjectPlugin.middleware!({});
    const names = ((mw as { tools: Array<{ name: string }> }).tools ?? [])
      .map((t) => t.name)
      .sort();
    expect(names).toEqual([
      "create_object",
      "create_record",
      "create_store",
      "delete_object",
      "delete_record",
      "get_object",
      "get_record",
      "grant_store",
      "list_objects",
      "list_store_grants",
      "list_stores",
      "query_records",
      "test_store",
      "update_object",
      "update_record",
    ]);
  });

  it("exposes only read-only tools to Open, all present in the middleware", async () => {
    const expose = (businessObjectPlugin.meta.openExpose ?? []).map((e) =>
      typeof e === "string" ? { name: e, readOnly: false } : e,
    );
    expect(expose.map((e) => e.name).sort()).toEqual([
      "get_object",
      "get_record",
      "list_objects",
      "list_store_grants",
      "list_stores",
      "query_records",
      "test_store",
    ]);
    for (const e of expose) expect(e.readOnly).toBe(true);

    const mw = await businessObjectPlugin.middleware!({});
    const toolNames = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    for (const e of expose) expect(toolNames).toContain(e.name);
  });

  it("ships a business-objects-builder agent wired to the plugin and modeling skill", () => {
    const agent = businessObjectPlugin.agents?.["business-objects-builder"];
    expect(agent).toBeDefined();
    expect(agent!.type).toBe(AgentType.DEEP_AGENT);
    const types = (agent!.middleware ?? []).map((m) => m.type).sort();
    expect(types).toEqual([
      "ask_user_to_clarify",
      "business-objects",
      "filesystem",
      "skill",
      "task",
    ]);
    const skillMw = (agent!.middleware ?? []).find((m) => m.type === "skill");
    expect((skillMw!.config as { skills: string[] }).skills).toContain("business-objects-modeling");
  });

  it("names the modeling skill with the plugin prefix", () => {
    for (const key of Object.keys(businessObjectPlugin.skills ?? {})) {
      expect(key.startsWith("business-objects-")).toBe(true);
    }
  });

  it("BO connection discovery maps object definitions into selectable entities", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [
        { objectKey: "customer", displayName: "Customer", storeKey: "crm_store" },
      ],
    } as unknown as Response);

    const discover = businessObjectPlugin.connection!.discover as unknown as (
      config: Record<string, unknown>,
    ) => Promise<Array<{ id: string; name: string; description?: string }>>;

    const result = await discover({ baseUrl: "http://svc:5707", boConnectionKey: "tenant" });

    expect(result).toEqual([{ id: "customer", name: "Customer", description: "store: crm_store" }]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://svc:5707/api/v1/bo/objects",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-BO-Connection-Key": "tenant" }),
      }),
    );
    const [, init] = fetchMock.mock.calls[0];
    expect((init?.headers as Record<string, string>)["X-Tenant-Id"]).toBeUndefined();
  });
});
```

Note: the previous file ended with the old discover test (which used `businessObjectRecordsPlugin`); the replacement block above already contains that discover test, so no duplicate remains. After this step the file's final `});` closes the new describe.

- [ ] **Step 3: Run tests to verify they fail**

Run: `npx jest src/agents/platform_service/__tests__/registration.test.ts`
Expected: FAIL — `Cannot find module '../business_objects/plugin'`.

- [ ] **Step 4: Create the unified plugin**

Create `agent/src/agents/platform_service/business_objects/plugin.ts`:

```ts
import { PluginRegistry } from "@axiom-lattice/core";
import { AgentType, type Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  boConnectionKey,
  boObjectCreate,
  boObjectDelete,
  boObjectGet,
  boObjectList,
  boObjectUpdate,
  boRecordCreate,
  boRecordDelete,
  boRecordGet,
  boRecordQuery,
  boRecordUpdate,
  boStoreCreate,
  boStoreGrantList,
  boStoreGrantUpsert,
  boStoreList,
  boStoreTest,
} from "./executors";
import { BUSINESS_OBJECTS_BUILDER_PROMPT } from "./prompt";
import { BUSINESS_OBJECTS_MODELING_SKILL } from "./skill";

const identifier = z.string().regex(/^[a-z][a-z0-9_]{0,62}$/);

const field = z.object({
  key: identifier,
  type: z.enum([
    "string",
    "text",
    "integer",
    "long",
    "decimal",
    "boolean",
    "date",
    "datetime",
    "json",
  ]),
  required: z.boolean().optional(),
  maxLength: z.number().int().optional(),
  precision: z.number().int().optional(),
  scale: z.number().int().optional(),
  description: z.string().optional(),
});

const index = z.object({
  name: identifier.optional(),
  fields: z.array(identifier).min(1),
  unique: z.boolean().optional(),
});

const objectDefinition = z.object({
  storeKey: identifier.optional(),
  objectKey: identifier,
  displayName: z.string().optional(),
  description: z.string().optional(),
  fields: z.array(field).min(1).optional(),
  indexes: z.array(index).optional(),
  status: z.number().int().optional(),
});

const filter = z.object({
  field: identifier,
  op: z.enum(["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]).optional(),
  value: z.unknown(),
});

const sort = z.object({
  field: identifier,
  direction: z.enum(["asc", "desc"]).optional(),
});

const schemas = {
  empty: z.object({}),
  storeCreate: z.object({
    storeKey: identifier,
    name: z.string().optional(),
    description: z.string().optional(),
    jdbcUrl: z.string().startsWith("jdbc:postgresql:"),
    username: z.string(),
    password: z.string(),
    status: z.number().int().optional(),
  }),
  storeKey: z.object({ storeKey: identifier }),
  storeGrant: z.object({
    storeKey: identifier,
    granteeKey: identifier.optional(),
    canRead: z.boolean().optional(),
    canWrite: z.boolean().optional(),
    canManage: z.boolean().optional(),
    status: z.number().int().optional(),
  }),
  objectGet: z.object({ objectKey: identifier }),
  objectCreate: objectDefinition.extend({ storeKey: identifier }),
  objectUpdate: objectDefinition.extend({ fields: z.array(field).min(1) }),
  objectDelete: z.object({
    objectKey: identifier,
    confirm: z.boolean().optional(),
  }),
  recordCreate: z.object({
    objectKey: identifier,
    data: z.record(z.unknown()),
  }),
  recordGet: z.object({
    objectKey: identifier,
    id: z.string(),
  }),
  recordUpdate: z.object({
    objectKey: identifier,
    id: z.string(),
    data: z.record(z.unknown()),
  }),
  recordDelete: z.object({
    objectKey: identifier,
    id: z.string(),
    confirm: z.boolean().optional(),
  }),
  recordQuery: z.object({
    objectKey: identifier,
    filters: z.array(filter).optional(),
    sort: z.array(sort).optional(),
    page: z.number().int().optional(),
    pageSize: z.number().int().optional(),
  }),
};

export const businessObjectPlugin: Plugin = {
  meta: {
    type: "business-objects",
    name: "Business Objects",
    description:
      "Business Object stores, grants, object definitions and record CRUDQ. PERMISSION MODEL — query-only agents enable this middleware with allowedTools set to the read tools; schema/record writes belong to the built-in 'business-objects-builder' agent.",
    version: "1.0.0",
    category: "data",
    capabilityBundleEligible: true,
    tools: [
      { name: "list_stores", description: "List Business Object stores." },
      {
        name: "create_store",
        description:
          "Create a Business Object store backed by one PostgreSQL database and create the default store grant.",
      },
      { name: "test_store", description: "Test connectivity to a Business Object store." },
      { name: "list_store_grants", description: "List grants for one Business Object store." },
      { name: "grant_store", description: "Create or update a Business Object store grant." },
      {
        name: "list_objects",
        description: "List Business Object definitions visible to the configured BO grant key.",
      },
      { name: "get_object", description: "Get one Business Object definition by objectKey." },
      {
        name: "create_object",
        description: "Create a Business Object definition and synchronize it into PostgreSQL DDL.",
      },
      {
        name: "update_object",
        description:
          "Update a Business Object definition. v1 supports additive columns only; no field removal/type changes.",
      },
      {
        name: "delete_object",
        description:
          "Soft-delete a Business Object definition. Requires user confirmation, then pass confirm:true.",
      },
      { name: "query_records", description: "Query Business Object records by objectKey." },
      { name: "get_record", description: "Get one Business Object record by objectKey and id." },
      {
        name: "create_record",
        description:
          "Create one Business Object record. Data is validated by the platform-service object schema.",
      },
      { name: "update_record", description: "Patch one Business Object record by objectKey and id." },
      {
        name: "delete_record",
        description:
          "Soft-delete one Business Object record. Requires user confirmation, then pass confirm:true.",
      },
    ],
    openExpose: [
      { name: "list_stores", readOnly: true },
      { name: "test_store", readOnly: true },
      { name: "list_store_grants", readOnly: true },
      { name: "list_objects", readOnly: true },
      { name: "get_object", readOnly: true },
      { name: "query_records", readOnly: true },
      { name: "get_record", readOnly: true },
    ],
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "Connect all available connections" },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
  },
  connection: {
    ...platformServiceConnection,
    discover: async (config) => {
      const conn = connectionFromConfig(config);
      const rows = await request<Array<{ objectKey: string; displayName?: string; storeKey?: string }>>({
        conn,
        method: "GET",
        path: "/api/v1/bo/objects",
        headers: { "X-BO-Connection-Key": boConnectionKey(conn) },
      });
      return rows.map((row) => ({
        id: row.objectKey,
        name: row.displayName || row.objectKey,
        description: row.storeKey ? `store: ${row.storeKey}` : undefined,
      }));
    },
  },
  skills: {
    "business-objects-modeling": BUSINESS_OBJECTS_MODELING_SKILL,
  },
  agents: {
    "business-objects-builder": {
      key: "business-objects-builder",
      name: "Business Objects Builder",
      description:
        "Interactively design and build Business Object stores, object definitions, fields and indexes; verify each step with record queries.",
      type: AgentType.DEEP_AGENT,
      prompt: BUSINESS_OBJECTS_BUILDER_PROMPT,
      middleware: [
        {
          id: "business-objects",
          type: "business-objects",
          name: "Business Objects",
          description: "Manage stores/objects and run record CRUDQ for verification",
          enabled: true,
          config: { connections: [], connectAll: true },
        },
        {
          id: "skill",
          type: "skill",
          name: "Skill",
          description: "Load the business-objects-modeling policy",
          enabled: true,
          config: { readAll: false, skills: ["business-objects-modeling", "task-definition"] },
        },
        {
          id: "task",
          type: "task",
          name: "Task",
          description: "Persistent TaskItems as the planning surface",
          enabled: true,
          config: {},
        },
        {
          id: "ask_user_to_clarify",
          type: "ask_user_to_clarify",
          name: "Ask User",
          description: "Confirm modeling decisions before writes",
          enabled: true,
          config: {},
        },
        {
          id: "filesystem",
          type: "filesystem",
          name: "Filesystem",
          description: "Read user-provided data dictionaries or sample data",
          enabled: true,
          config: {},
        },
      ],
    },
  },
  middleware: (rawConfig) =>
    createMiddleware({
      name: "BusinessObjects",
      tools: [
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boStoreList(input, exeConfig, rawConfig), {
          name: "list_stores",
          description: "List Business Object stores.",
          schema: schemas.empty,
        }),
        tool(
          (input: z.infer<typeof schemas.storeCreate>, exeConfig) =>
            boStoreCreate(input, exeConfig, rawConfig),
          {
            name: "create_store",
            description:
              "Create a Business Object store backed by one PostgreSQL database and create the default store grant.",
            schema: schemas.storeCreate,
          },
        ),
        tool((input: z.infer<typeof schemas.storeKey>, exeConfig) => boStoreTest(input, exeConfig, rawConfig), {
          name: "test_store",
          description: "Test connectivity to a Business Object store.",
          schema: schemas.storeKey,
        }),
        tool(
          (input: z.infer<typeof schemas.storeKey>, exeConfig) =>
            boStoreGrantList(input, exeConfig, rawConfig),
          {
            name: "list_store_grants",
            description: "List grants for one Business Object store.",
            schema: schemas.storeKey,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.storeGrant>, exeConfig) =>
            boStoreGrantUpsert(input, exeConfig, rawConfig),
          {
            name: "grant_store",
            description: "Create or update a Business Object store grant.",
            schema: schemas.storeGrant,
          },
        ),
        tool((input: z.infer<typeof schemas.empty>, exeConfig) => boObjectList(input, exeConfig, rawConfig), {
          name: "list_objects",
          description: "List Business Object definitions visible to the configured BO grant key.",
          schema: schemas.empty,
        }),
        tool(
          (input: z.infer<typeof schemas.objectGet>, exeConfig) =>
            boObjectGet(input, exeConfig, rawConfig),
          {
            name: "get_object",
            description: "Get one Business Object definition by objectKey.",
            schema: schemas.objectGet,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectCreate>, exeConfig) =>
            boObjectCreate(input, exeConfig, rawConfig),
          {
            name: "create_object",
            description: "Create a Business Object definition and synchronize it into PostgreSQL DDL.",
            schema: schemas.objectCreate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectUpdate>, exeConfig) =>
            boObjectUpdate(input, exeConfig, rawConfig),
          {
            name: "update_object",
            description:
              "Update a Business Object definition. v1 supports additive columns only; no field removal/type changes.",
            schema: schemas.objectUpdate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.objectDelete>, exeConfig) =>
            boObjectDelete(input, exeConfig, rawConfig),
          {
            name: "delete_object",
            description:
              "Soft-delete a Business Object definition. Requires user confirmation, then pass confirm:true.",
            schema: schemas.objectDelete,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordQuery>, exeConfig) =>
            boRecordQuery(input, exeConfig, rawConfig),
          {
            name: "query_records",
            description:
              "Query Business Object records by objectKey. The object definition resolves the store; do not pass storeKey.",
            schema: schemas.recordQuery,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordGet>, exeConfig) =>
            boRecordGet(input, exeConfig, rawConfig),
          {
            name: "get_record",
            description: "Get one Business Object record by objectKey and id.",
            schema: schemas.recordGet,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordCreate>, exeConfig) =>
            boRecordCreate(input, exeConfig, rawConfig),
          {
            name: "create_record",
            description:
              "Create one Business Object record. Data is validated by the platform-service object schema.",
            schema: schemas.recordCreate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordUpdate>, exeConfig) =>
            boRecordUpdate(input, exeConfig, rawConfig),
          {
            name: "update_record",
            description: "Patch one Business Object record by objectKey and id.",
            schema: schemas.recordUpdate,
          },
        ),
        tool(
          (input: z.infer<typeof schemas.recordDelete>, exeConfig) =>
            boRecordDelete(input, exeConfig, rawConfig),
          {
            name: "delete_record",
            description:
              "Soft-delete one Business Object record. Requires user confirmation, then pass confirm:true.",
            schema: schemas.recordDelete,
          },
        ),
      ],
    }),
};

PluginRegistry.register(businessObjectPlugin);
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx jest src/agents/platform_service/__tests__/registration.test.ts`
Expected: PASS (all registration + business objects tests).

- [ ] **Step 6: Commit**

```bash
git add agent/src/agents/platform_service/business_objects/plugin.ts \
        agent/src/agents/platform_service/business_objects/executors.ts \
        agent/src/agents/platform_service/__tests__/registration.test.ts
git commit -m "feat(agent): unified business-objects plugin with builder agent"
```

---

## Task 3: Switch barrel registration and delete legacy plugins

**Files:**
- Modify: `agent/src/agents/platform_service/index.ts`
- Modify: `agent/src/agents/platform_service/__tests__/barrel.test.ts:18`
- Delete: `agent/src/agents/platform_service/business_objects/schemaPlugin.ts`
- Delete: `agent/src/agents/platform_service/business_objects/recordsPlugin.ts`

- [ ] **Step 1: Update the barrel test to the new expected set**

In `agent/src/agents/platform_service/__tests__/barrel.test.ts`, change line 18:

```ts
    expect(types).toEqual(["business-objects", "storage", "webhooks"]);
```

- [ ] **Step 2: Run the barrel test to verify it fails**

Run: `npx jest src/agents/platform_service/__tests__/barrel.test.ts`
Expected: FAIL — received `["business-object-records", "business-object-schema", "storage", "webhooks"]`.

- [ ] **Step 3: Switch the barrel import**

Replace the contents of `agent/src/agents/platform_service/index.ts` with:

```ts
import "./storage/plugin";
import "./webhooks/plugin";
import "./business_objects/plugin";
```

- [ ] **Step 4: Delete the legacy plugin files**

```bash
git rm agent/src/agents/platform_service/business_objects/schemaPlugin.ts \
       agent/src/agents/platform_service/business_objects/recordsPlugin.ts
```

- [ ] **Step 5: Run the barrel test to verify it passes**

Run: `npx jest src/agents/platform_service/__tests__/barrel.test.ts`
Expected: PASS (1 test).

- [ ] **Step 6: Run the whole platform_service suite**

Run: `npx jest src/agents/platform_service`
Expected: PASS (barrel, registration, builder_assets, business_objects, client, storage, webhooks).

- [ ] **Step 7: Commit**

```bash
git add agent/src/agents/platform_service/index.ts \
        agent/src/agents/platform_service/__tests__/barrel.test.ts
git commit -m "refactor(agent): replace BO schema/records plugins with unified business-objects"
```

---

## Task 4: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full platform_service test suite**

Run: `npx jest src/agents/platform_service`
Expected: all suites PASS.

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: exactly one error remains, the pre-existing unrelated one:
`src/tools/metricsToolClient.ts(195,18): error TS18047: 'value' is possibly 'null'.`
No errors from `business_objects/`. (The previous `recordsPlugin.ts:106` error is gone because that file was deleted.)

- [ ] **Step 3: Confirm no dangling references**

Run: `npx jest src/agents/platform_service/__tests__/barrel.test.ts src/agents/platform_service/__tests__/registration.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit any verification fixups (only if needed)**

If Step 1-3 required edits, stage and commit them; otherwise nothing to commit.

---

## Notes for the implementer

- Do not edit `agent/src/agents/platform_service/__tests__/business_objects.test.ts` — executor behavior is unchanged.
- The plugin registers itself on import (`PluginRegistry.register(businessObjectPlugin)`); tests that import it get the mock registry from the `jest.mock("@axiom-lattice/core", ...)` at the top of `barrel.test.ts` / `registration.test.ts`.
- `openExpose` must stay a subset of the middleware tool names; the registration test asserts this invariant.
- `allowedTools` scoping for query-only consumers is documented in the spec §7; no code change is required for it.
