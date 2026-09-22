# Gateway Customer Tag Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the gateway's mock customer tags with real Business Object data: a new `tag_definition` master table, a rewritten `PUT /voice-tagging/:id/tags` that resolves/creates tags and reconciles `customer_tag`, and a `GET /customers/:customerId/tags` backed by BO.

**Architecture:** The gateway calls `business-objects_*` open-mcp tools through a thin adapter (`src/upstream/boTools.ts`) over the existing `McpCaller`. On tag edit, ids are left to the caller when present (validated against `tag_definition`, master never mutated) and created when absent; the task `result` is overwritten and the customer's `customer_tag` rows are reconciled to the union of all that customer's task tags (eventual consistency, no stale voice rows).

**Tech Stack:** Node 20 + TypeScript, Fastify 5, Vitest, `@modelcontextprotocol/sdk`, `zod` (not needed here). Design spec: `docs/superpowers/specs/2026-09-22-el-gateway-customer-tag-sync-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `el-ai-gateway/src/upstream/boTools.ts` | New: thin BO adapter over `McpCaller` (`getRecord`/`queryRecords`/`createRecord`/`deleteRecords`) |
| `el-ai-gateway/src/upstream/taskTools.ts` | Add `updatedAt` to `TaskRecord` + mappings |
| `el-ai-gateway/src/routes/tasks.ts` | Rewrite `PUT /tags`; normalize task tag shape; add reconcile |
| `el-ai-gateway/src/routes/customers.ts` | Read `customer_tag` via `boTools` |
| `el-ai-gateway/src/server.ts`, `src/index.ts` | Wire `boTools` into deps |
| `el-ai-gateway/src/mock/customerTags.ts` | Delete |
| `el-ai-gateway/test/boTools.test.ts` | New unit tests |
| `el-ai-gateway/test/tasksRoute.test.ts`, `test/customersRoute.test.ts`, `test/filesRoute.test.ts`, `test/cors.test.ts`, `test/upload.integration.test.ts` | Update deps + assertions |
| `el-ai-gateway/API.md` | Customer-facing docs |

---

### Task 1: BO adapter (`boTools`)

**Files:**
- Create: `el-ai-gateway/src/upstream/boTools.ts`
- Test: `el-ai-gateway/test/boTools.test.ts`

- [ ] **Step 1: Write the failing test**

Create `el-ai-gateway/test/boTools.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { createBoTools } from "../src/upstream/boTools";
import type { McpCaller } from "../src/upstream/mcp";
import { GatewayError } from "../src/lib/errors";

function mcpReturning(payload: unknown): McpCaller {
  return { callTool: vi.fn(async () => ({ text: JSON.stringify(payload), isError: false })) };
}

describe("boTools", () => {
  it("getRecord merges data + id and calls business-objects_get_record", async () => {
    const mcp = mcpReturning({ id: "t1", objectKey: "tag_definition", data: { tag_id: "t1", tag_group: "g", tag_name: "n" } });
    const bo = createBoTools(mcp);
    await expect(bo.getRecord("tag_definition", "t1")).resolves.toEqual({
      tag_id: "t1", tag_group: "g", tag_name: "n", id: "t1",
    });
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_get_record", { objectKey: "tag_definition", id: "t1" });
  });

  it("getRecord returns undefined for a not-found payload", async () => {
    const bo = createBoTools(mcpReturning({ success: false, error: "record not found" }));
    await expect(bo.getRecord("tag_definition", "nope")).resolves.toBeUndefined();
  });

  it("queryRecords returns rows and forwards filters", async () => {
    const mcp = mcpReturning({ objectKey: "customer_tag", total: 1, rows: [{ id: "c1", tag_id: "t1" }] });
    const bo = createBoTools(mcp);
    await expect(bo.queryRecords("customer_tag", [{ field: "customer_no", op: "eq", value: "cus_1" }]))
      .resolves.toEqual([{ id: "c1", tag_id: "t1" }]);
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_query_records", {
      objectKey: "customer_tag",
      filters: [{ field: "customer_no", op: "eq", value: "cus_1" }],
    });
  });

  it("createRecord merges data + id", async () => {
    const mcp = mcpReturning({ id: "new1", objectKey: "customer_tag", data: { tag_id: "t1" } });
    const bo = createBoTools(mcp);
    await expect(bo.createRecord("customer_tag", { customer_no: "c" })).resolves.toEqual({ tag_id: "t1", id: "new1" });
  });

  it("deleteRecords passes confirm:true and returns the count", async () => {
    const mcp = mcpReturning({ objectKey: "customer_tag", deleted: 2, ids: ["a", "b"] });
    const bo = createBoTools(mcp);
    await expect(bo.deleteRecords("customer_tag", ["a", "b"])).resolves.toBe(2);
    expect(mcp.callTool).toHaveBeenCalledWith("business-objects_delete_records", {
      objectKey: "customer_tag", ids: ["a", "b"], confirm: true,
    });
  });

  it("deleteRecords short-circuits on an empty id list", async () => {
    const mcp = mcpReturning({ deleted: 0 });
    const bo = createBoTools(mcp);
    await expect(bo.deleteRecords("customer_tag", [])).resolves.toBe(0);
    expect(mcp.callTool).not.toHaveBeenCalled();
  });

  it("queryRecords maps an error payload to 502", async () => {
    const bo = createBoTools(mcpReturning({ success: false, error: "boom" }));
    await expect(bo.queryRecords("customer_tag", []))
      .rejects.toMatchObject({ statusCode: 502 });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm exec vitest run test/boTools.test.ts`
Expected: FAIL — `../src/upstream/boTools` cannot be resolved.

- [ ] **Step 3: Implement `boTools`**

Create `el-ai-gateway/src/upstream/boTools.ts`:

```ts
import { GatewayError } from "../lib/errors";
import type { McpCaller } from "./mcp";

export type BoRecord = Record<string, unknown>;

export type BoFilter = { field: string; op?: string; value: unknown };

export type BoTools = {
  getRecord(objectKey: string, id: string): Promise<BoRecord | undefined>;
  queryRecords(objectKey: string, filters: BoFilter[]): Promise<BoRecord[]>;
  createRecord(objectKey: string, data: BoRecord): Promise<BoRecord>;
  deleteRecords(objectKey: string, ids: string[]): Promise<number>;
};

function parse(text: string, structured: unknown): any {
  if (structured && typeof structured === "object") return structured;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export function createBoTools(mcp: McpCaller): BoTools {
  async function call(tool: string, args: Record<string, unknown>): Promise<any> {
    const { text, structured } = await mcp.callTool(`business-objects_${tool}`, args);
    return parse(text, structured);
  }

  function fail(data: any, tool: string): never {
    throw new GatewayError(502, "UPSTREAM_ERROR", String(data?.error ?? `${tool} failed`));
  }

  return {
    async getRecord(objectKey, id) {
      let data: any;
      try {
        data = await call("get_record", { objectKey, id });
      } catch (err) {
        if (err instanceof GatewayError && /not found/i.test(err.message)) return undefined;
        throw err;
      }
      if (!data || data.success === false || data.error) return undefined;
      const record = data.record ?? data;
      if (!record || typeof record !== "object" || typeof record.id !== "string") return undefined;
      return { ...(record.data ?? {}), id: record.id };
    },

    async queryRecords(objectKey, filters) {
      const data = await call("query_records", { objectKey, filters });
      if (!data || data.success === false || data.error) fail(data, "query_records");
      const rows = data.rows ?? data.data?.rows ?? [];
      return Array.isArray(rows) ? rows : [];
    },

    async createRecord(objectKey, input) {
      const data = await call("create_record", { objectKey, data: input });
      if (!data || data.success === false || data.error) fail(data, "create_record");
      const record = data.record ?? data;
      return { ...(record.data ?? {}), id: record.id };
    },

    async deleteRecords(objectKey, ids) {
      if (ids.length === 0) return 0;
      const data = await call("delete_records", { objectKey, ids, confirm: true });
      if (!data || data.success === false || data.error) fail(data, "delete_records");
      return typeof data.deleted === "number" ? data.deleted : 0;
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm exec vitest run test/boTools.test.ts`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/upstream/boTools.ts el-ai-gateway/test/boTools.test.ts
git commit -m "feat(gateway): add business-objects MCP adapter"
```

---

### Task 2: Wire `boTools` into server deps

**Files:**
- Modify: `el-ai-gateway/src/server.ts`, `el-ai-gateway/src/index.ts`
- Modify tests that call `buildServer`: `test/tasksRoute.test.ts`, `test/customersRoute.test.ts`, `test/filesRoute.test.ts`, `test/cors.test.ts`, `test/upload.integration.test.ts`

- [ ] **Step 1: Add `boTools` to `ServerDeps`**

In `el-ai-gateway/src/server.ts`, add the import and field:

```ts
import type { BoTools } from "./upstream/boTools";
```

```ts
export type ServerDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  agentRuns: AgentRunsClient;
  taskTools: TaskToolClient;
  boTools: BoTools;
};
```

(`registerTaskRoutes`/`registerCustomerRoutes` receive `deps` and will type their own deps in Tasks 3/4; no change needed here yet.)

- [ ] **Step 2: Wire it in `index.ts`**

In `el-ai-gateway/src/index.ts`:

```ts
import { createBoTools } from "./upstream/boTools";
```

```ts
  const app = buildServer({
    config,
    authenticator: createAuthenticator(config),
    platformFiles: createPlatformFilesClient(config),
    agentRuns: createAgentRunsClient(config),
    taskTools: createTaskToolClient(mcp),
    boTools: createBoTools(mcp),
  });
```

- [ ] **Step 3: Add a `boTools` stub to every test deps object**

In each of the five test files that call `buildServer`, add this to the deps object passed to `buildServer` (place it next to `taskTools`):

```ts
    boTools: {
      getRecord: vi.fn(async () => undefined),
      queryRecords: vi.fn(async () => []),
      createRecord: vi.fn(async () => ({})),
      deleteRecords: vi.fn(async () => 0),
    } as any,
```

For files that don't already import `vi`, add `vi` to the existing `vitest` import.

- [ ] **Step 4: Typecheck + run the suite**

Run: `cd el-ai-gateway && pnpm typecheck`
Expected: no errors.

Run: `cd el-ai-gateway && pnpm test`
Expected: PASS (boTools tests + existing suites).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/server.ts el-ai-gateway/src/index.ts el-ai-gateway/test
git commit -m "feat(gateway): wire boTools into server deps"
```

---

### Task 3: Rewrite `PUT /voice-tagging/:id/tags` + reconcile

**Files:**
- Modify: `el-ai-gateway/src/upstream/taskTools.ts`
- Modify: `el-ai-gateway/src/routes/tasks.ts`
- Test: `el-ai-gateway/test/tasksRoute.test.ts`

- [ ] **Step 1: Add `updatedAt` to the task record**

In `el-ai-gateway/src/upstream/taskTools.ts`, add `updatedAt?: string;` to `TaskRecord` (after `createdAt?: string;`), and include it in both mappings:

```ts
        createdAt: task.createdAt,
        updatedAt: task.updatedAt,
        activities: data?.activities ?? task.activities ?? [],
```

```ts
        createdAt: task.createdAt,
        updatedAt: task.updatedAt,
        activities: task.activities ?? [],
```

- [ ] **Step 2: Write the failing tests**

Replace the entire `describe("PUT /api/v1/voice-tagging/:id/tags", ...)` block in `el-ai-gateway/test/tasksRoute.test.ts` with:

```ts
describe("PUT /api/v1/voice-tagging/:id/tags", () => {
  const TASK = "b3ca978f-3832-4a2c-959b-40fa48c43352";
  const DEF_ID = "9ce355bfacca49c4a9e9322a9317c196";

  function boStub(overrides: Record<string, unknown> = {}) {
    return {
      getRecord: vi.fn(async (_objectKey: string, id: string) =>
        id === DEF_ID
          ? { id: DEF_ID, tag_id: DEF_ID, tag_group: "concerns", tag_name: "抗老/紧致" }
          : undefined,
      ),
      queryRecords: vi.fn(async () => []),
      createRecord: vi.fn(async () => ({})),
      deleteRecords: vi.fn(async () => 0),
      ...overrides,
    };
  }

  function statefulDeps(boOverrides: Record<string, unknown> = {}) {
    let current: any = {
      id: TASK,
      status: "in_progress",
      title: "Voice tagging: 2ccf6fef88b64a16b62fe491a8f7a132",
      createdAt: "2026-09-15T05:18:00Z",
      updatedAt: "2026-09-15T05:20:00Z",
      metadata: { uuid: "2ccf6fef88b64a16b62fe491a8f7a132", baId: "ba_001", customerId: "cus_8899" },
      result: "[]",
      activities: [
        { id: "act-1", action: "activity", detail: { markdown: "## t" }, createdAt: "2026-09-15T05:20:00Z" },
      ],
      raw: {},
    };
    return deps({
      boTools: boStub(boOverrides),
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => current),
        updateResult: vi.fn(async ({ result }: { result: string }) => {
          current = {
            ...current,
            result,
            activities: [
              { id: "act-2", action: "updated", detail: { markdown: "" }, createdAt: "2026-09-15T07:20:00Z" },
              ...current.activities,
            ],
          };
          return { raw: {} };
        }),
        listTasks: vi.fn(async () => [current]),
      },
    });
  }

  it("resolves an existing tagId against tag_definition and writes {tagId,tagKey,tagValue}", async () => {
    const d = statefulDeps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagId: DEF_ID, tagValue: "ignored" }] },
    });
    expect(res.statusCode).toBe(200);
    expect(d.boTools.getRecord).toHaveBeenCalledWith("tag_definition", DEF_ID);
    expect(d.taskTools.updateResult).toHaveBeenCalledWith({
      id: TASK,
      result: JSON.stringify([{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }]),
    });
    expect(res.json().tags).toEqual([{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }]);
  });

  it("creates a new tag_definition (group 客户画像) when no tagId is given", async () => {
    const createRecord = vi.fn(async () => ({}));
    const d = statefulDeps({ queryRecords: vi.fn(async () => []), createRecord });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagValue: "新标签" }] },
    });
    expect(res.statusCode).toBe(200);
    const created = createRecord.mock.calls.find((c) => c[0] === "tag_definition");
    expect(created?.[1]).toMatchObject({
      tag_group: "客户画像", category: "自定义标签", tag_name: "新标签",
    });
    expect((created?.[1] as any).tag_id).toMatch(/^[0-9a-f]{32}$/);
  });

  it("reuses an existing master entry for a new tagValue", async () => {
    const createRecord = vi.fn(async () => ({}));
    const d = statefulDeps({
      queryRecords: vi.fn(async () => [{ id: "rec1", tag_id: "abc", tag_group: "客户画像", tag_name: "新标签" }]),
      createRecord,
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagValue: "新标签" }] },
    });
    expect(res.statusCode).toBe(200);
    expect(createRecord.mock.calls.some((c) => c[0] === "tag_definition")).toBe(false);
    expect(res.json().tags).toEqual([{ tagId: "abc", tagKey: "客户画像", tagValue: "新标签" }]);
  });

  it("rejects an unknown tagId", async () => {
    const app = buildServer(statefulDeps());
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: ["00000000000000000000000000000000"] },
    });
    expect(res.statusCode).toBe(400);
    expect(res.json().code).toBe("BAD_REQUEST");
  });

  it("rejects a non-array tags body", async () => {
    const app = buildServer(statefulDeps());
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: "nope" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("reconciles customer_tag to the union of the customer's task tags", async () => {
    const staleId = "cccccccccccccccccccccccccccccccc";
    const createRecord = vi.fn(async () => ({}));
    const deleteRecords = vi.fn(async () => 1);
    const d = statefulDeps({
      getRecord: vi.fn(async (_o: string, id: string) =>
        id === DEF_ID
          ? { id: DEF_ID, tag_id: DEF_ID, tag_group: "concerns", tag_name: "抗老/紧致" }
          : { id, tag_id: id, tag_group: "g", tag_name: "n" },
      ),
      queryRecords: vi.fn(async (objectKey: string) =>
        objectKey === "customer_tag"
          ? [{ id: "row-stale", tag_id: staleId, source: "voice", customer_no: "cus_8899" }]
          : [],
      ),
      createRecord,
      deleteRecords,
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagId: DEF_ID, tagValue: "x" }] },
    });
    expect(res.statusCode).toBe(200);
    const customerCreate = createRecord.mock.calls.find((c) => c[0] === "customer_tag");
    expect(customerCreate?.[1]).toMatchObject({
      customer_no: "cus_8899", tag_id: DEF_ID, tag_key: "concerns", tag_value: "抗老/紧致", source: "voice",
    });
    expect(deleteRecords).toHaveBeenCalledWith("customer_tag", ["row-stale"]);
  });

  it("returns 404 for an unknown task", async () => {
    const d = deps();
    d.taskTools.getTask = vi.fn(async () => {
      throw new GatewayError(404, "NOT_FOUND", "Task 'nope' not found");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: "/api/v1/voice-tagging/nope/tags",
      headers: { authorization: "Bearer secret" },
      payload: { tags: [] },
    });
    expect(res.statusCode).toBe(404);
  });
});
```

Also update the shared `deps()` helper at the top of the file so its default task includes `baId`/`customerId` metadata (needed by list/detail tests too):

```ts
        metadata: { uuid: "471c20082b524316accc1b23cba8a4de", baId: "ba_001", customerId: "cus_8899" },
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: FAIL — old handler ignores BO, `{tagId,tagKey,tagValue}` shape not produced.

- [ ] **Step 4: Rewrite the route**

In `el-ai-gateway/src/routes/tasks.ts`:

Remove `import { resolveTags } from "../mock/customerTags";`. Add imports:

```ts
import { randomBytes } from "node:crypto";
import type { BoTools } from "../upstream/boTools";
```

Add `boTools: BoTools;` to `TaskRouteDeps`.

Replace `parseTags` and add tag types/helpers (keep `toDetail`/`mapActivities` but normalize tag shape):

```ts
export type TaskTag = { tagId: string; tagKey?: string; tagValue?: string; name?: string; dimension?: string };

function parseTags(result: string | undefined): TaskTag[] {
  if (!result) return [];
  try {
    const parsed = JSON.parse(result);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function normalizeTags(result: string | undefined): Array<{ tagId: string; tagKey: string; tagValue: string }> {
  return parseTags(result)
    .filter((t) => typeof t?.tagId === "string")
    .map((t) => ({
      tagId: t.tagId,
      tagKey: (t.tagKey ?? t.dimension ?? "") as string,
      tagValue: (t.tagValue ?? t.name ?? "") as string,
    }));
}
```

In `toDetail`, change `tags: parseTags(task.result)` to `tags: normalizeTags(task.result)`. In the list route, change `tags: parseTags(task.result)` to `tags: normalizeTags(task.result)`.

Add the reconcile helper above `registerTaskRoutes`:

```ts
async function reconcileCustomerTags(
  deps: TaskRouteDeps,
  input: { ownerId: string; baId: string; customerId: string },
): Promise<void> {
  try {
    const tasks = await deps.taskTools.listTasks({
      ownerId: input.ownerId,
      baId: input.baId,
      customerId: input.customerId,
    });
    const union = new Map<string, string | undefined>();
    for (const task of tasks) {
      const when = task.updatedAt ?? task.createdAt;
      for (const tag of parseTags(task.result)) {
        if (typeof tag?.tagId !== "string") continue;
        const prev = union.get(tag.tagId);
        if (!prev || (when !== undefined && when > prev)) union.set(tag.tagId, when);
      }
    }
    const existing = await deps.boTools.queryRecords("customer_tag", [
      { field: "customer_no", op: "eq", value: input.customerId },
    ]);
    const existingIds = new Set(existing.map((row) => String(row.tag_id)));
    for (const [tagId, when] of union) {
      if (existingIds.has(tagId)) continue;
      const def = await deps.boTools.getRecord("tag_definition", tagId);
      await deps.boTools.createRecord("customer_tag", {
        customer_no: input.customerId,
        tag_key: String(def?.tag_group ?? ""),
        tag_value: String(def?.tag_name ?? ""),
        tag_id: tagId,
        source: "voice",
        confidence: null,
        tagged_at: when ?? new Date().toISOString(),
      });
    }
    const staleIds = existing
      .filter((row) => row.source === "voice" && !union.has(String(row.tag_id)))
      .map((row) => String(row.id));
    if (staleIds.length > 0) await deps.boTools.deleteRecords("customer_tag", staleIds);
  } catch (err) {
    console.error(
      `[voice-tagging] customer tag reconcile failed for ${input.customerId}: ${(err as Error).message}`,
    );
  }
}
```

Replace the `PUT` handler with:

```ts
  // Replace a task's tags (stored as JSON in the task `result`), then reconcile
  // the customer's aggregate tags (customer_tag) to the union of task tags.
  app.put("/api/v1/voice-tagging/:id/tags", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });

    const metadata = (task.metadata ?? {}) as Record<string, unknown>;
    const baId = typeof metadata.baId === "string" ? metadata.baId : undefined;
    const customerId = typeof metadata.customerId === "string" ? metadata.customerId : undefined;
    if (!baId || !customerId) {
      throw new GatewayError(400, "BAD_REQUEST", "task is missing baId/customerId metadata");
    }

    const body = (request.body ?? {}) as { tags?: unknown };
    if (!Array.isArray(body.tags)) {
      throw new GatewayError(400, "BAD_REQUEST", "tags must be an array");
    }

    const out: Array<{ tagId: string; tagKey: string; tagValue: string }> = [];
    const seen = new Set<string>();
    for (const item of body.tags) {
      const input = (typeof item === "string" ? { tagId: item } : (item ?? {})) as {
        tagId?: unknown;
        tagValue?: unknown;
      };
      let tagId: string | undefined;
      let tagKey: string;
      let tagValue: string;
      if (typeof input.tagId === "string" && input.tagId.trim() !== "") {
        const def = await deps.boTools.getRecord("tag_definition", input.tagId);
        if (!def) {
          throw new GatewayError(400, "BAD_REQUEST", `Unknown tagId: ${input.tagId}`);
        }
        tagId = String(def.tag_id ?? input.tagId);
        tagKey = String(def.tag_group ?? "");
        tagValue = String(def.tag_name ?? "");
      } else {
        if (typeof input.tagValue !== "string" || input.tagValue.trim() === "") {
          throw new GatewayError(400, "BAD_REQUEST", "each tag needs a tagId or a tagValue");
        }
        tagValue = input.tagValue.trim();
        tagKey = "客户画像";
        const existing = await deps.boTools.queryRecords("tag_definition", [
          { field: "tag_group", op: "eq", value: tagKey },
          { field: "tag_name", op: "eq", value: tagValue },
        ]);
        if (existing.length > 0) {
          tagId = String(existing[0].tag_id);
        } else {
          tagId = randomBytes(16).toString("hex");
          await deps.boTools.createRecord("tag_definition", {
            category: "自定义标签",
            tag_group: tagKey,
            tag_name: tagValue,
            tag_id: tagId,
          });
        }
      }
      if (!tagId || seen.has(tagId)) continue;
      seen.add(tagId);
      out.push({ tagId, tagKey, tagValue });
    }

    await deps.taskTools.updateResult({ id, result: JSON.stringify(out) });
    await reconcileCustomerTags(deps, { ownerId: principal.tenantId, baId, customerId });

    const updated = await deps.taskTools.getTask({ id });
    const activities = mapActivities(updated.activities);
    return { ...toDetail(updated), activity: activities[0] };
  });
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add el-ai-gateway/src/upstream/taskTools.ts el-ai-gateway/src/routes/tasks.ts el-ai-gateway/test/tasksRoute.test.ts
git commit -m "feat(gateway): resolve/create tags and reconcile customer_tag on PUT"
```

---

### Task 4: Rewrite `GET /customers/:customerId/tags` + remove mock

**Files:**
- Modify: `el-ai-gateway/src/routes/customers.ts`
- Delete: `el-ai-gateway/src/mock/customerTags.ts` (and the now-empty `src/mock/` directory)
- Test: `el-ai-gateway/test/customersRoute.test.ts`

- [ ] **Step 1: Write the failing test**

Replace `el-ai-gateway/test/customersRoute.test.ts` with:

```ts
import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesUrl: "http://files",
  maxUploadBytes: 1024,
  agentRunsUrl: "http://agent/api/runs",
  agentAuthUrl: "http://agent/api/auth/login",
  agentLoginEmail: "svc@example.com",
  agentLoginPassword: "secret",
  agentTenantId: "estee_lauder",
  agentWorkspaceId: "default-workspace",
  agentProjectId: "default",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "m",
  upstreamTimeoutMs: 1000,
};

function build(auth: (h?: string) => { tenantId: string; keyLabel: string } | null, rows: any[] = []) {
  return buildServer({
    config,
    authenticator: auth,
    platformFiles: { upload: vi.fn(), presign: vi.fn(), list: vi.fn() } as any,
    agentRuns: { startRun: vi.fn() } as any,
    taskTools: { createTask: vi.fn(), getTask: vi.fn() } as any,
    boTools: {
      getRecord: vi.fn(),
      queryRecords: vi.fn(async () => rows),
      createRecord: vi.fn(),
      deleteRecords: vi.fn(),
    } as any,
  });
}

const auth = (h?: string) =>
  h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null;

const ROWS = [
  {
    id: "row1",
    customer_no: "cus_8899",
    tag_key: "concerns",
    tag_value: "抗老/紧致",
    tag_id: "9ce355bfacca49c4a9e9322a9317c196",
    source: "voice",
    confidence: null,
    tagged_at: "2026-09-15T06:00:00Z",
  },
];

describe("GET /api/v1/customers/:customerId/tags", () => {
  it("maps customer_tag rows to tagId/tagKey/tagValue", async () => {
    const app = build(auth, ROWS);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/customers/cus_8899/tags",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.customerId).toBe("cus_8899");
    expect(body.total).toBe(1);
    expect(body.tags[0]).toEqual({
      tagId: "9ce355bfacca49c4a9e9322a9317c196",
      tagKey: "concerns",
      tagValue: "抗老/紧致",
      source: "voice",
      confidence: null,
      taggedAt: "2026-09-15T06:00:00Z",
    });
  });

  it("returns 200 with an empty list when the customer has no tags", async () => {
    const app = build(auth, []);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/customers/nobody/tags",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ customerId: "nobody", total: 0, tags: [] });
  });

  it("returns 401 without a valid key", async () => {
    const app = build(() => null, ROWS);
    const res = await app.inject({ method: "GET", url: "/api/v1/customers/cus_8899/tags" });
    expect(res.statusCode).toBe(401);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm exec vitest run test/customersRoute.test.ts`
Expected: FAIL — route still reads the mock (`name`/`dimension`, 404 path).

- [ ] **Step 3: Rewrite the route**

Replace `el-ai-gateway/src/routes/customers.ts` with:

```ts
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import type { BoTools } from "../upstream/boTools";

export type CustomerRouteDeps = {
  authenticator: Authenticator;
  boTools: BoTools;
};

export function registerCustomerRoutes(app: FastifyInstance, deps: CustomerRouteDeps): void {
  // All business tags of a customer, read from the customer_tag aggregate.
  app.get("/api/v1/customers/:customerId/tags", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { customerId } = request.params as { customerId: string };
    const rows = await deps.boTools.queryRecords("customer_tag", [
      { field: "customer_no", op: "eq", value: customerId },
    ]);
    const tags = rows.map((row) => ({
      tagId: row.tag_id,
      tagKey: row.tag_key,
      tagValue: row.tag_value,
      source: row.source,
      confidence: row.confidence ?? null,
      taggedAt: row.tagged_at,
    }));
    return { customerId, total: tags.length, tags };
  });
}
```

- [ ] **Step 4: Delete the mock**

```bash
git rm el-ai-gateway/src/mock/customerTags.ts
```

- [ ] **Step 5: Run tests + typecheck**

Run: `cd el-ai-gateway && pnpm exec vitest run test/customersRoute.test.ts`
Expected: PASS.

Run: `cd el-ai-gateway && pnpm test && pnpm typecheck`
Expected: all suites PASS; typecheck clean.

- [ ] **Step 6: Commit**

```bash
git add -A el-ai-gateway/src/routes/customers.ts el-ai-gateway/src/mock el-ai-gateway/test/customersRoute.test.ts
git commit -m "feat(gateway): read customer tags from BO and remove mock"
```

---

### Task 5: Update `API.md`

**Files:**
- Modify: `el-ai-gateway/API.md`

- [ ] **Step 1: Update §2 and §5 (customer tags are derived from task tags)**

In §2, change the two bullets describing "客户标签由系统内部更新" to:

```
- 任务标签是"某条语音的结论"；客户标签是该客户所有任务标签的汇总（最终一致）。两者不要混用。
- 客户标签由**任务标签汇总**而来：编辑某任务的标签（`PUT /voice-tagging/:taskId/tags`）后，系统会重算该客户的标签汇总。外部**没有客户级写接口**，只能查询。
```

In §5, change the rows to:

```
| 回显客户现有标签 | `GET /customers/:customerId/tags` | 客户级、只读 |
| 补充 / 更新客户标签 | 无（由任务标签自动汇总） | 编辑任务标签后，客户汇总会自动更新 |
| 若是修正"某条语音"的标签 | `GET /voice-tagging/:taskId` 读取 → 合并 → `PUT /voice-tagging/:taskId/tags` | 任务级、覆盖式 |
```

- [ ] **Step 2: Update §8.2 / §8.3 / §8.5 tag shapes**

Replace every task tag example/description that uses `name`/`dimension` with `tagKey`/`tagValue`:

```json
{ "tagId": "9ce355bfacca49c4a9e9322a9317c196", "tagKey": "concerns", "tagValue": "抗老/紧致" }
```

In §8.3 change the `tags` bullet to:

```
- `tags`：该任务已生成的标签（`tagId` 32 位 hex / `tagKey` 标签组 / `tagValue` 标签名）
```

In §8.5 change the request block and notes to:

```json
{ "tags": [ { "tagId": "9ce355bfacca49c4a9e9322a9317c196", "tagValue": "抗老/紧致" } ] }
```

```
- 有 `tagId`：必须是标签主数据中存在的 id，标签组/标签名以主数据为准。
- 无 `tagId`：按 `tagValue` 新建标签（标签组固定为 `客户画像`，类别为 `自定义标签`），并把生成的 `tagId` 回写到任务标签。
- `tags` 非数组 / 元素同时缺 `tagId` 与 `tagValue` / `tagId` 不存在 → `400 BAD_REQUEST`
- 任务不存在 → `404 NOT_FOUND`
- 说明：该接口整体替换本任务标签，并自动追加一条 activity；同时**重算该客户的标签汇总**。
```

Update the §8.5 response `tags` example to the `tagKey`/`tagValue` shape.

- [ ] **Step 3: Update §9 (customer tags)**

Replace §9.1 body/response/table with the BO-backed contract:

```json
{
  "customerId": "cus_8899",
  "total": 1,
  "tags": [
    {
      "tagId": "9ce355bfacca49c4a9e9322a9317c196",
      "tagKey": "concerns",
      "tagValue": "抗老/紧致",
      "source": "voice",
      "confidence": null,
      "taggedAt": "2026-09-15T06:00:00Z"
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `customerId` | 客户 id |
| `total` | 标签数量 |
| `tags[].tagId` | 标签值 id（32 位十六进制） |
| `tags[].tagKey` | 标签组 |
| `tags[].tagValue` | 标签名 |
| `tags[].source` | 来源（`voice` / `manual`） |
| `tags[].confidence` | 置信度（可空） |
| `tags[].taggedAt` | 打标时间（ISO 8601） |

- 删除"客户不存在 → 404"和"当前为 mock 数据"两条，改为：
  `- 客户没有标签时返回 `200` + 空数组（`total: 0`）。`
  `- 数据来源为标签主数据与任务标签的汇总，最终一致（刚编辑完可能有短暂延迟）。`

- [ ] **Step 4: Update §10 error codes**

Change the `400` row description to:

```
| 400 | `BAD_REQUEST` | 参数非法（缺 `uuid` / `baId` / `customerId`；`tags` 非法或 `tagId` 不存在等） |
```

Remove `客户不存在` from the `404` row:
```
| 404 | `NOT_FOUND` | 任务不存在 |
```

- [ ] **Step 5: Verify no stale references**

Run: `cd el-ai-gateway && grep -n '"dimension"\|"evidence"\|"name"' API.md`
Expected: no matches (tag fields `dimension`/`evidence`/`name` gone).

Run: `cd el-ai-gateway && grep -n "mock 数据" API.md`
Expected: no matches (the customer-tag mock note is removed; the transcription "转写为 mock" notes at the top are unrelated and stay).

- [ ] **Step 6: Commit**

```bash
git add el-ai-gateway/API.md
git commit -m "docs(gateway): document BO-backed tag sync"
```

---

### Task 6: Ops — create `tag_definition`, rebuild `customer_tag`, deploy, verify

**Files:** none (production data + deploy)

- [ ] **Step 1: Deploy the gateway**

Merge to `main` (CI rebuilds `ghcr.io/409zhangshu/fina-demo-el-ai-gateway`), then on 38:
`bash docker-deploy-38.sh 7`

- [ ] **Step 2: Create a temporary MANAGE store key for the API calls**

`bo_store_api_keys` stores `key_hash = sha256(rawKey)` (hex). On the platform DB, insert a temporary key for store `elc` (`store_id = 1`):

```sql
INSERT INTO bo_store_api_keys (store_id, key_name, key_hash, permissions_json, status)
VALUES (1, 'tag_sync_tmp', encode(digest('<RAW_KEY>', 'sha256'), 'hex'), '["READ","WRITE","MANAGE"]', 1)
RETURNING id;
```

(If `pgcrypto` is unavailable, compute the SHA-256 hex externally and insert the literal.) Use `X-BO-Connection-Key: <RAW_KEY>` for the POSTs below; delete the row when done (`DELETE FROM bo_store_api_keys WHERE key_name = 'tag_sync_tmp'`).

- [ ] **Step 3: Create the `tag_definition` object (elc store)**

Using the platform-service API with the temporary key, POST to `/api/v1/bo/objects`:

```json
{
  "storeKey": "elc",
  "objectKey": "tag_definition",
  "displayName": "Tag Definition",
  "description": "标签主数据：类别、标签组、标签、标签id、小肤标签",
  "fields": [
    { "key": "category", "type": "string", "required": false, "maxLength": 64, "description": "类别" },
    { "key": "tag_group", "type": "string", "required": true, "maxLength": 64, "description": "标签组（tagKey）" },
    { "key": "tag_name", "type": "string", "required": true, "maxLength": 255, "description": "标签名" },
    { "key": "tag_id", "type": "string", "required": true, "maxLength": 64, "description": "标签值 id" },
    { "key": "is_xiaofu", "type": "boolean", "required": false, "description": "小肤标签" }
  ],
  "indexes": [
    { "name": "uk_tag_id", "fields": ["tag_id"], "unique": true },
    { "name": "tag_group", "fields": ["tag_group"] }
  ],
  "deleteMode": "hard"
}
```

- [ ] **Step 4: Rebuild `customer_tag` with the new schema**

Drop the old table + definition, recreate (data is empty):

```sql
DROP TABLE IF EXISTS bo.bo_customer_tag;
```

```sql
DELETE FROM bo_object_ddl_history WHERE object_key = 'customer_tag';
DELETE FROM bo_object_definitions WHERE store_id = 1 AND object_key = 'customer_tag';
```

Then POST:

```json
{
  "storeKey": "elc",
  "objectKey": "customer_tag",
  "displayName": "Customer Tag",
  "description": "客户标签：标签组/标签名/标签id（汇总自任务标签）",
  "fields": [
    { "key": "customer_no", "type": "string", "required": true, "maxLength": 64, "description": "关联客户编号" },
    { "key": "tag_key", "type": "string", "required": true, "maxLength": 64, "description": "标签组" },
    { "key": "tag_value", "type": "string", "required": true, "maxLength": 255, "description": "标签名" },
    { "key": "tag_id", "type": "string", "required": true, "maxLength": 64, "description": "标签值 id" },
    { "key": "source", "type": "string", "required": false, "maxLength": 32, "description": "来源 voice/manual" },
    { "key": "confidence", "type": "decimal", "required": false, "precision": 5, "scale": 4, "description": "置信度" },
    { "key": "tagged_at", "type": "datetime", "required": false, "description": "打标时间" }
  ],
  "indexes": [
    { "name": "uk_customer_tag", "fields": ["customer_no", "tag_id"], "unique": true },
    { "name": "tag_id", "fields": ["tag_id"] },
    { "name": "customer_no", "fields": ["customer_no"] }
  ],
  "deleteMode": "hard"
}
```

- [ ] **Step 5: Live-verify end-to-end**

1. Create a task via `POST /api/v1/voice-tagging` (baId + customerId), or reuse one.
2. `PUT /api/v1/voice-tagging/:id/tags` with `{ "tags": [ { "tagId": "<existing tag_definition tag_id>" }, { "tagValue": "新标签" } ] }` → expect `200` with `tagKey`/`tagValue` resolved, and a new row in `tag_definition` for `新标签`.
3. `GET /api/v1/customers/<customerId>/tags` → expect the tags.
4. `PUT` again with a subset → confirm the removed `customer_tag` row is deleted (reconcile).
5. Confirm `bo_customer_tag` unique index is `(customer_no, tag_id)` and `bo_tag_definition` has `uk_tag_id`.

- [ ] **Step 6: Clean up probe data**

Delete probe rows (tasks/records) and confirm tables are back to the expected state.

---

## Notes for the implementer

- Gateway test/typecheck commands: `cd el-ai-gateway && pnpm test` and `cd el-ai-gateway && pnpm typecheck`.
- The BO MCP key already has `business-objects` read+write grants; no gateway config change is needed for BO access.
- `getRecord` must NOT throw on a missing record — the PUT handler relies on `undefined` to emit `400`.
- Reconcile is best-effort and must never fail the PUT (it catches internally and logs).
