# Task Result Aggregate (transcript/tags/like) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a voice-tagging task's `result` a single JSON object `{ transcript, tags, like }` and expose those three fields through the task detail/list and the tag/like edit endpoints.

**Architecture:** The gateway owns read-modify-write of `result`. `GET` endpoints parse `result` (object, or legacy array) into `{ transcript, tags, like }`. `PUT /tags` replaces `tags` while preserving `transcript`/`like` and inheriting `evidence` by `tagId`; a new `PUT /like` flips `like` between `true` and `null`.

**Tech Stack:** TypeScript / Fastify 5 / Vitest. Design spec: `docs/superpowers/specs/2026-09-22-gateway-task-result-aggregate-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `el-ai-gateway/src/routes/tasks.ts` | Result parsing/aggregation, detail/list shape, `PUT tags` RMW, new `PUT like` |
| `el-ai-gateway/test/tasksRoute.test.ts` | Detail/list aggregate tests, updated `PUT tags` tests, `PUT like` tests |
| `el-ai-gateway/API.md` | Customer-facing docs for the three fields + `PUT like` |

No other files change. `taskTools.ts` already exposes `getTask`/`listTasks`/`updateResult`.

---

### Task 1: Aggregate `result` parsing + detail/list fields

**Files:**
- Modify: `el-ai-gateway/src/routes/tasks.ts`
- Test: `el-ai-gateway/test/tasksRoute.test.ts`

- [ ] **Step 1: Write the failing tests**

Append this `describe` block to `el-ai-gateway/test/tasksRoute.test.ts`:

```ts
describe("task result aggregate (transcript/tags/like)", () => {
  const TAG_ID = "9ce355bfacca49c4a9e9322a9317c196";
  const OBJ_RESULT = JSON.stringify({
    transcript: "王女士提到皮肤偏干。",
    tags: [{ tagId: TAG_ID, tagKey: "concerns", tagValue: "抗老/紧致", evidence: "皮肤偏干" }],
    like: true,
  });

  function taskWith(result: string) {
    return {
      id: "task-agg",
      status: "completed",
      metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
      result,
      activities: [],
      raw: {},
    };
  }

  it("detail returns transcript / tags / like from an object result", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => taskWith(OBJ_RESULT)),
        listTasks: vi.fn(async () => []),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/task-agg",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.transcript).toBe("王女士提到皮肤偏干。");
    expect(body.like).toBe(true);
    expect(body.tags).toEqual([
      { tagId: TAG_ID, tagKey: "concerns", tagValue: "抗老/紧致", evidence: "皮肤偏干" },
    ]);
  });

  it("treats a legacy array result as tags only", async () => {
    const legacy = JSON.stringify([{ tagId: TAG_ID, name: "抗老/紧致", dimension: "concerns" }]);
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => taskWith(legacy)),
        listTasks: vi.fn(async () => []),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/task-agg",
      headers: { authorization: "Bearer secret" },
    });
    const body = res.json();
    expect(body.transcript).toBeNull();
    expect(body.like).toBeNull();
    expect(body.tags).toEqual([{ tagId: TAG_ID, tagKey: "concerns", tagValue: "抗老/紧致" }]);
  });

  it("list includes tags/like but not transcript", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(),
        updateResult: vi.fn(),
        listTasks: vi.fn(async () => [taskWith(OBJ_RESULT)]),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging?baId=ba_001&customerId=cus_8899",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const task = res.json().tasks[0];
    expect(task.like).toBe(true);
    expect(task.tags[0]).toMatchObject({ tagId: TAG_ID, tagKey: "concerns", tagValue: "抗老/紧致" });
    expect(task.transcript).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: FAIL — detail has no `transcript`/`like`; list has no `like` (transcript already absent, so the detail tests are the key failures).

- [ ] **Step 3: Implement the result helpers**

In `el-ai-gateway/src/routes/tasks.ts`, replace the block from `export type TaskTag = ...` through the end of `normalizeTags` (lines 34–54) with:

```ts
export type ResultTag = { tagId: string; tagKey: string; tagValue: string; evidence?: string };
export type TaskResult = { transcript: string | null; tags: ResultTag[]; like: boolean | null };

function normalizeResultTags(input: unknown): ResultTag[] {
  if (!Array.isArray(input)) return [];
  const out: ResultTag[] = [];
  for (const item of input) {
    if (!item || typeof item !== "object") continue;
    const t = item as Record<string, unknown>;
    if (typeof t.tagId !== "string") continue;
    const tag: ResultTag = {
      tagId: t.tagId,
      tagKey: String(t.tagKey ?? t.dimension ?? ""),
      tagValue: String(t.tagValue ?? t.name ?? ""),
    };
    if (typeof t.evidence === "string") tag.evidence = t.evidence;
    out.push(tag);
  }
  return out;
}

export function parseResult(result: string | undefined): TaskResult {
  const empty: TaskResult = { transcript: null, tags: [], like: null };
  if (!result) return empty;
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return empty;
  }
  if (Array.isArray(parsed)) {
    return { ...empty, tags: normalizeResultTags(parsed) };
  }
  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;
    return {
      transcript: typeof obj.transcript === "string" ? obj.transcript : null,
      tags: normalizeResultTags(obj.tags),
      like: obj.like === true ? true : null,
    };
  }
  return empty;
}
```

- [ ] **Step 4: Update `toDetail`**

Replace `toDetail` with:

```ts
function toDetail(task: TaskRecord) {
  const metadata = (task.metadata ?? {}) as Record<string, unknown>;
  const result = parseResult(task.result);
  return {
    taskId: task.id,
    fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
    status: task.status,
    createdAt: task.createdAt,
    title: task.title,
    transcript: result.transcript,
    tags: result.tags,
    like: result.like,
  };
}
```

- [ ] **Step 5: Update the list mapping**

In the `GET /api/v1/voice-tagging` handler, replace the `records.map(...)` body with:

```ts
    const tasks = records.map((task) => {
      const metadata = (task.metadata ?? {}) as Record<string, unknown>;
      const result = parseResult(task.result);
      return {
        taskId: task.id,
        fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
        status: task.status,
        createdAt: task.createdAt,
        tags: result.tags,
        like: result.like,
      };
    });
```

- [ ] **Step 6: Update reconcile to use `parseResult`**

In `reconcileCustomerTags`, replace `for (const tag of parseTags(task.result)) {` with:

```ts
      for (const tag of parseResult(task.result).tags) {
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add el-ai-gateway/src/routes/tasks.ts el-ai-gateway/test/tasksRoute.test.ts
git commit -m "feat(gateway): aggregate task result into transcript/tags/like"
```

---

### Task 2: `PUT tags` read-modify-write (preserve transcript/like, inherit evidence)

**Files:**
- Modify: `el-ai-gateway/src/routes/tasks.ts`
- Test: `el-ai-gateway/test/tasksRoute.test.ts`

- [ ] **Step 1: Write the failing tests**

In `el-ai-gateway/test/tasksRoute.test.ts`, add these two tests inside the existing `describe("PUT /api/v1/voice-tagging/:id/tags", ...)` block (e.g. right before its final `});`):

```ts
  it("preserves transcript/like and inherits evidence by tagId", async () => {
    let current: any = {
      id: TASK,
      status: "in_progress",
      createdAt: "2026-09-15T05:18:00Z",
      updatedAt: "2026-09-15T05:20:00Z",
      metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
      result: JSON.stringify({
        transcript: "原文A",
        tags: [{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致", evidence: "证据A" }],
        like: true,
      }),
      activities: [],
      raw: {},
    };
    const updateResult = vi.fn(async ({ result }: { result: string }) => {
      current = { ...current, result };
      return { raw: {} };
    });
    const d = deps({
      boTools: boStub({ queryRecords: vi.fn(async () => [DEF_ROW]) }),
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => current),
        updateResult,
        listTasks: vi.fn(async () => [current]),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagId: DEF_ID }] },
    });
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(updateResult.mock.calls[0][0].result)).toEqual({
      transcript: "原文A",
      like: true,
      tags: [{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致", evidence: "证据A" }],
    });
  });

  it("writes transcript/like as null for a legacy array result", async () => {
    const d = statefulDeps({ queryRecords: vi.fn(async () => [DEF_ROW]) });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagId: DEF_ID }] },
    });
    expect(res.statusCode).toBe(200);
    const written = JSON.parse(d.taskTools.updateResult.mock.calls[0][0].result);
    expect(written).toMatchObject({ transcript: null, like: null });
    expect(written.tags).toEqual([{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }]);
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: FAIL — the handler currently writes `result` as a bare tags array.

- [ ] **Step 3: Rewrite the `PUT /tags` handler**

In `el-ai-gateway/src/routes/tasks.ts`, replace the whole `app.put("/api/v1/voice-tagging/:id/tags", ...)` handler with:

```ts
  // Replace a task's tags (stored in the task `result` object), preserving
  // transcript/like; then reconcile the customer's aggregate tags.
  app.put("/api/v1/voice-tagging/:id/tags", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const prev = parseResult(task.result);

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

    const evidenceByTag = new Map(
      prev.tags.filter((t) => t.evidence).map((t) => [t.tagId, t.evidence as string]),
    );
    const out: ResultTag[] = [];
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
        const rows = await deps.boTools.queryRecords("tag_definition", [
          { field: "tag_id", op: "eq", value: input.tagId },
        ]);
        if (rows.length === 0) {
          throw new GatewayError(400, "BAD_REQUEST", `Unknown tagId: ${input.tagId}`);
        }
        const def = rows[0];
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
      const tag: ResultTag = { tagId, tagKey, tagValue };
      const evidence = evidenceByTag.get(tagId);
      if (evidence) tag.evidence = evidence;
      out.push(tag);
    }

    const next: TaskResult = { transcript: prev.transcript, tags: out, like: prev.like };
    await deps.taskTools.updateResult({ id, result: JSON.stringify(next) });
    await reconcileCustomerTags(deps, { ownerId: principal.tenantId, baId, customerId });

    const updated = await deps.taskTools.getTask({ id });
    const activities = mapActivities(updated.activities);
    return { ...toDetail(updated), activity: activities[0] };
  });
```

- [ ] **Step 4: Update existing `PUT tags` assertions**

The existing tests in that describe assert `updateResult` was called with a bare array. Update them to the object shape. Specifically:

In the test `"resolves an existing tagId via the tag_id field and writes {tagId,tagKey,tagValue}"`, change the assertion to:

```ts
    expect(d.taskTools.updateResult).toHaveBeenCalledWith({
      id: TASK,
      result: JSON.stringify({
        transcript: null,
        like: null,
        tags: [{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }],
      }),
    });
```

(All other existing tests in the block already pass because they don't assert the exact `result` string.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add el-ai-gateway/src/routes/tasks.ts el-ai-gateway/test/tasksRoute.test.ts
git commit -m "feat(gateway): PUT tags preserves transcript/like and inherits evidence"
```

---

### Task 3: `PUT /voice-tagging/:id/like`

**Files:**
- Modify: `el-ai-gateway/src/routes/tasks.ts`
- Test: `el-ai-gateway/test/tasksRoute.test.ts`

- [ ] **Step 1: Write the failing tests**

Append this `describe` block to `el-ai-gateway/test/tasksRoute.test.ts`:

```ts
describe("PUT /api/v1/voice-tagging/:id/like", () => {
  const TASK = "b3ca978f-3832-4a2c-959b-40fa48c43352";

  function likeDeps(result: string) {
    let current: any = {
      id: TASK,
      status: "completed",
      metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
      result,
      activities: [{ id: "act-1", action: "updated", detail: { markdown: "" }, createdAt: "t" }],
      raw: {},
    };
    return deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => current),
        listTasks: vi.fn(async () => []),
        updateResult: vi.fn(async ({ result: r }: { result: string }) => {
          current = { ...current, result: r };
          return { raw: {} };
        }),
      },
    });
  }

  it("sets like true and preserves tags/transcript", async () => {
    const d = likeDeps(
      JSON.stringify({ transcript: "原文", tags: [{ tagId: "x", tagKey: "g", tagValue: "v" }], like: null }),
    );
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/like`,
      headers: { authorization: "Bearer secret" },
      payload: { like: true },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().like).toBe(true);
    expect(JSON.parse(d.taskTools.updateResult.mock.calls[0][0].result)).toEqual({
      transcript: "原文",
      tags: [{ tagId: "x", tagKey: "g", tagValue: "v" }],
      like: true,
    });
  });

  it("clears like with null", async () => {
    const d = likeDeps(JSON.stringify({ transcript: null, tags: [], like: true }));
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/like`,
      headers: { authorization: "Bearer secret" },
      payload: { like: null },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().like).toBeNull();
  });

  it("rejects false / missing / non-boolean like", async () => {
    const app = buildServer(likeDeps("{}"));
    for (const payload of [{ like: false }, {}, { like: "x" }]) {
      const res = await app.inject({
        method: "PUT",
        url: `/api/v1/voice-tagging/${TASK}/like`,
        headers: { authorization: "Bearer secret" },
        payload,
      });
      expect(res.statusCode).toBe(400);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd el-ai-gateway && pnpm exec vitest run test/tasksRoute.test.ts`
Expected: FAIL — the route does not exist (404).

- [ ] **Step 3: Implement the like route**

In `el-ai-gateway/src/routes/tasks.ts`, add this handler inside `registerTaskRoutes`, immediately after the `PUT /tags` handler:

```ts
  // Set the user's like feedback on the task result (true | null).
  app.put("/api/v1/voice-tagging/:id/like", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const body = (request.body ?? {}) as { like?: unknown };
    if (!("like" in body) || (body.like !== true && body.like !== null)) {
      throw new GatewayError(400, "BAD_REQUEST", "like must be true or null");
    }
    const prev = parseResult(task.result);
    const next: TaskResult = {
      transcript: prev.transcript,
      tags: prev.tags,
      like: body.like === true ? true : null,
    };
    await deps.taskTools.updateResult({ id, result: JSON.stringify(next) });
    const updated = await deps.taskTools.getTask({ id });
    const activities = mapActivities(updated.activities);
    return { ...toDetail(updated), activity: activities[0] };
  });
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd el-ai-gateway && pnpm test && pnpm typecheck`
Expected: all suites PASS; typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/routes/tasks.ts el-ai-gateway/test/tasksRoute.test.ts
git commit -m "feat(gateway): add PUT like endpoint (true | null)"
```

---

### Task 4: Update `API.md`

**Files:**
- Modify: `el-ai-gateway/API.md`

- [ ] **Step 1: Update §8.2 (list) and §8.3 (detail) examples**

In the §8.2 `tasks[]` example and the §8.3 response example, add `like` (and, for §8.3 only, `transcript`), and change the tag elements to include `tagKey`/`tagValue` (already done) plus optional `evidence`. Use:

```json
      "tags": [
        {
          "tagId": "9ce355bfacca49c4a9e9322a9317c196",
          "tagKey": "concerns",
          "tagValue": "抗老/紧致",
          "evidence": "很喜欢用黑钻光灿面霜"
        }
      ],
      "like": true
```

For §8.3 (detail) only, add before `"tags"`:

```json
  "transcript": "……完整语音原文……",
```

And add the field explanations under §8.3:

```
- `transcript`：语音原文（string，无则 `null`）。
- `tags`：`{tagId, tagKey, tagValue, evidence?}` 列表。
- `like`：用户点赞，`true` 或 `null`（从未点赞/取消均为 `null`）。
```

Under §8.2 add:

```
- `tasks[].tags` / `tasks[].like` 同上；**列表不含 `transcript`**（原文请用任务详情接口获取）。
```

- [ ] **Step 2: Update §8.5 (`PUT tags`)**

Change the response note to clarify RMW, adding:

```
- 只替换 `tags`；`transcript` 与 `like` 保持不变；每个标签按 `tagId` 保留原有的 `evidence`。
```

- [ ] **Step 3: Add §8.7 `PUT like`**

After §8.6, add:

```markdown
### 8.7 任务点赞

对任务结果点赞 / 取消点赞（用户反馈）。

```
PUT /voice-tagging/:taskId/like
Content-Type: application/json
```

```json
{ "like": true }
```

| 值 | 含义 |
|---|---|
| `true` | 点赞 |
| `null` | 取消点赞 / 未点赞 |

**响应 `200`**：更新后的任务详情（同 [§8.3](#83-查询任务状态)，含 `transcript`/`tags`/`like`）+ `activity`。

- `like` 缺失或非 `true`/`null`（如 `false`）→ `400 BAD_REQUEST`
- 任务不存在 → `404 NOT_FOUND`
- 说明：只更新 `like`，`tags`/`transcript` 保持不变。
```

- [ ] **Step 4: Verify docs**

Run: `cd el-ai-gateway && grep -n "transcript\|like" API.md | head -30`
Expected: `transcript`/`like` documented in §8.2/§8.3/§8.5/§8.7 with no contradictions.

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/API.md
git commit -m "docs(gateway): document result transcript/tags/like and PUT like"
```

---

### Task 5: Prerequisites (outside this repo's code)

**Files:** none (platform-side config + data)

- [ ] **Step 1: Confirm the tagging agent writes `result`**

Update the `voice-tagging-agent` prompt (platform side) so that on completion it writes the task `result` as:

```json
{
  "transcript": "<完整转写原文>",
  "tags": [ { "tagId": "…", "tagKey": "…", "tagValue": "…", "evidence": "…" } ],
  "like": null
}
```

- [ ] **Step 2: Import the BA tag master into `tag_definition`**

Load the BA tag master (类别 / 标签组 / 标签 / 标签id / 小肤标签) into the `tag_definition` BO object so the agent's `tagId`s resolve on subsequent `PUT tags`. Use the platform-service BO API (`POST /api/v1/bo/objects/tag_definition/records` or `create_records` batch) with records like:

```json
{ "category": "进阶标签", "tag_group": "产品系列", "tag_name": "彩妆系列", "tag_id": "15551121d88343678efde53a9138395c", "is_xiaofu": false }
```

- [ ] **Step 3: Verify end-to-end**

Create a task, let the agent write `result`, then `GET /voice-tagging/:id` → `transcript`/`tags`/`like` populated; `PUT like {"like":true}` → `like:true`; `PUT tags` with an agent `tagId` → resolves (not `400`).

---

## Notes for the implementer

- Commands: `cd el-ai-gateway && pnpm test` and `pnpm typecheck`.
- Do NOT change `reconcileCustomerTags` behavior beyond switching to `parseResult(...).tags`.
- `PUT tags` must still reconcile `customer_tag` and still 400 on unknown `tagId` / non-array `tags`.
- `like` accepts only `true` or `null` (a present `null` is valid; a missing key is `400`).
