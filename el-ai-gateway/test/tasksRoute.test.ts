import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";
import { GatewayError } from "../src/lib/errors";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesUrl: "http://files",
  maxUploadBytes: 1024 * 1024,
  agentRunsUrl: "http://agent/api/runs",
  agentAuthUrl: "http://agent/api/auth/login",
  agentLoginEmail: "svc@example.com",
  agentLoginPassword: "secret",
  agentTenantId: "estee_lauder",
  agentWorkspaceId: "default-workspace",
  agentProjectId: "default",
  voiceTaggingAssistantId: "voice-agent",
  voiceTaggingFileUuid: "2ccf6fef88b64a16b62fe491a8f7a132",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

const auth = (h?: string) =>
  h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null;

function deps(overrides: Record<string, unknown> = {}) {
  return {
    config,
    authenticator: auth,
    platformFiles: {
      upload: vi.fn(),
      presign: vi.fn(async () => ({ url: "https://signed", kind: "presigned", expiresInSeconds: 600 })),
    },
    agentRuns: { startRun: vi.fn(async () => ({ messageId: "msg-1", queued: true })) },
    taskTools: {
      createTask: vi.fn(async () => ({ taskId: "task-1", raw: {} })),
      getTask: vi.fn(async () => ({
        id: "task-1",
        status: "in_progress",
        title: "Voice tagging: 471c20082b524316accc1b23cba8a4de",
        createdAt: "2026-09-15T06:13:00Z",
        metadata: { uuid: "471c20082b524316accc1b23cba8a4de", baId: "ba_001", customerId: "cus_8899" },
        result: JSON.stringify([
          { tagId: "9ce355bfacca49c4a9e9322a9317c196", name: "抗老/紧致", dimension: "concerns" },
        ]),
        activities: [
          { id: "act-1", action: "activity", detail: { markdown: "## t" }, createdAt: "2026-09-15T06:15:00Z" },
        ],
        raw: {},
      })),
      updateResult: vi.fn(async () => ({ raw: {} })),
      listTasks: vi.fn(async () => []),
    },
    boTools: {
      getRecord: vi.fn(async () => undefined),
      queryRecords: vi.fn(async () => []),
      createRecord: vi.fn(async (_objectKey: string, _data: any) => ({})),
      deleteRecords: vi.fn(async (_objectKey: string, _ids: string[]) => 0),
    } as any,
    ...overrides,
  } as any;
}

describe("POST /api/v1/voice-tagging", () => {
  it("presigns, creates the task, triggers A2A and returns the task id", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1", title: "My task", baId: "ba_001", customerId: "cus_8899", durationSec: 12.5 },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.taskId).toBe("task-1");
    expect(body.file).toEqual({ uuid: "u1", url: "https://signed" });
    expect(d.platformFiles.presign).toHaveBeenCalledWith({ tenantId: "tenant_a", uuid: "u1" });
    expect(d.taskTools.createTask).toHaveBeenCalledWith({
      title: "My task",
      description: undefined,
      status: "in_progress",
      ownerId: "tenant_a",
      metadata: { uuid: "u1", url: "https://signed", baId: "ba_001", customerId: "cus_8899", durationSec: 12.5 },
    });
    const runArg = d.agentRuns.startRun.mock.calls[0][0];
    expect(runArg.assistantId).toBe("voice-agent");
    expect(runArg.text).toContain("task-1");
  });

  it("falls back to the configured file uuid and sends only the task id", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { baId: "ba_001", customerId: "cus_8899", durationSec: 3 },
    });
    expect(res.statusCode).toBe(200);
    expect(d.platformFiles.presign).toHaveBeenCalledWith({
      tenantId: "tenant_a",
      uuid: "2ccf6fef88b64a16b62fe491a8f7a132",
    });
    const runArg = d.agentRuns.startRun.mock.calls[0][0];
    expect(runArg.text).toContain("task-1");
    expect(runArg.text).not.toContain("https://signed");
    expect(runArg.text).not.toContain("2ccf6fef88b64a16b62fe491a8f7a132");
  });

  it("returns 400 when uuid is missing and no default is configured", async () => {
    const app = buildServer(deps({ config: { ...config, voiceTaggingFileUuid: undefined } }));
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { baId: "ba_001", customerId: "cus_8899", durationSec: 3 },
    });
    expect(res.statusCode).toBe(400);
  });

  it("returns 400 when durationSec is missing", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("does not fail the request when the A2A trigger fails (fire-and-forget)", async () => {
    const d = deps();
    d.agentRuns.startRun = vi.fn(async () => {
      throw new Error("a2a down");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1", baId: "ba_001", customerId: "cus_8899", durationSec: 5 },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().taskId).toBe("task-1");
    expect(res.json().agent).toEqual({ dispatched: true });
  });
});

describe("GET /api/v1/voice-tagging/:id", () => {
  it("returns task detail (fileId / status / tags) from the task service", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/task-1",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.taskId).toBe("task-1");
    expect(body.fileId).toBe("471c20082b524316accc1b23cba8a4de");
    expect(body.status).toBe("in_progress");
    expect(body.tags).toHaveLength(1);
    expect(body.tags[0]).toMatchObject({
      tagId: "9ce355bfacca49c4a9e9322a9317c196",
      tagKey: "concerns",
      tagValue: "抗老/紧致",
    });
    expect(body.activities).toBeUndefined();
  });

  it("returns 404 for an unknown task", async () => {
    const d = deps();
    d.taskTools.getTask = vi.fn(async () => {
      throw new GatewayError(404, "NOT_FOUND", "Task 'nope' not found");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/nope",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(404);
    expect(res.json().code).toBe("NOT_FOUND");
  });
});

describe("GET /api/v1/voice-tagging?baId=&customerId=", () => {
  function listDeps() {
    return deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(),
        updateResult: vi.fn(),
        listTasks: vi.fn(async () => [
          {
            id: "task-1",
            status: "completed",
            title: "Voice tagging: 471c20082b524316accc1b23cba8a4de",
            createdAt: "2026-09-15T06:13:00Z",
            metadata: { uuid: "471c20082b524316accc1b23cba8a4de", baId: "ba_001", customerId: "cus_8899" },
            result: JSON.stringify([
              { tagId: "9ce355bfacca49c4a9e9322a9317c196", name: "抗老/紧致", dimension: "concerns" },
            ]),
            activities: [],
            raw: {},
          },
        ]),
      },
    });
  }

  it("returns tasks with fileId / taskId / status / tags", async () => {
    const d = listDeps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging?baId=ba_001&customerId=cus_8899",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.baId).toBe("ba_001");
    expect(body.customerId).toBe("cus_8899");
    expect(body.total).toBe(1);
    expect(d.taskTools.listTasks).toHaveBeenCalledWith({
      ownerId: "tenant_a",
      baId: "ba_001",
      customerId: "cus_8899",
    });
    const task = body.tasks[0];
    expect(typeof task.taskId).toBe("string");
    expect(typeof task.fileId).toBe("string");
    expect(typeof task.status).toBe("string");
    expect(Array.isArray(task.tags)).toBe(true);
    for (const tag of task.tags) expect(tag.tagId).toMatch(/^[0-9a-f]{32}$/);
  });

  it("returns an empty list for an unknown ba/customer combo", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging?baId=ba_x&customerId=cus_y",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ total: 0, tasks: [] });
  });

  it("requires baId and customerId", async () => {
    const app = buildServer(deps());
    const missingBa = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging?customerId=cus_8899",
      headers: { authorization: "Bearer secret" },
    });
    expect(missingBa.statusCode).toBe(400);
    const missingCus = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging?baId=ba_001",
      headers: { authorization: "Bearer secret" },
    });
    expect(missingCus.statusCode).toBe(400);
  });
});

describe("PUT /api/v1/voice-tagging/:id/tags", () => {
  const TASK = "b3ca978f-3832-4a2c-959b-40fa48c43352";
  const DEF_ID = "9ce355bfacca49c4a9e9322a9317c196";

  function boStub(overrides: Record<string, unknown> = {}) {
    return {
      getRecord: vi.fn(async () => undefined),
      queryRecords: vi.fn(async (objectKey: string) =>
        objectKey === "tag_definition" ? [] : [],
      ),
      createRecord: vi.fn(async (_objectKey: string, _data: any) => ({})),
      updateRecord: vi.fn(async (_objectKey: string, _id: string, _data: any) => ({})),
      deleteRecords: vi.fn(async (_objectKey: string, _ids: string[]) => 0),
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

  const DEF_ROW = { id: "rec-def", tag_id: DEF_ID, tag_group: "concerns", tag_name: "抗老/紧致" };

  it("resolves an existing tagId via the tag_id field and writes {tagId,tagKey,tagValue}", async () => {
    const d = statefulDeps({
      queryRecords: vi.fn(async (objectKey: string) => (objectKey === "tag_definition" ? [DEF_ROW] : [])),
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: [{ tagId: DEF_ID, tagValue: "ignored" }] },
    });
    expect(res.statusCode).toBe(200);
    expect(d.boTools.queryRecords).toHaveBeenCalledWith("tag_definition", [
      { field: "tag_id", op: "eq", value: DEF_ID },
    ]);
    expect(d.taskTools.updateResult).toHaveBeenCalledWith({
      id: TASK,
      result: JSON.stringify({
        transcript: null,
        tags: [{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }],
        like: null,
      }),
    });
    expect(res.json().tags).toEqual([{ tagId: DEF_ID, tagKey: "concerns", tagValue: "抗老/紧致" }]);
  });

  it("creates a new tag_definition (group 客户画像) when no tagId is given", async () => {
    const createRecord = vi.fn(async (_objectKey: string, _data: any) => ({}));
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
    const createRecord = vi.fn(async (_objectKey: string, _data: any) => ({}));
    const d = statefulDeps({
      queryRecords: vi.fn(async (objectKey: string) =>
        objectKey === "tag_definition"
          ? [{ id: "rec1", tag_id: "abc", tag_group: "客户画像", tag_name: "新标签" }]
          : [],
      ),
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

  it("reconciles customer_tag to the union: creates new, deletes stale, refreshes denormalized", async () => {
    const staleId = "cccccccccccccccccccccccccccccccc";
    const createRecord = vi.fn(async (_objectKey: string, _data: any) => ({}));
    const updateRecord = vi.fn(async (_objectKey: string, _id: string, _data: any) => ({}));
    const deleteRecords = vi.fn(async (_objectKey: string, _ids: string[]) => 1);
    const d = statefulDeps({
      queryRecords: vi.fn(async (objectKey: string) => {
        if (objectKey === "tag_definition") {
          return [{ id: "rec-def", tag_id: DEF_ID, tag_group: "concerns", tag_name: "抗老/紧致" }];
        }
        return [
          { id: "row-stale", tag_id: staleId, source: "voice", customer_no: "cus_8899" },
          { id: "row-old", tag_id: DEF_ID, tag_key: "old", tag_value: "old", source: "voice", customer_no: "cus_8899" },
        ];
      }),
      createRecord,
      updateRecord,
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
    expect(updateRecord).toHaveBeenCalledWith("customer_tag", "row-old", {
      tag_key: "concerns",
      tag_value: "抗老/紧致",
    });
    expect(deleteRecords).toHaveBeenCalledWith("customer_tag", ["row-stale"]);
    expect(createRecord.mock.calls.some((c) => c[0] === "tag_definition")).toBe(false);
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


  it("ignores transcript/like fields in the PUT request body", async () => {
    let current: any = {
      id: TASK,
      status: "completed",
      metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
      result: JSON.stringify({ transcript: "原文", tags: [], like: true }),
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
      payload: { tags: [{ tagId: DEF_ID }], transcript: "伪造原文", like: null },
    });
    expect(res.statusCode).toBe(200);
    const written = JSON.parse(updateResult.mock.calls[0][0].result);
    expect(written.transcript).toBe("原文");
    expect(written.like).toBe(true);
  });

});

describe("DELETE /api/v1/voice-tagging/:id", () => {
  const TASK = "b3ca978f-3832-4a2c-959b-40fa48c43352";

  function deleteDeps() {
    const deleteTask = vi.fn(async (_input: { id: string }) => ({ raw: {} }));
    return deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => ({
          id: TASK,
          status: "completed",
          metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
          result: "[]",
          activities: [],
          raw: {},
        })),
        listTasks: vi.fn(async () => []),
        deleteTask,
      },
    });
  }

  it("deletes the task and reconciles the customer's tags", async () => {
    const d = deleteDeps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "DELETE",
      url: `/api/v1/voice-tagging/${TASK}`,
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ taskId: TASK, deleted: true });
    expect(d.taskTools.deleteTask).toHaveBeenCalledWith({ id: TASK });
    expect(d.boTools.queryRecords).toHaveBeenCalledWith("customer_tag", [
      { field: "customer_no", op: "eq", value: "cus_8899" },
    ]);
  });

  it("returns 404 for an unknown task", async () => {
    const d = deps();
    d.taskTools.getTask = vi.fn(async () => {
      throw new GatewayError(404, "NOT_FOUND", "Task 'nope' not found");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "DELETE",
      url: "/api/v1/voice-tagging/nope",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(404);
  });
});

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

function multipartBody(filename: string, content: string) {
  const boundary = "----elgtest";
  const payload = Buffer.concat([
    Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
        "Content-Type: audio/wav\r\n\r\n",
    ),
    Buffer.from(content),
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);
  return { payload, contentType: `multipart/form-data; boundary=${boundary}` };
}

describe("POST /api/v1/voice-tagging/upload", () => {
  it("uploads, presigns, creates the task and dispatches", async () => {
    const upload = vi.fn(async (_i: any) => ({ uuid: "u" }));
    const d = deps({
      platformFiles: {
        upload,
        presign: vi.fn(async () => ({ url: "https://signed", kind: "presigned", expiresInSeconds: 600 })),
        list: vi.fn(),
      },
    });
    const app = buildServer(d);
    const { payload, contentType } = multipartBody("clip.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging/upload?baId=ba_001&customerId=cus_8899&durationSec=9&title=My",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().file.url).toBe("https://signed");
    expect(upload).toHaveBeenCalled();
    expect(d.taskTools.createTask.mock.calls[0][0].metadata).toMatchObject({
      baId: "ba_001",
      customerId: "cus_8899",
      durationSec: 9,
      url: "https://signed",
    });
    expect(d.agentRuns.startRun).toHaveBeenCalled();
  });

  it("requires baId / customerId / durationSec", async () => {
    const app = buildServer(deps());
    const { payload, contentType } = multipartBody("clip.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging/upload?customerId=cus_8899&durationSec=9",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(400);
  });
});

describe("GET /api/v1/voice-tagging/:id/audio", () => {
  const TASK = "t-audio";
  function taskWith(meta: Record<string, unknown>) {
    return { id: TASK, status: "completed", metadata: meta, result: "{}", activities: [], raw: {} };
  }

  it("proxies the presigned audio and passes Range/206", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => taskWith({ uuid: "u1", baId: "ba_001", customerId: "cus_1" })),
        listTasks: vi.fn(async () => []),
        updateResult: vi.fn(),
      },
    });
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 206,
      headers: new Headers({
        "content-type": "audio/wav",
        "content-range": "bytes 0-2/3",
        "content-length": "3",
        "accept-ranges": "bytes",
      }),
      body: (async function* () {
        yield Buffer.from("abc");
      })(),
    } as any);
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: `/api/v1/voice-tagging/${TASK}/audio`,
      headers: { authorization: "Bearer secret", range: "bytes=0-2" },
    });
    expect(res.statusCode).toBe(206);
    expect(res.headers["content-range"]).toBe("bytes 0-2/3");
    expect(res.headers["accept-ranges"]).toBe("bytes");
    expect(res.body).toBe("abc");
    const [calledUrl, init] = fetchSpy.mock.calls[0];
    expect(String(calledUrl)).toBe("https://signed");
    expect((init as any).headers.Range).toBe("bytes=0-2");
    fetchSpy.mockRestore();
  });

  it("answers HEAD with size + accept-ranges via a GET probe", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => taskWith({ uuid: "u1", baId: "ba_001", customerId: "cus_1" })),
        listTasks: vi.fn(async () => []),
        updateResult: vi.fn(),
      },
    });
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 206,
      headers: new Headers({
        "content-type": "audio/wav",
        "content-range": "bytes 0-0/8236",
        "accept-ranges": "bytes",
      }),
      body: (async function* () {
        yield Buffer.from("R");
      })(),
    } as any);
    const app = buildServer(d);
    const res = await app.inject({
      method: "HEAD",
      url: `/api/v1/voice-tagging/${TASK}/audio`,
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.headers["content-length"]).toBe("8236");
    expect(res.headers["accept-ranges"]).toBe("bytes");
    expect(res.headers["content-type"]).toBe("audio/wav");
    expect((fetchSpy.mock.calls[0][1] as any).headers.Range).toBe("bytes=0-0");
    fetchSpy.mockRestore();
  });

  it("returns 404 when the task has no file", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => taskWith({})),
        listTasks: vi.fn(async () => []),
        updateResult: vi.fn(),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: `/api/v1/voice-tagging/${TASK}/audio`,
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(404);
  });
});

describe("task duration/audio fields", () => {
  it("detail includes durationSec and audioUrl", async () => {
    const d = deps({
      taskTools: {
        createTask: vi.fn(),
        getTask: vi.fn(async () => ({
          id: "t1",
          status: "completed",
          metadata: { uuid: "u1", baId: "ba_001", customerId: "c", durationSec: 8 },
          result: "{}",
          activities: [],
          raw: {},
        })),
        listTasks: vi.fn(async () => []),
        updateResult: vi.fn(),
      },
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/t1",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.json().durationSec).toBe(8);
    expect(res.json().audioUrl).toBe("/voice-tagging/t1/audio");
  });
});
