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
        metadata: { uuid: "471c20082b524316accc1b23cba8a4de" },
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
      createRecord: vi.fn(async () => ({})),
      deleteRecords: vi.fn(async () => 0),
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
      payload: { uuid: "u1", title: "My task", baId: "ba_001", customerId: "cus_8899" },
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
      metadata: { uuid: "u1", url: "https://signed", baId: "ba_001", customerId: "cus_8899" },
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
      payload: { baId: "ba_001", customerId: "cus_8899" },
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
      payload: { baId: "ba_001", customerId: "cus_8899" },
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
      payload: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
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
    expect(body.tags[0]).toMatchObject({ tagId: "9ce355bfacca49c4a9e9322a9317c196", name: "抗老/紧致" });
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

  function statefulDeps() {
    let current: any = {
      id: TASK,
      status: "in_progress",
      title: "Voice tagging: 2ccf6fef88b64a16b62fe491a8f7a132",
      createdAt: "2026-09-15T05:18:00Z",
      metadata: { uuid: "2ccf6fef88b64a16b62fe491a8f7a132" },
      result: JSON.stringify([
        { tagId: "2f0a7d1c6b4e48a2b3c5d6e7f8091a2b", name: "保湿", dimension: "concerns" },
      ]),
      activities: [
        { id: "act-1", action: "activity", detail: { markdown: "## t" }, createdAt: "2026-09-15T05:20:00Z" },
      ],
      raw: {},
    };
    return deps({
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
      },
    });
  }

  it("replaces tags (stored in result) and records an activity", async () => {
    const d = statefulDeps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: {
        tags: ["9ce355bfacca49c4a9e9322a9317c196", "4a1b2c3d5e6f47089a0b1c2d3e4f5061"],
      },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.tags).toHaveLength(2);
    expect(body.tags[0]).toMatchObject({ tagId: "9ce355bfacca49c4a9e9322a9317c196", name: "抗老/紧致" });
    expect(body.activity.action).toBe("updated");
    expect(d.taskTools.updateResult).toHaveBeenCalledWith({
      id: TASK,
      result: JSON.stringify([
        { tagId: "9ce355bfacca49c4a9e9322a9317c196", name: "抗老/紧致", dimension: "concerns" },
        { tagId: "4a1b2c3d5e6f47089a0b1c2d3e4f5061", name: "黑钻光灿面霜", dimension: "interested_products" },
      ]),
    });

    const acts = await app.inject({
      method: "GET",
      url: `/api/v1/voice-tagging/${TASK}/activities`,
      headers: { authorization: "Bearer secret" },
    });
    expect(acts.statusCode).toBe(200);
    expect(acts.json().total).toBe(2);
    expect(acts.json().activities[0].action).toBe("updated");
  });

  it("rejects an unknown tagId", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: ["00000000000000000000000000000000"] },
    });
    expect(res.statusCode).toBe(400);
  });

  it("rejects a non-array tags body", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "PUT",
      url: `/api/v1/voice-tagging/${TASK}/tags`,
      headers: { authorization: "Bearer secret" },
      payload: { tags: "nope" },
    });
    expect(res.statusCode).toBe(400);
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
