import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesUrl: "http://files",
  maxUploadBytes: 1024 * 1024,
  a2aBaseUrl: "http://a2a",
  a2aApiKey: "a2a_x",
  a2aVoiceTaggingAssistantId: "voice-agent",
  a2aVoiceTaggingFileUuid: "2ccf6fef88b64a16b62fe491a8f7a132",
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
    a2a: { sendTask: vi.fn(async () => ({ taskId: "a2a-1", state: "submitted", raw: {} })) },
    taskTools: {
      createTask: vi.fn(async () => ({ taskId: "task-1", raw: {} })),
      getTask: vi.fn(async () => ({
        id: "task-1",
        status: "completed",
        title: "T",
        activities: [{ id: "act-1" }],
        raw: {},
      })),
    },
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
      payload: { uuid: "u1", title: "My task" },
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
      metadata: { uuid: "u1", url: "https://signed" },
    });
    const a2aArg = d.a2a.sendTask.mock.calls[0][0];
    expect(a2aArg.assistantId).toBe("voice-agent");
    expect(a2aArg.text).toContain("task-1");
  });

  it("falls back to the configured file uuid and sends only the task id", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: {},
    });
    expect(res.statusCode).toBe(200);
    expect(d.platformFiles.presign).toHaveBeenCalledWith({
      tenantId: "tenant_a",
      uuid: "2ccf6fef88b64a16b62fe491a8f7a132",
    });
    const a2aArg = d.a2a.sendTask.mock.calls[0][0];
    expect(a2aArg.text).toContain("task-1");
    expect(a2aArg.text).not.toContain("https://signed");
    expect(a2aArg.text).not.toContain("2ccf6fef88b64a16b62fe491a8f7a132");
  });

  it("returns 400 when uuid is missing and no default is configured", async () => {
    const app = buildServer(deps({ config: { ...config, a2aVoiceTaggingFileUuid: undefined } }));
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: {},
    });
    expect(res.statusCode).toBe(400);
  });

  it("keeps the task id when A2A fails", async () => {
    const d = deps();
    d.a2a.sendTask = vi.fn(async () => {
      throw new Error("a2a down");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1" },
    });
    expect(res.statusCode).toBe(502);
    expect(res.json().upstream).toEqual({ taskId: "task-1" });
  });
});

describe("GET /api/v1/voice-tagging/:id", () => {
  it("returns task status and activities", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/voice-tagging/task-1",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({
      taskId: "task-1",
      status: "completed",
      activities: [{ id: "act-1" }],
    });
  });
});

describe("POST /api/v1/voice-tagging/:id/feedback", () => {
  it("relays feedback to the agent over A2A", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging/task-1/feedback",
      headers: { authorization: "Bearer secret" },
      payload: { content: "tag corrected" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ taskId: "task-1", forwarded: true });
    const a2aArg = d.a2a.sendTask.mock.calls[0][0];
    expect(a2aArg.assistantId).toBe("voice-agent");
    expect(a2aArg.text).toContain("task-1");
    expect(a2aArg.text).toContain("tag corrected");
  });

  it("returns 400 when content is empty", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/voice-tagging/task-1/feedback",
      headers: { authorization: "Bearer secret" },
      payload: { content: "   " },
    });
    expect(res.statusCode).toBe(400);
  });
});
