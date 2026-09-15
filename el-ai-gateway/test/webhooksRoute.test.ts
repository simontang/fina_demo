import { afterEach, describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

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
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

function build() {
  return buildServer({
    config,
    authenticator: () => null,
    platformFiles: { upload: vi.fn(), presign: vi.fn() } as any,
    agentRuns: { startRun: vi.fn() } as any,
    taskTools: { createTask: vi.fn(), getTask: vi.fn() } as any,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("POST /api/v1/webhooks/:topic", () => {
  it("prints the payload and returns 200 without gateway auth", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const app = build();
    const payload = { eventType: "job.completed", data: { taskId: "task-1" } };

    const res = await app.inject({
      method: "POST",
      url: "/api/v1/webhooks/voice-tagging",
      headers: { "svix-id": "msg_1" },
      payload,
    });

    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ ok: true, topic: "voice-tagging" });
    expect(log).toHaveBeenCalledTimes(1);
    const printed = log.mock.calls[0].join(" ");
    expect(printed).toContain("voice-tagging");
    expect(printed).toContain("job.completed");
    expect(printed).toContain("task-1");
  });

  it("accepts an empty body", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    const app = build();
    const res = await app.inject({ method: "POST", url: "/api/v1/webhooks/ping" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ ok: true, topic: "ping" });
  });
});
