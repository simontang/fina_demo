import { describe, expect, it, vi } from "vitest";
import { createAgentRunsClient } from "../src/upstream/agentRuns";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map(),
  authDisabled: false,
  authDevTenant: "estee_lauder",
  platformFilesUrl: "http://files",
  maxUploadBytes: 1000,
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

function makeToken(exp: number): string {
  const payload = Buffer.from(JSON.stringify({ userId: "u", tenantId: "estee_lauder", exp })).toString(
    "base64url",
  );
  return `${payload}.sig`;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("agentRuns.startRun", () => {
  it("logs in, then POSTs a background run with tenant headers and the task id", async () => {
    const token = makeToken(Date.now() + 3600_000);
    const fetchImpl = vi.fn(async (url: string | URL) =>
      String(url).includes("/auth/login")
        ? json({ success: true, data: { token } })
        : json({ success: true, messageId: "m1", queued: true }, 202),
    ) as unknown as typeof fetch;
    const client = createAgentRunsClient(config, fetchImpl);

    const out = await client.startRun({ assistantId: "voice", threadId: "t1", text: "hi", taskId: "task-1" });

    expect(out).toEqual({ messageId: "m1", queued: true });
    const runCall = (fetchImpl as any).mock.calls.find((c: any[]) => String(c[0]).includes("/api/runs"));
    expect(runCall[1].headers.Authorization).toBe(`Bearer ${token}`);
    expect(runCall[1].headers["x-tenant-id"]).toBe("estee_lauder");
    expect(runCall[1].headers["x-workspace-id"]).toBe("default-workspace");
    expect(runCall[1].headers["x-project-id"]).toBe("default");
    expect(runCall[1].headers["x-user-id"]).toBe("estee_lauder");
    const body = JSON.parse(runCall[1].body);
    expect(body.assistant_id).toBe("voice");
    expect(body.thread_id).toBe("t1");
    expect(body.background).toBe(true);
    expect(body.custom_run_config).toEqual({ taskId: "task-1" });
  });

  it("reuses the cached token across runs", async () => {
    const token = makeToken(Date.now() + 3600_000);
    const fetchImpl = vi.fn(async (url: string | URL) =>
      String(url).includes("/auth/login")
        ? json({ success: true, data: { token } })
        : json({ success: true, messageId: "m1", queued: true }, 202),
    ) as unknown as typeof fetch;
    const client = createAgentRunsClient(config, fetchImpl);

    await client.startRun({ assistantId: "voice", threadId: "t1", text: "a", taskId: "x" });
    await client.startRun({ assistantId: "voice", threadId: "t2", text: "b", taskId: "y" });

    const logins = (fetchImpl as any).mock.calls.filter((c: any[]) => String(c[0]).includes("/auth/login"));
    expect(logins).toHaveLength(1);
  });

  it("posts without Authorization when no login credentials are configured", async () => {
    const noCreds: Config = { ...config, agentLoginEmail: undefined, agentLoginPassword: undefined };
    const fetchImpl = vi.fn(async () => json({ success: true, messageId: "m1", queued: true }, 202)) as unknown as typeof fetch;
    const client = createAgentRunsClient(noCreds, fetchImpl);

    const out = await client.startRun({ assistantId: "voice", threadId: "t1", text: "hi", taskId: "task-1" });

    expect(out).toEqual({ messageId: "m1", queued: true });
    const calls = (fetchImpl as any).mock.calls;
    expect(calls.every((c: any[]) => !String(c[0]).includes("/auth/login"))).toBe(true);
    expect(calls[0][1].headers.Authorization).toBeUndefined();
    expect(calls[0][1].headers["x-user-id"]).toBe("estee_lauder");
  });

  it("re-logs in and retries once on 401", async () => {
    const t1 = makeToken(Date.now() + 3600_000);
    const t2 = makeToken(Date.now() + 3600_000);
    let runCount = 0;
    const fetchImpl = vi.fn(async (url: string | URL) => {
      if (String(url).includes("/auth/login")) {
        return json({ success: true, data: { token: runCount === 0 ? t1 : t2 } });
      }
      runCount += 1;
      return runCount === 1
        ? json({ success: false, error: "Unauthorized - Missing or invalid token" }, 401)
        : json({ success: true, messageId: "m2", queued: true }, 202);
    }) as unknown as typeof fetch;
    const client = createAgentRunsClient(config, fetchImpl);

    const out = await client.startRun({ assistantId: "voice", threadId: "t1", text: "hi", taskId: "task-1" });
    expect(out).toEqual({ messageId: "m2", queued: true });
    const logins = (fetchImpl as any).mock.calls.filter((c: any[]) => String(c[0]).includes("/auth/login"));
    expect(logins).toHaveLength(2);
  });
});
