import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const baseConfig: Config = {
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

function build(config: Config) {
  return buildServer({
    config,
    authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
    platformFiles: { upload: vi.fn(), presign: vi.fn() } as any,
    agentRuns: { startRun: vi.fn() } as any,
    taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
    boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
    events: { customerTagUpdated: vi.fn(async () => {}) } as any,
  });
}

function preflight(app: ReturnType<typeof build>, origin: string) {
  return app.inject({
    method: "OPTIONS",
    url: "/api/v1/customers/cus_8899/tags",
    headers: {
      origin,
      "access-control-request-method": "GET",
      "access-control-request-headers": "authorization,content-type",
    },
  });
}

describe("CORS", () => {
  it("allows any origin by default", async () => {
    const res = await preflight(build(baseConfig), "https://shop.example.com");
    expect(res.headers["access-control-allow-origin"]).toBe("*");
  });

  it("reflects the origin and allows the Authorization header", async () => {
    const res = await preflight(build(baseConfig), "https://shop.example.com");
    expect(String(res.headers["access-control-allow-headers"]).toLowerCase()).toContain(
      "authorization",
    );
  });

  it("honours a configured origin allowlist", async () => {
    const config: Config = { ...baseConfig, corsOrigins: "https://app.example.com" };
    const app = build(config);

    const allowed = await preflight(app, "https://app.example.com");
    expect(allowed.headers["access-control-allow-origin"]).toBe("https://app.example.com");

    const denied = await preflight(app, "https://evil.example.com");
    expect(denied.headers["access-control-allow-origin"]).toBeUndefined();
  });
});
