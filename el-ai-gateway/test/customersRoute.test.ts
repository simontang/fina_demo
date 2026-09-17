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

function build(auth: (h?: string) => { tenantId: string; keyLabel: string } | null) {
  return buildServer({
    config,
    authenticator: auth,
    platformFiles: { upload: vi.fn(), presign: vi.fn(), list: vi.fn() } as any,
    agentRuns: { startRun: vi.fn() } as any,
    taskTools: { createTask: vi.fn(), getTask: vi.fn() } as any,
  });
}

const auth = (h?: string) =>
  h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null;

describe("GET /api/v1/customers/:customerId/tags", () => {
  it("returns the customer's tags with name + tag uuid", async () => {
    const app = build(auth);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/customers/cus_8899/tags",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.customerId).toBe("cus_8899");
    expect(body.total).toBe(body.tags.length);
    expect(body.total).toBeGreaterThan(0);
    for (const tag of body.tags) {
      expect(tag.tagId).toMatch(/^[0-9a-f]{32}$/);
      expect(typeof tag.name).toBe("string");
      expect(typeof tag.dimension).toBe("string");
    }
  });

  it("returns 404 for an unknown customer", async () => {
    const app = build(auth);
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/customers/nope/tags",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(404);
    expect(res.json().code).toBe("NOT_FOUND");
  });

  it("returns 401 without a valid key", async () => {
    const app = build(() => null);
    const res = await app.inject({ method: "GET", url: "/api/v1/customers/cus_8899/tags" });
    expect(res.statusCode).toBe(401);
  });
});
