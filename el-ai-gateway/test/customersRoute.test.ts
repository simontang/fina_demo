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

function build(
  auth: (h?: string) => { tenantId: string; keyLabel: string } | null,
  rows: any[] = [],
) {
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
    events: { customerTagUpdated: vi.fn(async () => {}) } as any,
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
