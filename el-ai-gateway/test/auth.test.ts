import { describe, expect, it } from "vitest";
import { createAuthenticator, requirePrincipal } from "../src/auth";
import type { Config } from "../src/types";

function config(overrides: Partial<Config> = {}): Config {
  return {
    port: 5708,
    gatewayApiKeys: new Map([["secret", "tenant_a"]]),
    authDisabled: false,
    authDevTenant: "tenant_dev",
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
    mcpApiKey: "a2a_m",
    upstreamTimeoutMs: 1000,
    ...overrides,
  };
}

describe("createAuthenticator", () => {
  it("accepts a configured bearer key", () => {
    const auth = createAuthenticator(config());
    expect(auth("Bearer secret")).toEqual({ tenantId: "tenant_a", keyLabel: "secret" });
  });

  it("rejects missing or unknown keys", () => {
    const auth = createAuthenticator(config());
    expect(auth(undefined)).toBeNull();
    expect(auth("Bearer nope")).toBeNull();
  });

  it("uses the dev tenant when auth is disabled", () => {
    const auth = createAuthenticator(config({ authDisabled: true }));
    expect(auth(undefined)).toEqual({ tenantId: "tenant_dev", keyLabel: "dev" });
  });
});

describe("requirePrincipal", () => {
  it("throws 401 when unauthenticated", () => {
    const auth = createAuthenticator(config());
    expect(() => requirePrincipal(auth, undefined)).toThrowError(/Missing or invalid API key/);
  });
});
