import { describe, expect, it } from "vitest";
import { loadConfig, parseApiKeys } from "../src/config";

const base = {
  AGENT_LOGIN_EMAIL: "svc@example.com",
  AGENT_LOGIN_PASSWORD: "secret",
  MCP_API_KEY: "a2a_mcp",
};

describe("parseApiKeys", () => {
  it("parses key:tenant pairs", () => {
    const map = parseApiKeys("k1:t1, k2:t2");
    expect(map.get("k1")).toBe("t1");
    expect(map.get("k2")).toBe("t2");
  });

  it("throws on malformed entries", () => {
    expect(() => parseApiKeys("nocolon")).toThrow(/Invalid GATEWAY_API_KEYS/);
  });
});

describe("loadConfig", () => {
  it("applies defaults and requires agent login + MCP key", () => {
    const config = loadConfig(base as NodeJS.ProcessEnv);
    expect(config.port).toBe(5708);
    expect(config.authDisabled).toBe(false);
    expect(config.maxUploadBytes).toBe(52428800);
    expect(config.agentRunsUrl).toBe("http://127.0.0.1:5702/api/runs");
    expect(config.agentTenantId).toBe("estee_lauder");
    expect(config.mcpServerUrl).toBe("http://127.0.0.1:5702/open/mcp");
  });

  it("fails when AGENT_LOGIN_EMAIL is missing", () => {
    expect(() =>
      loadConfig({ AGENT_LOGIN_PASSWORD: "x", MCP_API_KEY: "y" } as NodeJS.ProcessEnv),
    ).toThrow(/AGENT_LOGIN_EMAIL/);
  });

  it("fails when MCP_API_KEY is missing", () => {
    expect(() =>
      loadConfig({ AGENT_LOGIN_EMAIL: "a@b.c", AGENT_LOGIN_PASSWORD: "x" } as NodeJS.ProcessEnv),
    ).toThrow(/MCP_API_KEY/);
  });

  it("parses AUTH_DISABLED=true", () => {
    const config = loadConfig({ ...base, AUTH_DISABLED: "true" } as NodeJS.ProcessEnv);
    expect(config.authDisabled).toBe(true);
  });
});
