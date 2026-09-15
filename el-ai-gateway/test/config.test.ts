import { describe, expect, it } from "vitest";
import { loadConfig, parseApiKeys } from "../src/config";

const base = {
  A2A_API_KEY: "a2a_test",
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
  it("applies defaults and requires outbound keys", () => {
    const config = loadConfig(base as NodeJS.ProcessEnv);
    expect(config.port).toBe(5708);
    expect(config.authDisabled).toBe(false);
    expect(config.maxUploadBytes).toBe(52428800);
    expect(config.mcpServerUrl).toBe("http://127.0.0.1:5702/open/mcp");
    expect(config.a2aApiKey).toBe("a2a_test");
  });

  it("fails when A2A_API_KEY is missing", () => {
    expect(() => loadConfig({ MCP_API_KEY: "x" } as NodeJS.ProcessEnv)).toThrow(/A2A_API_KEY/);
  });

  it("parses AUTH_DISABLED=true", () => {
    const config = loadConfig({ ...base, AUTH_DISABLED: "true" } as NodeJS.ProcessEnv);
    expect(config.authDisabled).toBe(true);
  });
});
