import { describe, expect, it } from "vitest";
import { loadConfig, parseApiKeys } from "../src/config";

const PLATFORM_KEY = "a2a_f47409c1c94f49ffad37adea3259646f";

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
  it("boots with built-in defaults (no env)", () => {
    const config = loadConfig({} as NodeJS.ProcessEnv);
    expect(config.port).toBe(5708);
    expect(config.gatewayApiKeys.get(PLATFORM_KEY)).toBe("estee_lauder");
    expect(config.mcpApiKey).toBe(PLATFORM_KEY);
    expect(config.agentRunsUrl).toBe("http://127.0.0.1:5702/api/runs");
    expect(config.voiceTaggingAssistantId).toBe("voice-tagging-agent");
    expect(config.authDisabled).toBe(false);
  });

  it("allows env overrides", () => {
    const config = loadConfig({
      GATEWAY_API_KEYS: "k:t",
      MCP_API_KEY: "other",
      AUTH_DISABLED: "true",
    } as NodeJS.ProcessEnv);
    expect(config.gatewayApiKeys.get("k")).toBe("t");
    expect(config.mcpApiKey).toBe("other");
    expect(config.authDisabled).toBe(true);
  });
});
