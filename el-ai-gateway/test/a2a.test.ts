import { describe, expect, it, vi } from "vitest";
import { createA2AClient } from "../src/upstream/a2a";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map(),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files",
  maxUploadBytes: 1000,
  a2aBaseUrl: "http://agent:5702",
  a2aApiKey: "a2a_secret",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

describe("a2a.sendTask", () => {
  it("posts a message/send JSON-RPC and parses the task", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: "1",
          result: { id: "task-1", status: { state: "submitted" } },
        }),
        { status: 200 },
      ),
    ) as unknown as typeof fetch;
    const client = createA2AClient(config, fetchImpl);

    const out = await client.sendTask({ assistantId: "voice", text: "transcribe https://x" });

    expect(out.taskId).toBe("task-1");
    expect(out.state).toBe("submitted");
    const [url, init] = (fetchImpl as any).mock.calls[0];
    expect(url).toBe("http://agent:5702/api/a2a/agents/voice/jsonrpc");
    expect(init.headers.Authorization).toBe("Bearer a2a_secret");
    const body = JSON.parse(init.body);
    expect(body.method).toBe("message/send");
    expect(body.params.message.parts[0].text).toBe("transcribe https://x");
  });

  it("maps JSON-RPC errors to 502 A2A_ERROR", async () => {
    const fetchImpl = (async () =>
      new Response(
        JSON.stringify({ jsonrpc: "2.0", id: "1", error: { code: -32001, message: "Unauthorized" } }),
        { status: 200 },
      )) as typeof fetch;
    const client = createA2AClient(config, fetchImpl);
    await expect(client.sendTask({ assistantId: "voice", text: "x" })).rejects.toMatchObject({
      statusCode: 502,
      code: "A2A_ERROR",
    });
  });
});
