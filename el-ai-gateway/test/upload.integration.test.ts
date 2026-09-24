import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import { createPlatformFilesClient } from "../src/upstream/platformFiles";
import type { Config } from "../src/types";

let server: Server;
let port: number;
const captured: { method?: string; url?: string; headers?: Record<string, unknown>; body?: Buffer } = {};

function config(overrides: Partial<Config> = {}): Config {
  return {
    port: 5708,
    gatewayApiKeys: new Map([["secret", "tenant_a"]]),
    authDisabled: false,
    authDevTenant: "tenant_demo",
    platformFilesUrl: `http://127.0.0.1:${port}/api/v1/files`,
    fileServiceApiKey: "internal",
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
    upstreamTimeoutMs: 5000,
    ...overrides,
  };
}

function multipartBody(filename: string, content: string) {
  const boundary = "----itest";
  const payload = Buffer.concat([
    Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
        "Content-Type: audio/wav\r\n\r\n",
    ),
    Buffer.from(content),
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);
  return { payload, contentType: `multipart/form-data; boundary=${boundary}` };
}

beforeAll(async () => {
  server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(chunk as Buffer));
    req.on("end", () => {
      captured.method = req.method;
      captured.url = req.url;
      captured.headers = req.headers as Record<string, unknown>;
      captured.body = Buffer.concat(chunks);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          uuid: String(req.url).split("/").pop(),
          size: captured.body.length,
          status: "active",
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  port = (server.address() as { port: number }).port;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

describe("upload integration (real fetch streaming)", () => {
  it("streams the multipart file to platform-service as a raw PUT", async () => {
    const cfg = config();
    const app = buildServer({
      config: cfg,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: createPlatformFilesClient(cfg),
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });

    const content = "RIFFfakewavbytes-0123456789";
    const { payload, contentType } = multipartBody("clip.wav", content);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?path=voice&fileName=clip.wav&baId=u_b&customerId=c_1",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });

    expect(res.statusCode).toBe(200);
    expect(res.json().size).toBe(Buffer.byteLength(content));
    expect(captured.method).toBe("PUT");
    expect(captured.url).toMatch(/^\/api\/v1\/files\/[0-9a-f]{32}$/);
    expect(captured.headers?.["x-tenant-id"]).toBe("tenant_a");
    expect(captured.headers?.["x-api-key"]).toBe("internal");
    expect(captured.headers?.["x-file-path"]).toBe("voice");
    expect(captured.headers?.["x-file-name"]).toBe("clip.wav");
    expect(captured.headers?.["content-type"]).toBe("audio/wav");
    expect(captured.body?.toString()).toBe(content);
  });

  it("rejects uploads without a valid key and does not call upstream", async () => {
    captured.method = undefined;
    const cfg = config();
    const app = buildServer({
      config: cfg,
      authenticator: () => null,
      platformFiles: createPlatformFilesClient(cfg),
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const { payload, contentType } = multipartBody("clip.wav", "x");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files",
      headers: { "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(401);
    expect(captured.method).toBeUndefined();
  });
});
