import { Readable } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { createPlatformFilesClient } from "../src/upstream/platformFiles";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map(),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesUrl: "http://files:5707/api/v1/files",
  fileServiceApiKey: "internal",
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
};

describe("platformFiles.upload", () => {
  it("PUTs the raw stream with tenant and file headers", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ uuid: "abc", version: 1 }), { status: 200 }),
    ) as unknown as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);

    const receipt = await client.upload({
      tenantId: "tenant_a",
      body: Readable.from([Buffer.from("hello")]),
      uuid: "abc",
      filename: "a.wav",
      mime: "audio/wav",
      path: "voice",
      fileName: "a.wav",
      fileCategory: "raw",
      usage: "voice-tagging",
      meta: { userId: "u1", customerId: "c2" },
    });

    expect(receipt).toEqual({ uuid: "abc", version: 1 });
    const [url, init] = (fetchImpl as any).mock.calls[0];
    expect(url).toBe("http://files:5707/api/v1/files/abc");
    expect(init.method).toBe("PUT");
    expect(init.headers["X-Tenant-Id"]).toBe("tenant_a");
    expect(init.headers["X-Api-Key"]).toBe("internal");
    expect(init.headers["Content-Type"]).toBe("audio/wav");
    expect(init.headers["X-File-Path"]).toBe("voice");
    expect(init.headers["X-File-Name"]).toBe("a.wav");
    expect(init.headers["X-File-Category"]).toBe("raw");
    expect(init.headers["X-File-Usage"]).toBe("voice-tagging");
    expect(init.headers["X-File-Meta"]).toBe(JSON.stringify({ userId: "u1", customerId: "c2" }));
    expect(init.duplex).toBe("half");
  });

  it("maps upstream errors to 502 with the upstream code", async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ code: "TENANT_REQUIRED", message: "no tenant" }), {
        status: 400,
      })) as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);
    await expect(
      client.upload({
        tenantId: "t",
        body: Readable.from([Buffer.from("x")]),
        uuid: "abc",
        filename: "a.wav",
        mime: "audio/wav",
      }),
    ).rejects.toMatchObject({ statusCode: 502, code: "TENANT_REQUIRED" });
  });
});

describe("platformFiles.presign", () => {
  it("returns the presigned url", async () => {
    const fetchImpl = (async () =>
      new Response(
        JSON.stringify({ url: "https://signed", kind: "presigned", expiresInSeconds: 600 }),
        { status: 200 },
      )) as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);
    const link = await client.presign({ tenantId: "t", uuid: "abc" });
    expect(link.url).toBe("https://signed");
  });
});

describe("platformFiles.list", () => {
  it("GETs the list with the meta filter and tenant header", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ files: [], total: 0, page: 1, size: 20 }), { status: 200 }),
    ) as unknown as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);
    const out = await client.list({
      tenantId: "tenant_a",
      meta: JSON.stringify({ baId: "u1", customerId: "c2" }),
      path: "voice",
      page: "1",
    });
    expect(out).toEqual({ files: [], total: 0, page: 1, size: 20 });
    const [url, init] = (fetchImpl as any).mock.calls[0];
    expect(String(url)).toContain("meta=");
    expect(String(url)).toContain("path=voice");
    expect(String(url)).toContain("page=1");
    expect(init.method).toBe("GET");
    expect(init.headers["X-Tenant-Id"]).toBe("tenant_a");
  });
});
