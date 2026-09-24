import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
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

function multipartBody(filename: string, content: string) {
  const boundary = "----elgtest";
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

describe("POST /api/v1/files", () => {
  it("streams the upload and returns the platform receipt", async () => {
    const upload = vi.fn(async (_input: any) => ({ uuid: "abc", version: 1, filename: "a.wav" }));
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });

    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url:
        "/api/v1/files?path=voice&fileName=a.wav&fileCategory=raw&usage=voice-tagging" +
        `&baId=u1&customerId=c2&meta=${encodeURIComponent('{"src":"wms"}')}`,
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });

    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ uuid: "abc", version: 1, filename: "a.wav" });
    expect(upload).toHaveBeenCalledTimes(1);
    const arg = upload.mock.calls[0][0];
    expect(arg.tenantId).toBe("tenant_a");
    expect(arg.mime).toBe("audio/wav");
    expect(arg.path).toBe("voice");
    expect(arg.fileName).toBe("a.wav");
    expect(arg.fileCategory).toBe("raw");
    expect(arg.usage).toBe("voice-tagging");
    expect(arg.meta).toEqual({ src: "wms", baId: "u1", customerId: "c2" });
  });

  it("records durationSec into the file meta", async () => {
    const upload = vi.fn(async (_input: any) => ({ uuid: "abc" }));
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });

    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?baId=u1&customerId=c2&durationSec=42",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });

    expect(res.statusCode).toBe(200);
    expect(upload.mock.calls[0][0].meta).toEqual({ baId: "u1", customerId: "c2", durationSec: 42 });
  });

  it("rejects invalid meta JSON", async () => {
    const upload = vi.fn(async (_input: any) => ({ uuid: "abc" }));
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: `/api/v1/files?baId=u1&customerId=c2&meta=${encodeURIComponent("not json")}`,
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(400);
    expect(upload).not.toHaveBeenCalled();
  });

  it("requires baId", async () => {
    const upload = vi.fn();
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?customerId=c2",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(400);
    expect(upload).not.toHaveBeenCalled();
  });

  it("requires customerId", async () => {
    const upload = vi.fn();
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?baId=u1",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(400);
    expect(upload).not.toHaveBeenCalled();
  });

  it("returns 401 without a valid key", async () => {
    const app = buildServer({
      config,
      authenticator: () => null,
      platformFiles: { upload: vi.fn(), presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files",
      headers: { "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(401);
    expect(res.json().code).toBe("UNAUTHORIZED");
  });
});

describe("GET /api/v1/files", () => {
  const auth = (h?: string) =>
    h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null;

  it("requires baId", async () => {
    const app = buildServer({
      config,
      authenticator: auth,
      platformFiles: { upload: vi.fn(), presign: vi.fn(), list: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/files?customerId=c2",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("requires customerId", async () => {
    const app = buildServer({
      config,
      authenticator: auth,
      platformFiles: { upload: vi.fn(), presign: vi.fn(), list: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/files?baId=u1",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(400);
  });

  it("lists files filtered by baId + customerId", async () => {
    const list = vi.fn(async () => ({ files: [{ uuid: "abc" }], total: 1, page: 1, size: 20 }));
    const app = buildServer({
      config,
      authenticator: auth,
      platformFiles: { upload: vi.fn(), presign: vi.fn(), list } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/files?baId=u1&customerId=c2&page=1&size=20",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json().total).toBe(1);
    expect(list).toHaveBeenCalledWith({
      tenantId: "tenant_a",
      meta: JSON.stringify({ baId: "u1", customerId: "c2" }),
      path: undefined,
      q: undefined,
      recursive: undefined,
      page: "1",
      size: "20",
    });
  });
});

describe("GET /api/v1/files/:uuid/url", () => {
  const auth = (h?: string) =>
    h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null;
  const uuid = "471c20082b524316accc1b23cba8a4de";

  it("returns a playable (presigned) url", async () => {
    const presign = vi.fn(async () => ({
      url: "https://signed",
      kind: "presigned",
      expiresInSeconds: 600,
    }));
    const app = buildServer({
      config,
      authenticator: auth,
      platformFiles: { upload: vi.fn(), list: vi.fn(), presign } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const res = await app.inject({
      method: "GET",
      url: `/api/v1/files/${uuid}/url`,
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      uuid,
      url: "https://signed",
      kind: "presigned",
      expiresInSeconds: 600,
    });
    expect(presign).toHaveBeenCalledWith({ tenantId: "tenant_a", uuid, ttlSeconds: undefined });
  });

  it("rejects a bad uuid", async () => {
    const app = buildServer({
      config,
      authenticator: auth,
      platformFiles: { upload: vi.fn(), list: vi.fn(), presign: vi.fn() } as any,
      agentRuns: { startRun: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
      boTools: { getRecord: vi.fn(async () => undefined), queryRecords: vi.fn(async () => []), createRecord: vi.fn(async () => ({})), deleteRecords: vi.fn(async () => 0) } as any,
      events: { customerTagUpdated: vi.fn(async () => {}) } as any,
    });
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/files/nope/url",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(400);
  });
});
