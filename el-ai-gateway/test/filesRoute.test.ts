import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files",
  maxUploadBytes: 1024 * 1024,
  a2aBaseUrl: "http://a2a",
  a2aApiKey: "a2a_x",
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
    const upload = vi.fn(async () => ({ uuid: "abc", version: 1, filename: "a.wav" }));
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      a2a: { sendTask: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
    });

    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?path=voice&fileName=a.wav",
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
  });

  it("returns 401 without a valid key", async () => {
    const app = buildServer({
      config,
      authenticator: () => null,
      platformFiles: { upload: vi.fn(), presign: vi.fn() } as any,
      a2a: { sendTask: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
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
