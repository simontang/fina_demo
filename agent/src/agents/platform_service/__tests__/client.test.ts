import {
  resolveConnection,
  tenantFromExeConfig,
  tenantFromRequest,
  connectionFromConfig,
  PlatformServiceError,
  errorResult,
  request,
} from "../client";

describe("resolveConnection", () => {
  const OLD = process.env;
  beforeEach(() => {
    process.env = { ...OLD };
  });
  afterAll(() => {
    process.env = OLD;
  });

  it("prefers invoke-time _resolvedConnections over build-time", () => {
    const rawConfig = {
      _resolvedConnections: [{ config: { baseUrl: "http://build:5707" } }],
    };
    const exeConfig = {
      configurable: {
        runConfig: { _resolvedConnections: [{ config: { baseUrl: "http://invoke:5707" } }] },
      },
    };
    expect(resolveConnection(rawConfig, exeConfig).baseUrl).toBe("http://invoke:5707");
  });

  it("falls back to env then default", () => {
    process.env.PLATFORM_SERVICE_URL = "http://env:5707/";
    expect(resolveConnection(undefined, undefined).baseUrl).toBe("http://env:5707");
    delete process.env.PLATFORM_SERVICE_URL;
    expect(resolveConnection().baseUrl).toBe("http://127.0.0.1:5707");
  });

  it("treats an empty baseUrl as absent and falls back", () => {
    process.env.PLATFORM_SERVICE_URL = "http://env:5707/";
    expect(
      resolveConnection({ _resolvedConnections: [{ config: { baseUrl: "" } }] }).baseUrl,
    ).toBe("http://env:5707");
    delete process.env.PLATFORM_SERVICE_URL;
    expect(
      resolveConnection({ _resolvedConnections: [{ config: { baseUrl: "   " } }] }).baseUrl,
    ).toBe("http://127.0.0.1:5707");
  });

  it("uses connection apiKey, else env, else undefined", () => {
    process.env.FILE_SERVICE_API_KEY = "envkey";
    expect(resolveConnection({ _resolvedConnections: [{ config: { apiKey: "connkey" } }] }).apiKey).toBe("connkey");
    expect(resolveConnection().apiKey).toBe("envkey");
    delete process.env.FILE_SERVICE_API_KEY;
    expect(resolveConnection().apiKey).toBeUndefined();
  });

  it("keeps business object connection key separate from the platform service api key", () => {
    const conn = resolveConnection({
      _resolvedConnections: [{ config: { apiKey: "platform_key", boConnectionKey: "tenant" } }],
    });
    expect(conn.apiKey).toBe("platform_key");
    expect(conn.boConnectionKey).toBe("tenant");
  });

  it("reads selectedEntities from the resolved connection config", () => {
    const conn = resolveConnection({
      _resolvedConnections: [{ config: { selectedEntities: ["ep_1", 42] } }],
    });
    expect(conn.selectedEntities).toEqual(["ep_1"]);
  });
});

describe("tenant extraction", () => {
  it("reads session tenant from runConfig", () => {
    expect(
      tenantFromExeConfig({ configurable: { runConfig: { tenantId: " t1 " } } }),
    ).toBe("t1");
  });

  it("throws when session tenant missing", () => {
    expect(() => tenantFromExeConfig({ configurable: { runConfig: {} } })).toThrow(
      "tenant context is missing",
    );
  });

  it("reads authenticated tenant from the request (never the header)", () => {
    expect(tenantFromRequest({ user: { tenantId: "t2" } })).toBe("t2");
    expect(() => tenantFromRequest({})).toThrow("tenant context is missing");
    expect(() =>
      tenantFromRequest({ headers: { "x-tenant-id": "evil" } } as any),
    ).toThrow("tenant context is missing");
  });

  it("connectionFromConfig normalizes a bare connection config", () => {
    expect(connectionFromConfig({ baseUrl: "http://x:1/" }).baseUrl).toBe("http://x:1");
  });
});

describe("request", () => {
  afterEach(() => jest.restoreAllMocks());

  const conn = { baseUrl: "http://svc:5707", apiKey: "k", selectedEntities: [] };

  it("builds url+query and injects tenant and api key", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({ a: 1 }) } as Response);
    const out = await request({
      conn,
      tenantId: "t1",
      method: "GET",
      path: "/api/v1/files",
      query: { path: "ops", page: 1, empty: undefined },
    });
    expect(out).toEqual({ a: 1 });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://svc:5707/api/v1/files?path=ops&page=1");
    expect((init.headers as Record<string, string>)["X-Tenant-Id"]).toBe("t1");
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBe("k");
  });

  it("wraps a network failure with the target url and NETWORK_ERROR", async () => {
    jest.spyOn(global, "fetch").mockRejectedValue(new Error("connect ECONNREFUSED"));
    const err = await request({
      conn,
      tenantId: "t1",
      method: "GET",
      path: "/api/v1/files",
    }).catch((e) => e);
    const parsed = JSON.parse(errorResult(err));
    expect(parsed.code).toBe("NETWORK_ERROR");
    expect(parsed.status).toBe(0);
    expect(parsed.message).toContain("http://svc:5707/api/v1/files");
  });

  it("maps a JSON error body to PlatformServiceError", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      status: 404,
      text: async () => JSON.stringify({ code: "NOT_FOUND", message: "no active file" }),
    } as Response);
    await expect(
      request({ conn, tenantId: "t1", method: "GET", path: "/api/v1/files/x" }),
    ).rejects.toMatchObject({ status: 404, code: "NOT_FOUND" });
  });

  it("never sends X-Api-Key when apiKey is absent", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({}) } as Response);
    await request({ conn: { ...conn, apiKey: undefined }, tenantId: "t1", method: "GET", path: "/x" });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBeUndefined();
  });

  it("identity headers cannot be overridden by custom headers", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({}) } as Response);
    await request({
      conn,
      tenantId: "t1",
      method: "GET",
      path: "/x",
      headers: { "X-Tenant-Id": "evil", "X-Api-Key": "evil" },
    });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["X-Tenant-Id"]).toBe("t1");
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBe("k");
  });

  it("strips X-Api-Key injected by custom headers when apiKey is absent", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({}) } as Response);
    await request({
      conn: { ...conn, apiKey: undefined },
      tenantId: "t1",
      method: "GET",
      path: "/x",
      headers: { "X-Tenant-Id": "evil", "X-Api-Key": "evil" },
    });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["X-Tenant-Id"]).toBe("t1");
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBeUndefined();
  });

  it("returns undefined on a 204 response", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({ ok: true, status: 204 } as Response);
    await expect(
      request({ conn, tenantId: "t1", method: "DELETE", path: "/api/v1/files/x" }),
    ).resolves.toBeUndefined();
  });

  it("sends a raw body with contentType and custom headers", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({}) } as Response);
    const buf = Buffer.from("x");
    await request({
      conn,
      tenantId: "t1",
      method: "PUT",
      path: "/api/v1/files/x",
      body: buf,
      contentType: "text/csv",
      headers: { "X-File-Name": "r.csv" },
    });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.body).toBe(buf);
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("text/csv");
    expect((init.headers as Record<string, string>)["X-File-Name"]).toBe("r.csv");
  });
});

describe("errorResult", () => {
  it("serializes PlatformServiceError without leaking secrets", () => {
    const out = errorResult(new PlatformServiceError(404, "NOT_FOUND", "nope"));
    expect(JSON.parse(out)).toEqual({
      ok: false,
      status: 404,
      code: "NOT_FOUND",
      message: "nope",
    });
    expect(out).not.toContain("secret-key");
  });

  it("serializes a generic Error with a code and message", () => {
    const out = errorResult(new Error("boom"));
    expect(JSON.parse(out)).toEqual({ ok: false, code: "ERROR", message: "boom" });
    expect(out).not.toContain("secret-key");
  });
});
