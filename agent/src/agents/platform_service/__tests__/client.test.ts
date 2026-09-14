import {
  resolveConnection,
  tenantFromExeConfig,
  tenantFromRequest,
  connectionFromConfig,
  PlatformServiceError,
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
});
