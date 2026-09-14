import {
  resolveConnection,
  tenantFromExeConfig,
  tenantFromRequest,
  connectionFromConfig,
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

  it("uses connection apiKey, else env, else undefined", () => {
    process.env.FILE_SERVICE_API_KEY = "envkey";
    expect(resolveConnection({ _resolvedConnections: [{ config: { apiKey: "connkey" } }] }).apiKey).toBe("connkey");
    expect(resolveConnection().apiKey).toBe("envkey");
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
  });

  it("connectionFromConfig normalizes a bare connection config", () => {
    expect(connectionFromConfig({ baseUrl: "http://x:1/" }).baseUrl).toBe("http://x:1");
  });
});
