jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { SemanticMetricsV2Client, resolveMetricsClientFromSelector } from "../tools/SemanticMetricsV2Client";
import { ConnectionRegistry } from "@axiom-lattice/core";

function mockResponse(body: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    statusText: "OK",
    json: jest.fn().mockResolvedValue(body),
    text: jest.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

describe("SemanticMetricsV2Client", () => {
  let fetchMock: jest.Mock;

  beforeEach(() => {
    fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("unwraps a success envelope and resolves to its data payload", async () => {
    fetchMock.mockResolvedValue(mockResponse({
      code: 200, message: "success",
      data: [{ id: 15, name: "Analytics" }],
    }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const result = await client.listDatasources();
    expect(result).toEqual([{ id: 15, name: "Analytics" }]);
  });

  it("throws a business error carrying the server message when the envelope code is not 200", async () => {
    fetchMock.mockResolvedValue(mockResponse({
      code: 400, message: "Metric 'x' not found in catalog", data: null,
    }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).rejects.toThrow(/API 400: Metric 'x' not found in catalog/);
  });

  it("returns a non-envelope JSON body as-is", async () => {
    fetchMock.mockResolvedValue(mockResponse([{ id: 1 }]));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const result = await client.listDatasources();
    expect(result).toEqual([{ id: 1 }]);
  });

  it("returns a plain-text success body as-is", async () => {
    fetchMock.mockResolvedValue({
      ok: true, status: 200, statusText: "OK",
      json: jest.fn().mockResolvedValue("ok"),
      text: jest.fn().mockResolvedValue("ok"),
    });
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).resolves.toBe("ok");
  });

  it("reads selectedEntities (UI contract) as the connection's resource scope", () => {
    const viaEntities = new SemanticMetricsV2Client({
      serverUrl: "https://m.example/api/v1",
      selectedEntities: ["15", 16],
    });
    expect(viaEntities.getSelectedEntities()).toEqual([15, 16]);

    const unrestricted = new SemanticMetricsV2Client({ serverUrl: "https://m.example/api/v1" });
    expect(unrestricted.getSelectedEntities()).toEqual([]);
  });

  it("builds headers with Accept, apiKey bearer, and custom headers (no X-Tenant-Id)", async () => {
    fetchMock.mockResolvedValue(mockResponse([{ id: 1 }]));
    const client = new SemanticMetricsV2Client({
      serverUrl: "https://metrics.example/api/v1/",
      apiKey: "secret",
      headers: { "X-Custom": "1" },
    });

    await client.listDatasources();

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources");
    expect(init.headers).toEqual({
      Accept: "application/json",
      "X-Custom": "1",
      Authorization: "Bearer secret",
    });
  });

  it("rejects a non-SELECT datasource query before fetch", async () => {
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(
      client.queryDatasource(15, { sql: "drop table t" }),
    ).rejects.toThrow(/SQL_NOT_ALLOWED/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("posts a datasource query to the correct path with body", async () => {
    fetchMock.mockResolvedValue(mockResponse({ rows: [] }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });

    await client.queryDatasource(15, { sql: "select count(*) from t", maxRows: 10 });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/query");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ sql: "select count(*) from t", maxRows: 10 });
  });

  it.each([
    ["listDatasources", "GET", "/datasources"],
    ["getTableGrants", "GET", "/datasources/15/table-grants"],
    ["getPoolStatus", "GET", "/datasources/15/pool"],
    ["getRuntimeMeta", "GET", "/datasources/15/meta"],
  ] as const)("routes %s to GET %s", async (_name, method, path) => {
    fetchMock.mockResolvedValue(mockResponse({}));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const call: Record<string, () => Promise<unknown>> = {
      listDatasources: () => client.listDatasources(),
      getTableGrants: () => client.getTableGrants(15),
      getPoolStatus: () => client.getPoolStatus(15),
      getRuntimeMeta: () => client.getRuntimeMeta(15),
    };
    await call[_name]();
    const [url, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe(method);
    expect(url).toBe(`https://metrics.example/api/v1${path}`);
  });

  it("routes listTables to the paged collection path", async () => {
    fetchMock.mockResolvedValue(mockResponse({ items: [], total: 0, page: 1, pageSize: 100 }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.listTables(15);
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/tables?page=1&pageSize=100");
  });

  it("routes listMetrics to the paged collection path", async () => {
    fetchMock.mockResolvedValue(mockResponse({ items: [], total: 0, page: 1, pageSize: 100 }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.listMetrics(15);
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/metrics?page=1&pageSize=100");
  });

  it("aggregates all pages of the paged meta collection", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse({
        items: [{ objectKey: "a" }, { objectKey: "b" }],
        total: 3, page: 1, pageSize: 100,
      }))
      .mockResolvedValueOnce(mockResponse({
        items: [{ objectKey: "c" }],
        total: 3, page: 2, pageSize: 100,
      }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const result = await client.listMetrics(15) as { items: Array<{ objectKey: string }>; total: number };
    expect(result.items.map((i) => i.objectKey)).toEqual(["a", "b", "c"]);
    expect(result.total).toBe(3);
    const [firstUrl, secondUrl] = fetchMock.mock.calls.map((c) => c[0]);
    expect(firstUrl).toContain("page=1&pageSize=100");
    expect(secondUrl).toContain("page=2&pageSize=100");
  });

  it("rejects a customSql with a non-read query", async () => {
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(
      client.customSql({ datasourceId: 15, customSql: "update t set x=1" }),
    ).rejects.toThrow(/SQL_NOT_ALLOWED/);
  });

  it("throws an error including status and body on non-ok response", async () => {
    fetchMock.mockResolvedValue(mockResponse({ message: "nope" }, false, 500));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).rejects.toThrow(/HTTP 500/);
  });

  it("resolves to undefined on a 200 response with an empty body", async () => {
    fetchMock.mockResolvedValue({
      ok: true, status: 200, statusText: "OK",
      json: jest.fn().mockRejectedValue(new SyntaxError("Unexpected end of JSON input")),
      text: jest.fn().mockResolvedValue(""),
    });
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).resolves.toBeUndefined();
  });

  it("updateTable merges the incoming fields over the stored payload (read-modify-write)", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse([{ payload: { schemaName: "public", displayName: "Old" }, objectKey: "t" }]))
      .mockResolvedValueOnce(mockResponse({ ok: true }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.updateTable(15, "hankel_distr_sell_in", { displayName: "New" });
    const [getUrl] = fetchMock.mock.calls[0];
    expect(getUrl).toBe("https://metrics.example/api/v1/datasources/15/meta/tables/hankel_distr_sell_in");
    const [, putInit] = fetchMock.mock.calls[1];
    expect(putInit.method).toBe("PUT");
    expect(JSON.parse(putInit.body)).toEqual({
      payload: { schemaName: "public", displayName: "New" },
    });
  });

  it("updateMetric merges the incoming fields over the stored payload (read-modify-write)", async () => {
    fetchMock
      .mockResolvedValueOnce(mockResponse([{ payload: { sourceTable: "t", display_name: "Old" } }]))
      .mockResolvedValueOnce(mockResponse({ ok: true }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.updateMetric(15, "hankel_sell_in_nes", { displayName: "New" });
    const [, putInit] = fetchMock.mock.calls[1];
    expect(putInit.method).toBe("PUT");
    expect(JSON.parse(putInit.body)).toEqual({
      payload: { sourceTable: "t", display_name: "Old", displayName: "New" },
    });
  });

  it("GETs a two-segment table with an encoded table key", async () => {
    fetchMock.mockResolvedValue(mockResponse({}));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.getTable(15, "a b/c");
    const [url, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("GET");
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/tables/a%20b%2Fc");
  });

  it("GETs a two-segment metric with an encoded metric key", async () => {
    fetchMock.mockResolvedValue(mockResponse({}));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.getMetric(15, "a b/c");
    const [url, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("GET");
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/metrics/a%20b%2Fc");
  });

  it("POSTs testDatasource with an empty body", async () => {
    fetchMock.mockResolvedValue(mockResponse({ connected: true }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.testDatasource(15);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/test");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({});
  });

  it("POSTs createTable to the tables collection", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: true }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const payload = { objectKey: "hankel_distr_sell_in" };
    await client.createTable(15, payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/tables");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(payload);
  });

  it("POSTs createMetric to the metrics collection", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: true }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const payload = { objectKey: "hankel_sell_in_nes" };
    await client.createMetric(15, payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/datasources/15/meta/metrics");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(payload);
  });

  it("POSTs queryMetrics to /metrics/query", async () => {
    fetchMock.mockResolvedValue(mockResponse({ rows: [] }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    const request = { datasourceId: 15, metrics: ["hankel_sell_in_nes"], groupBy: ["sales_team"], limit: 10 };
    await client.queryMetrics(request);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/metrics/query");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(request);
  });

  it("POSTs customSql success to /metrics/query", async () => {
    fetchMock.mockResolvedValue(mockResponse({ rows: [] }));
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await client.customSql({ datasourceId: 15, customSql: "select count(*) from t", limit: 20 });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://metrics.example/api/v1/metrics/query");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ datasourceId: 15, customSql: "select count(*) from t", limit: 20 });
  });

  it("surfaces the response body text in the error", async () => {
    fetchMock.mockResolvedValue({
      ok: false, status: 400, statusText: "Bad Request",
      json: jest.fn(), text: jest.fn().mockResolvedValue("SQL rejected by server"),
    });
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).rejects.toThrow(/HTTP 400: SQL rejected by server/);
  });

  it("falls back to statusText when the error body is empty", async () => {
    fetchMock.mockResolvedValue({
      ok: false, status: 502, statusText: "Bad Gateway",
      json: jest.fn(), text: jest.fn().mockResolvedValue(""),
    });
    const client = new SemanticMetricsV2Client({ serverUrl: "https://metrics.example/api/v1" });
    await expect(client.listDatasources()).rejects.toThrow(/HTTP 502: Bad Gateway/);
  });
});

describe("resolveMetricsClientFromSelector", () => {
  const SCOPE = { tenantId: "tenant-1" };
  const entry = (key: string) => ({
    id: key, tenantId: "tenant-1", type: "semantic-metrics", key,
    name: key, config: { serverUrl: "https://metrics.example/api/v1" },
    createdAt: "", updatedAt: "",
  });

  afterEach(() => jest.restoreAllMocks());

  it("resolves a connection by key from the tenant-scoped Connection Store", async () => {
    const get = jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(entry("primary"));
    const client = await resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connections: ["primary"] },
      SCOPE,
    );
    expect(client).toBeInstanceOf(SemanticMetricsV2Client);
    expect(get).toHaveBeenCalledWith("semantic-metrics", "primary", "tenant-1");
  });

  it("lists every connection when connectAll is set and requires an explicit key", async () => {
    const list = jest.spyOn(ConnectionRegistry, "list").mockResolvedValue([entry("a"), entry("b")]);
    await expect(resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connectAll: true },
      SCOPE,
    )).rejects.toThrow(/connectionKey is required\. Available connections: a, b/);
    expect(list).toHaveBeenCalledWith("semantic-metrics", "tenant-1");
  });

  it("throws when the key is not found and lists available keys", async () => {
    jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(entry("a"));
    await expect(resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connections: ["a"] },
      SCOPE,
      "missing",
    )).rejects.toThrow(/Connection "missing" not found\. Available connections: a/);
  });

  it("defaults a blank key to the only configured connection", async () => {
    jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(entry("only"));
    const client = await resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connections: ["only"] },
      SCOPE,
    );
    expect(client).toBeInstanceOf(SemanticMetricsV2Client);
  });

  it("surfaces the configuration hint when no tenant is available", async () => {
    await expect(resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connections: ["primary"] },
      {},
    )).rejects.toThrow(/No semantic-metrics connection is configured/);
  });

  it("maps an unconfigured Connection Store to the configuration hint", async () => {
    jest.spyOn(ConnectionRegistry, "get").mockRejectedValue(new Error("ConnectionStore not configured"));
    await expect(resolveMetricsClientFromSelector(
      { connectionType: "semantic-metrics", connections: ["primary"] },
      SCOPE,
    )).rejects.toThrow(/No semantic-metrics connection is configured/);
  });

  it("re-resolves on every call so config-layer changes apply", async () => {
    const get = jest.spyOn(ConnectionRegistry, "get")
      .mockResolvedValueOnce(entry("primary"))
      .mockResolvedValueOnce({
        ...entry("primary"),
        config: { serverUrl: "https://fresh.example/api/v1" },
      });
    const selector = { connectionType: "semantic-metrics", connections: ["primary"] };
    await resolveMetricsClientFromSelector(selector, SCOPE);
    await resolveMetricsClientFromSelector(selector, SCOPE);
    expect(get).toHaveBeenCalledTimes(2);
  });
});
