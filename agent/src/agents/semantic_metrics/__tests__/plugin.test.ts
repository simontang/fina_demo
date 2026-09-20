jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { createSemanticMetricsMiddleware, semanticMetricsPlugin } from "../plugin";
import { PluginRegistry } from "@axiom-lattice/core";

function mockResponse(body: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    statusText: "OK",
    json: jest.fn().mockResolvedValue(body),
    text: jest.fn().mockResolvedValue(JSON.stringify(body)),
  };
}

describe("semanticMetricsMiddleware", () => {
  it("compiles three tools with the expected names", () => {
    const mw = createSemanticMetricsMiddleware({ connections: ["primary"] });
    const names = (mw.tools ?? []).map((t) => (t as { name?: string }).name).sort();
    expect(names).toEqual(["metrics_datasource_tool", "metrics_meta_tool", "metrics_runtime_tool"]);
  });

  it("registers itself as the semantic-metrics plugin on import", () => {
    expect(PluginRegistry.has("semantic-metrics")).toBe(true);
  });

  it("compiles tools with an empty connection selector (no crash)", () => {
    const mw = createSemanticMetricsMiddleware({});
    expect((mw.tools ?? []).length).toBe(3);
  });
});

describe("semanticMetricsPlugin", () => {
  it("declares the standard connection contract", () => {
    expect(semanticMetricsPlugin.meta.type).toBe("semantic-metrics");
    expect(semanticMetricsPlugin.meta.capabilityBundleEligible).toBe(true);
    expect(semanticMetricsPlugin.meta.tools?.map((t) => t.name)).toEqual([
      "metrics_datasource_tool", "metrics_meta_tool", "metrics_runtime_tool",
    ]);
    expect(semanticMetricsPlugin.connection?.fields?.map((f) => f.key)).toEqual([
      "serverUrl", "datasourceKey",
    ]);
    expect(typeof semanticMetricsPlugin.connection?.test).toBe("function");
  });

  it("declares the full standard plugin surface", () => {
    expect(semanticMetricsPlugin.meta.category).toBe("data");
    expect(semanticMetricsPlugin.meta.version).toBe("1.0.0");
    expect(semanticMetricsPlugin.meta.configSchema?.properties?.connections?.widget).toBe("connectionSelect");
    expect(semanticMetricsPlugin.meta.configSchema?.properties).not.toHaveProperty("resourceKeys");
    expect(semanticMetricsPlugin.meta.defaultConfig).toEqual({ connections: [], connectAll: false });
    expect(semanticMetricsPlugin.connection?.fields?.[0]).toMatchObject({ key: "serverUrl", type: "string", required: true });
    expect(semanticMetricsPlugin.connection?.fields?.[1]).toMatchObject({ key: "datasourceKey", type: "password", required: true });
    expect(semanticMetricsPlugin.connection).not.toHaveProperty("discover");
    expect(semanticMetricsPlugin.middleware).toBeInstanceOf(Function);
  });
});

describe("semanticMetricsPlugin connection", () => {
  let fetchMock: jest.Mock;

  beforeEach(() => {
    fetchMock = jest.fn();
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    (global as { fetch?: unknown }).fetch = undefined;
  });

  it("connection.test reports ok for the authorized datasource and sends the datasource key", async () => {
    fetchMock.mockResolvedValue(mockResponse({ code: 200, data: { id: 15, name: "Hankel PostgreSQL" } }));
    const result = await semanticMetricsPlugin.connection!.test!({
      serverUrl: "https://m.example/api/v1",
      datasourceKey: "secret",
    });
    expect(result.ok).toBe(true);
    expect(result.message).toContain("datasource Hankel PostgreSQL authorized");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://m.example/api/v1/datasources/current");
    expect(init.headers).toMatchObject({ Accept: "application/json", "X-Metrics-Datasource-Key": "secret" });
  });

  it("connection.test keeps apiKey as a legacy datasource key alias", async () => {
    fetchMock.mockResolvedValue(mockResponse({ code: 200, data: { id: 15, name: "Hankel PostgreSQL" } }));
    const result = await semanticMetricsPlugin.connection!.test!({
      serverUrl: "https://m.example/api/v1",
      apiKey: "legacy-secret",
    });
    expect(result.ok).toBe(true);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers).toEqual({ Accept: "application/json", "X-Metrics-Datasource-Key": "legacy-secret" });
  });

  it("connection.test merges custom config.headers for header-gated servers", async () => {
    fetchMock.mockResolvedValue(mockResponse({ code: 200, data: { id: 15, name: "Hankel PostgreSQL" } }));
    const result = await semanticMetricsPlugin.connection!.test!({
      serverUrl: "https://m.example/api/v1",
      datasourceKey: "secret",
      headers: { "X-Api-Key": "header-key", "X-Tenant": "acme" },
    });
    expect(result.ok).toBe(true);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers).toEqual({
      Accept: "application/json",
      "X-Api-Key": "header-key",
      "X-Tenant": "acme",
      "X-Metrics-Datasource-Key": "secret",
    });
  });

  it("connection.test reports failure when the server is unreachable", async () => {
    fetchMock.mockRejectedValue(new Error("network down"));
    const result = await semanticMetricsPlugin.connection!.test!({
      serverUrl: "https://m.example/api/v1",
    });
    expect(result.ok).toBe(false);
    expect(result.message).toContain("network down");
  });

  it("connection.test reports failure with an actionable message when serverUrl is absent", async () => {
    const result = await semanticMetricsPlugin.connection!.test!({});
    expect(result.ok).toBe(false);
    expect(result.message).toContain("serverUrl is required");
  });

  it("connection.test reports failure on an unexpected payload", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: jest.fn().mockResolvedValue("<html>nope</html>"),
    });
    const result = await semanticMetricsPlugin.connection!.test!({
      serverUrl: "https://m.example/api/v1",
      datasourceKey: "secret",
    });
    expect(result.ok).toBe(false);
    expect(result.message).toContain("Unexpected token");
  });
});
