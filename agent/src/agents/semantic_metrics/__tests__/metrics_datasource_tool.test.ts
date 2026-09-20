jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import type { ConnectionEntry } from "@axiom-lattice/protocols";
import { createMetricsDatasourceTool } from "../tools/metrics_datasource_tool";
import { SemanticMetricsV2Client } from "../tools/SemanticMetricsV2Client";
import { ConnectionRegistry } from "@axiom-lattice/core";

const CONNECTION: ConnectionEntry = {
  id: "c1", tenantId: "tenant-1", type: "semantic-metrics", key: "primary",
  name: "Primary", config: { serverUrl: "https://metrics.example/api/v1" },
  createdAt: "", updatedAt: "",
};

function toolParams(extra: Record<string, unknown> = {}) {
  return { connectionType: "semantic-metrics", connections: ["primary"], ...extra };
}

function runtimeConfig(metricsDataSource?: { datasourceId?: string }) {
  return { configurable: { runConfig: { tenantId: "tenant-1", metricsDataSource } } };
}

describe("metrics_datasource_tool", () => {
  beforeEach(() => {
    jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(CONNECTION);
  });
  afterEach(() => jest.restoreAllMocks());

  it("uses a flat object schema so OpenAI function calling accepts it (no anyOf)", () => {
    const tool = createMetricsDatasourceTool(toolParams());
    const def = (tool.schema as { _def?: { typeName?: string; shape?: unknown } })._def ?? {};
    expect(def.typeName).toBe("ZodObject");
    expect(def.shape).toBeDefined();
  });

  it("dispatches list_datasources", async () => {
    const spy = jest
      .spyOn(SemanticMetricsV2Client.prototype, "listDatasources")
      .mockResolvedValue([{ id: 15, name: "hankel" }]);
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke({ action: "list_datasources", connectionKey: "primary" }, runtimeConfig());
    expect(spy).toHaveBeenCalled();
    expect(out).toContain("15");
  });

  it("redacts sensitive fields from list_datasources output", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "listDatasources").mockResolvedValue([
      {
        id: 15, name: "Hankel PostgreSQL", sourceType: "cdp_postgres",
        description: "Hankel dataset", statusLabel: "active",
        url: "jdbc:postgresql://pgm-xxx.pg.rds.aliyuncs.com:5432/postgres", username: "postgres_fuli",
        connected: true, createdAt: "2026", updatedAt: "2026",
      },
    ]);
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke({ action: "list_datasources", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("15");
    expect(out).toContain("Hankel PostgreSQL");
    expect(out).toContain("cdp_postgres");
    expect(out).not.toContain("jdbc");
    expect(out).not.toContain("postgres_fuli");
    expect(out).not.toContain("aliyuncs");
  });

  it("filters list_datasources to the connection's selectedEntities", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "listDatasources").mockResolvedValue([
      { id: 15, name: "hankel" }, { id: 1, name: "SAP Demo" },
    ]);
    jest.spyOn(SemanticMetricsV2Client.prototype, "getSelectedEntities").mockReturnValue([15]);
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke({ action: "list_datasources", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("15");
    expect(out).not.toContain("SAP Demo");
  });

  it("rejects a datasource outside the effective scope", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "getSelectedEntities").mockReturnValue([15]);
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke(
      { action: "get_grants", connectionKey: "primary", datasourceId: 1 },
      runtimeConfig(),
    );
    expect(out).toContain("DATASOURCE_NOT_SELECTED");
    expect(out).toContain("15");
  });

  it("dispatches get_grants with explicit datasourceId", async () => {
    const spy = jest
      .spyOn(SemanticMetricsV2Client.prototype, "getTableGrants")
      .mockResolvedValue([{ id: 1, schemaName: "public", tablePattern: "hankel_" }]);
    const tool = createMetricsDatasourceTool(toolParams());
    await tool.invoke({ action: "get_grants", connectionKey: "primary", datasourceId: "15" }, runtimeConfig());
    expect(spy).toHaveBeenCalledWith(15);
  });

  it("falls back to runConfig.metricsDataSource.datasourceId", async () => {
    const spy = jest
      .spyOn(SemanticMetricsV2Client.prototype, "getPoolStatus")
      .mockResolvedValue({ active: 1 });
    const tool = createMetricsDatasourceTool(toolParams());
    await tool.invoke({ action: "pool_status", connectionKey: "primary" }, runtimeConfig({ datasourceId: "15" }));
    expect(spy).toHaveBeenCalledWith(15);
  });

  it("forwards query sql and maxRows to queryDatasource", async () => {
    const spy = jest
      .spyOn(SemanticMetricsV2Client.prototype, "queryDatasource")
      .mockResolvedValue({ rows: [] });
    const tool = createMetricsDatasourceTool(toolParams());
    await tool.invoke(
      { action: "query_sql", connectionKey: "primary", datasourceId: "15", sql: "select count(*) from t", maxRows: 10 },
      runtimeConfig(),
    );
    expect(spy).toHaveBeenCalledWith(15, { sql: "select count(*) from t", maxRows: 10 });
  });

  it("returns an error when datasourceId is missing", async () => {
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke({ action: "get_grants", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("datasourceId");
  });

  it("returns an error when connectionKey is not found", async () => {
    const tool = createMetricsDatasourceTool(toolParams());
    const out = await tool.invoke({ action: "list_datasources", connectionKey: "missing" }, runtimeConfig());
    expect(out).toContain("not found");
  });
});
