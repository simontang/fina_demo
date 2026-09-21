jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import type { ConnectionEntry } from "@axiom-lattice/protocols";
import { createMetricsRuntimeTool } from "../tools/metrics_runtime_tool";
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

describe("metrics_runtime_tool", () => {
  beforeEach(() => {
    jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(CONNECTION);
  });
  afterEach(() => jest.restoreAllMocks());

  it("uses a flat object schema so OpenAI function calling accepts it (no anyOf)", () => {
    const tool = createMetricsRuntimeTool(toolParams());
    const def = (tool.schema as { _def?: { typeName?: string; shape?: unknown } })._def ?? {};
    expect(def.typeName).toBe("ZodObject");
    expect(def.shape).toBeDefined();
  });

  it("dispatches list_datasources with redacted output", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "listDatasources").mockResolvedValue([
      { id: 15, name: "Hankel PostgreSQL", sourceType: "cdp_postgres", description: "d", statusLabel: "active",
        url: "jdbc:postgresql://secret-host:5432/db", username: "secret_user" },
    ]);
    const tool = createMetricsRuntimeTool(toolParams());
    const out = await tool.invoke({ action: "list_datasources", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("15");
    expect(out).not.toContain("jdbc");
    expect(out).not.toContain("secret_user");
  });

  it("dispatches read_semantic_catalog and returns index only (no full definitions)", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "getRuntimeMeta").mockResolvedValue({
      index: { metrics: [{ metricName: "m1" }], tables: [{ tableName: "t1" }] },
      metricsDetails: [{ metricName: "m1", aiAgentContext: "huge" }],
      tablesDetails: [{ tableName: "t1" }],
    });
    const tool = createMetricsRuntimeTool(toolParams());
    const out = await tool.invoke({ action: "read_semantic_catalog", connectionKey: "primary", datasourceId: "15" }, runtimeConfig());
    expect(spy).toHaveBeenCalledWith(15);
    expect(out).toContain("\"index\"");
    expect(out).toContain("m1");
    expect(out).not.toContain("metricsDetails");
    expect(out).not.toContain("tablesDetails");
    expect(out).not.toContain("aiAgentContext");
  });

  it("reads one metric's full definition on demand", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "getMetric")
      .mockResolvedValue([{ metricName: "m1", supportedDimensions: ["d1"] }]);
    const tool = createMetricsRuntimeTool(toolParams());
    const out = await tool.invoke(
      { action: "read_metric", connectionKey: "primary", datasourceId: "15", metricName: "m1" },
      runtimeConfig(),
    );
    expect(spy).toHaveBeenCalledWith(15, "m1");
    expect(out).toContain("supportedDimensions");
  });

  it("reads one table's full definition on demand", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "getTable")
      .mockResolvedValue([{ tableName: "t1" }]);
    const tool = createMetricsRuntimeTool(toolParams());
    const out = await tool.invoke(
      { action: "read_table", connectionKey: "primary", datasourceId: "15", tableName: "t1" },
      runtimeConfig(),
    );
    expect(spy).toHaveBeenCalledWith(15, "t1");
    expect(out).toContain("t1");
  });

  it("dispatches query_metrics with the semantic request", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "queryMetrics").mockResolvedValue({ rows: [] });
    const tool = createMetricsRuntimeTool(toolParams());
    const request = {
      datasourceId: "15",
      metrics: ["hankel_sell_in_nes"],
      groupBy: ["sales_team"],
      filters: [{ dimension: "posting_year", operator: "GTE", values: [2024] }],
      limit: 10,
    };
    await tool.invoke({ action: "query_metrics", connectionKey: "primary", query: request }, runtimeConfig());
    expect(spy).toHaveBeenCalledWith(request);
  });

  it("does not accept custom_sql (removed; schema rejects it)", async () => {
    const tool = createMetricsRuntimeTool(toolParams());
    await expect(
      tool.invoke(
        { action: "custom_sql", connectionKey: "primary", datasourceId: "15", sql: "select 1" } as never,
        runtimeConfig(),
      ),
    ).rejects.toThrow();
  });

  it("returns an error when the query_metrics action has no query object", async () => {
    const tool = createMetricsRuntimeTool(toolParams());
    const out = await tool.invoke({ action: "query_metrics", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("query");
  });
});
