jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import type { ConnectionEntry } from "@axiom-lattice/protocols";
import { createMetricsMetaTool } from "../tools/metrics_meta_tool";
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

describe("metrics_meta_tool", () => {
  beforeEach(() => {
    jest.spyOn(ConnectionRegistry, "get").mockResolvedValue(CONNECTION);
  });
  afterEach(() => jest.restoreAllMocks());

  it("uses a flat object schema so OpenAI function calling accepts it (no anyOf)", () => {
    const tool = createMetricsMetaTool(toolParams());
    const def = (tool.schema as { _def?: { typeName?: string; shape?: unknown } })._def ?? {};
    expect(def.typeName).toBe("ZodObject");
    expect(def.shape).toBeDefined();
  });

  it("dispatches list_tables", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "listTables").mockResolvedValue([]);
    const tool = createMetricsMetaTool(toolParams());
    await tool.invoke({ action: "list_tables", connectionKey: "primary", datasourceId: "15" }, runtimeConfig());
    expect(spy).toHaveBeenCalledWith(15);
  });

  it("rejects a datasource outside the connection's selectedEntities scope", async () => {
    jest.spyOn(SemanticMetricsV2Client.prototype, "getSelectedEntities").mockReturnValue([15]);
    const tool = createMetricsMetaTool(toolParams());
    const out = await tool.invoke(
      { action: "create_table", connectionKey: "primary", datasourceId: 1, payload: { objectKey: "x" }, modelingSkillLoaded: true },
      runtimeConfig(),
    );
    expect(out).toContain("DATASOURCE_NOT_SELECTED");
  });

  it("dispatches create_table with the full payload", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "createTable").mockResolvedValue({ ok: true });
    const tool = createMetricsMetaTool(toolParams());
    const payload = { objectType: "table_view_detail", objectKey: "hankel_distr_sell_in", status: 1, payload: {} };
    await tool.invoke(
      { action: "create_table", connectionKey: "primary", datasourceId: "15", payload },
      runtimeConfig(),
    );
    expect(spy).toHaveBeenCalledWith(15, payload);
  });

  it("dispatches update_metric with objectKey as the path segment", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "updateMetric").mockResolvedValue({ ok: true });
    const tool = createMetricsMetaTool(toolParams());
    await tool.invoke(
      { action: "update_metric", connectionKey: "primary", datasourceId: "15", objectKey: "hankel_sell_in_nes", payload: { displayName: "Sell-in NES" } },
      runtimeConfig(),
    );
    expect(spy).toHaveBeenCalledWith(15, "hankel_sell_in_nes", { displayName: "Sell-in NES" });
  });

  it("falls back to runConfig.metricsDataSource.datasourceId", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "getMetric").mockResolvedValue({ metricName: "hankel_sell_in_nes" });
    const tool = createMetricsMetaTool(toolParams());
    await tool.invoke(
      { action: "read_metric_meta", connectionKey: "primary", objectKey: "hankel_sell_in_nes" },
      runtimeConfig({ datasourceId: "15" }),
    );
    expect(spy).toHaveBeenCalledWith(15, "hankel_sell_in_nes");
  });

  it("returns an error when datasourceId is missing", async () => {
    const tool = createMetricsMetaTool(toolParams());
    const out = await tool.invoke({ action: "list_tables", connectionKey: "primary" }, runtimeConfig());
    expect(out).toContain("datasourceId");
  });

  it("does not accept delete actions (schema rejects them)", async () => {
    const tool = createMetricsMetaTool(toolParams());
    await expect(
      tool.invoke({ action: "delete_table", connectionKey: "primary", datasourceId: "15", objectKey: "x" }, runtimeConfig()),
    ).rejects.toThrow();
    await expect(
      tool.invoke({ action: "delete_metric", connectionKey: "primary", datasourceId: "15", objectKey: "x" }, runtimeConfig()),
    ).rejects.toThrow();
  });

  function builderRunConfig(assistantId?: string) {
    return { configurable: { runConfig: { tenantId: "tenant-1", assistant_id: assistantId, metricsDataSource: { datasourceId: "15" } } } };
  }

  it("denies a builder create_table without modelingSkillLoaded", async () => {
    const tool = createMetricsMetaTool(toolParams());
    const out = await tool.invoke(
      { action: "create_table", connectionKey: "primary", payload: { objectKey: "x" } },
      builderRunConfig("semantic-metrics-builder"),
    );
    expect(out).toContain("semantic-metrics-modeling");
  });

  it("allows a builder create_table when modelingSkillLoaded is true", async () => {
    const spy = jest.spyOn(SemanticMetricsV2Client.prototype, "createTable").mockResolvedValue({ ok: true });
    const tool = createMetricsMetaTool(toolParams());
    await tool.invoke(
      { action: "create_table", connectionKey: "primary", payload: { objectKey: "x" }, modelingSkillLoaded: true },
      builderRunConfig("semantic-metrics-builder"),
    );
    expect(spy).toHaveBeenCalledWith(15, { objectKey: "x" });
  });

  it("does not enforce the guard for non-builder assistant ids", async () => {
    const metricSpy = jest.spyOn(SemanticMetricsV2Client.prototype, "createMetric").mockResolvedValue({ ok: true });
    const tool = createMetricsMetaTool(toolParams());
    await tool.invoke(
      { action: "create_metric", connectionKey: "primary", payload: { objectKey: "m" } },
      builderRunConfig("some-other-agent"),
    );
    expect(metricSpy).toHaveBeenCalledWith(15, { objectKey: "m" });
  });
});
