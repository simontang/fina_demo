import type { ConnectionStore, MetricsServerConfigEntry, MetricsServerConfigStore } from "@axiom-lattice/protocols";
import { migrateLegacyMetricsConfigsToConnections } from "../connectionMigration";

describe("semantic metrics legacy connection migration", () => {
  const originalPrefix = process.env.DEFAULT_METRICS_DATASOURCE_KEY_PREFIX;
  const originalExplicit = process.env.DEFAULT_METRICS_DATASOURCE_KEY;

  afterEach(() => {
    restoreEnv("DEFAULT_METRICS_DATASOURCE_KEY_PREFIX", originalPrefix);
    restoreEnv("DEFAULT_METRICS_DATASOURCE_KEY", originalExplicit);
    jest.restoreAllMocks();
  });

  it("creates semantic-metrics connections from legacy metrics configs", async () => {
    process.env.DEFAULT_METRICS_DATASOURCE_KEY_PREFIX = "metrics-datasource-default-";
    delete process.env.DEFAULT_METRICS_DATASOURCE_KEY;
    const metricsStore = metricsStoreWith([
      legacyEntry({
        tenantId: "hankel",
        key: "semantic-metrics",
        config: {
          type: "semantic",
          serverUrl: "http://metrics:5704/api/v1",
          selectedDataSources: ["15"],
        },
      }),
    ]);
    const connectionStore = connectionStoreWithExisting(null);

    const result = await migrateLegacyMetricsConfigsToConnections(metricsStore, connectionStore);

    expect(result).toEqual({ scanned: 1, created: 1, updated: 0, skipped: 0 });
    expect(connectionStore.create).toHaveBeenCalledWith(expect.objectContaining({
      tenantId: "hankel",
      type: "semantic-metrics",
      key: "semantic-metrics",
      name: "semantic-metrics",
      config: expect.objectContaining({
        type: "semantic",
        serverUrl: "http://metrics:5704/api/v1",
        datasourceKey: "metrics-datasource-default-15",
        selectedDataSources: ["15"],
      }),
    }));
  });

  it("updates existing migrated connections instead of creating duplicates", async () => {
    process.env.DEFAULT_METRICS_DATASOURCE_KEY = "explicit-datasource-key";
    const metricsStore = metricsStoreWith([
      legacyEntry({
        tenantId: "tenant_5",
        key: "argo",
        name: "Argo Metrics",
        config: {
          type: "semantic",
          serverUrl: "http://metrics.example/api/v1",
        },
      }),
    ]);
    const connectionStore = connectionStoreWithExisting({ id: "existing" });

    const result = await migrateLegacyMetricsConfigsToConnections(metricsStore, connectionStore);

    expect(result).toEqual({ scanned: 1, created: 0, updated: 1, skipped: 0 });
    expect(connectionStore.update).toHaveBeenCalledWith(
      "tenant_5",
      "semantic-metrics",
      "argo",
      expect.objectContaining({
        name: "Argo Metrics",
        config: expect.objectContaining({
          datasourceKey: "explicit-datasource-key",
          serverUrl: "http://metrics.example/api/v1",
        }),
      }),
    );
    expect(connectionStore.create).not.toHaveBeenCalled();
  });

  it("skips non-semantic metrics configs", async () => {
    const metricsStore = metricsStoreWith([
      legacyEntry({
        tenantId: "default",
        key: "prometheus",
        config: { type: "prometheus", serverUrl: "http://prometheus:9090" },
      }),
    ]);
    const connectionStore = connectionStoreWithExisting(null);

    const result = await migrateLegacyMetricsConfigsToConnections(metricsStore, connectionStore);

    expect(result).toEqual({ scanned: 1, created: 0, updated: 0, skipped: 1 });
    expect(connectionStore.create).not.toHaveBeenCalled();
    expect(connectionStore.update).not.toHaveBeenCalled();
  });
});

function metricsStoreWith(entries: MetricsServerConfigEntry[]): MetricsServerConfigStore {
  return {
    getAllConfigsWithoutTenant: jest.fn().mockResolvedValue(entries),
  } as unknown as MetricsServerConfigStore;
}

function connectionStoreWithExisting(existing: { id: string } | null): ConnectionStore {
  return {
    getByKey: jest.fn().mockResolvedValue(existing),
    create: jest.fn().mockImplementation(async (entry) => ({
      id: "created",
      createdAt: new Date(0).toISOString(),
      updatedAt: new Date(0).toISOString(),
      ...entry,
    })),
    update: jest.fn().mockResolvedValue(existing),
  } as unknown as ConnectionStore;
}

function legacyEntry(overrides: {
  tenantId: string;
  key: string;
  name?: string;
  config: Record<string, unknown>;
}): MetricsServerConfigEntry {
  return {
    id: `legacy-${overrides.key}`,
    tenantId: overrides.tenantId,
    key: overrides.key,
    name: overrides.name,
    config: overrides.config,
    createdAt: new Date(0),
    updatedAt: new Date(0),
  } as MetricsServerConfigEntry;
}

function restoreEnv(key: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[key];
  } else {
    process.env[key] = value;
  }
}
