import type {
  ConnectionStore,
  MetricsServerConfig,
  MetricsServerConfigEntry,
  MetricsServerConfigStore,
} from "@axiom-lattice/protocols";

const SEMANTIC_METRICS_CONNECTION_TYPE = "semantic-metrics";

type MetricsStoreWithAllTenants = MetricsServerConfigStore & {
  getAllConfigsWithoutTenant?: () => Promise<MetricsServerConfigEntry[]>;
};

export interface LegacyMetricsConnectionMigrationResult {
  scanned: number;
  created: number;
  updated: number;
  skipped: number;
}

export async function migrateLegacyMetricsConfigsToConnections(
  metricsStore: MetricsServerConfigStore,
  connectionStore: ConnectionStore,
): Promise<LegacyMetricsConnectionMigrationResult> {
  const result: LegacyMetricsConnectionMigrationResult = {
    scanned: 0,
    created: 0,
    updated: 0,
    skipped: 0,
  };

  const entries = await listLegacyMetricsConfigs(metricsStore);
  for (const entry of entries) {
    result.scanned += 1;
    const connection = toSemanticMetricsConnection(entry);
    if (!connection) {
      result.skipped += 1;
      continue;
    }

    const existing = await connectionStore.getByKey(
      connection.tenantId,
      SEMANTIC_METRICS_CONNECTION_TYPE,
      connection.key,
    );
    if (existing) {
      await connectionStore.update(
        connection.tenantId,
        SEMANTIC_METRICS_CONNECTION_TYPE,
        connection.key,
        {
          name: connection.name,
          description: connection.description,
          config: connection.config,
        },
      );
      result.updated += 1;
    } else {
      await connectionStore.create(connection);
      result.created += 1;
    }
  }

  return result;
}

async function listLegacyMetricsConfigs(
  metricsStore: MetricsServerConfigStore,
): Promise<MetricsServerConfigEntry[]> {
  const store = metricsStore as MetricsStoreWithAllTenants;
  if (typeof store.getAllConfigsWithoutTenant === "function") {
    return store.getAllConfigsWithoutTenant();
  }
  return [];
}

function toSemanticMetricsConnection(entry: MetricsServerConfigEntry) {
  if (entry.config.type !== "semantic") {
    return undefined;
  }
  const config = buildConnectionConfig(entry.config);
  if (!hasString(config.serverUrl)) {
    return undefined;
  }

  return {
    tenantId: entry.tenantId,
    type: SEMANTIC_METRICS_CONNECTION_TYPE,
    key: entry.key,
    name: entry.name || entry.key,
    description: entry.description || "Migrated from legacy metrics server config.",
    config,
  };
}

function buildConnectionConfig(config: MetricsServerConfig): Record<string, unknown> {
  const migrated: Record<string, unknown> = {
    ...config,
    type: "semantic",
  };

  const datasourceKey = resolveDatasourceKey(config);
  if (datasourceKey) {
    migrated.datasourceKey = datasourceKey;
    delete migrated.apiKey;
  }

  return migrated;
}

function resolveDatasourceKey(config: MetricsServerConfig): string | undefined {
  const current = (config as { datasourceKey?: unknown }).datasourceKey;
  if (typeof current === "string" && current.trim()) {
    return current.trim();
  }

  const explicitFallback = process.env.DEFAULT_METRICS_DATASOURCE_KEY;
  if (explicitFallback?.trim()) {
    return explicitFallback.trim();
  }

  const selected = Array.isArray((config as { selectedDataSources?: unknown }).selectedDataSources)
    ? (config as { selectedDataSources: unknown[] }).selectedDataSources.map(String).filter(Boolean)
    : [];
  const prefix = process.env.DEFAULT_METRICS_DATASOURCE_KEY_PREFIX;
  if (prefix?.trim() && selected.length === 1) {
    return `${prefix.trim()}${selected[0]}`;
  }

  if (typeof config.apiKey === "string" && config.apiKey.trim()) {
    return config.apiKey.trim();
  }

  return undefined;
}

function hasString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}
