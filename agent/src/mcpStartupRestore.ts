import {
  getStoreLattice,
  mcpManager,
} from "@axiom-lattice/core";
import type {
  McpServerConfig,
  McpServerConfigEntry,
  McpServerConfigStore,
} from "@axiom-lattice/protocols";

type McpConnection = Parameters<typeof mcpManager.addServer>[1];

export type RestoreMcpSummary = {
  totalConfigs: number;
  connectedConfigs: number;
  restoredServers: number;
  skippedDuplicateKeys: string[];
};

export function convertConfigToMcpConnection(config: McpServerConfig): McpConnection {
  const baseConfig = {
    env: config.env,
  };

  if (config.transport === "stdio") {
    return {
      ...baseConfig,
      transport: "stdio",
      command: config.command,
      args: config.args || [],
    } as McpConnection;
  }

  return {
    ...baseConfig,
    transport: config.transport === "streamable_http" ? "http" : config.transport,
    url: config.url,
  } as McpConnection;
}

function preferDefaultTenant(
  existing: McpServerConfigEntry,
  candidate: McpServerConfigEntry,
): McpServerConfigEntry {
  if (candidate.tenantId === "default" && existing.tenantId !== "default") {
    return candidate;
  }

  return existing;
}

export function selectConnectedMcpConfigs(
  configs: McpServerConfigEntry[],
): {
  selected: McpServerConfigEntry[];
  duplicateKeys: string[];
} {
  const byKey = new Map<string, McpServerConfigEntry>();
  const duplicateKeys = new Set<string>();

  for (const config of configs) {
    if (config.status !== "connected") {
      continue;
    }

    const existing = byKey.get(config.key);
    if (!existing) {
      byKey.set(config.key, config);
      continue;
    }

    duplicateKeys.add(config.key);
    byKey.set(config.key, preferDefaultTenant(existing, config));
  }

  return {
    selected: Array.from(byKey.values()),
    duplicateKeys: Array.from(duplicateKeys).sort(),
  };
}

export async function restoreConnectedMcpServersAcrossTenants(): Promise<RestoreMcpSummary> {
  const storeLattice = getStoreLattice("default", "mcp");
  const store = storeLattice?.store as McpServerConfigStore | undefined;

  if (!store) {
    console.info("[MCP] MCP store not configured, skipping tenant-wide restoration");
    return {
      totalConfigs: 0,
      connectedConfigs: 0,
      restoredServers: 0,
      skippedDuplicateKeys: [],
    };
  }

  const configs = await store.getAllConfigsWithoutTenant();
  const { selected, duplicateKeys } = selectConnectedMcpConfigs(configs);

  if (selected.length === 0) {
    console.info("[MCP] No connected MCP server configs found for tenant-wide restoration");
    return {
      totalConfigs: configs.length,
      connectedConfigs: 0,
      restoredServers: 0,
      skippedDuplicateKeys: duplicateKeys,
    };
  }

  if (duplicateKeys.length > 0) {
    console.warn(
      `[MCP] Duplicate connected MCP server keys detected across tenants: ${duplicateKeys.join(", ")}. ` +
      "Tool keys are global, so only one config per key can be restored.",
    );
  }

  console.info(`[MCP] Restoring ${selected.length} connected MCP server config(s) across tenants...`);

  for (const config of selected) {
    try {
      const connection = convertConfigToMcpConnection(config.config);
      mcpManager.addServer(config.key, connection);
    } catch (error) {
      console.warn(`[MCP] Failed to prepare MCP server "${config.key}" for restoration`, error);
    }
  }

  await mcpManager.connect();

  let restoredServers = 0;
  for (const config of selected) {
    try {
      await mcpManager.registerToolsToToolLattice(config.key, config.selectedTools);
      restoredServers += 1;
      console.info(
        `[MCP] Restored server "${config.key}" for tenant "${config.tenantId}" ` +
        `with ${config.selectedTools.length} selected tool(s)`,
      );
    } catch (error) {
      console.warn(`[MCP] Failed to register tools for MCP server "${config.key}"`, error);
    }
  }

  console.info(`[MCP] Tenant-wide MCP restoration complete: ${restoredServers}/${selected.length} server(s)`);

  return {
    totalConfigs: configs.length,
    connectedConfigs: selected.length,
    restoredServers,
    skippedDuplicateKeys: duplicateKeys,
  };
}
