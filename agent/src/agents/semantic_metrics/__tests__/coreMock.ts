/**
 * Lightweight `@axiom-lattice/core` stand-in for jest tests.
 *
 * fina's ts-jest setup transforms only `.ts` files, so importing the real core
 * dist pulls ESM-only transitive deps (e2b -> chalk) into the test runtime. The
 * plugin only needs a handful of core symbols; this factory provides the same
 * shapes so tests can spy/mock them exactly like the core-internal tests did.
 *
 * Use with: `jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());`
 */
export function createCoreMock() {
  const registry = new Map<string, unknown>();

  const ConnectionRegistry = {
    get: async (_type: string, _key: string, _tenantId: string): Promise<unknown> => null,
    list: async (_type: string, _tenantId: string): Promise<unknown[]> => [],
  };

  return {
    ConnectionRegistry,
    // Mirrors `resolvePluginConnections` from @axiom-lattice/core.
    resolvePluginConnections: async (
      type: string,
      config: { connections?: unknown; connectAll?: unknown },
      scope: { tenantId?: string },
    ): Promise<Array<{ key: string; config: Record<string, unknown> }>> => {
      const tenantId = scope?.tenantId;
      if (!tenantId) return [];
      const connectAll = config.connectAll === true;
      const keys = Array.isArray(config.connections)
        ? config.connections.filter((k): k is string => typeof k === "string")
        : [];
      if (!connectAll && keys.length === 0) return [];
      const entries = connectAll
        ? await ConnectionRegistry.list(type, tenantId)
        : await Promise.all(keys.map((k) => ConnectionRegistry.get(type, k, tenantId)));
      return (entries as Array<{
        key: string;
        config: Record<string, unknown>;
        type: string;
        tenantId: string;
      }>)
        .filter((e) => e !== null && e !== undefined && e.type === type && e.tenantId === tenantId)
        .map((e) => ({ key: e.key, config: e.config }));
    },
    PluginRegistry: {
      register: (plugin: { meta: { type: string } }) => {
        registry.set(plugin.meta.type, plugin);
      },
      get: (key: string) => registry.get(key),
      has: (key: string) => registry.has(key),
      list: () => [...registry.keys()],
      listMeta: () => [],
    },
  };
}
