/**
 * Semantic Metrics Middleware & Plugin (connection-backed, /api/v1).
 *
 * Exposes datasource exploration, meta publishing, and runtime query tools.
 * Credentials are resolved by the plugin's own tools at invocation time from
 * the tenant-scoped Connection Store (see `resolveMetricsClientFromSelector`).
 */
import { createMiddleware, type AgentMiddleware } from "langchain";
import { z } from "zod";
import { PluginRegistry } from "@axiom-lattice/core";
import { createMetricsDatasourceTool } from "./tools/metrics_datasource_tool";
import { createMetricsMetaTool } from "./tools/metrics_meta_tool";
import { createMetricsRuntimeTool } from "./tools/metrics_runtime_tool";
import { SEMANTIC_METRICS_MODELING_SKILL } from "./skill";
import { SEMANTIC_METRICS_BUILDER_PROMPT } from "./prompt";
import { AgentType, type Plugin } from "@axiom-lattice/protocols";

/** Local copy of the core middleware context contract (`configurable.runConfig`). */
const contextSchema = z.object({ runConfig: z.any() });

/**
 * Create the Semantic Metrics middleware exposing three action-dispatch tools.
 *
 * @param config - Plugin config; reads `connections`/`connectAll` (resolved per
 * invocation from the tenant-scoped Connection Store). The datasource scope is
 * enforced by the connection's datasource key on the Metrics Server.
 * @returns Middleware with the datasource/meta/runtime tool set.
 */
export function createSemanticMetricsMiddleware(
  config: Record<string, unknown>,
): AgentMiddleware {
  const connections = (config.connections as string[] | undefined) ?? [];
  const connectAll = config.connectAll === true;
  const selector = { connectionType: "semantic-metrics", connections, connectAll };

  return createMiddleware({
    name: "SemanticMetrics",
    contextSchema,
    tools: [
      createMetricsDatasourceTool(selector),
      createMetricsMetaTool(selector),
      createMetricsRuntimeTool(selector),
    ],
  });
}

/**
 * Build request headers for a connection's Metrics Server call.
 *
 * Mirrors the {@link SemanticMetricsV2Client} behavior: `Accept: application/json`,
 * any custom `config.headers`, plus `X-Metrics-Datasource-Key` when a datasource
 * key is configured. Legacy `apiKey` is accepted as a datasource-key alias.
 *
 * @param config - Connection config (may carry optional `apiKey` and `headers`).
 * @returns Merged headers for the outgoing fetch call.
 */
function buildConnectionHeaders(config: Record<string, unknown>): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json" };
  const customHeaders = config.headers;
  if (customHeaders && typeof customHeaders === "object" && !Array.isArray(customHeaders)) {
    for (const [key, value] of Object.entries(customHeaders)) {
      if (typeof value === "string") headers[key] = value;
    }
  }
  const datasourceKey = resolveDatasourceKey(config);
  if (datasourceKey) headers["X-Metrics-Datasource-Key"] = datasourceKey;
  return headers;
}

function resolveDatasourceKey(config: Record<string, unknown>): string | undefined {
  for (const key of ["datasourceKey", "apiKey"]) {
    const value = config[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  const fallback = process.env.DEFAULT_METRICS_DATASOURCE_KEY;
  return fallback?.trim() || undefined;
}

function valueToId(value: unknown): string | undefined {
  if (typeof value === "string" || typeof value === "number") {
    const id = String(value).trim();
    return id ? id : undefined;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    return valueToId(record.id ?? record.key ?? record.value);
  }
  return undefined;
}

function parseCurrentDatasourceEnvelope(text: string): Record<string, unknown> {
  const parsed = JSON.parse(text) as { code?: number; message?: string; data?: unknown } | Record<string, unknown>;
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Metrics server returned an unexpected payload");
  }
  const envelope = parsed as { code?: number; message?: string; data?: unknown };
  if (typeof envelope.code === "number" && envelope.code !== 200) {
    throw new Error(envelope.message || `API ${envelope.code}`);
  }
  const data = "data" in envelope ? envelope.data : parsed;
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("Metrics server returned an unexpected payload (expected one datasource)");
  }
  return data as Record<string, unknown>;
}

/**
 * Resolve the Metrics Server base URL from a connection config.
 *
 * @param config - Connection config carrying a required `serverUrl`.
 * @returns The base URL with any trailing slash removed.
 * @throws An actionable error when `serverUrl` is absent.
 */
function resolveBaseUrl(config: Record<string, unknown>): string {
  const serverUrl = config.serverUrl;
  if (typeof serverUrl !== "string" || !serverUrl.trim()) {
    throw new Error("serverUrl is required");
  }
  return serverUrl.trim().replace(/\/$/, "");
}

/** Built-in Semantic Metrics plugin definition. */
export const semanticMetricsPlugin: Plugin = {
  meta: {
    type: "semantic-metrics",
    category: "data",
    capabilityBundleEligible: true,
    openExpose: [
      // Query-only surface: no writes reach either tool (SQL is read-only).
      { name: "metrics_datasource_tool", readOnly: true },
      { name: "metrics_runtime_tool", readOnly: true },
    ],
    name: "Semantic Metrics",
    description:
      "Semantic metrics datasource exploration, meta publishing, and runtime querying. " +
      "PERMISSION MODEL — query-only agents (metric definitions + data): enable this middleware with allowedTools ['metrics_runtime_tool']. " +
      "Metric designers (Builder): keep all three tools (metrics_datasource_tool, metrics_meta_tool, metrics_runtime_tool) for exploration, publishing, and verification, or use the built-in 'semantic-metrics-builder' agent.",
    version: "1.0.0",
    tools: [
      { name: "metrics_datasource_tool", description: "Explore datasource structure within the tenant's granted scope" },
      { name: "metrics_meta_tool", description: "Publish and maintain runtime semantic tables and metrics" },
      { name: "metrics_runtime_tool", description: "Read runtime meta and execute semantic metric queries" },
    ],
    configSchema: {
      type: "object",
      title: "Semantic Metrics Configuration",
      description:
        "First select connections (connections or connectAll). Then scope tools by role via allowedTools: " +
        "QUERY-ONLY agents (metric definitions + data) set allowedTools to ['metrics_runtime_tool'] — it is self-contained (list_datasources, read_semantic_catalog, query_metrics); " +
        "metric DESIGNERS keep all three tools (datasource exploration, meta publishing, runtime verification).",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          items: { type: "string" },
          widget: "connectionSelect",
        },
        connectAll: {
          type: "boolean",
          title: "Connect All",
          default: false,
          widget: "switch",
        },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
  },

  connection: {
    // PluginConnectionFieldSchema.type supports only string/number/boolean/password,
    // so headers is NOT a form field; the V2 client still reads config.headers at runtime.
    fields: [
      { key: "serverUrl", type: "string", title: "Metrics Server Base URL (include /api/v1)", required: true },
      {
        key: "datasourceKey",
        type: "password",
        title: "Datasource Key",
        required: true,
        helpText: "Authorizes exactly one Metrics datasource/account. Use one connection per datasource.",
      },
    ],
    test: async (config) => {
      try {
        const baseUrl = resolveBaseUrl(config);
        const res = await fetch(`${baseUrl}/datasources/current`, {
          headers: buildConnectionHeaders(config),
        });
        const text = await res.text().catch(() => "");
        let ok = res.ok;
        let detail = "";
        let datasourceName = "";
        try {
          const datasource = parseCurrentDatasourceEnvelope(text || "{}");
          const id = valueToId(datasource.id);
          datasourceName = typeof datasource.name === "string" && datasource.name.trim()
            ? datasource.name.trim()
            : id || "";
        } catch (err) {
          ok = false;
          detail = ` (${err instanceof Error ? err.message : String(err)})`;
        }
        return {
          ok,
          message: ok
            ? `Reachable at ${baseUrl}${datasourceName ? `; datasource ${datasourceName} authorized` : ""}`
            : `HTTP ${res.status}${detail}`,
        };
      } catch (err) {
        return {
          ok: false,
          message: `Cannot connect: ${err instanceof Error ? err.message : String(err)}`,
        };
      }
    },
  },

  skills: {
    "semantic-metrics-modeling": SEMANTIC_METRICS_MODELING_SKILL,
  },

  agents: {
    "semantic-metrics-builder": {
      key: "semantic-metrics-builder",
      name: "Semantic Metrics Builder",
      description:
        "Explore a granted datasource and build semantic tables and metrics interactively, " +
        "tracking the work as persistent tasks with belief state and verifying each step.",
      type: AgentType.DEEP_AGENT,
      prompt: SEMANTIC_METRICS_BUILDER_PROMPT,
      middleware: [
        {
          id: "semantic-metrics",
          type: "semantic-metrics",
          name: "Semantic Metrics",
          description: "Explore datasources, publish table/meta metrics, query runtime",
          enabled: true,
          config: { connections: [], connectAll: true },
        },
        {
          id: "skill",
          type: "skill",
          name: "Skill",
          description: "Load the semantic-metrics-modeling policy and task-definition reference",
          enabled: true,
          config: { readAll: false, skills: ["semantic-metrics-modeling", "task-definition"] },
        },
        {
          id: "task",
          type: "task",
          name: "Task",
          description: "Persistent TaskItems as the planning surface; update Belief State in task descriptions",
          enabled: true,
          config: {},
        },
        {
          id: "ask_user_to_clarify",
          type: "ask_user_to_clarify",
          name: "Ask User",
          description: "Wait for user input at modeling approval gates",
          enabled: true,
          config: {},
        },
        {
          id: "filesystem",
          type: "filesystem",
          name: "Filesystem",
          description: "Read user-provided schema docs, data dictionaries, or sample data files",
          enabled: true,
          config: {},
        },
        {
          id: "code_eval",
          type: "code_eval",
          name: "Code Execution",
          description: "Run code in the sandbox to inspect/transform data files and verify results",
          enabled: true,
          config: {},
        },
      ],
    },
  },

  middleware: (config) => createSemanticMetricsMiddleware(config),
};

PluginRegistry.register(semanticMetricsPlugin);
