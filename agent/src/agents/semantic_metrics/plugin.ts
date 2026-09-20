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
 * invocation from the tenant-scoped Connection Store). Resource scope is owned
 * by the connection's own `selectedEntities`.
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
 * any custom `config.headers`, plus a Bearer `Authorization` header when an API
 * key is configured (the API key overrides a custom `Authorization`).
 *
 * @param config - Connection config (may carry optional `apiKey` and `headers`).
 * @returns Merged headers for the outgoing fetch call.
 */
function buildConnectionHeaders(config: Record<string, unknown>): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(config.headers !== undefined && config.headers !== null
      ? (config.headers as Record<string, string>)
      : {}),
  };
  if (config.apiKey) headers["Authorization"] = `Bearer ${config.apiKey}`;
  return headers;
}

/**
 * Resolve the Metrics Server base URL from a connection config.
 *
 * @param config - Connection config carrying a required `serverUrl`.
 * @returns The base URL with any trailing slash removed.
 * @throws An actionable error when `serverUrl` is absent.
 */
function resolveBaseUrl(config: Record<string, unknown>): string {
  const serverUrl = config.serverUrl as string | undefined;
  if (!serverUrl) {
    throw new Error("serverUrl is required");
  }
  return serverUrl.replace(/\/$/, "");
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
      { key: "apiKey", type: "password", title: "API Key (optional)" },
    ],
    test: async (config) => {
      try {
        const baseUrl = resolveBaseUrl(config);
        const res = await fetch(`${baseUrl}/datasources`, {
          headers: buildConnectionHeaders(config),
        });
        const text = await res.text().catch(() => "");
        let ok = res.ok;
        let detail = "";
        try {
          const parsed: unknown = JSON.parse(text);
          if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
            const code = (parsed as Record<string, unknown>).code;
            if (typeof code === "number") {
              ok = res.ok && code === 200;
              if (!ok) detail = ` (${(parsed as Record<string, unknown>).message ?? code})`;
            }
          }
        } catch {
          // Non-JSON body — fall back to the HTTP status.
        }
        return {
          ok,
          message: ok ? `Reachable at ${baseUrl}` : `HTTP ${res.status}${detail}`,
        };
      } catch (err) {
        return {
          ok: false,
          message: `Cannot connect: ${err instanceof Error ? err.message : String(err)}`,
        };
      }
    },
    discover: async (config) => {
      try {
        const baseUrl = resolveBaseUrl(config);
        const res = await fetch(`${baseUrl}/datasources`, {
          headers: buildConnectionHeaders(config),
        });
        const text = await res.text();
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}${text ? `: ${text.slice(0, 200)}` : ""}`);
        }
        const parsed: unknown = JSON.parse(text);
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("Metrics server returned an unexpected payload (expected a JSON envelope)");
        }
        const envelope = parsed as { code?: unknown; message?: unknown; data?: unknown };
        if (typeof envelope.code === "number" && envelope.code !== 200) {
          throw new Error(envelope.message as string || `API ${envelope.code}`);
        }
        const data = envelope.data;
        if (!Array.isArray(data)) {
          throw new Error("Metrics server returned an unexpected payload (expected a datasource array)");
        }
        // Resource identity is the metrics-server datasource id. A name is NOT a
        // valid key: the runtime coerces keys to numbers, so a non-numeric name
        // would be dropped and silently widen the effective scope to "all".
        // Entries without an id are skipped rather than keyed by name.
        return data
          .filter((d): d is { id: string | number; name?: string } => {
            if (d === null || typeof d !== "object") return false;
            const id = (d as { id?: unknown }).id;
            return id !== undefined && id !== null && id !== "";
          })
          .map((item) => ({
            id: String(item.id),
            name: item.name ?? String(item.id),
          }));
      } catch (err) {
        throw new Error(
          `Cannot discover datasources: ${err instanceof Error ? err.message : String(err)}`,
        );
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
      ],
    },
  },

  middleware: (config) => createSemanticMetricsMiddleware(config),
};

PluginRegistry.register(semanticMetricsPlugin);
