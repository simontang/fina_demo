import { metricsServerManager } from "@axiom-lattice/core";
import type { MetricsServerConfig, SemanticMetricsServerConfig } from "@axiom-lattice/protocols";

const DEFAULT_TENANT_ID = process.env.TENANT_ID || "default";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface MetricsServerContext {
  tenantId: string;
  serverKey: string;
  config: SemanticMetricsServerConfig;
}

export function getTenantIdFromExecutionConfig(exeConfig?: unknown): string {
  const tenantId = (exeConfig as {
    configurable?: { runConfig?: { tenantId?: unknown } };
  })?.configurable?.runConfig?.tenantId;

  return typeof tenantId === "string" && tenantId.trim()
    ? tenantId
    : DEFAULT_TENANT_ID;
}

export function getMetricsDataSourceFromExecutionConfig(exeConfig?: unknown):
  | { serverKey?: string; datasourceId?: string | number }
  | undefined {
  const value = (exeConfig as {
    configurable?: {
      runConfig?: {
        metricsDataSource?: { serverKey?: unknown; datasourceId?: unknown };
      };
    };
  })?.configurable?.runConfig?.metricsDataSource;

  if (!value || typeof value !== "object") {
    return undefined;
  }
  return {
    serverKey: typeof value.serverKey === "string" ? value.serverKey : undefined,
    datasourceId:
      typeof value.datasourceId === "string" || typeof value.datasourceId === "number"
        ? value.datasourceId
        : undefined,
  };
}

export async function resolveSemanticMetricsServer(
  exeConfig: unknown,
  inputServerKey?: string,
): Promise<MetricsServerContext> {
  const tenantId = getTenantIdFromExecutionConfig(exeConfig);
  const runConfigSource = getMetricsDataSourceFromExecutionConfig(exeConfig);
  const serverKey = runConfigSource?.serverKey || inputServerKey || await resolveDefaultServerKey(tenantId);

  const config = await metricsServerManager.getConfig(tenantId, serverKey) as MetricsServerConfig;
  if (config.type !== "semantic") {
    throw new Error(`Metrics server "${serverKey}" is not a semantic metrics server`);
  }
  const semanticConfig = config as SemanticMetricsServerConfig;
  if (!semanticConfig.selectedDataSources || semanticConfig.selectedDataSources.length === 0) {
    throw new Error(`Metrics server "${serverKey}" has no selectedDataSources configured`);
  }
  return { tenantId, serverKey, config: semanticConfig };
}

export async function resolveAllowedDatasourceId(
  server: MetricsServerContext,
  exeConfig: unknown,
  inputDatasourceId?: string | number,
): Promise<string> {
  const runConfigSource = getMetricsDataSourceFromExecutionConfig(exeConfig);
  const datasourceId = runConfigSource?.datasourceId ?? inputDatasourceId;
  if (datasourceId === undefined || datasourceId === null || String(datasourceId).trim() === "") {
    const selected = server.config.selectedDataSources || [];
    if (selected.length === 1) {
      return String(selected[0]);
    }
    throw new Error("datasourceId is required when multiple datasources are selected");
  }
  assertDatasourceAllowed(server.config, datasourceId);
  return String(datasourceId);
}

export function assertDatasourceAllowed(config: SemanticMetricsServerConfig, datasourceId: string | number): void {
  const selected = (config.selectedDataSources || []).map(String);
  if (!selected.includes(String(datasourceId))) {
    throw new Error(`datasourceId "${datasourceId}" is not allowed for this tenant metrics server`);
  }
}

export async function metricsFetch(
  server: MetricsServerContext,
  path: string,
  options?: RequestInit,
): Promise<unknown> {
  const url = `${server.config.serverUrl.replace(/\/$/, "")}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      ...buildHeaders(server.config),
      ...options?.headers,
      "X-Tenant-Id": server.tenantId,
    },
  });

  const text = await res.text();
  const parsed = parseJsonOrText(text);
  if (!res.ok) {
    throw new Error(`Metrics API error ${res.status}: ${stringifyErrorBody(parsed)}`);
  }
  if (isApiEnvelope(parsed) && parsed.code !== 200) {
    throw new Error(`Metrics API error ${parsed.code}: ${parsed.message || stringifyErrorBody(parsed)}`);
  }
  return isApiEnvelope(parsed) ? parsed.data : parsed;
}

export function validateReadOnlySql(sql: string): void {
  if (!sql || !sql.trim()) {
    throw new Error("customSql is required");
  }
  const stripped = stripCommentsAndLiterals(sql).trim();
  if (stripped.includes(";")) {
    throw new Error("Only a single read-only SQL statement is allowed");
  }
  const lower = stripped.toLowerCase();
  if (!(lower.startsWith("select") || lower.startsWith("with"))) {
    throw new Error("Only SELECT or WITH read-only SQL is allowed");
  }
  if (/\b(insert|update|delete|drop|alter|create|truncate|merge|call|execute|grant|revoke|copy|vacuum|analyze)\b/i.test(stripped)) {
    throw new Error("Write or DDL SQL is not allowed");
  }
}

export function serializeMetricParameters(parameters?: unknown): string | undefined {
  if (parameters === undefined || parameters === null) {
    return undefined;
  }
  return typeof parameters === "string" ? parameters : JSON.stringify(parameters);
}

async function resolveDefaultServerKey(tenantId: string): Promise<string> {
  const servers = await metricsServerManager.getServerKeys(tenantId);
  const semanticServers = servers.filter((server) => server.type === "semantic");
  if (semanticServers.length === 1) {
    return semanticServers[0].key;
  }
  if (semanticServers.length === 0) {
    throw new Error(`No semantic metrics server is configured for tenant "${tenantId}"`);
  }
  throw new Error(`serverKey is required; available semantic servers: ${semanticServers.map((server) => server.key).join(", ")}`);
}

function buildHeaders(config: MetricsServerConfig): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...config.headers,
  };
  if (config.apiKey) {
    headers.Authorization = `Bearer ${config.apiKey}`;
  } else if (config.username && config.password) {
    headers.Authorization = `Basic ${Buffer.from(`${config.username}:${config.password}`).toString("base64")}`;
  }
  return headers;
}

function parseJsonOrText(text: string): unknown {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function stringifyErrorBody(body: unknown): string {
  return typeof body === "string" ? body : JSON.stringify(body);
}

function isApiEnvelope(value: unknown): value is { code: number; message?: string; data?: unknown } {
  return Boolean(value)
    && typeof value === "object"
    && "code" in value
    && typeof (value as { code?: unknown }).code === "number";
}

function stripCommentsAndLiterals(sql: string): string {
  let out = "";
  let inSingle = false;
  let inDouble = false;
  let inBracketIdentifier = false;
  let inLineComment = false;
  let inBlockComment = false;

  for (let i = 0; i < sql.length; i += 1) {
    const c = sql[i];
    const next = sql[i + 1] || "";

    if (inLineComment) {
      if (c === "\n" || c === "\r") {
        inLineComment = false;
        out += c;
      } else {
        out += " ";
      }
      continue;
    }
    if (inBlockComment) {
      if (c === "*" && next === "/") {
        inBlockComment = false;
        out += "  ";
        i += 1;
      } else {
        out += " ";
      }
      continue;
    }
    if (inSingle) {
      if (c === "'" && next === "'") {
        out += "  ";
        i += 1;
      } else if (c === "'") {
        inSingle = false;
        out += " ";
      } else {
        out += " ";
      }
      continue;
    }
    if (inDouble) {
      if (c === "\"" && next === "\"") {
        out += "  ";
        i += 1;
      } else if (c === "\"") {
        inDouble = false;
        out += " ";
      } else {
        out += " ";
      }
      continue;
    }
    if (inBracketIdentifier) {
      if (c === "]" && next === "]") {
        out += "  ";
        i += 1;
      } else if (c === "]") {
        inBracketIdentifier = false;
        out += " ";
      } else {
        out += " ";
      }
      continue;
    }

    if (c === "-" && next === "-") {
      inLineComment = true;
      out += "  ";
      i += 1;
    } else if (c === "/" && next === "*") {
      inBlockComment = true;
      out += "  ";
      i += 1;
    } else if (c === "'") {
      inSingle = true;
      out += " ";
    } else if (c === "\"") {
      inDouble = true;
      out += " ";
    } else if (c === "[") {
      inBracketIdentifier = true;
      out += " ";
    } else {
      out += c;
    }
  }
  return out;
}
