/**
 * Semantic Metrics V2 HTTP client.
 *
 * Talks to the Metrics Server `/api/v1` surface. This iteration does NOT send
 * `X-Tenant-Id`; tenant isolation relies on the connection's own credentials.
 */
import { SemanticMetricsV2Config, validateSql } from "./types";
import {
  resolvePluginConnections,
  type ResolvedPluginConnection,
} from "@axiom-lattice/core";

/** User-facing hint shown when no connection can be resolved for the tenant. */
const NO_CONNECTION_HINT =
  "No semantic-metrics connection is configured for this tenant. " +
  "Add a connection of type 'semantic-metrics' for the tenant, then select it (connections) or enable connectAll in the agent's middleware config.";

/** Connection selector captured from the plugin's middleware config. */
export interface SemanticMetricsConnectionSelector {
  /** Plugin type used as `ConnectionEntry.type`. */
  connectionType: string;
  /** Explicit connection keys. */
  connections?: string[];
  /** When true, resolve every connection of the type for the tenant. */
  connectAll?: boolean;
}

/**
 * Resolve a client from the plugin's connection selector at invocation time.
 *
 * Connections are read from the tenant-scoped Connection Store on every call
 * (see {@link resolvePluginConnections}), so connection config changes apply to
 * an already-compiled agent without a rebuild.
 *
 * @param selector - Plugin connection selector (`connectionType` + `connections`/`connectAll`).
 * @param scope - Run identity; `tenantId` is required for any connection to resolve.
 * @param connectionKey - Explicit connection key from the tool input (optional).
 * @returns A configured {@link SemanticMetricsV2Client} for the resolved connection.
 * @throws When nothing is configured or resolvable, the key is missing with multiple
 * connections available, or no connection matches the key. Error messages list the
 * available keys; a blank key defaults to the only configured connection.
 */
export async function resolveMetricsClientFromSelector(
  selector: SemanticMetricsConnectionSelector,
  scope: { tenantId?: string },
  connectionKey?: string,
): Promise<SemanticMetricsV2Client> {
  let resolved: ResolvedPluginConnection[];
  try {
    resolved = await resolvePluginConnections(selector.connectionType, selector, scope);
  } catch (error) {
    // Keep the actionable configuration hint when the Connection Store itself is
    // not configured, instead of surfacing a raw infrastructure error.
    if (error instanceof Error && error.message === "ConnectionStore not configured") {
      throw new Error(NO_CONNECTION_HINT);
    }
    throw error;
  }
  const available = resolved.map((c) => c.key).join(", ");

  if (resolved.length === 0) {
    throw new Error(NO_CONNECTION_HINT);
  }

  // Single-connection convenience: an omitted/blank key defaults to the only connection.
  const key = connectionKey?.trim() || (resolved.length === 1 ? resolved[0].key : "");
  if (!key) {
    throw new Error(
      `connectionKey is required. Available connections: ${available}`,
    );
  }

  const entry = resolved.find((c) => c.key === key);
  if (!entry) {
    throw new Error(`Connection "${key}" not found. Available connections: ${available}`);
  }
  const config = entry.config as unknown as SemanticMetricsV2Config;
  return new SemanticMetricsV2Client(config);
}

/**
 * HTTP client for the Semantic Metrics V2 `/api/v1` server surface.
 *
 * Handles read-only SQL validation before any query is sent, JSON request/response
 * encoding, and uniform error reporting for non-OK HTTP responses.
 *
 * @remarks
 * - A trailing slash on `serverUrl` is normalized away.
 * - When `apiKey` is set it is sent as a `Bearer` `Authorization` header.
 * - Custom `headers` are merged on top of the default `Accept` header; per-call
 *   headers merge on top of both.
 */
export class SemanticMetricsV2Client {
  private config: SemanticMetricsV2Config;
  private baseUrl: string;

  /**
   * @param config - Connection configuration (server URL, optional API key and headers).
   */
  constructor(config: SemanticMetricsV2Config) {
    this.config = config;
    this.baseUrl = config.serverUrl.replace(/\/$/, "");
  }

  /**
   * Build default request headers: `Accept: application/json`, custom headers,
   * and a Bearer `Authorization` header when an API key is configured.
   *
   * @returns Merged headers.
   */
  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...(this.config.headers ?? {}),
    };
    if (this.config.apiKey) {
      headers["Authorization"] = `Bearer ${this.config.apiKey}`;
    }
    return headers;
  }

  /**
   * Numeric selected resource ids of this connection (normalized).
   *
   * Reads `selectedEntities` — the standard connection resource-selection
   * field written by the connection UI (`connection.discover` + selection).
   *
   * @returns The selected resource ids; empty when unrestricted.
   */
  getSelectedEntities(): number[] {
    const raw = (this.config as unknown as Record<string, unknown>).selectedEntities;
    if (!Array.isArray(raw)) return [];
    return raw
      .map((v) => Number(v))
      .filter((n) => Number.isFinite(n));
  }

  /**
   * Perform a JSON request against the base URL.
   *
   * The Metrics Server wraps every response as `{ code, message, data }`. A
   * `code` of 200 resolves to `data`; any other `code` throws a business error
   * carrying the server message (regardless of HTTP status). Non-envelope
   * bodies are returned as-is.
   *
   * @param path - Path appended to the base URL (must start with `/`).
   * @param init - Optional `fetch` init overrides (method, body, per-call headers).
   * @returns The unwrapped `data` payload of the response envelope.
   * @throws When the server returns a non-200 `code`, or the HTTP response is not OK.
   */
  private async request(path: string, init: RequestInit = {}): Promise<unknown> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      method: init.method ?? "GET",
      headers: { ...this.getHeaders(), ...(init.headers ?? {}) },
    });
    const text = await response.text().catch(() => "");
    const parsed = this.parseBody(text);
    if (parsed?.envelope === true) {
      if (parsed.code !== 200) {
        throw new Error(`API ${parsed.code}: ${parsed.message || "Unknown error"}`);
      }
      return parsed.data;
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
    }
    return parsed?.value ?? undefined;
  }

  /**
   * Parse a response body text. Detects the server envelope `{ code, message, data }`.
   *
   * @param text - Raw response body text.
   * @returns `{ envelope: true, code, message, data }` for an envelope body,
   *   `{ envelope: false, value }` for anything else (or `{ envelope: false }` when empty/invalid).
   */
  private parseBody(text: string):
    | { envelope: true; code: number; message?: string; data?: unknown }
    | { envelope: false; value?: unknown } {
    if (!text.trim()) return { envelope: false };
    try {
      const value: unknown = JSON.parse(text);
      if (value !== null && typeof value === "object" && !Array.isArray(value)) {
        const record = value as Record<string, unknown>;
        if (typeof record.code === "number") {
          return {
            envelope: true,
            code: record.code,
            message: typeof record.message === "string" ? record.message : undefined,
            data: record.data,
          };
        }
      }
      return { envelope: false, value };
    } catch {
      return { envelope: false, value: text };
    }
  }

  /**
   * POST a JSON body to a path.
   *
   * @param path - Path appended to the base URL.
   * @param body - Payload serialized as JSON.
   * @returns Parsed JSON response body.
   */
  private post(path: string, body: unknown): Promise<unknown> {
    return this.request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  /**
   * PUT a JSON body to a path.
   *
   * @param path - Path appended to the base URL.
   * @param body - Payload serialized as JSON.
   * @returns Parsed JSON response body.
   */
  private put(path: string, body: unknown): Promise<unknown> {
    return this.request(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  /**
   * List all datasources.
   *
   * @returns Parsed JSON response body.
   */
  listDatasources(): Promise<unknown> {
    return this.request("/datasources");
  }

  /**
   * Get table grants for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @returns Parsed JSON response body.
   */
  getTableGrants(datasourceId: number): Promise<unknown> {
    return this.request(`/datasources/${encodeURIComponent(datasourceId)}/table-grants`);
  }

  /**
   * Run a read-only SQL query against a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @param input - Query input: `sql` (SELECT/WITH only), optional `params`, `maxRows`, `debug`.
   * @returns Parsed JSON response body.
   * @throws When `sql` fails read-only validation (rejected before any fetch).
   */
  async queryDatasource(
    datasourceId: number,
    input: { sql: string; params?: Record<string, unknown>; maxRows?: number; debug?: boolean },
  ): Promise<unknown> {
    const validation = validateSql(input.sql);
    if (!validation.ok) throw new Error(`SQL_NOT_ALLOWED: ${validation.reason}`);
    return this.post(`/datasources/${encodeURIComponent(datasourceId)}/query`, input);
  }

  /**
   * Test a datasource connection.
   *
   * @param datasourceId - Datasource identifier.
   * @returns Parsed JSON response body.
   */
  testDatasource(datasourceId: number): Promise<unknown> {
    return this.post(`/datasources/${encodeURIComponent(datasourceId)}/test`, {});
  }

  /**
   * Get connection-pool status for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @returns Parsed JSON response body.
   */
  getPoolStatus(datasourceId: number): Promise<unknown> {
    return this.request(`/datasources/${encodeURIComponent(datasourceId)}/pool`);
  }

  /**
   * List table metadata for a datasource, auto-paginating.
   *
   * The server pages this collection (`page`/`pageSize`, default page size 20),
   * so requests aggregate every page into one `{ items, total }` result.
   *
   * @param datasourceId - Datasource identifier.
   * @returns `{ items: [...], total }` covering all pages.
   */
  async listTables(datasourceId: number): Promise<unknown> {
    return this.listMetaPage(`/datasources/${encodeURIComponent(datasourceId)}/meta/tables`);
  }

  /**
   * Shared pagination walk for the paged meta collections (`{items,total,page,pageSize}`).
   *
   * @param basePath - Collection path (tables or metrics).
   * @returns `{ items, total }` aggregated across all pages.
   */
  private async listMetaPage(basePath: string): Promise<unknown> {
    const pageSize = 100;
    const items: unknown[] = [];
    let total = Number.POSITIVE_INFINITY;
    for (let page = 1; items.length < total; page += 1) {
      const body = await this.request(`${basePath}?page=${page}&pageSize=${pageSize}`);
      const record = body !== null && typeof body === "object" ? body as Record<string, unknown> : {};
      const batch = Array.isArray(record.items) ? record.items : [];
      items.push(...batch);
      total = typeof record.total === "number" ? record.total : items.length;
      if (batch.length === 0) break;
    }
    return { items, total: items.length };
  }

  /**
   * Get metadata for a single table.
   *
   * @param datasourceId - Datasource identifier.
   * @param tableKey - Table key.
   * @returns Parsed JSON response body.
   */
  getTable(datasourceId: number, tableKey: string): Promise<unknown> {
    return this.request(
      `/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(tableKey)}`,
    );
  }

  /**
   * Create a table definition for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @param payload - Table definition payload.
   * @returns Parsed JSON response body.
   */
  createTable(datasourceId: number, payload: Record<string, unknown>): Promise<unknown> {
    return this.post(`/datasources/${encodeURIComponent(datasourceId)}/meta/tables`, payload);
  }

  /**
   * Update a table definition for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @param tableKey - Table key.
   * @param payload - Table definition payload.
   * @returns Parsed JSON response body.
   */
  /**
   * Update a published table meta via read-modify-write.
   *
   * The server's PUT replaces the whole payload, so the current payload is
   * fetched first and the incoming fields are merged on top; fields not
   * mentioned are preserved.
   */
  async updateTable(datasourceId: number, tableKey: string, payload: Record<string, unknown>): Promise<unknown> {
    const existing = await this.readMetaPayload(`${this.baseUrl}/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(tableKey)}`);
    return this.put(
      `/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(tableKey)}`,
      { payload: { ...existing, ...payload } },
    );
  }

  /**
   * List metric definitions for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @returns Parsed JSON response body.
   */
  /**
   * List metric metadata for a datasource, auto-paginating.
   *
   * The server pages this collection (`page`/`pageSize`, default page size 20),
   * so requests aggregate every page into one `{ items, total }` result.
   *
   * @param datasourceId - Datasource identifier.
   * @returns `{ items: [...], total }` covering all pages.
   */
  async listMetrics(datasourceId: number): Promise<unknown> {
    return this.listMetaPage(`/datasources/${encodeURIComponent(datasourceId)}/meta/metrics`);
  }

  /**
   * Get a single metric definition.
   *
   * @param datasourceId - Datasource identifier.
   * @param metricKey - Metric key.
   * @returns Parsed JSON response body.
   */
  getMetric(datasourceId: number, metricKey: string): Promise<unknown> {
    return this.request(
      `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(metricKey)}`,
    );
  }

  /**
   * Create a metric definition for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @param payload - Metric definition payload.
   * @returns Parsed JSON response body.
   */
  createMetric(datasourceId: number, payload: Record<string, unknown>): Promise<unknown> {
    return this.post(`/datasources/${encodeURIComponent(datasourceId)}/meta/metrics`, payload);
  }

  /**
   * Update a metric definition for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @param metricKey - Metric key.
   * @param payload - Metric definition payload.
   * @returns Parsed JSON response body.
   */
  /**
   * Update a published metric meta via read-modify-write.
   *
   * The server's PUT replaces the whole payload, so the current payload is
   * fetched first and the incoming fields are merged on top; fields not
   * mentioned are preserved.
   */
  async updateMetric(datasourceId: number, metricKey: string, payload: Record<string, unknown>): Promise<unknown> {
    const existing = await this.readMetaPayload(`${this.baseUrl}/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(metricKey)}`);
    return this.put(
      `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(metricKey)}`,
      { payload: { ...existing, ...payload } },
    );
  }

  /**
   * Fetch the current `payload` object of a published meta entry.
   *
   * The single-entry read returns `{ code, data: [entry] }` (already unwrapped
   * to the array by {@link request}); the payload lives on the first entry.
   *
   * @param absoluteUrl - Full URL of the single-entry meta read.
   * @returns The stored payload object (empty when none exists).
   */
  private async readMetaPayload(absoluteUrl: string): Promise<Record<string, unknown>> {
    const response = await fetch(absoluteUrl, { headers: this.getHeaders() });
    const text = await response.text().catch(() => "");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
    }
    const parsed = this.parseBody(text);
    const data = parsed?.envelope === true ? parsed.data : parsed?.value;
    const entry = Array.isArray(data) ? data[0] : data;
    const payload = (entry as { payload?: unknown } | undefined)?.payload;
    return payload !== null && typeof payload === "object" && !Array.isArray(payload)
      ? payload as Record<string, unknown>
      : {};
  }

  /**
   * Get runtime metadata for a datasource.
   *
   * @param datasourceId - Datasource identifier.
   * @returns Parsed JSON response body.
   */
  getRuntimeMeta(datasourceId: number): Promise<unknown> {
    return this.request(`/datasources/${encodeURIComponent(datasourceId)}/meta`);
  }

  /**
   * Run a metrics query.
   *
   * @param request - Metrics query request payload.
   * @returns Parsed JSON response body.
   */
  queryMetrics(request: Record<string, unknown>): Promise<unknown> {
    return this.post("/metrics/query", request);
  }

  /**
   * Run an ad-hoc custom SQL query, validated as read-only before sending.
   *
   * @param request - Request with `datasourceId`, `customSql` (SELECT/WITH only),
   * and optional `limit` / `debug`.
   * @returns Parsed JSON response body.
   * @throws When `customSql` fails read-only validation (rejected before any fetch).
   */
  async customSql(request: {
    datasourceId: number;
    customSql: string;
    limit?: number;
    debug?: boolean;
  }): Promise<unknown> {
    const validation = validateSql(request.customSql);
    if (!validation.ok) throw new Error(`SQL_NOT_ALLOWED: ${validation.reason}`);
    return this.post("/metrics/query", request);
  }
}
