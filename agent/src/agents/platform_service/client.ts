export interface PlatformServiceConn {
  baseUrl: string;
  apiKey?: string;
  boStoreKey?: string;
  selectedEntities: string[];
}

/**
 * Plugin connection selector captured from the plugin's middleware config.
 *
 * `connectionType` is the plugin's `meta.type`; `connections`/`connectAll` come
 * straight from the agent's middleware config.
 */
export interface PlatformServiceSelector {
  connectionType?: string;
  connections?: unknown;
  connectAll?: unknown;
}

const DEFAULT_BASE_URL = "http://127.0.0.1:5707";

function firstResolvedConfig(container?: unknown): Record<string, unknown> {
  const c = container as
    | { _resolvedConnections?: Array<{ config?: Record<string, unknown> }> }
    | undefined;
  return c?._resolvedConnections?.[0]?.config ?? {};
}

function runConfigOf(exeConfig?: unknown): Record<string, unknown> {
  return (
    ((exeConfig as { configurable?: { runConfig?: Record<string, unknown> } })?.configurable
      ?.runConfig as Record<string, unknown>) ?? {}
  );
}

function envBaseUrl(): string {
  return process.env.PLATFORM_SERVICE_URL?.trim() || DEFAULT_BASE_URL;
}

function envApiKey(): string | undefined {
  const v = process.env.FILE_SERVICE_API_KEY;
  return v && v.trim() ? v.trim() : undefined;
}

function normalize(config: Record<string, unknown>): PlatformServiceConn {
  const baseUrlRaw = config.baseUrl;
  const baseUrl = (
    typeof baseUrlRaw === "string" && baseUrlRaw.trim() ? baseUrlRaw.trim() : envBaseUrl()
  ).replace(/\/+$/, "");
  const apiKeyRaw = config.apiKey;
  const apiKey =
    typeof apiKeyRaw === "string" && apiKeyRaw.trim() ? apiKeyRaw.trim() : envApiKey();
  const boStoreKeyRaw = config.boStoreKey;
  const boStoreKey =
    typeof boStoreKeyRaw === "string" && boStoreKeyRaw.trim()
      ? boStoreKeyRaw.trim()
      : undefined;
  const entities = Array.isArray(config.selectedEntities)
    ? config.selectedEntities.filter((x): x is string => typeof x === "string")
    : [];
  return { baseUrl, apiKey, boStoreKey, selectedEntities: entities };
}

/** Connection config from a bare config object (connection.test / discover). */
export function connectionFromConfig(config: Record<string, unknown>): PlatformServiceConn {
  return normalize(config ?? {});
}

function noConnectionHint(connectionType: string, tenantId: string): string {
  return (
    `No "${connectionType}" connection is configured for tenant "${tenantId}". ` +
    `Add a connection of type "${connectionType}" and select it (connections) or enable connectAll ` +
    `in the agent's middleware config.`
  );
}

/**
 * Resolve the platform-service connection for a tool invocation.
 *
 * Prefers a host-injected, pre-resolved connection (`_resolvedConnections`).
 * Otherwise resolves dynamically from the tenant-scoped Connection Store using
 * the plugin selector (`connectionType` + `connections`/`connectAll`), so
 * connection changes apply without rebuilding the agent. Falls back to env /
 * default only when no selector is configured.
 *
 * @param pluginConfig - Plugin middleware config (selector) or pre-resolved container.
 * @param exeConfig - LangChain execution config; `runConfig.tenantId` scopes resolution.
 * @throws When a selector is configured but resolves to no connection for the tenant.
 */
export async function resolveConnection(
  pluginConfig?: unknown,
  exeConfig?: unknown,
): Promise<PlatformServiceConn> {
  const preResolved = {
    ...firstResolvedConfig(pluginConfig),
    ...firstResolvedConfig(runConfigOf(exeConfig)),
  };
  if (Object.keys(preResolved).length > 0) {
    return normalize(preResolved);
  }

  const selector = (pluginConfig ?? {}) as PlatformServiceSelector;
  const connectionType =
    typeof selector.connectionType === "string" ? selector.connectionType : undefined;
  const connections = Array.isArray(selector.connections)
    ? selector.connections.filter((key): key is string => typeof key === "string")
    : [];
  const connectAll = selector.connectAll === true;

  if (!connectionType || (!connectAll && connections.length === 0)) {
    return normalize({});
  }

  const tenantId = tenantFromExeConfig(exeConfig);
  // Imported lazily so unit tests that only exercise pre-resolved/env paths do not
  // load the full core dependency graph (chalk and other ESM-only modules).
  const { resolvePluginConnections } = await import("@axiom-lattice/core");
  const resolved = await resolvePluginConnections(
    connectionType,
    { connections, connectAll },
    { tenantId },
  );
  if (resolved.length === 0) {
    throw new Error(noConnectionHint(connectionType, tenantId));
  }
  return normalize(resolved[0].config);
}

export function tenantFromExeConfig(exeConfig?: unknown): string {
  const t = runConfigOf(exeConfig).tenantId;
  if (typeof t === "string" && t.trim()) return t.trim();
  throw new Error("tenant context is missing");
}

/** Tenant for the gateway proxy: only the authenticated request context. */
export function tenantFromRequest(request: { user?: { tenantId?: unknown } }): string {
  const t = request?.user?.tenantId;
  if (typeof t === "string" && t.trim()) return t.trim();
  throw new Error("tenant context is missing");
}

export class PlatformServiceError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "PlatformServiceError";
  }
}

export interface RequestOptions {
  conn: PlatformServiceConn;
  tenantId?: string;
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  path: string;
  query?: Record<string, string | number | boolean | undefined | null>;
  json?: unknown;
  body?: RequestInit["body"];
  contentType?: string;
  headers?: Record<string, string>;
}

function buildUrl(base: string, path: string, query?: RequestOptions["query"]): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== undefined && v !== null) qs.append(k, String(v));
  }
  const tail = qs.toString();
  return `${base}${path}${tail ? `?${tail}` : ""}`;
}

export async function request<T = unknown>(opts: RequestOptions): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers ?? {}) };
  if (opts.tenantId) {
    headers["X-Tenant-Id"] = opts.tenantId;
  } else {
    delete headers["X-Tenant-Id"];
  }
  if (opts.conn.apiKey) {
    headers["X-Api-Key"] = opts.conn.apiKey;
  } else {
    delete headers["X-Api-Key"];
  }

  let body = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  } else if (opts.contentType) {
    headers["Content-Type"] = opts.contentType;
  }

  let res: Response;
  try {
    res = await fetch(buildUrl(opts.conn.baseUrl, opts.path, opts.query), {
      method: opts.method,
      headers,
      body,
    });
  } catch (err) {
    throw new PlatformServiceError(
      0,
      "NETWORK_ERROR",
      `${opts.conn.baseUrl}${opts.path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let code = "HTTP_ERROR";
    let message = text || res.statusText;
    try {
      const parsed = JSON.parse(text) as { code?: string; message?: string };
      if (parsed.code) code = parsed.code;
      if (parsed.message) message = parsed.message;
    } catch {
      /* keep raw text */
    }
    throw new PlatformServiceError(res.status, code, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function errorResult(err: unknown): string {
  if (err instanceof PlatformServiceError) {
    return JSON.stringify({ ok: false, status: err.status, code: err.code, message: err.message });
  }
  return JSON.stringify({
    ok: false,
    code: "ERROR",
    message: err instanceof Error ? err.message : String(err),
  });
}
