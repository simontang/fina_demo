export interface PlatformServiceConn {
  baseUrl: string;
  apiKey?: string;
  selectedEntities: string[];
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
  const entities = Array.isArray(config.selectedEntities)
    ? config.selectedEntities.filter((x): x is string => typeof x === "string")
    : [];
  return { baseUrl, apiKey, selectedEntities: entities };
}

/** Connection config from a bare config object (connection.test / discover). */
export function connectionFromConfig(config: Record<string, unknown>): PlatformServiceConn {
  return normalize(config ?? {});
}

/** Merge build-time and invoke-time resolved connections, then env fallbacks. */
export function resolveConnection(rawConfig?: unknown, exeConfig?: unknown): PlatformServiceConn {
  const merged = {
    ...firstResolvedConfig(rawConfig),
    ...firstResolvedConfig(runConfigOf(exeConfig)),
  };
  return normalize(merged);
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
