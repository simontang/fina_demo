import { randomUUID } from "node:crypto";

export const DEFAULT_BASE_URL = "https://openspeech.bytedance.com";
export const DEFAULT_RESOURCE_ID = "volc.seedasr.auc";
export const SUBMIT_PATH = "/api/v3/auc/bigmodel/submit";
export const QUERY_PATH = "/api/v3/auc/bigmodel/query";
export const FLASH_PATH = "/api/v3/auc/bigmodel/recognize/flash";
export const FLASH_RESOURCE_ID = "volc.bigasr.auc_turbo";

export const STATUS_SUCCESS = "20000000";
export const STATUS_PROCESSING = ["20000001", "20000002"];

const CONNECTION_TYPE = "volcano-asr";

export interface VolcanoAsrConn {
  apiKey: string;
  baseUrl: string;
  resourceId: string;
}

function firstResolvedConfig(container?: unknown): Record<string, unknown> {
  const c = container as
    | { _resolvedConnections?: Array<{ config?: Record<string, unknown> }> }
    | undefined;
  return c?._resolvedConnections?.[0]?.config ?? {};
}

export function runConfigOf(exeConfig?: unknown): Record<string, unknown> {
  return (
    ((exeConfig as { configurable?: { runConfig?: Record<string, unknown> } })?.configurable
      ?.runConfig as Record<string, unknown>) ?? {}
  );
}

function tenantFromExeConfig(exeConfig?: unknown): string {
  const t = runConfigOf(exeConfig).tenantId;
  if (typeof t === "string" && t.trim()) return t.trim();
  throw new Error("tenant context is missing");
}

export function normalizeConnection(config: Record<string, unknown>): VolcanoAsrConn {
  const apiKeyRaw = config.apiKey;
  const envKey = process.env.VOLC_ASR_API_KEY?.trim();
  const apiKey =
    typeof apiKeyRaw === "string" && apiKeyRaw.trim() ? apiKeyRaw.trim() : envKey || "";
  const baseUrlRaw = config.baseUrl;
  const baseUrl = (
    typeof baseUrlRaw === "string" && baseUrlRaw.trim() ? baseUrlRaw.trim() : DEFAULT_BASE_URL
  ).replace(/\/+$/, "");
  const resourceIdRaw = config.resourceId;
  const resourceId =
    typeof resourceIdRaw === "string" && resourceIdRaw.trim()
      ? resourceIdRaw.trim()
      : DEFAULT_RESOURCE_ID;
  return { apiKey, baseUrl, resourceId };
}

export function connectionFromConfig(config: Record<string, unknown>): VolcanoAsrConn {
  return normalizeConnection(config ?? {});
}

function noConnectionHint(connectionType: string, tenantId: string): string {
  return (
    `No "${connectionType}" connection is configured for tenant "${tenantId}". ` +
    `Add a connection of type "${connectionType}" and select it (connections) or enable connectAll ` +
    `in the agent's middleware config.`
  );
}

/**
 * Resolve the Volcano ASR connection for a tool invocation.
 *
 * Prefers a host-injected, pre-resolved connection (`_resolvedConnections` in the
 * plugin config or `runConfig`). Otherwise resolves the selected `volcano-asr`
 * connection from the tenant-scoped Connection Store. Falls back to env/default
 * only when no selector is configured.
 */
export async function resolveConnection(
  pluginConfig?: unknown,
  exeConfig?: unknown,
): Promise<VolcanoAsrConn> {
  const preResolved = {
    ...firstResolvedConfig(pluginConfig),
    ...firstResolvedConfig(runConfigOf(exeConfig)),
  };
  if (Object.keys(preResolved).length > 0) return normalizeConnection(preResolved);

  const selector = (pluginConfig ?? {}) as {
    connectionType?: unknown;
    connections?: unknown;
    connectAll?: unknown;
  };
  const connectionType =
    typeof selector.connectionType === "string" ? selector.connectionType : CONNECTION_TYPE;
  const connections = Array.isArray(selector.connections)
    ? selector.connections.filter((k): k is string => typeof k === "string")
    : [];
  const connectAll = selector.connectAll === true;

  if (!connectAll && connections.length === 0) return normalizeConnection({});

  const tenantId = tenantFromExeConfig(exeConfig);
  const { resolvePluginConnections } = await import("@axiom-lattice/core");
  const resolved = await resolvePluginConnections(
    connectionType,
    { connections, connectAll },
    { tenantId },
  );
  if (resolved.length === 0) throw new Error(noConnectionHint(connectionType, tenantId));
  return normalizeConnection(resolved[0].config);
}

export function newRequestId(): string {
  return randomUUID();
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface CallApiOptions {
  baseUrl: string;
  apiKey: string;
  resourceId: string;
  path: string;
  body?: unknown;
  requestId: string;
  sequence?: number;
}

export interface CallApiResult {
  httpStatus: number;
  statusCode: string;
  message: string;
  requestId: string;
  body?: unknown;
  rawBody: string;
}

export async function callApi(opts: CallApiOptions): Promise<CallApiResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Api-Key": opts.apiKey,
    "X-Api-Resource-Id": opts.resourceId,
    "X-Api-Request-Id": opts.requestId,
  };
  if (opts.sequence !== undefined) headers["X-Api-Sequence"] = String(opts.sequence);

  const res = await fetch(opts.baseUrl + opts.path, {
    method: "POST",
    headers,
    body: JSON.stringify(opts.body ?? {}),
  });

  const statusCode = res.headers.get("x-api-status-code") ?? "";
  const message = res.headers.get("x-api-message") ?? "";
  const respRequestId = res.headers.get("x-api-request-id") ?? opts.requestId;
  const rawBody = await res.text();

  let body: unknown;
  try {
    body = rawBody ? JSON.parse(rawBody) : undefined;
  } catch {
    body = undefined;
  }

  return { httpStatus: res.status, statusCode, message, requestId: respRequestId, body, rawBody };
}

export function isSuccess(statusCode: string): boolean {
  return statusCode === STATUS_SUCCESS;
}

export function isProcessing(statusCode: string): boolean {
  return STATUS_PROCESSING.includes(statusCode);
}

export function extractText(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const result = (body as { result?: unknown }).result ?? body;
  const text = (result as { text?: unknown }).text;
  if (typeof text === "string") return text;
  const nested = (result as { result?: { text?: unknown } }).result;
  if (nested && typeof nested.text === "string") return nested.text;
  const utterances = (result as { utterances?: unknown }).utterances;
  if (Array.isArray(utterances)) {
    return utterances
      .map((u) =>
        u && typeof (u as { text?: unknown }).text === "string" ? (u as { text: string }).text : "",
      )
      .filter(Boolean)
      .join("");
  }
  return "";
}

export function extractUtterances(body: unknown): unknown[] | undefined {
  if (!body || typeof body !== "object") return undefined;
  const result = (body as { result?: unknown }).result ?? body;
  const direct = (result as { utterances?: unknown }).utterances;
  if (Array.isArray(direct)) return direct;
  const nested = (result as { result?: { utterances?: unknown } }).result;
  if (nested && Array.isArray(nested.utterances)) return nested.utterances;
  return undefined;
}

export function errorResult(err: unknown): string {
  return JSON.stringify({
    ok: false,
    code: "ERROR",
    message: err instanceof Error ? err.message : String(err),
  });
}

export function maskSecret(value: string): string {
  const s = String(value || "");
  if (!s) return "(empty)";
  if (s.length <= 8) return s.slice(0, 2) + "***";
  return s.slice(0, 4) + "***" + s.slice(-4) + " (len=" + s.length + ")";
}
