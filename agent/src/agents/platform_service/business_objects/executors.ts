import {
  errorResult,
  request,
  resolveConnection,
  type PlatformServiceConn,
} from "../client";

export type JsonObject = Record<string, unknown>;

export interface FieldDefinition {
  key: string;
  type: "string" | "text" | "integer" | "long" | "decimal" | "boolean" | "date" | "datetime" | "json";
  required?: boolean;
  maxLength?: number;
  precision?: number;
  scale?: number;
  description?: string;
}

export interface IndexDefinition {
  name?: string;
  fields: string[];
  unique?: boolean;
}

export interface ObjectDefinitionInput {
  storeKey?: string;
  objectKey: string;
  displayName?: string;
  description?: string;
  fields?: FieldDefinition[];
  indexes?: IndexDefinition[];
  status?: number;
}

export interface QueryRecordsInput {
  objectKey: string;
  filters?: Array<{ field: string; op?: string; value?: unknown }>;
  sort?: Array<{ field: string; direction?: "asc" | "desc" }>;
  page?: number;
  pageSize?: number;
}

export interface RecordInput {
  objectKey: string;
  data: JsonObject;
}

export interface RecordIdInput {
  objectKey: string;
  id: string;
}

export interface RecordBatchInput {
  objectKey: string;
  records: JsonObject[];
}

export interface RecordIdBatchInput {
  objectKey: string;
  ids: string[];
  confirm?: boolean;
}

export interface StoreInput {
  storeKey: string;
  name?: string;
  description?: string;
  jdbcUrl: string;
  username: string;
  password: string;
  status?: number;
}

export interface StoreKeyInput {
  storeKey: string;
  keyName: string;
  rawKey?: string;
  permissions?: Array<"READ" | "WRITE" | "MANAGE">;
  status?: number;
}

export interface StoreKeyUpdateInput {
  storeKey: string;
  keyId: number;
  keyName?: string;
  rawKey?: string;
  permissions?: Array<"READ" | "WRITE" | "MANAGE">;
  status?: number;
}

async function connection(rawConfig: unknown, exeConfig: unknown): Promise<PlatformServiceConn> {
  return resolveConnection(rawConfig, exeConfig);
}

export function boStoreKey(conn: PlatformServiceConn): string {
  const key = conn.boStoreKey;
  if (!key) {
    throw new Error("Business Object store key is required in the platform-service connection");
  }
  return key;
}

function requireObjectAllowed(conn: PlatformServiceConn, objectKey: string): void {
  if (conn.selectedEntities.length === 0) return;
  if (!conn.selectedEntities.includes(objectKey)) {
    throw new Error(`business object "${objectKey}" is not allowed by the selected connection scope`);
  }
}

function requireMaybeObjectAllowed(conn: PlatformServiceConn, input: { objectKey?: string }): void {
  if (input.objectKey) requireObjectAllowed(conn, input.objectKey);
}

async function callPlatform(
  rawConfig: unknown,
  exeConfig: unknown,
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  json?: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: await connection(rawConfig, exeConfig),
      method,
      path,
      ...(json === undefined ? {} : { json }),
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

async function callBoObjectApi(
  rawConfig: unknown,
  exeConfig: unknown,
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE",
  path: string,
  json?: unknown,
): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    const result = await request({
      conn,
      method,
      path,
      headers: { "X-BO-Connection-Key": boStoreKey(conn) },
      ...(json === undefined ? {} : { json }),
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

/**
 * Store fields safe to surface to the agent. Connection credentials
 * (`jdbcUrl`, `username`, `id`) are intentionally omitted so they never enter
 * the model context or conversation history.
 */
const STORE_PUBLIC_FIELDS = ["storeKey", "name", "description", "schemaName", "status"] as const;

function sanitizeStoreResult(raw: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return raw;
  }
  if (parsed && typeof parsed === "object" && (parsed as { ok?: unknown }).ok === false) {
    return raw;
  }
  const pick = (store: unknown): Record<string, unknown> => {
    if (!store || typeof store !== "object" || Array.isArray(store)) return {};
    const record = store as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of STORE_PUBLIC_FIELDS) {
      if (key in record) out[key] = record[key];
    }
    return out;
  };
  return JSON.stringify(Array.isArray(parsed) ? parsed.map(pick) : pick(parsed));
}

export async function boStoreList(_input: {}, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return sanitizeStoreResult(await callPlatform(rawConfig, exeConfig, "GET", "/api/v1/bo/stores"));
}

export async function boStoreCreate(input: StoreInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return sanitizeStoreResult(await callPlatform(rawConfig, exeConfig, "POST", "/api/v1/bo/stores", input));
}

export async function boStoreTest(input: { storeKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "POST", `/api/v1/bo/stores/${encodeURIComponent(input.storeKey)}/test`);
}

export async function boStoreKeyList(input: { storeKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "GET", `/api/v1/bo/stores/${encodeURIComponent(input.storeKey)}/keys`);
}

export async function boStoreKeyCreate(input: StoreKeyInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  const { storeKey, ...body } = input;
  return callPlatform(rawConfig, exeConfig, "POST", `/api/v1/bo/stores/${encodeURIComponent(storeKey)}/keys`, body);
}

export async function boStoreKeyUpdate(input: StoreKeyUpdateInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  const { storeKey, keyId, ...body } = input;
  return callPlatform(rawConfig, exeConfig, "PUT", `/api/v1/bo/stores/${encodeURIComponent(storeKey)}/keys/${encodeURIComponent(keyId)}`, body);
}

export async function boStoreKeyDelete(input: { storeKey: string; keyId: number; confirm?: boolean }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message: "Explicit user confirmation is required before deleting a store key.",
    });
  }
  return callPlatform(rawConfig, exeConfig, "DELETE", `/api/v1/bo/stores/${encodeURIComponent(input.storeKey)}/keys/${encodeURIComponent(input.keyId)}`);
}

export async function boObjectList(_input: {}, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callBoObjectApi(rawConfig, exeConfig, "GET", "/api/v1/bo/objects");
}

export async function boObjectGet(input: { objectKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "GET", `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boObjectCreate(input: ObjectDefinitionInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireMaybeObjectAllowed(conn, input);
    return await callBoObjectApi(rawConfig, exeConfig, "POST", "/api/v1/bo/objects", input);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boObjectUpdate(input: ObjectDefinitionInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "PUT", `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}`, input);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boObjectDelete(input: { objectKey: string; confirm?: boolean }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message: "Explicit user confirmation is required before deleting an object definition.",
    });
  }
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "DELETE", `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordCreate(input: RecordInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "POST",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records`,
      { data: input.data });
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordGet(input: RecordIdInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "GET",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/${encodeURIComponent(input.id)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordUpdate(input: RecordIdInput & { data: JsonObject }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "PATCH",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/${encodeURIComponent(input.id)}`,
      { data: input.data });
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordDelete(input: RecordIdInput & { confirm?: boolean }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message: "Explicit user confirmation is required before deleting a record.",
    });
  }
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "DELETE",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/${encodeURIComponent(input.id)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordCreateMany(input: RecordBatchInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "POST",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/batch`,
      { records: input.records });
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordDeleteMany(input: RecordIdBatchInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message: "Explicit user confirmation is required before deleting records.",
    });
  }
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "POST",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/batch-delete`,
      { ids: input.ids });
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordQuery(input: QueryRecordsInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = await connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    const { objectKey, ...body } = input;
    return await callBoObjectApi(rawConfig, exeConfig, "POST",
      `/api/v1/bo/objects/${encodeURIComponent(objectKey)}/records/query`,
      body);
  } catch (err) {
    return errorResult(err);
  }
}
