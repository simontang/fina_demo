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
  filters?: Array<{ field: string; op?: string; value: unknown }>;
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

export interface StoreInput {
  storeKey: string;
  name?: string;
  description?: string;
  jdbcUrl: string;
  username: string;
  password: string;
  status?: number;
}

export interface StoreGrantInput {
  storeKey: string;
  granteeKey?: string;
  canRead?: boolean;
  canWrite?: boolean;
  canManage?: boolean;
  status?: number;
}

function connection(rawConfig: unknown, exeConfig: unknown): PlatformServiceConn {
  return resolveConnection(rawConfig, exeConfig);
}

export function boConnectionKey(conn: PlatformServiceConn): string {
  const key = conn.boConnectionKey;
  if (!key) {
    throw new Error("Business Object connection key is required in the platform-service connection");
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
      conn: connection(rawConfig, exeConfig),
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
    const conn = connection(rawConfig, exeConfig);
    const result = await request({
      conn,
      method,
      path,
      headers: { "X-BO-Connection-Key": boConnectionKey(conn) },
      ...(json === undefined ? {} : { json }),
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boStoreList(_input: {}, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "GET", "/api/v1/bo/stores");
}

export async function boStoreCreate(input: StoreInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "POST", "/api/v1/bo/stores", input);
}

export async function boStoreTest(input: { storeKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "POST", `/api/v1/bo/stores/${encodeURIComponent(input.storeKey)}/test`);
}

export async function boStoreGrantList(input: { storeKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callPlatform(rawConfig, exeConfig, "GET", `/api/v1/bo/stores/${encodeURIComponent(input.storeKey)}/grants`);
}

export async function boStoreGrantUpsert(input: StoreGrantInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  const { storeKey, ...body } = input;
  return callPlatform(rawConfig, exeConfig, "POST", `/api/v1/bo/stores/${encodeURIComponent(storeKey)}/grants`, body);
}

export async function boObjectList(_input: {}, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  return callBoObjectApi(rawConfig, exeConfig, "GET", "/api/v1/bo/objects");
}

export async function boObjectGet(input: { objectKey: string }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "GET", `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boObjectCreate(input: ObjectDefinitionInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
    requireMaybeObjectAllowed(conn, input);
    return await callBoObjectApi(rawConfig, exeConfig, "POST", "/api/v1/bo/objects", input);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boObjectUpdate(input: ObjectDefinitionInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
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
    const conn = connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "DELETE", `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordCreate(input: RecordInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
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
    const conn = connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "GET",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/${encodeURIComponent(input.id)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordUpdate(input: RecordIdInput & { data: JsonObject }, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
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
    const conn = connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    return await callBoObjectApi(rawConfig, exeConfig, "DELETE",
      `/api/v1/bo/objects/${encodeURIComponent(input.objectKey)}/records/${encodeURIComponent(input.id)}`);
  } catch (err) {
    return errorResult(err);
  }
}

export async function boRecordQuery(input: QueryRecordsInput, exeConfig: unknown, rawConfig: unknown): Promise<string> {
  try {
    const conn = connection(rawConfig, exeConfig);
    requireObjectAllowed(conn, input.objectKey);
    const { objectKey, ...body } = input;
    return await callBoObjectApi(rawConfig, exeConfig, "POST",
      `/api/v1/bo/objects/${encodeURIComponent(objectKey)}/records/query`,
      body);
  } catch (err) {
    return errorResult(err);
  }
}
