import { randomUUID } from "node:crypto";
import { getSandBoxManager } from "@axiom-lattice/core";
import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

const isAscii = (v?: string) => v === undefined || /^[\x00-\x7F]*$/.test(v);

export const UUID_PATTERN = /^[0-9a-fA-F]{32}$/;

const MIME_BY_EXT: Record<string, string> = {
  csv: "text/csv",
  txt: "text/plain",
  json: "application/json",
  tsv: "text/tab-separated-values",
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  zip: "application/zip",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
};

function mimeFor(fileName: string): string {
  const ext = fileName.split(".").pop()?.toLowerCase() ?? "";
  return MIME_BY_EXT[ext] ?? "application/octet-stream";
}

function maxUploadBytes(): number {
  const raw = Number(process.env.STORAGE_TOOL_MAX_UPLOAD_BYTES);
  return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_MAX_UPLOAD_BYTES;
}

interface SandboxFileHandle {
  path: string;
  size?: number;
}

interface SandboxLike {
  file: {
    downloadFile: (args: { file: string }) => Promise<Buffer>;
    listPath?: (
      path: string,
      options?: { recursive?: boolean },
    ) => Promise<{ files: Array<SandboxFileHandle> }>;
  };
}

async function resolveSandbox(exeConfig: unknown): Promise<SandboxLike> {
  const rc = ((exeConfig as { configurable?: { runConfig?: Record<string, unknown> } })
    ?.configurable?.runConfig ?? {}) as Record<string, unknown>;
  return getSandBoxManager().getSandboxFromConfig({
    assistant_id: (rc.assistant_id as string) || "",
    thread_id: (rc.thread_id as string) || "",
    tenantId: rc.tenantId as string | undefined,
    workspaceId: rc.workspaceId as string | undefined,
    projectId: rc.projectId as string | undefined,
  });
}

async function sandboxFileSize(
  sandbox: SandboxLike,
  sandboxPath: string,
): Promise<number | undefined> {
  if (!sandbox.file.listPath) return undefined;
  const idx = sandboxPath.lastIndexOf("/");
  const dir = idx > 0 ? sandboxPath.slice(0, idx) : "/";
  try {
    const { files } = await sandbox.file.listPath(dir);
    const entry = files.find(
      (f) => f.path === sandboxPath || f.path.endsWith("/" + sandboxPath.slice(idx + 1)),
    );
    return entry?.size;
  } catch {
    return undefined;
  }
}

export interface StorageUploadInput {
  sandboxPath: string;
  logicalPath?: string;
  fileName?: string;
  fileCategory?: string;
  usage?: string;
  meta?: string;
}

export async function storageUpload(
  input: StorageUploadInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const tenantId = tenantFromExeConfig(exeConfig);
    const conn = await resolveConnection(rawConfig, exeConfig);
    const fileName = input.fileName?.trim() || input.sandboxPath.split("/").pop() || "file";
    if (
      !isAscii(fileName) ||
      !isAscii(input.logicalPath) ||
      !isAscii(input.fileCategory) ||
      !isAscii(input.usage) ||
      !isAscii(input.meta)
    ) {
      return JSON.stringify({
        ok: false,
        code: "NON_ASCII_METADATA",
        message:
          "Metadata (file name/path/meta) currently supports ASCII only; non-ASCII characters will fail and require server-side decoding support",
      });
    }
    const cap = maxUploadBytes();
    const sandbox = await resolveSandbox(exeConfig);
    const known = await sandboxFileSize(sandbox, input.sandboxPath);
    if (known !== undefined && known > cap) {
      return JSON.stringify({
        ok: false,
        code: "FILE_TOO_LARGE",
        message: `file is ${known} bytes; limit is ${cap}`,
      });
    }
    const bytes = await sandbox.file.downloadFile({ file: input.sandboxPath });
    if (bytes.length > cap) {
      return JSON.stringify({
        ok: false,
        code: "FILE_TOO_LARGE",
        message: `file is ${bytes.length} bytes; limit is ${cap}`,
      });
    }
    const uuid = randomUUID().replace(/-/g, "");
    const result = await request({
      conn,
      tenantId,
      method: "PUT",
      path: `/api/v1/files/${uuid}`,
      body: bytes,
      contentType: mimeFor(fileName),
      headers: {
        "X-File-Name": fileName,
        ...(input.logicalPath ? { "X-File-Path": input.logicalPath } : {}),
        ...(input.fileCategory ? { "X-File-Category": input.fileCategory } : {}),
        ...(input.usage ? { "X-File-Usage": input.usage } : {}),
        ...(input.meta ? { "X-File-Meta": input.meta } : {}),
      },
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export interface StorageListInput {
  path?: string;
  q?: string;
  recursive?: boolean;
  fileCategory?: string;
  usage?: string;
  from?: string;
  to?: string;
  page?: number;
  size?: number;
}

export async function storageList(
  input: StorageListInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: "/api/v1/files",
      query: { ...input },
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function storageGetMetadata(
  input: { uuid: string },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  if (!UUID_PATTERN.test(input.uuid)) {
    return JSON.stringify({ ok: false, code: "BAD_REQUEST", message: "uuid must be 32 hex characters" });
  }
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: `/api/v1/files/${input.uuid}`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function storageGetDownloadUrl(
  input: { uuid: string; ttlSeconds?: number },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  if (!UUID_PATTERN.test(input.uuid)) {
    return JSON.stringify({ ok: false, code: "BAD_REQUEST", message: "uuid must be 32 hex characters" });
  }
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "POST",
      path: "/api/v1/files/presign",
      json: { uuid: input.uuid, ttlSeconds: input.ttlSeconds },
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function storageDelete(
  input: { uuid: string; confirm?: boolean },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message:
        "Explicit user confirmation is required before deleting; ask the user to confirm, then retry with confirm:true.",
    });
  }
  if (!UUID_PATTERN.test(input.uuid)) {
    return JSON.stringify({ ok: false, code: "BAD_REQUEST", message: "uuid must be 32 hex characters" });
  }
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "DELETE",
      path: `/api/v1/files/${input.uuid}`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}
