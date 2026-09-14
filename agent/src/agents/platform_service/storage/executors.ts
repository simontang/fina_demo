import { randomUUID } from "node:crypto";
import { getSandBoxManager } from "@axiom-lattice/core";
import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

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

async function downloadFromSandbox(exeConfig: unknown, sandboxPath: string): Promise<Buffer> {
  const rc = ((exeConfig as { configurable?: { runConfig?: Record<string, unknown> } })
    ?.configurable?.runConfig ?? {}) as Record<string, unknown>;
  const sandbox = await getSandBoxManager().getSandboxFromConfig({
    assistant_id: (rc.assistant_id as string) || "",
    thread_id: (rc.thread_id as string) || "",
    tenantId: rc.tenantId as string | undefined,
    workspaceId: rc.workspaceId as string | undefined,
    projectId: rc.projectId as string | undefined,
  });
  return sandbox.file.downloadFile({ file: sandboxPath });
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
    const conn = resolveConnection(rawConfig, exeConfig);
    const fileName = input.fileName?.trim() || input.sandboxPath.split("/").pop() || "file";
    const bytes = await downloadFromSandbox(exeConfig, input.sandboxPath);
    if (bytes.length > maxUploadBytes()) {
      return JSON.stringify({
        ok: false,
        code: "FILE_TOO_LARGE",
        message: `file is ${bytes.length} bytes; limit is ${maxUploadBytes()}`,
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
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}
