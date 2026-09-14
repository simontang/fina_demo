import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { platformServiceConnection } from "../connection";
import {
  UUID_PATTERN,
  storageDelete,
  storageGetDownloadUrl,
  storageGetMetadata,
  storageList,
  storageUpload,
} from "./executors";

const SCHEMAS = {
  upload: z.object({
    sandboxPath: z.string().describe("Path to the file inside the sandbox, e.g. /project/out/report.csv"),
    logicalPath: z.string().optional().describe("Logical directory (metadata), e.g. reports/2026-09"),
    fileName: z
      .string()
      .optional()
      .describe("Stored file name; defaults to the basename of sandboxPath"),
    fileCategory: z.string().optional(),
    usage: z.string().optional(),
    meta: z.string().optional().describe("Free-form JSON string"),
  }),
  list: z.object({
    path: z.string().optional(),
    q: z.string().optional().describe("File name substring (case-insensitive)"),
    recursive: z.boolean().optional(),
    fileCategory: z.string().optional(),
    usage: z.string().optional(),
    from: z.string().optional().describe("YYYY-MM-DD"),
    to: z.string().optional().describe("YYYY-MM-DD"),
    page: z.number().int().optional(),
    size: z.number().int().optional(),
  }),
  metadata: z.object({ uuid: z.string().regex(UUID_PATTERN).describe("32-character hexadecimal uuid") }),
  presign: z.object({
    uuid: z.string().regex(UUID_PATTERN),
    ttlSeconds: z.number().int().optional(),
  }),
  delete: z.object({
    uuid: z.string().regex(UUID_PATTERN),
    confirm: z.boolean().optional().describe("Must be true to execute; otherwise a confirmation prompt is returned"),
  }),
};

export const storagePlugin: Plugin = {
  meta: {
    type: "storage",
    name: "Unified File Storage",
    description:
      "Tenant-scoped durable object store: uuid-addressed, content-deduplicated, versioned, with presigned download links. Distinct from the project sandbox's scratch files—use the sandbox file tools to write temporary files.",
    version: "1.0.0",
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "Connect all available connections" },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
    openExpose: [
      { name: "storage_list", readOnly: true },
      { name: "storage_get_metadata", readOnly: true },
      { name: "storage_get_download_url", readOnly: true },
      { name: "storage_delete", destructive: true },
    ],
  },
  connection: platformServiceConnection,
  middleware: (rawConfig) =>
    createMiddleware({
      name: "Storage",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.upload>, exeConfig) =>
            storageUpload(input, exeConfig, rawConfig),
          {
            name: "storage_upload",
            description:
              "Upload a file from the agent sandbox to unified storage. The input is a sandbox path, not file contents. On success returns a FileReceipt (uuid/fullPath/sha256/size…). (Metadata currently supports ASCII only; non-ASCII file names/paths require server-side decoding support)",
            schema: SCHEMAS.upload,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.list>, exeConfig) =>
            storageList(input, exeConfig, rawConfig),
          {
            name: "storage_list",
            description:
              "List files in unified storage, filterable by directory/name/attributes/time, returned paginated.",
            schema: SCHEMAS.list,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.metadata>, exeConfig) =>
            storageGetMetadata(input, exeConfig, rawConfig),
          {
            name: "storage_get_metadata",
            description: "Get file metadata by uuid (without downloading contents).",
            schema: SCHEMAS.metadata,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.presign>, exeConfig) =>
            storageGetDownloadUrl(input, exeConfig, rawConfig),
          {
            name: "storage_get_download_url",
            description:
              "Get a time-limited download link for a file. The link is a credential: anyone who has the URL can download it, so do not share it.",
            schema: SCHEMAS.presign,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.delete>, exeConfig) =>
            storageDelete(input, exeConfig, rawConfig),
          {
            name: "storage_delete",
            description:
              "Soft-delete one version of a file (the version corresponding to that uuid), not the entire logical file. Requires user confirmation, then pass confirm:true.",
            schema: SCHEMAS.delete,
          },
        ),
      ],
    }),
};

PluginRegistry.register(storagePlugin);
