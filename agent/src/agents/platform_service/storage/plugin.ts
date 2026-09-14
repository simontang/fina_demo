import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import {
  storageDelete,
  storageGetDownloadUrl,
  storageGetMetadata,
  storageList,
  storageUpload,
} from "./executors";

const UUID_RE = /^[0-9a-fA-F]{32}$/;

const SCHEMAS = {
  upload: z.object({
    sandboxPath: z.string().describe("沙盒内的文件路径，如 /project/out/report.csv"),
    logicalPath: z.string().optional().describe("逻辑目录（元数据），如 reports/2026-09"),
    fileName: z.string().optional().describe("存储文件名，默认取 sandboxPath 的 basename"),
    fileCategory: z.string().optional(),
    usage: z.string().optional(),
    meta: z.string().optional().describe("自由 JSON 字符串"),
  }),
  list: z.object({
    path: z.string().optional(),
    q: z.string().optional().describe("文件名子串（忽略大小写）"),
    recursive: z.boolean().optional(),
    fileCategory: z.string().optional(),
    usage: z.string().optional(),
    from: z.string().optional().describe("YYYY-MM-DD"),
    to: z.string().optional().describe("YYYY-MM-DD"),
    page: z.number().int().optional(),
    size: z.number().int().optional(),
  }),
  metadata: z.object({ uuid: z.string().regex(UUID_RE).describe("32 位十六进制 uuid") }),
  presign: z.object({
    uuid: z.string().regex(UUID_RE),
    ttlSeconds: z.number().int().optional(),
  }),
  delete: z.object({
    uuid: z.string().regex(UUID_RE),
    confirm: z.boolean().optional().describe("必须为 true 才执行；否则返回确认提示"),
  }),
};

export const storagePlugin: Plugin = {
  meta: {
    type: "storage",
    name: "统一文件资源存储",
    description:
      "租户级持久资源库：uuid 寻址、内容去重、版本化、预签名下载链接。与项目沙盒的临时文件不同——写临时文件用 sandbox 文件工具。",
    version: "1.0.0",
    configSchema: {
      type: "object",
      properties: {
        connections: { type: "array", title: "连接", widget: "connectionSelect" },
      },
    },
    defaultConfig: { connections: [] },
    openExpose: [
      { name: "storage_list", readOnly: true },
      { name: "storage_get_metadata", readOnly: true },
      { name: "storage_get_download_url", readOnly: true },
      { name: "storage_delete", destructive: true },
    ],
  },
  connection: {
    fields: [
      {
        key: "baseUrl",
        type: "string",
        title: "Base URL",
        widget: "input",
        required: true,
        default: process.env.PLATFORM_SERVICE_URL ?? "http://127.0.0.1:5707",
      },
      {
        key: "apiKey",
        type: "password",
        title: "API Key",
        widget: "password",
        default: process.env.FILE_SERVICE_API_KEY ?? "",
        helpText: "对应 FILE_SERVICE_API_KEY；留空表示服务端未启用校验",
      },
    ],
    test: async (config) => {
      try {
        const base = String(config.baseUrl ?? "").replace(/\/+$/, "");
        const res = await fetch(`${base}/actuator/health`);
        return { ok: res.ok, message: res.ok ? "连接成功" : `HTTP ${res.status}` };
      } catch (err) {
        return { ok: false, message: err instanceof Error ? err.message : String(err) };
      }
    },
  },
  middleware: (rawConfig) =>
    createMiddleware({
      name: "Storage",
      tools: [
        tool((input, exeConfig) => storageUpload(input as never, exeConfig, rawConfig), {
          name: "storage_upload",
          description:
            "把 agent 沙盒里的文件上传到统一存储。输入是沙盒路径，不是文件内容。成功后返回 FileReceipt（uuid/fullPath/sha256/size…）。",
          schema: SCHEMAS.upload,
        } as never),
        tool((input, exeConfig) => storageList(input as never, exeConfig, rawConfig), {
          name: "storage_list",
          description: "列出统一存储中的文件，可按目录/名称/属性/时间过滤，分页返回。",
          schema: SCHEMAS.list,
        } as never),
        tool((input, exeConfig) => storageGetMetadata(input as never, exeConfig, rawConfig), {
          name: "storage_get_metadata",
          description: "按 uuid 获取文件元数据（不下载内容）。",
          schema: SCHEMAS.metadata,
        } as never),
        tool((input, exeConfig) => storageGetDownloadUrl(input as never, exeConfig, rawConfig), {
          name: "storage_get_download_url",
          description: "获取文件的限时下载链接。链接即凭据：拿到 URL 的任何人都可下载，勿分享。",
          schema: SCHEMAS.presign,
        } as never),
        tool((input, exeConfig) => storageDelete(input as never, exeConfig, rawConfig), {
          name: "storage_delete",
          description:
            "软删除文件的某一个版本（该 uuid 对应版本），非删除整个逻辑文件。必须用户确认后传 confirm:true。",
          schema: SCHEMAS.delete,
        } as never),
      ],
    }),
};

PluginRegistry.register(storagePlugin);
