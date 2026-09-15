import { z } from "zod";
import type { Config } from "./types";

const EnvSchema = z.object({
  PORT: z.coerce.number().int().positive().default(5708),
  GATEWAY_API_KEYS: z.string().default(""),
  AUTH_DISABLED: z.enum(["true", "false"]).default("false"),
  AUTH_DEV_TENANT: z.string().default("estee_lauder"),
  PLATFORM_FILES_URL: z.string().url().default("http://127.0.0.1:5707/api/v1/files"),
  FILE_SERVICE_API_KEY: z.string().optional(),
  GATEWAY_MAX_UPLOAD_BYTES: z.coerce.number().int().positive().default(52428800),
  A2A_BASE_URL: z.string().url().default("http://127.0.0.1:5702"),
  A2A_API_KEY: z.string().optional(),
  A2A_VOICE_TAGGING_ASSISTANT_ID: z.string().optional(),
  A2A_VOICE_TAGGING_FILE_UUID: z.string().optional(),
  A2A_TRIGGER_TIMEOUT_MS: z.coerce.number().int().positive().default(600000),
  A2A_MESSAGE_TEMPLATE: z.string().optional(),
  MCP_SERVER_URL: z.string().url().default("http://127.0.0.1:5702/open/mcp"),
  MCP_API_KEY: z.string().optional(),
  UPSTREAM_TIMEOUT_MS: z.coerce.number().int().positive().default(30000),
});

export function parseApiKeys(raw: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const part of raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)) {
    const idx = part.indexOf(":");
    if (idx <= 0 || idx === part.length - 1) {
      throw new Error(`Invalid GATEWAY_API_KEYS entry (expected key:tenant): ${part}`);
    }
    map.set(part.slice(0, idx), part.slice(idx + 1));
  }
  return map;
}

function required(value: string | undefined, name: string): string {
  if (!value || value.trim() === "") throw new Error(`${name} is required`);
  return value;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const parsed = EnvSchema.parse(env);
  return {
    port: parsed.PORT,
    gatewayApiKeys: parseApiKeys(parsed.GATEWAY_API_KEYS),
    authDisabled: parsed.AUTH_DISABLED === "true",
    authDevTenant: parsed.AUTH_DEV_TENANT,
    platformFilesUrl: parsed.PLATFORM_FILES_URL.replace(/\/$/, ""),
    fileServiceApiKey: parsed.FILE_SERVICE_API_KEY,
    maxUploadBytes: parsed.GATEWAY_MAX_UPLOAD_BYTES,
    a2aBaseUrl: parsed.A2A_BASE_URL.replace(/\/$/, ""),
    a2aApiKey: required(parsed.A2A_API_KEY, "A2A_API_KEY"),
    a2aVoiceTaggingAssistantId: parsed.A2A_VOICE_TAGGING_ASSISTANT_ID,
    a2aVoiceTaggingFileUuid: parsed.A2A_VOICE_TAGGING_FILE_UUID,
    a2aTriggerTimeoutMs: parsed.A2A_TRIGGER_TIMEOUT_MS,
    a2aMessageTemplate: parsed.A2A_MESSAGE_TEMPLATE,
    mcpServerUrl: parsed.MCP_SERVER_URL,
    mcpApiKey: required(parsed.MCP_API_KEY, "MCP_API_KEY"),
    upstreamTimeoutMs: parsed.UPSTREAM_TIMEOUT_MS,
  };
}
