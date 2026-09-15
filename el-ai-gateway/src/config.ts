import { z } from "zod";
import type { Config } from "./types";

// Platform-issued key used as both the inbound gateway API key and the MCP key.
// Override via env; baked default is for local/demo so the service boots with no config.
const DEFAULT_PLATFORM_KEY = "a2a_f47409c1c94f49ffad37adea3259646f";

const EnvSchema = z.object({
  PORT: z.coerce.number().int().positive().default(5708),
  GATEWAY_API_KEYS: z.string().default(`${DEFAULT_PLATFORM_KEY}:estee_lauder`),
  AUTH_DISABLED: z.enum(["true", "false"]).default("false"),
  AUTH_DEV_TENANT: z.string().default("estee_lauder"),
  PLATFORM_FILES_URL: z.string().url().default("http://127.0.0.1:5707/api/v1/files"),
  FILE_SERVICE_API_KEY: z.string().optional(),
  GATEWAY_MAX_UPLOAD_BYTES: z.coerce.number().int().positive().default(52428800),

  AGENT_RUNS_URL: z.string().url().default("http://127.0.0.1:5702/api/runs"),
  AGENT_AUTH_URL: z.string().url().default("http://127.0.0.1:5702/api/auth/login"),
  AGENT_LOGIN_EMAIL: z.string().optional(),
  AGENT_LOGIN_PASSWORD: z.string().optional(),
  AGENT_TENANT_ID: z.string().default("estee_lauder"),
  AGENT_WORKSPACE_ID: z.string().default("default-workspace"),
  AGENT_PROJECT_ID: z.string().default("default"),
  VOICE_TAGGING_ASSISTANT_ID: z.string().default("voice-tagging-agent"),
  VOICE_TAGGING_FILE_UUID: z.string().optional(),
  VOICE_TAGGING_MESSAGE_TEMPLATE: z.string().optional(),
  AGENT_TRIGGER_TIMEOUT_MS: z.coerce.number().int().positive().default(600000),

  MCP_SERVER_URL: z.string().url().default("http://127.0.0.1:5702/open/mcp"),
  MCP_API_KEY: z.string().default(DEFAULT_PLATFORM_KEY),
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
    agentRunsUrl: parsed.AGENT_RUNS_URL,
    agentAuthUrl: parsed.AGENT_AUTH_URL,
    agentLoginEmail: parsed.AGENT_LOGIN_EMAIL,
    agentLoginPassword: parsed.AGENT_LOGIN_PASSWORD,
    agentTenantId: parsed.AGENT_TENANT_ID,
    agentWorkspaceId: parsed.AGENT_WORKSPACE_ID,
    agentProjectId: parsed.AGENT_PROJECT_ID,
    voiceTaggingAssistantId: parsed.VOICE_TAGGING_ASSISTANT_ID,
    voiceTaggingFileUuid: parsed.VOICE_TAGGING_FILE_UUID,
    voiceTaggingMessageTemplate: parsed.VOICE_TAGGING_MESSAGE_TEMPLATE,
    agentTriggerTimeoutMs: parsed.AGENT_TRIGGER_TIMEOUT_MS,
    mcpServerUrl: parsed.MCP_SERVER_URL,
    mcpApiKey: parsed.MCP_API_KEY,
    upstreamTimeoutMs: parsed.UPSTREAM_TIMEOUT_MS,
  };
}
