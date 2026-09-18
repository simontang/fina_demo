export type Principal = { tenantId: string; keyLabel: string };

export type Config = {
  port: number;
  gatewayApiKeys: Map<string, string>;
  authDisabled: boolean;
  authDevTenant: string;
  platformFilesUrl: string;
  fileServiceApiKey?: string;
  maxUploadBytes: number;
  // Comma-separated allowed origins, or "*" for any (default).
  corsOrigins?: string;
  agentRunsUrl: string;
  agentAuthUrl: string;
  agentLoginEmail?: string;
  agentLoginPassword?: string;
  agentTenantId: string;
  agentWorkspaceId: string;
  agentProjectId: string;
  voiceTaggingAssistantId?: string;
  voiceTaggingFileUuid?: string;
  voiceTaggingMessageTemplate?: string;
  agentTriggerTimeoutMs?: number;
  mcpServerUrl: string;
  mcpApiKey: string;
  upstreamTimeoutMs: number;
};
