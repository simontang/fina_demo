export type Principal = { tenantId: string; keyLabel: string };

export type Config = {
  port: number;
  gatewayApiKeys: Map<string, string>;
  authDisabled: boolean;
  authDevTenant: string;
  platformFilesUrl: string;
  fileServiceApiKey?: string;
  maxUploadBytes: number;
  a2aBaseUrl: string;
  a2aApiKey: string;
  a2aVoiceTaggingAssistantId?: string;
  a2aVoiceTaggingFileUuid?: string;
  a2aMessageTemplate?: string;
  mcpServerUrl: string;
  mcpApiKey: string;
  upstreamTimeoutMs: number;
};
