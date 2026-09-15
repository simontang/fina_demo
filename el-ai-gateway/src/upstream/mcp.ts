import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { Config } from "../types";
import { GatewayError } from "../lib/errors";

export type McpToolResult = { text: string; structured?: unknown; isError: boolean };

export type McpCaller = {
  callTool(name: string, args: Record<string, unknown>): Promise<McpToolResult>;
};

export type McpSdkClient = {
  callTool(params: { name: string; arguments: Record<string, unknown> }): Promise<{
    content?: Array<{ type: string; text?: string }>;
    structuredContent?: unknown;
    isError?: boolean;
  }>;
  close(): Promise<void>;
};

async function connectDefault(config: Config): Promise<McpSdkClient> {
  const transport = new StreamableHTTPClientTransport(new URL(config.mcpServerUrl), {
    requestInit: { headers: { Authorization: `Bearer ${config.mcpApiKey}` } },
  });
  const client = new Client({ name: "el-ai-gateway", version: "0.1.0" }, { capabilities: {} });
  await client.connect(transport);
  return client as unknown as McpSdkClient;
}

export function createMcpClient(
  config: Config,
  connectImpl: (config: Config) => Promise<McpSdkClient> = connectDefault,
): McpCaller {
  let clientPromise: Promise<McpSdkClient> | null = null;

  async function getClient(): Promise<McpSdkClient> {
    if (!clientPromise) {
      clientPromise = connectImpl(config).catch((err) => {
        clientPromise = null;
        throw new GatewayError(502, "MCP_ERROR", `MCP connect failed: ${(err as Error).message}`);
      });
    }
    return clientPromise;
  }

  return {
    async callTool(name, args) {
      const client = await getClient();
      let res;
      try {
        res = await client.callTool({ name, arguments: args });
      } catch (err) {
        clientPromise = null;
        throw new GatewayError(502, "MCP_ERROR", `MCP tool "${name}" failed: ${(err as Error).message}`);
      }
      const content = res.content ?? [];
      const textPart = content.find((part) => part.type === "text");
      const text = textPart?.text ?? "";
      if (res.isError) {
        throw new GatewayError(502, "MCP_ERROR", text || `MCP tool "${name}" returned an error`);
      }
      return { text, structured: res.structuredContent, isError: false };
    },
  };
}
