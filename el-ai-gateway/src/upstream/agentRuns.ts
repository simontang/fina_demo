import type { Config } from "../types";
import { GatewayError, upstreamError } from "../lib/errors";
import { fetchWithTimeout, type FetchLike } from "../lib/http";

export type AgentRunResult = { messageId?: string; queued: boolean };

export type AgentRunsClient = {
  startRun(input: {
    assistantId: string;
    threadId: string;
    text: string;
    taskId: string;
    timeoutMs?: number;
  }): Promise<AgentRunResult>;
};

function tokenExpiryMs(token: string): number {
  try {
    const payload = JSON.parse(Buffer.from(token.split(".")[0], "base64url").toString("utf8"));
    return typeof payload?.exp === "number" ? payload.exp : 0;
  } catch {
    return 0;
  }
}

export function createAgentRunsClient(config: Config, fetchImpl: FetchLike = fetch): AgentRunsClient {
  let cached: { token: string; exp: number } | null = null;
  const hasCredentials = Boolean(config.agentLoginEmail && config.agentLoginPassword);

  async function login(): Promise<string> {
    const res = await fetchWithTimeout(
      fetchImpl,
      config.agentAuthUrl,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: config.agentLoginEmail, password: config.agentLoginPassword }),
      },
      config.upstreamTimeoutMs,
    );
    if (!res.ok) {
      throw new GatewayError(502, "AGENT_AUTH_ERROR", `Agent login failed: ${res.status}`);
    }
    const json = (await res.json()) as { data?: { token?: string } };
    const token = json?.data?.token;
    if (!token) throw new GatewayError(502, "AGENT_AUTH_ERROR", "Agent login returned no token");
    cached = { token, exp: tokenExpiryMs(token) || Date.now() + 3600_000 };
    return token;
  }

  async function currentToken(): Promise<string | null> {
    if (!hasCredentials) return null;
    if (cached && cached.exp - Date.now() > 60_000) return cached.token;
    return login();
  }

  async function post(
    token: string | null,
    input: { assistantId: string; threadId: string; text: string; taskId: string; timeoutMs?: number },
  ): Promise<Response> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "x-tenant-id": config.agentTenantId,
      "x-workspace-id": config.agentWorkspaceId,
      "x-project-id": config.agentProjectId,
      "x-user-id": config.agentTenantId,
    };
    if (token) headers.Authorization = `Bearer ${token}`;
    return fetchWithTimeout(
      fetchImpl,
      config.agentRunsUrl,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          assistant_id: input.assistantId,
          thread_id: input.threadId,
          message: input.text,
          background: true,
          custom_run_config: { taskId: input.taskId },
        }),
      },
      input.timeoutMs ?? config.upstreamTimeoutMs,
    );
  }

  return {
    async startRun(input) {
      let res = await post(await currentToken(), input);
      if (res.status === 401 && hasCredentials) {
        cached = null;
        res = await post(await login(), input);
      }
      if (!res.ok) throw await upstreamError(res);
      const json = (await res.json()) as { messageId?: string; queued?: boolean };
      return { messageId: json?.messageId, queued: !!json?.queued };
    },
  };
}
