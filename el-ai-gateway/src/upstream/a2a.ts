import { randomUUID } from "node:crypto";
import type { Config } from "../types";
import { GatewayError, upstreamError } from "../lib/errors";
import { fetchWithTimeout, type FetchLike } from "../lib/http";

export type A2ASendResult = { taskId?: string; state?: string; raw: unknown };

export type A2AClient = {
  sendTask(input: { assistantId: string; text: string }): Promise<A2ASendResult>;
};

export function createA2AClient(config: Config, fetchImpl: FetchLike = fetch): A2AClient {
  return {
    async sendTask({ assistantId, text }) {
      const url = `${config.a2aBaseUrl}/api/a2a/agents/${encodeURIComponent(assistantId)}/jsonrpc`;
      const payload = {
        jsonrpc: "2.0",
        id: randomUUID(),
        method: "message/send",
        params: {
          message: {
            role: "user",
            messageId: randomUUID(),
            parts: [{ kind: "text", text }],
          },
        },
      };
      const res = await fetchWithTimeout(
        fetchImpl,
        url,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            Authorization: `Bearer ${config.a2aApiKey}`,
          },
          body: JSON.stringify(payload),
        },
        config.upstreamTimeoutMs,
      );
      if (!res.ok) throw await upstreamError(res);
      const json = (await res.json()) as any;
      if (json?.error) {
        throw new GatewayError(502, "A2A_ERROR", json.error.message ?? "A2A request failed");
      }
      const result = json?.result ?? {};
      return { taskId: result.id, state: result.status?.state, raw: result };
    },
  };
}
