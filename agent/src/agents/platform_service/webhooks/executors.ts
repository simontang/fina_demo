import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

export const WEBHOOK_TOPICS = [
  "import.completed",
  "gate.passed",
  "decision.captured",
  "job.completed",
  "run.published",
] as const;

export async function webhooksListDestinations(
  _input: Record<string, never>,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = resolveConnection(rawConfig, exeConfig);
    const data = await request<Array<{ endpointId: string }>>({
      conn,
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: "/api/v1/webhooks/destinations",
    });
    const scoped =
      conn.selectedEntities.length > 0
        ? data.filter((d) => conn.selectedEntities.includes(d.endpointId))
        : data;
    return JSON.stringify(scoped);
  } catch (err) {
    return errorResult(err);
  }
}

export async function webhooksPublishEvent(
  input: { topic: string; data: Record<string, unknown>; endpointIds?: string[] },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = resolveConnection(rawConfig, exeConfig);
    const scope = conn.selectedEntities;
    const requested = input.endpointIds ?? [];
    if (scope.length > 0) {
      const out = requested.filter((id) => !scope.includes(id));
      if (out.length > 0) {
        return JSON.stringify({
          ok: false,
          code: "OUT_OF_SCOPE",
          message: `endpointIds 超出已选范围: ${out.join(", ")}`,
        });
      }
    }
    const effective = requested.length > 0 ? requested : scope;
    const result = await request({
      conn,
      tenantId: tenantFromExeConfig(exeConfig),
      method: "POST",
      path: "/api/v1/webhooks/publish",
      json: {
        topic: input.topic,
        data: input.data,
        ...(effective.length > 0 ? { endpointIds: effective } : {}),
      },
    });
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}

export async function webhooksListRecentEvents(
  input: { limit?: number },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: "/api/v1/webhooks/messages",
      query: { limit: input.limit },
    });
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}

export async function webhooksGetDeliveryStatus(
  input: { messageId: string },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: `/api/v1/webhooks/messages/${input.messageId}/attempts`,
    });
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}
