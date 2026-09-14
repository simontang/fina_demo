import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

export const MESSAGE_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

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
    if (scope.length === 0 && requested.length > 0) {
      return JSON.stringify({
        ok: false,
        code: "OUT_OF_SCOPE",
        message: "endpointIds cannot be specified explicitly when selectedEntities is empty",
      });
    }
    if (scope.length > 0) {
      const out = requested.filter((id) => !scope.includes(id));
      if (out.length > 0) {
        return JSON.stringify({
          ok: false,
          code: "OUT_OF_SCOPE",
          message: `endpointIds outside the selected scope: ${out.join(", ")}`,
        });
      }
    }
    const effective = requested.length > 0 ? requested : scope;
    // Empty selectedEntities intentionally means server-side topic fan-out
    // (fail-open within the tenant), per spec §6.3.
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
    return JSON.stringify(result ?? null);
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
    return JSON.stringify(result ?? null);
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
    if (!MESSAGE_ID_PATTERN.test(input.messageId)) {
      return JSON.stringify({
        ok: false,
        code: "BAD_REQUEST",
        message: "messageId is invalid",
      });
    }
    const result = await request({
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: `/api/v1/webhooks/messages/${input.messageId}/attempts`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}
