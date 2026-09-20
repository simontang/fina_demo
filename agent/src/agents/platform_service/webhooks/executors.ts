import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

export const MESSAGE_ID_PATTERN = /^[A-Za-z0-9_-]+$/;
export const ENDPOINT_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

export const WEBHOOK_EVENT_TYPES = [
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
    const result = await request<Array<{ endpointId: string }>>({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: "/api/v1/webhooks/destinations",
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function webhooksPublishEvent(
  input: { eventType: string; payload: Record<string, unknown>; channels?: string[] },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "POST",
      path: "/api/v1/webhooks/publish",
      json: {
        eventType: input.eventType,
        payload: input.payload,
        ...(input.channels && input.channels.length > 0 ? { channels: input.channels } : {}),
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
      conn: await resolveConnection(rawConfig, exeConfig),
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
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: `/api/v1/webhooks/messages/${input.messageId}/attempts`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function registerDestination(
  input: { url: string; filterTypes?: string[]; channels?: string[]; description?: string },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "POST",
      path: "/api/v1/webhooks/destinations",
      json: {
        url: input.url,
        ...(input.filterTypes ? { filterTypes: input.filterTypes } : {}),
        ...(input.channels ? { channels: input.channels } : {}),
        ...(input.description ? { description: input.description } : {}),
      },
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function deleteDestination(
  input: { endpointId: string; confirm?: boolean },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  if (input.confirm !== true) {
    return JSON.stringify({
      ok: false,
      code: "CONFIRM_REQUIRED",
      message:
        "Explicit user confirmation is required before deleting a delivery destination; confirm, then retry with confirm:true.",
    });
  }
  if (!ENDPOINT_ID_PATTERN.test(input.endpointId)) {
    return JSON.stringify({ ok: false, code: "BAD_REQUEST", message: "endpointId is invalid" });
  }
  try {
    const result = await request({
      conn: await resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "DELETE",
      path: `/api/v1/webhooks/destinations/${input.endpointId}`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}
