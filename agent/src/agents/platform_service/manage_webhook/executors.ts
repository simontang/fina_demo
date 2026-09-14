import {
  errorResult,
  request,
  resolveConnection,
  tenantFromExeConfig,
} from "../client";

const ENDPOINT_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

export async function manageWebhookListDestinations(
  _input: Record<string, never>,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "GET",
      path: "/api/v1/webhooks/destinations",
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function manageWebhookRegisterDestination(
  input: { url: string; topics: string[]; description?: string },
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const result = await request({
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "POST",
      path: "/api/v1/webhooks/destinations",
      json: {
        url: input.url,
        topics: input.topics,
        ...(input.description ? { description: input.description } : {}),
      },
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}

export async function manageWebhookDeleteDestination(
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
      conn: resolveConnection(rawConfig, exeConfig),
      tenantId: tenantFromExeConfig(exeConfig),
      method: "DELETE",
      path: `/api/v1/webhooks/destinations/${input.endpointId}`,
    });
    return JSON.stringify(result ?? null);
  } catch (err) {
    return errorResult(err);
  }
}
