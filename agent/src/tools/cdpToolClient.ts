const CDP_BASE =
  process.env.CDP_API_URL || "http://127.0.0.1:5706";

const DEFAULT_TENANT_ID = process.env.TENANT_ID || "default";

export function getTenantIdFromExecutionConfig(exeConfig?: unknown): string {
  const tenantId = (exeConfig as {
    configurable?: { runConfig?: { tenantId?: unknown } };
  })?.configurable?.runConfig?.tenantId;

  return typeof tenantId === "string" && tenantId.trim()
    ? tenantId
    : DEFAULT_TENANT_ID;
}

export async function cdpFetch(
  path: string,
  options?: RequestInit,
  tenantId: string = DEFAULT_TENANT_ID,
): Promise<unknown> {
  const url = `${CDP_BASE.replace(/\/$/, "")}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
      "X-Tenant-Id": tenantId,
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`CDP API error ${res.status}: ${text}`);
  }
  return res.json();
}
