export interface MockResponseOptions {
  httpStatus?: number;
  statusCode?: string;
  message?: string;
  requestId?: string;
  body?: unknown;
  raw?: string;
}

/** Minimal fetch Response stub whose `headers.get` is case-insensitive. */
export function mockResponse(opts: MockResponseOptions): Response {
  const headers = new Map<string, string>();
  if (opts.statusCode !== undefined) headers.set("x-api-status-code", opts.statusCode);
  if (opts.message !== undefined) headers.set("x-api-message", opts.message);
  if (opts.requestId !== undefined) headers.set("x-api-request-id", opts.requestId);
  const raw =
    opts.raw !== undefined ? opts.raw : opts.body !== undefined ? JSON.stringify(opts.body) : "";
  return {
    status: opts.httpStatus ?? 200,
    headers: { get: (k: string) => headers.get(k.toLowerCase()) ?? null },
    text: async () => raw,
  } as unknown as Response;
}
