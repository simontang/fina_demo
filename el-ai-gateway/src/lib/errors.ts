export class GatewayError extends Error {
  constructor(
    public statusCode: number,
    public code: string,
    message: string,
    public upstream?: unknown,
  ) {
    super(message);
    this.name = "GatewayError";
  }
}

export function toErrorResponse(err: unknown): {
  statusCode: number;
  body: { code: string; message: string; upstream?: unknown };
} {
  if (err instanceof GatewayError) {
    const body: { code: string; message: string; upstream?: unknown } = {
      code: err.code,
      message: err.message,
    };
    if (err.upstream !== undefined) body.upstream = err.upstream;
    return { statusCode: err.statusCode, body };
  }
  if (err && typeof err === "object" && (err as { code?: string }).code === "FST_REQ_FILE_TOO_LARGE") {
    return { statusCode: 413, body: { code: "PAYLOAD_TOO_LARGE", message: "Upload exceeds the size limit" } };
  }
  if (err && typeof err === "object" && typeof (err as { statusCode?: unknown }).statusCode === "number") {
    const statusCode = (err as { statusCode: number }).statusCode;
    if (statusCode >= 400 && statusCode < 500) {
      return {
        statusCode,
        body: {
          code: (err as { code?: string }).code ?? "BAD_REQUEST",
          message: (err as Error).message ?? "Bad request",
        },
      };
    }
  }
  return { statusCode: 500, body: { code: "INTERNAL_ERROR", message: "Internal error" } };
}

export async function upstreamError(res: Response): Promise<GatewayError> {
  let payload: unknown = undefined;
  try {
    payload = await res.json();
  } catch {
    /* non-JSON body */
  }
  const typed = payload as { code?: unknown; message?: unknown } | undefined;
  const code = typeof typed?.code === "string" ? typed.code : "UPSTREAM_ERROR";
  const message =
    typeof typed?.message === "string" ? typed.message : `Upstream returned ${res.status}`;
  return new GatewayError(502, code, message, { status: res.status, code });
}
