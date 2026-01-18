import { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

// We may send a decompressed body, so don't forward these verbatim.
const STRIP_RESPONSE_HEADERS = new Set(["content-length", "content-encoding"]);

function getPythonApiBaseUrl(): string {
  const raw =
    process.env.PYTHON_API_URL ||
    process.env.PREDICTION_API_URL ||
    process.env.PY_API_URL ||
    "http://localhost:8000";
  return raw.replace(/\/$/, "");
}

function shouldHaveBody(method: string): boolean {
  const m = method.toUpperCase();
  return !["GET", "HEAD", "OPTIONS"].includes(m);
}

function toSingleHeaderValue(v: undefined | string | string[]): string | undefined {
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.join(", ");
  return undefined;
}

/**
 * Reverse-proxy all `/api/v1/*` requests to the Python service (FastAPI).
 * This lets the frontend talk to a single backend origin (the agent),
 * while keeping Python compute endpoints intact.
 */
export function registerPythonProxyRoutes(app: FastifyInstance): void {
  const base = getPythonApiBaseUrl();

  const handler = async (request: FastifyRequest, reply: FastifyReply) => {
    // Fastify's request.url contains path + query string.
    const incomingUrl = request.raw.url || request.url;
    const targetUrl = new URL(incomingUrl, base).toString();

    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(request.headers)) {
      const key = k.toLowerCase();
      if (key === "host" || HOP_BY_HOP_HEADERS.has(key)) continue;
      const value = toSingleHeaderValue(v as any);
      if (value !== undefined) headers[k] = value;
    }
    // Let fetch compute these.
    delete headers["content-length"];

    let body: any = undefined;
    if (shouldHaveBody(request.method)) {
      const contentType = (request.headers["content-type"] as string | undefined) || "";
      const reqBody = (request as any).body;
      if (reqBody !== undefined && reqBody !== null) {
        if (Buffer.isBuffer(reqBody)) {
          body = reqBody;
        } else if (typeof reqBody === "string") {
          body = reqBody;
        } else if (contentType.includes("application/json")) {
          body = JSON.stringify(reqBody);
          if (!headers["Content-Type"] && !headers["content-type"]) headers["Content-Type"] = "application/json";
        } else {
          // Fallback: forward best-effort JSON.
          body = JSON.stringify(reqBody);
          if (!headers["Content-Type"] && !headers["content-type"]) headers["Content-Type"] = "application/json";
        }
      }
    }

    try {
      const res = await fetch(targetUrl, {
        method: request.method,
        headers,
        body,
        redirect: "manual",
      });

      reply.code(res.status);

      // Forward headers (excluding hop-by-hop and unsafe ones).
      for (const [k, v] of res.headers) {
        const key = k.toLowerCase();
        if (HOP_BY_HOP_HEADERS.has(key) || STRIP_RESPONSE_HEADERS.has(key)) continue;
        // Fastify will handle content-length; avoid conflicts.
        reply.header(k, v);
      }

      // `set-cookie` can be multi-valued; undici exposes it via getSetCookie().
      const anyHeaders = res.headers as any;
      if (typeof anyHeaders.getSetCookie === "function") {
        const cookies: string[] = anyHeaders.getSetCookie();
        if (cookies?.length) reply.header("set-cookie", cookies);
      }

      if (request.method.toUpperCase() === "HEAD") {
        return reply.send();
      }

      const buf = Buffer.from(await res.arrayBuffer());
      return reply.send(buf);
    } catch (e: any) {
      reply.code(502);
      return reply.send({
        success: false,
        error: "Upstream Python API unavailable",
        detail: e?.message || String(e),
        upstream: base,
      });
    }
  };

  // Proxy everything under /api/v1
  app.all("/api/v1", handler);
  app.all("/api/v1/*", handler);
}

