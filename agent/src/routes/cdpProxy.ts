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

const STRIP_RESPONSE_HEADERS = new Set(["content-length", "content-encoding"]);
const CDP_PROXY_PREFIX = "/api/cdp";
const CDP_UPSTREAM_PREFIX = "/api/v1";

function getCdpApiBaseUrl(): string {
  const raw = process.env.CDP_API_URL || "http://127.0.0.1:5706";
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

export function buildCdpTargetUrl(incomingUrl: string, base: string): string {
  const targetUrl = new URL(incomingUrl, base);
  const incomingPath = targetUrl.pathname;

  if (incomingPath !== CDP_PROXY_PREFIX && !incomingPath.startsWith(`${CDP_PROXY_PREFIX}/`)) {
    throw new Error(`Unexpected CDP proxy path: ${incomingPath}`);
  }

  targetUrl.pathname = `${CDP_UPSTREAM_PREFIX}${incomingPath.slice(CDP_PROXY_PREFIX.length)}`;
  return targetUrl.toString();
}

export function registerCdpProxyRoutes(app: FastifyInstance): void {
  const base = getCdpApiBaseUrl();

  const handler = async (request: FastifyRequest, reply: FastifyReply) => {
    const incomingUrl = request.raw.url || request.url;
    const targetUrl = buildCdpTargetUrl(incomingUrl, base);

    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(request.headers)) {
      const key = k.toLowerCase();
      if (key === "host" || HOP_BY_HOP_HEADERS.has(key)) continue;
      const value = toSingleHeaderValue(v as any);
      if (value !== undefined) headers[k] = value;
    }
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

      for (const [k, v] of res.headers) {
        const key = k.toLowerCase();
        if (HOP_BY_HOP_HEADERS.has(key) || STRIP_RESPONSE_HEADERS.has(key)) continue;
        reply.header(k, v);
      }

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
        error: "Upstream CDP API unavailable",
        detail: e?.message || String(e),
        upstream: base,
      });
    }
  };

  app.all("/api/cdp", handler);
  app.all("/api/cdp/*", handler);
}
