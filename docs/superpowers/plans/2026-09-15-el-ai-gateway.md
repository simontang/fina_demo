# el-ai-gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Fastify + TypeScript service `el-ai-gateway` that exposes 4 voice-file-tagging business APIs to an application, orchestrating platform-service files, an A2A voice agent, and agent-platform MCP task tools.

**Architecture:** Stateless edge gateway. Four REST endpoints under `/api/v1` authenticated by static env API keys. Outbound adapters: `platformFiles` (platform-service 5707, raw-stream PUT + presign), `a2a` (agent 5702 JSON-RPC `message/send`), and `mcp` (agent 5702 `/open/mcp` Streamable HTTP) wrapped by `taskTools` (`task_manage_task` create/get/add_activity). Routes are thin; every upstream is injected so it can be unit-tested without network.

**Tech Stack:** Node 20, Fastify 5, TypeScript, Zod, `@fastify/multipart`, `@modelcontextprotocol/sdk`, tsup, Vitest, pnpm.

**Spec:** `docs/superpowers/specs/2026-09-15-el-ai-gateway-design.md`

---

## File Structure

```
el-ai-gateway/
  package.json, tsconfig.json, .gitignore, .env.example, Dockerfile, README.md
  src/
    types.ts                  # Config, Principal, shared client interfaces
    config.ts                 # loadConfig(env), parseApiKeys
    auth.ts                   # createAuthenticator, requirePrincipal
    lib/errors.ts             # GatewayError, toErrorResponse, upstreamError
    lib/http.ts               # FetchLike, fetchWithTimeout
    upstream/platformFiles.ts # upload (PUT stream), presign
    upstream/a2a.ts           # sendTask (message/send)
    upstream/mcp.ts           # createMcpClient -> McpCaller
    upstream/taskTools.ts     # createTask/getTask/addActivity over McpCaller
    routes/files.ts           # POST /api/v1/files
    routes/tasks.ts           # POST /api/v1/tasks, GET+feedback /api/v1/tasks/:id
    server.ts                 # buildServer(deps)
    index.ts                  # entrypoint wiring real deps
  test/                       # Vitest tests, one per module
  scripts/mcp-probe.ts
  scripts/smoke.sh
```

---

## Task 1: Scaffold project + config

**Files:**
- Create: `el-ai-gateway/package.json`
- Create: `el-ai-gateway/tsconfig.json`
- Create: `el-ai-gateway/.gitignore`
- Create: `el-ai-gateway/.env.example`
- Create: `el-ai-gateway/src/types.ts`
- Create: `el-ai-gateway/src/config.ts`
- Test: `el-ai-gateway/test/config.test.ts`

- [ ] **Step 1: Create package.json, tsconfig, .gitignore, .env.example**

`el-ai-gateway/package.json`:
```json
{
  "name": "el-ai-gateway",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "packageManager": "pnpm@10.30.3",
  "description": "Estee Lauder AI project API gateway for voice-file tagging",
  "scripts": {
    "dev": "tsup src/index.ts --format esm --sourcemap --watch --onSuccess \"node dist/index.js\"",
    "build": "tsup src/index.ts --format esm --clean --sourcemap",
    "start": "node dist/index.js",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "tsc --noEmit",
    "typecheck": "tsc --noEmit",
    "mcp:probe": "tsx scripts/mcp-probe.ts"
  },
  "dependencies": {
    "@fastify/multipart": "^9.3.0",
    "@modelcontextprotocol/sdk": "^1.30.0",
    "dotenv": "^16.6.1",
    "fastify": "^5.5.0",
    "zod": "^3.25.76"
  },
  "devDependencies": {
    "@types/node": "^20.19.27",
    "tsx": "^4.19.2",
    "tsup": "^8.5.0",
    "typescript": "^5.8.2",
    "vitest": "^2.1.9"
  },
  "engines": { "node": ">=20.0.0" }
}
```

`el-ai-gateway/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "lib": ["ES2022", "DOM"],
    "outDir": "dist",
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "sourceMap": true,
    "noEmit": true,
    "types": ["node"]
  },
  "include": ["src", "test", "scripts"]
}
```

`el-ai-gateway/.gitignore`:
```
node_modules/
dist/
.env
*.log
```

`el-ai-gateway/.env.example`:
```
PORT=5708
GATEWAY_API_KEYS=dev_key:tenant_demo
AUTH_DISABLED=false
AUTH_DEV_TENANT=tenant_demo

PLATFORM_FILES_BASE_URL=http://127.0.0.1:5707
FILE_SERVICE_API_KEY=
GATEWAY_MAX_UPLOAD_BYTES=52428800

A2A_BASE_URL=http://127.0.0.1:5702
A2A_API_KEY=a2a_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
A2A_VOICE_TAGGING_ASSISTANT_ID=
A2A_MESSAGE_TEMPLATE=

MCP_SERVER_URL=http://127.0.0.1:5702/open/mcp
MCP_API_KEY=a2a_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

UPSTREAM_TIMEOUT_MS=30000
```

- [ ] **Step 2: Write types**

`el-ai-gateway/src/types.ts`:
```ts
export type Principal = { tenantId: string; keyLabel: string };

export type Config = {
  port: number;
  gatewayApiKeys: Map<string, string>;
  authDisabled: boolean;
  authDevTenant: string;
  platformFilesBaseUrl: string;
  fileServiceApiKey?: string;
  maxUploadBytes: number;
  a2aBaseUrl: string;
  a2aApiKey: string;
  a2aVoiceTaggingAssistantId?: string;
  a2aMessageTemplate?: string;
  mcpServerUrl: string;
  mcpApiKey: string;
  upstreamTimeoutMs: number;
};
```

- [ ] **Step 3: Write the failing config test**

`el-ai-gateway/test/config.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { loadConfig, parseApiKeys } from "../src/config";

const base = {
  A2A_API_KEY: "a2a_test",
  MCP_API_KEY: "a2a_mcp",
};

describe("parseApiKeys", () => {
  it("parses key:tenant pairs", () => {
    const map = parseApiKeys("k1:t1, k2:t2");
    expect(map.get("k1")).toBe("t1");
    expect(map.get("k2")).toBe("t2");
  });

  it("throws on malformed entries", () => {
    expect(() => parseApiKeys("nocolon")).toThrow(/Invalid GATEWAY_API_KEYS/);
  });
});

describe("loadConfig", () => {
  it("applies defaults and requires outbound keys", () => {
    const config = loadConfig(base as NodeJS.ProcessEnv);
    expect(config.port).toBe(5708);
    expect(config.authDisabled).toBe(false);
    expect(config.maxUploadBytes).toBe(52428800);
    expect(config.mcpServerUrl).toBe("http://127.0.0.1:5702/open/mcp");
    expect(config.a2aApiKey).toBe("a2a_test");
  });

  it("fails when A2A_API_KEY is missing", () => {
    expect(() => loadConfig({ MCP_API_KEY: "x" } as NodeJS.ProcessEnv)).toThrow(/A2A_API_KEY/);
  });

  it("parses AUTH_DISABLED=true", () => {
    const config = loadConfig({ ...base, AUTH_DISABLED: "true" } as NodeJS.ProcessEnv);
    expect(config.authDisabled).toBe(true);
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm install && pnpm test -- test/config.test.ts`
Expected: FAIL — cannot resolve `../src/config`.

- [ ] **Step 5: Implement config**

`el-ai-gateway/src/config.ts`:
```ts
import { z } from "zod";
import type { Config } from "./types";

const EnvSchema = z.object({
  PORT: z.coerce.number().int().positive().default(5708),
  GATEWAY_API_KEYS: z.string().default(""),
  AUTH_DISABLED: z.enum(["true", "false"]).default("false"),
  AUTH_DEV_TENANT: z.string().default("tenant_demo"),
  PLATFORM_FILES_BASE_URL: z.string().url().default("http://127.0.0.1:5707"),
  FILE_SERVICE_API_KEY: z.string().optional(),
  GATEWAY_MAX_UPLOAD_BYTES: z.coerce.number().int().positive().default(52428800),
  A2A_BASE_URL: z.string().url().default("http://127.0.0.1:5702"),
  A2A_API_KEY: z.string().optional(),
  A2A_VOICE_TAGGING_ASSISTANT_ID: z.string().optional(),
  A2A_MESSAGE_TEMPLATE: z.string().optional(),
  MCP_SERVER_URL: z.string().url().default("http://127.0.0.1:5702/open/mcp"),
  MCP_API_KEY: z.string().optional(),
  UPSTREAM_TIMEOUT_MS: z.coerce.number().int().positive().default(30000),
});

export function parseApiKeys(raw: string): Map<string, string> {
  const map = new Map<string, string>();
  for (const part of raw.split(",").map((s) => s.trim()).filter(Boolean)) {
    const idx = part.indexOf(":");
    if (idx <= 0 || idx === part.length - 1) {
      throw new Error(`Invalid GATEWAY_API_KEYS entry (expected key:tenant): ${part}`);
    }
    map.set(part.slice(0, idx), part.slice(idx + 1));
  }
  return map;
}

function required(value: string | undefined, name: string): string {
  if (!value || value.trim() === "") throw new Error(`${name} is required`);
  return value;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const parsed = EnvSchema.parse(env);
  return {
    port: parsed.PORT,
    gatewayApiKeys: parseApiKeys(parsed.GATEWAY_API_KEYS),
    authDisabled: parsed.AUTH_DISABLED === "true",
    authDevTenant: parsed.AUTH_DEV_TENANT,
    platformFilesBaseUrl: parsed.PLATFORM_FILES_BASE_URL.replace(/\/$/, ""),
    fileServiceApiKey: parsed.FILE_SERVICE_API_KEY,
    maxUploadBytes: parsed.GATEWAY_MAX_UPLOAD_BYTES,
    a2aBaseUrl: parsed.A2A_BASE_URL.replace(/\/$/, ""),
    a2aApiKey: required(parsed.A2A_API_KEY, "A2A_API_KEY"),
    a2aVoiceTaggingAssistantId: parsed.A2A_VOICE_TAGGING_ASSISTANT_ID,
    a2aMessageTemplate: parsed.A2A_MESSAGE_TEMPLATE,
    mcpServerUrl: parsed.MCP_SERVER_URL,
    mcpApiKey: required(parsed.MCP_API_KEY, "MCP_API_KEY"),
    upstreamTimeoutMs: parsed.UPSTREAM_TIMEOUT_MS,
  };
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/config.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add el-ai-gateway/package.json el-ai-gateway/tsconfig.json el-ai-gateway/.gitignore el-ai-gateway/.env.example el-ai-gateway/src/types.ts el-ai-gateway/src/config.ts el-ai-gateway/test/config.test.ts
git commit -m "feat(el-ai-gateway): scaffold project and config loader"
```

---

## Task 2: Errors + HTTP helper

**Files:**
- Create: `el-ai-gateway/src/lib/errors.ts`
- Create: `el-ai-gateway/src/lib/http.ts`
- Test: `el-ai-gateway/test/errors.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/errors.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { GatewayError, toErrorResponse } from "../src/lib/errors";
import { fetchWithTimeout } from "../src/lib/http";

describe("toErrorResponse", () => {
  it("maps GatewayError", () => {
    const res = toErrorResponse(new GatewayError(404, "NOT_FOUND", "nope", { taskId: "t1" }));
    expect(res.statusCode).toBe(404);
    expect(res.body).toEqual({ code: "NOT_FOUND", message: "nope", upstream: { taskId: "t1" } });
  });

  it("maps unknown errors to 500", () => {
    const res = toErrorResponse(new Error("boom"));
    expect(res.statusCode).toBe(500);
    expect(res.body.code).toBe("INTERNAL_ERROR");
  });

  it("maps multipart file-too-large to 413", () => {
    const res = toErrorResponse(Object.assign(new Error("too large"), { code: "FST_REQ_FILE_TOO_LARGE" }));
    expect(res.statusCode).toBe(413);
    expect(res.body.code).toBe("PAYLOAD_TOO_LARGE");
  });
});

describe("fetchWithTimeout", () => {
  it("returns the response on success", async () => {
    const fetchImpl = (async () => new Response("{}", { status: 200 })) as typeof fetch;
    const res = await fetchWithTimeout(fetchImpl, "http://x", { method: "GET" }, 1000);
    expect(res.status).toBe(200);
  });

  it("maps AbortError to 504", async () => {
    const fetchImpl = (async () => {
      const e = new Error("aborted");
      e.name = "AbortError";
      throw e;
    }) as typeof fetch;
    await expect(fetchWithTimeout(fetchImpl, "http://x", {}, 10)).rejects.toMatchObject({
      statusCode: 504,
      code: "UPSTREAM_TIMEOUT",
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/errors.test.ts`
Expected: FAIL — cannot resolve `../src/lib/errors`.

- [ ] **Step 3: Implement errors and http**

`el-ai-gateway/src/lib/errors.ts`:
```ts
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
  let payload: any = undefined;
  try {
    payload = await res.json();
  } catch {
    /* non-JSON body */
  }
  const code = typeof payload?.code === "string" ? payload.code : "UPSTREAM_ERROR";
  const message = typeof payload?.message === "string" ? payload.message : `Upstream returned ${res.status}`;
  return new GatewayError(502, code, message, { status: res.status, code });
}
```

`el-ai-gateway/src/lib/http.ts`:
```ts
import { GatewayError } from "./errors";

export type FetchLike = typeof fetch;

export async function fetchWithTimeout(
  fetchImpl: FetchLike,
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(url, { ...init, signal: controller.signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") {
      throw new GatewayError(504, "UPSTREAM_TIMEOUT", `Upstream timeout after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/errors.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/lib el-ai-gateway/test/errors.test.ts
git commit -m "feat(el-ai-gateway): error envelope and fetch timeout helper"
```

---

## Task 3: Inbound auth

**Files:**
- Create: `el-ai-gateway/src/auth.ts`
- Test: `el-ai-gateway/test/auth.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/auth.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { createAuthenticator, requirePrincipal } from "../src/auth";
import type { Config } from "../src/types";

function config(overrides: Partial<Config> = {}): Config {
  return {
    port: 5708,
    gatewayApiKeys: new Map([["secret", "tenant_a"]]),
    authDisabled: false,
    authDevTenant: "tenant_dev",
    platformFilesBaseUrl: "http://files",
    maxUploadBytes: 1000,
    a2aBaseUrl: "http://a2a",
    a2aApiKey: "a2a_x",
    mcpServerUrl: "http://mcp",
    mcpApiKey: "a2a_m",
    upstreamTimeoutMs: 1000,
    ...overrides,
  };
}

describe("createAuthenticator", () => {
  it("accepts a configured bearer key", () => {
    const auth = createAuthenticator(config());
    expect(auth("Bearer secret")).toEqual({ tenantId: "tenant_a", keyLabel: "secret" });
  });

  it("rejects missing or unknown keys", () => {
    const auth = createAuthenticator(config());
    expect(auth(undefined)).toBeNull();
    expect(auth("Bearer nope")).toBeNull();
  });

  it("uses the dev tenant when auth is disabled", () => {
    const auth = createAuthenticator(config({ authDisabled: true }));
    expect(auth(undefined)).toEqual({ tenantId: "tenant_dev", keyLabel: "dev" });
  });
});

describe("requirePrincipal", () => {
  it("throws 401 when unauthenticated", () => {
    const auth = createAuthenticator(config());
    expect(() => requirePrincipal(auth, undefined)).toThrowError(/Unauthorized/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/auth.test.ts`
Expected: FAIL — cannot resolve `../src/auth`.

- [ ] **Step 3: Implement auth**

`el-ai-gateway/src/auth.ts`:
```ts
import { GatewayError } from "./lib/errors";
import type { Config, Principal } from "./types";

export type Authenticator = (authorization: string | undefined) => Principal | null;

export function createAuthenticator(config: Config): Authenticator {
  return (authorization) => {
    if (config.authDisabled) {
      return { tenantId: config.authDevTenant, keyLabel: "dev" };
    }
    if (!authorization?.startsWith("Bearer ")) return null;
    const key = authorization.slice("Bearer ".length).trim();
    const tenantId = config.gatewayApiKeys.get(key);
    if (!tenantId) return null;
    return { tenantId, keyLabel: key };
  };
}

export function requirePrincipal(auth: Authenticator, authorization: string | undefined): Principal {
  const principal = auth(authorization);
  if (!principal) throw new GatewayError(401, "UNAUTHORIZED", "Missing or invalid API key");
  return principal;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/auth.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/auth.ts el-ai-gateway/test/auth.test.ts
git commit -m "feat(el-ai-gateway): inbound API key authentication"
```

---

## Task 4: platformFiles adapter

**Files:**
- Create: `el-ai-gateway/src/upstream/platformFiles.ts`
- Test: `el-ai-gateway/test/platformFiles.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/platformFiles.test.ts`:
```ts
import { Readable } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { createPlatformFilesClient } from "../src/upstream/platformFiles";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map(),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files:5707",
  fileServiceApiKey: "internal",
  maxUploadBytes: 1000,
  a2aBaseUrl: "http://a2a",
  a2aApiKey: "a2a_x",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

describe("platformFiles.upload", () => {
  it("PUTs the raw stream with tenant and file headers", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ uuid: "abc", version: 1 }), { status: 200 }),
    ) as unknown as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);

    const receipt = await client.upload({
      tenantId: "tenant_a",
      body: Readable.from([Buffer.from("hello")]),
      uuid: "abc",
      filename: "a.wav",
      mime: "audio/wav",
      path: "voice",
      fileName: "a.wav",
    });

    expect(receipt).toEqual({ uuid: "abc", version: 1 });
    const [url, init] = (fetchImpl as any).mock.calls[0];
    expect(url).toBe("http://files:5707/api/v1/files/abc");
    expect(init.method).toBe("PUT");
    expect(init.headers["X-Tenant-Id"]).toBe("tenant_a");
    expect(init.headers["X-Api-Key"]).toBe("internal");
    expect(init.headers["Content-Type"]).toBe("audio/wav");
    expect(init.headers["X-File-Path"]).toBe("voice");
    expect(init.headers["X-File-Name"]).toBe("a.wav");
    expect(init.duplex).toBe("half");
  });

  it("maps upstream errors to 502 with the upstream code", async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ code: "TENANT_REQUIRED", message: "no tenant" }), {
        status: 400,
      })) as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);
    await expect(
      client.upload({
        tenantId: "t",
        body: Readable.from([Buffer.from("x")]),
        uuid: "abc",
        filename: "a.wav",
        mime: "audio/wav",
      }),
    ).rejects.toMatchObject({ statusCode: 502, code: "TENANT_REQUIRED" });
  });
});

describe("platformFiles.presign", () => {
  it("returns the presigned url", async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ url: "https://signed", kind: "presigned", expiresInSeconds: 600 }), {
        status: 200,
      })) as typeof fetch;
    const client = createPlatformFilesClient(config, fetchImpl);
    const link = await client.presign({ tenantId: "t", uuid: "abc" });
    expect(link.url).toBe("https://signed");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/platformFiles.test.ts`
Expected: FAIL — cannot resolve `../src/upstream/platformFiles`.

- [ ] **Step 3: Implement platformFiles**

`el-ai-gateway/src/upstream/platformFiles.ts`:
```ts
import type { Readable } from "node:stream";
import type { Config } from "../types";
import { upstreamError } from "../lib/errors";
import { fetchWithTimeout, type FetchLike } from "../lib/http";

export type UploadReceipt = Record<string, unknown> & { uuid?: string };

export type PresignLink = { url: string; kind: string; expiresInSeconds: number };

export type PlatformFilesClient = {
  upload(input: {
    tenantId: string;
    body: Readable;
    uuid: string;
    filename: string;
    mime: string;
    path?: string;
    fileName?: string;
  }): Promise<UploadReceipt>;
  presign(input: { tenantId: string; uuid: string; ttlSeconds?: number }): Promise<PresignLink>;
};

export function createPlatformFilesClient(
  config: Config,
  fetchImpl: FetchLike = fetch,
): PlatformFilesClient {
  function baseHeaders(tenantId: string): Record<string, string> {
    const headers: Record<string, string> = { "X-Tenant-Id": tenantId };
    if (config.fileServiceApiKey) headers["X-Api-Key"] = config.fileServiceApiKey;
    return headers;
  }

  return {
    async upload(input) {
      const headers = baseHeaders(input.tenantId);
      headers["Content-Type"] = input.mime;
      if (input.path) headers["X-File-Path"] = input.path;
      headers["X-File-Name"] = input.fileName ?? input.filename;

      const res = await fetchWithTimeout(
        fetchImpl,
        `${config.platformFilesBaseUrl}/api/v1/files/${input.uuid}`,
        {
          method: "PUT",
          headers,
          body: input.body as unknown as BodyInit,
          duplex: "half",
        } as RequestInit,
        config.upstreamTimeoutMs,
      );
      if (!res.ok) throw await upstreamError(res);
      return (await res.json()) as UploadReceipt;
    },

    async presign(input) {
      const res = await fetchWithTimeout(
        fetchImpl,
        `${config.platformFilesBaseUrl}/api/v1/files/presign`,
        {
          method: "POST",
          headers: { ...baseHeaders(input.tenantId), "Content-Type": "application/json" },
          body: JSON.stringify({ uuid: input.uuid, ttlSeconds: input.ttlSeconds }),
        },
        config.upstreamTimeoutMs,
      );
      if (!res.ok) throw await upstreamError(res);
      return (await res.json()) as PresignLink;
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/platformFiles.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/upstream/platformFiles.ts el-ai-gateway/test/platformFiles.test.ts
git commit -m "feat(el-ai-gateway): platform-service files adapter"
```

---

## Task 5: A2A adapter

**Files:**
- Create: `el-ai-gateway/src/upstream/a2a.ts`
- Test: `el-ai-gateway/test/a2a.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/a2a.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";
import { createA2AClient } from "../src/upstream/a2a";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map(),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files",
  maxUploadBytes: 1000,
  a2aBaseUrl: "http://agent:5702",
  a2aApiKey: "a2a_secret",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

describe("a2a.sendTask", () => {
  it("posts a message/send JSON-RPC and parses the task", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({ jsonrpc: "2.0", id: "1", result: { id: "task-1", status: { state: "submitted" } } }),
        { status: 200 },
      ),
    ) as unknown as typeof fetch;
    const client = createA2AClient(config, fetchImpl);

    const out = await client.sendTask({ assistantId: "voice", text: "transcribe https://x" });

    expect(out.taskId).toBe("task-1");
    expect(out.state).toBe("submitted");
    const [url, init] = (fetchImpl as any).mock.calls[0];
    expect(url).toBe("http://agent:5702/api/a2a/agents/voice/jsonrpc");
    expect(init.headers.Authorization).toBe("Bearer a2a_secret");
    const body = JSON.parse(init.body);
    expect(body.method).toBe("message/send");
    expect(body.params.message.parts[0].text).toBe("transcribe https://x");
  });

  it("maps JSON-RPC errors to 502 A2A_ERROR", async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ jsonrpc: "2.0", id: "1", error: { code: -32001, message: "Unauthorized" } }), {
        status: 200,
      })) as typeof fetch;
    const client = createA2AClient(config, fetchImpl);
    await expect(client.sendTask({ assistantId: "voice", text: "x" })).rejects.toMatchObject({
      statusCode: 502,
      code: "A2A_ERROR",
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/a2a.test.ts`
Expected: FAIL — cannot resolve `../src/upstream/a2a`.

- [ ] **Step 3: Implement a2a**

`el-ai-gateway/src/upstream/a2a.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/a2a.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/upstream/a2a.ts el-ai-gateway/test/a2a.test.ts
git commit -m "feat(el-ai-gateway): A2A JSON-RPC adapter"
```

---

## Task 6: MCP client + taskTools

**Files:**
- Create: `el-ai-gateway/src/upstream/mcp.ts`
- Create: `el-ai-gateway/src/upstream/taskTools.ts`
- Test: `el-ai-gateway/test/taskTools.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/taskTools.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";
import { createTaskToolClient } from "../src/upstream/taskTools";
import type { McpCaller } from "../src/upstream/mcp";

function callerReturning(text: string): McpCaller {
  return {
    callTool: vi.fn(async () => ({ text, isError: false })),
  };
}

describe("taskTools", () => {
  it("creates a task and returns its id", async () => {
    const caller = callerReturning(JSON.stringify({ success: true, taskId: "task-1", task: { id: "task-1" } }));
    const tools = createTaskToolClient(caller);
    const out = await tools.createTask({ title: "Voice tagging", status: "in_progress", metadata: { uuid: "u" } });
    expect(out.taskId).toBe("task-1");
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "create",
      title: "Voice tagging",
      description: undefined,
      status: "in_progress",
      metadata: { uuid: "u" },
    });
  });

  it("reads a task and its activities", async () => {
    const caller = callerReturning(
      JSON.stringify({ success: true, task: { id: "t1", status: "completed", title: "T" }, activities: [{ id: "a1" }] }),
    );
    const tools = createTaskToolClient(caller);
    const task = await tools.getTask({ id: "t1" });
    expect(task).toMatchObject({ id: "t1", status: "completed", title: "T" });
    expect(task.activities).toEqual([{ id: "a1" }]);
  });

  it("maps a not-found task to 404", async () => {
    const caller = callerReturning(JSON.stringify({ success: false, error: "Task not found" }));
    const tools = createTaskToolClient(caller);
    await expect(tools.getTask({ id: "missing" })).rejects.toMatchObject({ statusCode: 404, code: "NOT_FOUND" });
  });

  it("adds an activity", async () => {
    const caller = callerReturning(JSON.stringify({ success: true }));
    const tools = createTaskToolClient(caller);
    await tools.addActivity({ id: "t1", content: "tag is wrong", summary: "feedback" });
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "add_activity",
      id: "t1",
      content: "tag is wrong",
      summary: "feedback",
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/taskTools.test.ts`
Expected: FAIL — cannot resolve `../src/upstream/taskTools`.

- [ ] **Step 3: Implement mcp and taskTools**

`el-ai-gateway/src/upstream/mcp.ts`:
```ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { Config } from "../types";
import { GatewayError } from "../lib/errors";

export type McpToolResult = { text: string; structured?: unknown; isError: boolean };

export type McpCaller = {
  callTool(name: string, args: Record<string, unknown>): Promise<McpToolResult>;
};

export type McpSdkClient = {
  callTool(params: { name: string; arguments: Record<string, unknown> }): Promise<{
    content?: Array<{ type: string; text?: string }>;
    structuredContent?: unknown;
    isError?: boolean;
  }>;
  close(): Promise<void>;
};

async function connectDefault(config: Config): Promise<McpSdkClient> {
  const transport = new StreamableHTTPClientTransport(new URL(config.mcpServerUrl), {
    requestInit: { headers: { Authorization: `Bearer ${config.mcpApiKey}` } },
  });
  const client = new Client({ name: "el-ai-gateway", version: "0.1.0" }, { capabilities: {} });
  await client.connect(transport);
  return client as unknown as McpSdkClient;
}

export function createMcpClient(
  config: Config,
  connectImpl: (config: Config) => Promise<McpSdkClient> = connectDefault,
): McpCaller {
  let clientPromise: Promise<McpSdkClient> | null = null;

  async function getClient(): Promise<McpSdkClient> {
    if (!clientPromise) {
      clientPromise = connectImpl(config).catch((err) => {
        clientPromise = null;
        throw new GatewayError(502, "MCP_ERROR", `MCP connect failed: ${(err as Error).message}`);
      });
    }
    return clientPromise;
  }

  return {
    async callTool(name, args) {
      const client = await getClient();
      let res;
      try {
        res = await client.callTool({ name, arguments: args });
      } catch (err) {
        clientPromise = null;
        throw new GatewayError(502, "MCP_ERROR", `MCP tool "${name}" failed: ${(err as Error).message}`);
      }
      const content = res.content ?? [];
      const textPart = content.find((part) => part.type === "text");
      const text = textPart?.text ?? "";
      if (res.isError) {
        throw new GatewayError(502, "MCP_ERROR", text || `MCP tool "${name}" returned an error`);
      }
      return { text, structured: res.structuredContent, isError: false };
    },
  };
}
```

`el-ai-gateway/src/upstream/taskTools.ts`:
```ts
import { GatewayError } from "../lib/errors";
import type { McpCaller } from "./mcp";

const TOOL = "task_manage_task";

export type TaskRecord = {
  id: string;
  status?: string;
  title?: string;
  result?: string;
  activities: unknown[];
  raw: unknown;
};

export type TaskToolClient = {
  createTask(input: {
    title: string;
    description?: string;
    status?: string;
    metadata?: Record<string, unknown>;
  }): Promise<{ taskId: string; raw: unknown }>;
  getTask(input: { id: string }): Promise<TaskRecord>;
  addActivity(input: { id: string; content: string; summary?: string }): Promise<{ raw: unknown }>;
};

function parseResult(text: string, structured: unknown): any {
  if (structured && typeof structured === "object") return structured;
  try {
    return JSON.parse(text);
  } catch {
    return { rawText: text };
  }
}

export function createTaskToolClient(mcp: McpCaller): TaskToolClient {
  async function invoke(args: Record<string, unknown>): Promise<any> {
    const { text, structured } = await mcp.callTool(TOOL, args);
    const data = parseResult(text, structured);
    if (data && (data.success === false || data.error)) {
      const message = String(data.error ?? "task tool failed");
      if (/not found|does not exist/i.test(message)) {
        throw new GatewayError(404, "NOT_FOUND", message);
      }
      throw new GatewayError(502, "MCP_ERROR", message);
    }
    return data;
  }

  return {
    async createTask(input) {
      const data = await invoke({
        action: "create",
        title: input.title,
        description: input.description,
        status: input.status ?? "in_progress",
        metadata: input.metadata,
      });
      const taskId = (data?.taskId ?? data?.task?.id ?? data?.id) as string | undefined;
      if (!taskId) throw new GatewayError(502, "MCP_ERROR", "create task returned no id");
      return { taskId, raw: data };
    },

    async getTask({ id }) {
      const data = await invoke({ action: "get", id });
      const task = data?.task ?? data ?? {};
      return {
        id: task.id ?? id,
        status: task.status,
        title: task.title,
        result: task.result,
        activities: data?.activities ?? task.activities ?? [],
        raw: data,
      };
    },

    async addActivity({ id, content, summary }) {
      return { raw: await invoke({ action: "add_activity", id, content, summary }) };
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/taskTools.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add el-ai-gateway/src/upstream/mcp.ts el-ai-gateway/src/upstream/taskTools.ts el-ai-gateway/test/taskTools.test.ts
git commit -m "feat(el-ai-gateway): MCP client and task/activity tools"
```

---

## Task 7: Server + files route

**Files:**
- Create: `el-ai-gateway/src/routes/files.ts`
- Create: `el-ai-gateway/src/server.ts`
- Test: `el-ai-gateway/test/filesRoute.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/filesRoute.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files",
  maxUploadBytes: 1024 * 1024,
  a2aBaseUrl: "http://a2a",
  a2aApiKey: "a2a_x",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

function multipartBody(filename: string, content: string) {
  const boundary = "----elgtest";
  const payload = Buffer.concat([
    Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
        "Content-Type: audio/wav\r\n\r\n",
    ),
    Buffer.from(content),
    Buffer.from(`\r\n--${boundary}--\r\n`),
  ]);
  return { payload, contentType: `multipart/form-data; boundary=${boundary}` };
}

describe("POST /api/v1/files", () => {
  it("streams the upload and returns the platform receipt", async () => {
    const upload = vi.fn(async () => ({ uuid: "abc", version: 1, filename: "a.wav" }));
    const app = buildServer({
      config,
      authenticator: (h) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null),
      platformFiles: { upload, presign: vi.fn() } as any,
      a2a: { sendTask: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
    });

    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files?path=voice&fileName=a.wav",
      headers: { authorization: "Bearer secret", "content-type": contentType },
      payload,
    });

    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ uuid: "abc", version: 1, filename: "a.wav" });
    expect(upload).toHaveBeenCalledTimes(1);
    const arg = upload.mock.calls[0][0];
    expect(arg.tenantId).toBe("tenant_a");
    expect(arg.mime).toBe("audio/wav");
    expect(arg.path).toBe("voice");
    expect(arg.fileName).toBe("a.wav");
  });

  it("returns 401 without a valid key", async () => {
    const app = buildServer({
      config,
      authenticator: () => null,
      platformFiles: { upload: vi.fn(), presign: vi.fn() } as any,
      a2a: { sendTask: vi.fn() } as any,
      taskTools: { createTask: vi.fn(), getTask: vi.fn(), addActivity: vi.fn() } as any,
    });
    const { payload, contentType } = multipartBody("a.wav", "RIFF");
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/files",
      headers: { "content-type": contentType },
      payload,
    });
    expect(res.statusCode).toBe(401);
    expect(res.json().code).toBe("UNAUTHORIZED");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/filesRoute.test.ts`
Expected: FAIL — cannot resolve `../src/server`.

- [ ] **Step 3: Implement files route and server**

`el-ai-gateway/src/routes/files.ts`:
```ts
import { randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";

export type FileRouteDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
};

export function registerFileRoutes(app: FastifyInstance, deps: FileRouteDeps): void {
  app.post("/api/v1/files", async (request, reply) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const data = await request.file();
    if (!data) throw new GatewayError(400, "BAD_REQUEST", "multipart field 'file' is required");

    const query = request.query as { path?: string; fileName?: string };
    const uuid = randomUUID().replace(/-/g, "");

    const receipt = await deps.platformFiles.upload({
      tenantId: principal.tenantId,
      body: data.file,
      uuid,
      filename: data.filename,
      mime: data.mimetype,
      path: query.path,
      fileName: query.fileName,
    });

    return reply.send(receipt);
  });
}
```

`el-ai-gateway/src/server.ts`:
```ts
import Fastify, { type FastifyInstance } from "fastify";
import multipart from "@fastify/multipart";
import type { Authenticator } from "./auth";
import { toErrorResponse } from "./lib/errors";
import type { Config } from "./types";
import type { PlatformFilesClient } from "./upstream/platformFiles";
import type { A2AClient } from "./upstream/a2a";
import type { TaskToolClient } from "./upstream/taskTools";
import { registerFileRoutes } from "./routes/files";
import { registerTaskRoutes } from "./routes/tasks";

export type ServerDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  a2a: A2AClient;
  taskTools: TaskToolClient;
};

export function buildServer(deps: ServerDeps): FastifyInstance {
  const app = Fastify({ logger: false, bodyLimit: deps.config.maxUploadBytes });

  app.register(multipart, { limits: { fileSize: deps.config.maxUploadBytes } });

  app.get("/health", async () => ({ status: "ok" }));

  registerFileRoutes(app, deps);
  registerTaskRoutes(app, deps);

  app.setErrorHandler((error, _request, reply) => {
    const { statusCode, body } = toErrorResponse(error);
    reply.status(statusCode).send(body);
  });

  return app;
}
```

Note: `server.ts` imports `registerTaskRoutes`, which is created in Task 8. If executing strictly task-by-task, create a stub now and replace in Task 8, OR do Task 8 before running Task 7's test. Simplest: in this task, also create `src/routes/tasks.ts` as an empty registrar `export function registerTaskRoutes(): void {}`, then replace it in Task 8.

- [ ] **Step 4: Create temporary tasks route stub**

`el-ai-gateway/src/routes/tasks.ts`:
```ts
export function registerTaskRoutes(..._args: unknown[]): void {
  // Implemented in Task 8.
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/filesRoute.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add el-ai-gateway/src/routes/files.ts el-ai-gateway/src/routes/tasks.ts el-ai-gateway/src/server.ts el-ai-gateway/test/filesRoute.test.ts
git commit -m "feat(el-ai-gateway): server bootstrap and file upload route"
```

---

## Task 8: Tasks routes (start / status / feedback)

**Files:**
- Modify: `el-ai-gateway/src/routes/tasks.ts` (replace stub)
- Test: `el-ai-gateway/test/tasksRoute.test.ts`

- [ ] **Step 1: Write the failing test**

`el-ai-gateway/test/tasksRoute.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";
import { buildServer } from "../src/server";
import type { Config } from "../src/types";

const config: Config = {
  port: 5708,
  gatewayApiKeys: new Map([["secret", "tenant_a"]]),
  authDisabled: false,
  authDevTenant: "tenant_demo",
  platformFilesBaseUrl: "http://files",
  maxUploadBytes: 1024 * 1024,
  a2aBaseUrl: "http://a2a",
  a2aApiKey: "a2a_x",
  a2aVoiceTaggingAssistantId: "voice-agent",
  mcpServerUrl: "http://mcp",
  mcpApiKey: "a2a_m",
  upstreamTimeoutMs: 1000,
};

const auth = (h?: string) => (h === "Bearer secret" ? { tenantId: "tenant_a", keyLabel: "k" } : null);

function deps(overrides: Record<string, unknown> = {}) {
  return {
    config,
    authenticator: auth,
    platformFiles: {
      upload: vi.fn(),
      presign: vi.fn(async () => ({ url: "https://signed", kind: "presigned", expiresInSeconds: 600 })),
    },
    a2a: { sendTask: vi.fn(async () => ({ taskId: "a2a-1", state: "submitted", raw: {} })) },
    taskTools: {
      createTask: vi.fn(async () => ({ taskId: "task-1", raw: {} })),
      getTask: vi.fn(async () => ({ id: "task-1", status: "completed", title: "T", activities: [{ id: "act-1" }], raw: {} })),
      addActivity: vi.fn(async () => ({ raw: {} })),
    },
    ...overrides,
  } as any;
}

describe("POST /api/v1/tasks", () => {
  it("presigns, creates the task, triggers A2A and returns the task id", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/tasks",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1", title: "My task" },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.taskId).toBe("task-1");
    expect(body.file).toEqual({ uuid: "u1", url: "https://signed" });
    expect(d.platformFiles.presign).toHaveBeenCalledWith({ tenantId: "tenant_a", uuid: "u1" });
    expect(d.taskTools.createTask).toHaveBeenCalledWith({
      title: "My task",
      description: undefined,
      status: "in_progress",
      metadata: { uuid: "u1", url: "https://signed" },
    });
    const a2aArg = d.a2a.sendTask.mock.calls[0][0];
    expect(a2aArg.assistantId).toBe("voice-agent");
    expect(a2aArg.text).toContain("https://signed");
  });

  it("returns 400 when uuid is missing", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/tasks",
      headers: { authorization: "Bearer secret" },
      payload: {},
    });
    expect(res.statusCode).toBe(400);
  });

  it("keeps the task id when A2A fails", async () => {
    const d = deps();
    d.a2a.sendTask = vi.fn(async () => {
      throw new Error("a2a down");
    });
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/tasks",
      headers: { authorization: "Bearer secret" },
      payload: { uuid: "u1" },
    });
    expect(res.statusCode).toBe(502);
    expect(res.json().upstream).toEqual({ taskId: "task-1" });
  });
});

describe("GET /api/v1/tasks/:id", () => {
  it("returns task status and activities", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "GET",
      url: "/api/v1/tasks/task-1",
      headers: { authorization: "Bearer secret" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ taskId: "task-1", status: "completed", activities: [{ id: "act-1" }] });
  });
});

describe("POST /api/v1/tasks/:id/feedback", () => {
  it("adds an activity", async () => {
    const d = deps();
    const app = buildServer(d);
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/tasks/task-1/feedback",
      headers: { authorization: "Bearer secret" },
      payload: { content: "tag corrected" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ taskId: "task-1", added: true });
    expect(d.taskTools.addActivity).toHaveBeenCalledWith({
      id: "task-1",
      content: "tag corrected",
      summary: undefined,
    });
  });

  it("returns 400 when content is empty", async () => {
    const app = buildServer(deps());
    const res = await app.inject({
      method: "POST",
      url: "/api/v1/tasks/task-1/feedback",
      headers: { authorization: "Bearer secret" },
      payload: { content: "   " },
    });
    expect(res.statusCode).toBe(400);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd el-ai-gateway && pnpm test -- test/tasksRoute.test.ts`
Expected: FAIL — routes not implemented (404).

- [ ] **Step 3: Implement tasks routes**

`el-ai-gateway/src/routes/tasks.ts`:
```ts
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";
import type { A2AClient } from "../upstream/a2a";
import type { TaskToolClient } from "../upstream/taskTools";

export type TaskRouteDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  a2a: A2AClient;
  taskTools: TaskToolClient;
};

export function renderA2AMessage(
  template: string | undefined,
  vars: { uuid: string; url: string; taskId: string },
): string {
  if (template) {
    return template
      .replaceAll("{uuid}", vars.uuid)
      .replaceAll("{url}", vars.url)
      .replaceAll("{taskId}", vars.taskId);
  }
  return `Voice file tagging task ${vars.taskId}.\nFile: ${vars.uuid}\nDownload URL: ${vars.url}\nTranscribe the audio and tag the resulting text.`;
}

export function registerTaskRoutes(app: FastifyInstance, deps: TaskRouteDeps): void {
  app.post("/api/v1/tasks", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const body = (request.body ?? {}) as {
      uuid?: string;
      title?: string;
      description?: string;
      assistantId?: string;
    };
    if (typeof body.uuid !== "string" || body.uuid.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "uuid is required");
    }

    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid: body.uuid });
    const title = body.title ?? `Voice tagging: ${body.uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: body.description,
      status: "in_progress",
      metadata: { uuid: body.uuid, url },
    });

    let a2a: { taskId?: string; state?: string };
    try {
      const assistantId = body.assistantId ?? deps.config.a2aVoiceTaggingAssistantId;
      if (!assistantId) {
        throw new GatewayError(400, "BAD_REQUEST", "assistantId is required (or set A2A_VOICE_TAGGING_ASSISTANT_ID)");
      }
      const text = renderA2AMessage(deps.config.a2aMessageTemplate, { uuid: body.uuid, url, taskId });
      a2a = await deps.a2a.sendTask({ assistantId, text });
    } catch (err) {
      if (err instanceof GatewayError && err.statusCode === 400) throw err;
      throw new GatewayError(502, "A2A_ERROR", (err as Error).message, { taskId });
    }

    return {
      taskId,
      status: "in_progress",
      file: { uuid: body.uuid, url },
      a2a: { taskId: a2a.taskId, state: a2a.state },
    };
  });

  app.get("/api/v1/tasks/:id", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    return {
      taskId: task.id,
      status: task.status,
      title: task.title,
      result: task.result,
      activities: task.activities,
    };
  });

  app.post("/api/v1/tasks/:id/feedback", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const body = (request.body ?? {}) as { content?: string; summary?: string };
    if (typeof body.content !== "string" || body.content.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "content is required");
    }
    await deps.taskTools.addActivity({ id, content: body.content, summary: body.summary });
    return { taskId: id, added: true };
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd el-ai-gateway && pnpm test -- test/tasksRoute.test.ts`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the whole suite**

Run: `cd el-ai-gateway && pnpm test`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add el-ai-gateway/src/routes/tasks.ts el-ai-gateway/test/tasksRoute.test.ts
git commit -m "feat(el-ai-gateway): task start, status and feedback routes"
```

---

## Task 9: Entrypoint, scripts, Dockerfile, README

**Files:**
- Create: `el-ai-gateway/src/index.ts`
- Create: `el-ai-gateway/scripts/mcp-probe.ts`
- Create: `el-ai-gateway/scripts/smoke.sh`
- Create: `el-ai-gateway/Dockerfile`
- Create: `el-ai-gateway/README.md`

- [ ] **Step 1: Implement the entrypoint**

`el-ai-gateway/src/index.ts`:
```ts
import "dotenv/config";
import { loadConfig } from "./config";
import { createAuthenticator } from "./auth";
import { createPlatformFilesClient } from "./upstream/platformFiles";
import { createA2AClient } from "./upstream/a2a";
import { createMcpClient } from "./upstream/mcp";
import { createTaskToolClient } from "./upstream/taskTools";
import { buildServer } from "./server";

async function main(): Promise<void> {
  const config = loadConfig();
  const mcp = createMcpClient(config);
  const app = buildServer({
    config,
    authenticator: createAuthenticator(config),
    platformFiles: createPlatformFilesClient(config),
    a2a: createA2AClient(config),
    taskTools: createTaskToolClient(mcp),
  });
  await app.listen({ port: config.port, host: "0.0.0.0" });
  app.log.info(`el-ai-gateway listening on :${config.port}`);
}

main().catch((error) => {
  console.error("Failed to start el-ai-gateway:", error);
  process.exit(1);
});
```

- [ ] **Step 2: Implement the MCP probe script**

`el-ai-gateway/scripts/mcp-probe.ts`:
```ts
import "dotenv/config";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url = process.env.MCP_SERVER_URL ?? "http://127.0.0.1:5702/open/mcp";
const key = process.env.MCP_API_KEY;
if (!key) {
  console.error("MCP_API_KEY is required");
  process.exit(1);
}

const transport = new StreamableHTTPClientTransport(new URL(url), {
  requestInit: { headers: { Authorization: `Bearer ${key}` } },
});
const client = new Client({ name: "el-ai-gateway-probe", version: "0.1.0" }, { capabilities: {} });
await client.connect(transport);

const tools = await client.listTools();
console.log("tools:", JSON.stringify(tools, null, 2));

if (process.env.MCP_PROBE_ACTION === "create_task") {
  const created = await client.callTool({
    name: "task_manage_task",
    arguments: { action: "create", title: "el-ai-gateway probe", status: "in_progress" },
  });
  console.log("create_task:", JSON.stringify(created, null, 2));
}

await client.close();
```

- [ ] **Step 3: Implement the smoke script**

`el-ai-gateway/scripts/smoke.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:5708}"
KEY="${KEY:-dev_key}"
FILE="${1:-/tmp/voice.wav}"

echo "== health =="
curl -sS "$BASE/health"; echo

echo "== upload =="
UPLOAD=$(curl -sS -X POST "$BASE/api/v1/files?path=voice" \
  -H "Authorization: Bearer $KEY" \
  -F "file=@${FILE}")
echo "$UPLOAD"
UUID=$(printf '%s' "$UPLOAD" | python3 -c 'import sys,json;print(json.load(sys.stdin)["uuid"])')

echo "== start task =="
TASK=$(curl -sS -X POST "$BASE/api/v1/tasks" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"uuid\":\"$UUID\"}")
echo "$TASK"
TASK_ID=$(printf '%s' "$TASK" | python3 -c 'import sys,json;print(json.load(sys.stdin)["taskId"])')

echo "== task status =="
curl -sS "$BASE/api/v1/tasks/$TASK_ID" -H "Authorization: Bearer $KEY"; echo

echo "== feedback =="
curl -sS -X POST "$BASE/api/v1/tasks/$TASK_ID/feedback" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"content":"Smoke test feedback."}'; echo
```

- [ ] **Step 4: Implement the Dockerfile**

`el-ai-gateway/Dockerfile`:
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --no-frozen-lockfile
COPY tsconfig.json ./
COPY src ./src
RUN pnpm build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
RUN corepack enable
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --prod --no-frozen-lockfile
COPY --from=build /app/dist ./dist
EXPOSE 5708
CMD ["node", "dist/index.js"]
```

- [ ] **Step 5: Implement the README**

`el-ai-gateway/README.md`:
````markdown
# el-ai-gateway

Estée Lauder AI project API gateway for voice-file tagging. Standalone Fastify + TypeScript
service that orchestrates:

- `platform-service` (5707) files — upload + presigned download URL
- `agent` (5702) A2A — triggers the voice agent (transcription + tagging)
- `agent` (5702) `/open/mcp` — task/activity tools (`task_manage_task`)

## Endpoints (all require `Authorization: Bearer <GATEWAY_API_KEYS key>`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/files?path=&fileName=` | multipart `file`; returns platform upload receipt (`uuid`, ...) |
| POST | `/api/v1/tasks` | `{uuid,title?,description?,assistantId?}`; presigns, creates a task, triggers A2A |
| GET | `/api/v1/tasks/:id` | task status + recent activities |
| POST | `/api/v1/tasks/:id/feedback` | `{content,summary?}`; appends an activity |

## Run

```bash
cp .env.example .env   # fill A2A_API_KEY / MCP_API_KEY / A2A_VOICE_TAGGING_ASSISTANT_ID
pnpm install
pnpm dev
```

## Probe the MCP server

```bash
pnpm mcp:probe                       # prints tools/list
MCP_PROBE_ACTION=create_task pnpm mcp:probe   # also tries task_manage_task create
```

## Test

```bash
pnpm test
pnpm typecheck
```

## Smoke (against running gateway + real upstreams)

```bash
./scripts/smoke.sh /path/to/voice.wav
```

## Notes

- The two webhook callbacks are fired by the agent platform, not this gateway.
- Task/activity state lives in the agent platform (`task_manage_task`); the gateway is stateless.
````

- [ ] **Step 6: Typecheck and test**

Run: `cd el-ai-gateway && pnpm typecheck && pnpm test`
Expected: `tsc` exits 0; all tests PASS.

- [ ] **Step 7: Probe the real MCP server (integration check)**

Put real credentials in `el-ai-gateway/.env` (`MCP_API_KEY` from the platform; optionally `A2A_API_KEY`).
Run: `cd el-ai-gateway && MCP_PROBE_ACTION=create_task pnpm mcp:probe`
Expected: `tools/list` includes `task_manage_task`; `create_task` returns success with a task id.

If `task_manage_task` create fails with a context/authority error, stop and record the exact error — the task-tool wiring (spec §12 item 2) needs adjustment before continuing.

- [ ] **Step 8: Commit**

```bash
git add el-ai-gateway/src/index.ts el-ai-gateway/scripts el-ai-gateway/Dockerfile el-ai-gateway/README.md
git commit -m "feat(el-ai-gateway): entrypoint, probe/smoke scripts, Dockerfile, README"
```

---

## Task 10: Final verification

- [ ] **Step 1: Full test + typecheck**

Run: `cd el-ai-gateway && pnpm typecheck && pnpm test`
Expected: typecheck clean; all tests PASS.

- [ ] **Step 2: Build**

Run: `cd el-ai-gateway && pnpm build`
Expected: `dist/index.js` produced, no errors.

- [ ] **Step 3: Confirm no repo-wide collateral changes**

Run: `git -C /Users/simon/code/fina_demo status --short`
Expected: only files under `el-ai-gateway/` and the spec/plan docs; no `docker-compose*.yml` or nginx changes.

- [ ] **Step 4: Commit any remaining files**

```bash
git add el-ai-gateway
git commit -m "chore(el-ai-gateway): finalize project" || echo "nothing to commit"
```

---

## Self-Review Notes

- **Spec coverage:** §5.1 files → Task 7; §5.2 tasks start → Task 8; §5.3 status → Task 8; §5.4 feedback → Task 8; §6 auth → Task 3; §7.1 platformFiles → Task 4; §7.2 a2a → Task 5; §7.3 mcp/taskTools → Task 6; §8 config → Task 1; §9 errors → Task 2/Task 7; §10 tests → every task; §11 deliverables → Task 9; §12 open items → Task 9 Step 7 (probe) + spec notes.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:** `Config`, `Principal`, `Authenticator`, `PlatformFilesClient`, `A2AClient`, `McpCaller`, `TaskToolClient` are defined once (Tasks 1/3/4/5/6) and reused verbatim in `server.ts` and routes.
