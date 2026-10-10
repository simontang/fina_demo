# Volcano ASR Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Axiom Lattice `Plugin` (`type: "volcano-asr"`) exposing Volcano Engine (Doubao Voice) recording-file ASR — async submit/query, one-shot recognize, and synchronous flash — with connection config and a real connection test.

**Architecture:** Self-contained plugin under `agent/src/agents/volcano_asr/`, mirroring `platform_service/storage/`: a `client.ts` (connection resolution + volc HTTP + text extraction), `executors.ts` (pure `(input, exeConfig, rawConfig) => Promise<string>` JSON-returning executors), and `plugin.ts` (Plugin object, `PluginConnection`, middleware, `PluginRegistry.register`). Registered via a side-effect import in `agent/src/agents/index.ts`.

**Tech Stack:** TypeScript, `langchain` (`createMiddleware`/`tool`), `zod` 3.x, `@axiom-lattice/core` `PluginRegistry`, `@axiom-lattice/protocols` `Plugin`, Jest + ts-jest (isolatedModules), mocked `fetch` (no network).

**Spec:** `docs/superpowers/specs/2026-10-10-volcano-asr-plugin-design.md`

**Workdir for all commands:** `agent/` (prefix commands with `cd agent` or use `workdir`). Tests run with `npx jest`.

---

## File Structure

```
agent/src/agents/
  index.ts                              # MODIFY: add import "./volcano_asr"
  volcano_asr/
    index.ts                            # CREATE: import "./plugin"
    client.ts                           # CREATE: constants + connection resolve + callApi + extract/error helpers
    executors.ts                        # CREATE: submit / get / recognize / recognize_flash
    plugin.ts                           # CREATE: Plugin + connection.test + middleware + register
    __tests__/
      mockResponse.ts                   # CREATE: test-only fetch Response factory
      client.test.ts                    # CREATE
      executors.test.ts                 # CREATE
      plugin.test.ts                    # CREATE
```

## Volcano ASR API reference (from the sandbox implementation)

- Standard edition (async, poll): `POST {baseUrl}/api/v3/auc/bigmodel/submit` and `POST {baseUrl}/api/v3/auc/bigmodel/query`; resource id `volc.seedasr.auc` (2.0) / `volc.bigasr.auc` (1.0); submit sends `X-Api-Sequence: -1`.
- Flash edition (sync): `POST {baseUrl}/api/v3/auc/bigmodel/recognize/flash`; resource id `volc.bigasr.auc_turbo`; `audio.format` is required.
- Auth/status travel in headers: request `X-Api-Key`, `X-Api-Resource-Id`, `X-Api-Request-Id`, `X-Api-Sequence`; response `X-Api-Status-Code`, `X-Api-Message`, `X-Api-Request-Id`.
- Status codes: `20000000` success; `20000001` / `20000002` still processing.
- Body shape: `{ audio: { url, format?, codec?, rate?, bits?, channel?, language? }, request: {...} }`.

---

## Task 1: `client.ts` — connection resolution and volc HTTP client

**Files:**
- Create: `agent/src/agents/volcano_asr/client.ts`
- Create: `agent/src/agents/volcano_asr/__tests__/mockResponse.ts`
- Test: `agent/src/agents/volcano_asr/__tests__/client.test.ts`

- [ ] **Step 1: Write the test Response factory helper**

Create `agent/src/agents/volcano_asr/__tests__/mockResponse.ts`:

```ts
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
```

- [ ] **Step 2: Write the failing client tests**

Create `agent/src/agents/volcano_asr/__tests__/client.test.ts`:

```ts
import { callApi, extractText, isProcessing, isSuccess } from "../client";
import { mockResponse } from "./mockResponse";

describe("volcano-asr client", () => {
  afterEach(() => jest.restoreAllMocks());

  it("classifies success and processing status codes", () => {
    expect(isSuccess("20000000")).toBe(true);
    expect(isSuccess("20000001")).toBe(false);
    expect(isProcessing("20000001")).toBe(true);
    expect(isProcessing("20000002")).toBe(true);
    expect(isProcessing("20000000")).toBe(false);
  });

  it("extracts text from flat, nested, and utterance payloads", () => {
    expect(extractText({ result: { text: "nested" } })).toBe("nested");
    expect(extractText({ text: "flat" })).toBe("flat");
    expect(extractText({ result: { utterances: [{ text: "a" }, { text: "b" }] } })).toBe("ab");
    expect(extractText(null)).toBe("");
  });

  it("sends auth headers and parses response headers", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue(
      mockResponse({
        statusCode: "20000000",
        message: "OK",
        requestId: "rid-1",
        body: { result: { text: "hey" } },
      }),
    );

    const res = await callApi({
      baseUrl: "https://x",
      apiKey: "k",
      resourceId: "volc.seedasr.auc",
      path: "/submit",
      body: { a: 1 },
      requestId: "req-1",
      sequence: -1,
    });

    expect(res.statusCode).toBe("20000000");
    expect(res.requestId).toBe("rid-1");
    expect(res.body).toEqual({ result: { text: "hey" } });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://x/submit");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("k");
    expect(headers["X-Api-Resource-Id"]).toBe("volc.seedasr.auc");
    expect(headers["X-Api-Request-Id"]).toBe("req-1");
    expect(headers["X-Api-Sequence"]).toBe("-1");
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx jest src/agents/volcano_asr/__tests__/client.test.ts`
Expected: FAIL — `Cannot find module '../client'`.

- [ ] **Step 4: Implement `client.ts`**

Create `agent/src/agents/volcano_asr/client.ts`:

```ts
import { randomUUID } from "node:crypto";

export const DEFAULT_BASE_URL = "https://openspeech.bytedance.com";
export const DEFAULT_RESOURCE_ID = "volc.seedasr.auc";
export const SUBMIT_PATH = "/api/v3/auc/bigmodel/submit";
export const QUERY_PATH = "/api/v3/auc/bigmodel/query";
export const FLASH_PATH = "/api/v3/auc/bigmodel/recognize/flash";
export const FLASH_RESOURCE_ID = "volc.bigasr.auc_turbo";

export const STATUS_SUCCESS = "20000000";
export const STATUS_PROCESSING = ["20000001", "20000002"];

const CONNECTION_TYPE = "volcano-asr";

export interface VolcanoAsrConn {
  apiKey: string;
  baseUrl: string;
  resourceId: string;
}

function firstResolvedConfig(container?: unknown): Record<string, unknown> {
  const c = container as
    | { _resolvedConnections?: Array<{ config?: Record<string, unknown> }> }
    | undefined;
  return c?._resolvedConnections?.[0]?.config ?? {};
}

export function runConfigOf(exeConfig?: unknown): Record<string, unknown> {
  return (
    ((exeConfig as { configurable?: { runConfig?: Record<string, unknown> } })?.configurable
      ?.runConfig as Record<string, unknown>) ?? {}
  );
}

function tenantFromExeConfig(exeConfig?: unknown): string {
  const t = runConfigOf(exeConfig).tenantId;
  if (typeof t === "string" && t.trim()) return t.trim();
  throw new Error("tenant context is missing");
}

export function normalizeConnection(config: Record<string, unknown>): VolcanoAsrConn {
  const apiKeyRaw = config.apiKey;
  const envKey = process.env.VOLC_ASR_API_KEY?.trim();
  const apiKey =
    typeof apiKeyRaw === "string" && apiKeyRaw.trim() ? apiKeyRaw.trim() : envKey || "";
  const baseUrlRaw = config.baseUrl;
  const baseUrl = (
    typeof baseUrlRaw === "string" && baseUrlRaw.trim() ? baseUrlRaw.trim() : DEFAULT_BASE_URL
  ).replace(/\/+$/, "");
  const resourceIdRaw = config.resourceId;
  const resourceId =
    typeof resourceIdRaw === "string" && resourceIdRaw.trim() ? resourceIdRaw.trim() : DEFAULT_RESOURCE_ID;
  return { apiKey, baseUrl, resourceId };
}

export function connectionFromConfig(config: Record<string, unknown>): VolcanoAsrConn {
  return normalizeConnection(config ?? {});
}

function noConnectionHint(connectionType: string, tenantId: string): string {
  return (
    `No "${connectionType}" connection is configured for tenant "${tenantId}". ` +
    `Add a connection of type "${connectionType}" and select it (connections) or enable connectAll ` +
    `in the agent's middleware config.`
  );
}

/**
 * Resolve the Volcano ASR connection for a tool invocation.
 *
 * Prefers a host-injected, pre-resolved connection (`_resolvedConnections` in the
 * plugin config or `runConfig`). Otherwise resolves the selected `volcano-asr`
 * connection from the tenant-scoped Connection Store. Falls back to env/default
 * only when no selector is configured.
 */
export async function resolveConnection(
  pluginConfig?: unknown,
  exeConfig?: unknown,
): Promise<VolcanoAsrConn> {
  const preResolved = {
    ...firstResolvedConfig(pluginConfig),
    ...firstResolvedConfig(runConfigOf(exeConfig)),
  };
  if (Object.keys(preResolved).length > 0) return normalizeConnection(preResolved);

  const selector = (pluginConfig ?? {}) as {
    connectionType?: unknown;
    connections?: unknown;
    connectAll?: unknown;
  };
  const connectionType =
    typeof selector.connectionType === "string" ? selector.connectionType : CONNECTION_TYPE;
  const connections = Array.isArray(selector.connections)
    ? selector.connections.filter((k): k is string => typeof k === "string")
    : [];
  const connectAll = selector.connectAll === true;

  if (!connectAll && connections.length === 0) return normalizeConnection({});

  const tenantId = tenantFromExeConfig(exeConfig);
  const { resolvePluginConnections } = await import("@axiom-lattice/core");
  const resolved = await resolvePluginConnections(
    connectionType,
    { connections, connectAll },
    { tenantId },
  );
  if (resolved.length === 0) throw new Error(noConnectionHint(connectionType, tenantId));
  return normalizeConnection(resolved[0].config);
}

export function newRequestId(): string {
  return randomUUID();
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface CallApiOptions {
  baseUrl: string;
  apiKey: string;
  resourceId: string;
  path: string;
  body?: unknown;
  requestId: string;
  sequence?: number;
}

export interface CallApiResult {
  httpStatus: number;
  statusCode: string;
  message: string;
  requestId: string;
  body?: unknown;
  rawBody: string;
}

export async function callApi(opts: CallApiOptions): Promise<CallApiResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Api-Key": opts.apiKey,
    "X-Api-Resource-Id": opts.resourceId,
    "X-Api-Request-Id": opts.requestId,
  };
  if (opts.sequence !== undefined) headers["X-Api-Sequence"] = String(opts.sequence);

  const res = await fetch(opts.baseUrl + opts.path, {
    method: "POST",
    headers,
    body: JSON.stringify(opts.body ?? {}),
  });

  const statusCode = res.headers.get("x-api-status-code") ?? "";
  const message = res.headers.get("x-api-message") ?? "";
  const respRequestId = res.headers.get("x-api-request-id") ?? opts.requestId;
  const rawBody = await res.text();

  let body: unknown;
  try {
    body = rawBody ? JSON.parse(rawBody) : undefined;
  } catch {
    body = undefined;
  }

  return { httpStatus: res.status, statusCode, message, requestId: respRequestId, body, rawBody };
}

export function isSuccess(statusCode: string): boolean {
  return statusCode === STATUS_SUCCESS;
}

export function isProcessing(statusCode: string): boolean {
  return STATUS_PROCESSING.includes(statusCode);
}

export function extractText(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const result = (body as { result?: unknown }).result ?? body;
  const text = (result as { text?: unknown }).text;
  if (typeof text === "string") return text;
  const nested = (result as { result?: { text?: unknown } }).result;
  if (nested && typeof nested.text === "string") return nested.text;
  const utterances = (result as { utterances?: unknown }).utterances;
  if (Array.isArray(utterances)) {
    return utterances
      .map((u) =>
        u && typeof (u as { text?: unknown }).text === "string" ? (u as { text: string }).text : "",
      )
      .filter(Boolean)
      .join("");
  }
  return "";
}

export function extractUtterances(body: unknown): unknown[] | undefined {
  if (!body || typeof body !== "object") return undefined;
  const result = (body as { result?: unknown }).result ?? body;
  const direct = (result as { utterances?: unknown }).utterances;
  if (Array.isArray(direct)) return direct;
  const nested = (result as { result?: { utterances?: unknown } }).result;
  if (nested && Array.isArray(nested.utterances)) return nested.utterances;
  return undefined;
}

export function errorResult(err: unknown): string {
  return JSON.stringify({
    ok: false,
    code: "ERROR",
    message: err instanceof Error ? err.message : String(err),
  });
}

export function maskSecret(value: string): string {
  const s = String(value || "");
  if (!s) return "(empty)";
  if (s.length <= 8) return s.slice(0, 2) + "***";
  return s.slice(0, 4) + "***" + s.slice(-4) + " (len=" + s.length + ")";
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx jest src/agents/volcano_asr/__tests__/client.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add agent/src/agents/volcano_asr/client.ts agent/src/agents/volcano_asr/__tests__/mockResponse.ts agent/src/agents/volcano_asr/__tests__/client.test.ts
git commit -m "feat(agent): add volcano-asr client helpers"
```

---

## Task 2: `executors.ts` — submit / get / recognize / recognize_flash

**Files:**
- Create: `agent/src/agents/volcano_asr/executors.ts`
- Test: `agent/src/agents/volcano_asr/__tests__/executors.test.ts`

- [ ] **Step 1: Write the failing executor tests**

Create `agent/src/agents/volcano_asr/__tests__/executors.test.ts`:

```ts
jest.mock("@axiom-lattice/core", () => ({
  resolvePluginConnections: jest.fn(),
}));

import { resolvePluginConnections } from "@axiom-lattice/core";
import {
  getTranscription,
  recognize,
  recognizeFlash,
  submitTranscription,
} from "../executors";
import { mockResponse } from "./mockResponse";

const RESOLVED = {
  _resolvedConnections: [
    { config: { apiKey: "test-key", baseUrl: "https://openspeech.bytedance.com" } },
  ],
};

beforeAll(() => {
  delete process.env.VOLC_ASR_API_KEY;
});

describe("volcano-asr executors", () => {
  afterEach(() => {
    jest.restoreAllMocks();
    (resolvePluginConnections as jest.Mock).mockReset();
  });

  it("submit_transcription posts to submit with sequence -1 and returns requestId", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", message: "ok", requestId: "task-1" }));

    const out = JSON.parse(await submitTranscription({ audioUrl: "https://a.mp3" }, {}, RESOLVED));

    expect(out.requestId).toBe("task-1");
    expect(out.audioUrl).toBe("https://a.mp3");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-Api-Key"]).toBe("test-key");
    expect(headers["X-Api-Sequence"]).toBe("-1");
    const body = JSON.parse(init.body as string);
    expect(body.audio.url).toBe("https://a.mp3");
    expect(body.request.enable_itn).toBe(true);
  });

  it("returns NO_API_KEY when the connection has no apiKey", async () => {
    const out = JSON.parse(
      await submitTranscription(
        { audioUrl: "https://a.mp3" },
        {},
        { _resolvedConnections: [{ config: { baseUrl: "https://x" } }] },
      ),
    );
    expect(out.ok).toBe(false);
    expect(out.code).toBe("NO_API_KEY");
  });

  it("returns SUBMIT_FAILED on a non-success status code", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000010", message: "bad key" }));

    const out = JSON.parse(await submitTranscription({ audioUrl: "https://a.mp3" }, {}, RESOLVED));

    expect(out.ok).toBe(false);
    expect(out.code).toBe("SUBMIT_FAILED");
    expect(out.statusCode).toBe("45000010");
  });

  it("get_transcription polls a processing task until it completes", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse({ statusCode: "20000001", message: "processing" }))
      .mockResolvedValueOnce(
        mockResponse({ statusCode: "20000000", body: { result: { text: "final text" } } }),
      );

    const out = JSON.parse(
      await getTranscription({ requestId: "task-1", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.status).toBe("completed");
    expect(out.text).toBe("final text");
    expect(out.attempts).toBe(2);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("get_transcription returns timeout when the task never completes", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000001", message: "processing" }));

    const out = JSON.parse(
      await getTranscription(
        { requestId: "task-1", timeoutSeconds: 0.05, intervalSeconds: 0 },
        {},
        RESOLVED,
      ),
    );

    expect(out.status).toBe("timeout");
    expect(out.hint).toContain("get_transcription");
  });

  it("get_transcription returns QUERY_FAILED on a terminal error status", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000001", message: "not found" }));

    const out = JSON.parse(
      await getTranscription({ requestId: "task-1", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.ok).toBe(false);
    expect(out.code).toBe("QUERY_FAILED");
  });

  it("recognize submits and then polls to completion", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse({ statusCode: "20000000", requestId: "task-9" }))
      .mockResolvedValueOnce(
        mockResponse({ statusCode: "20000000", body: { result: { text: "done" } } }),
      );

    const out = JSON.parse(
      await recognize({ audioUrl: "https://a.mp3", intervalSeconds: 0 }, {}, RESOLVED),
    );

    expect(out.status).toBe("completed");
    expect(out.requestId).toBe("task-9");
    expect(out.text).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("recognize_flash requires format", async () => {
    const out = JSON.parse(await recognizeFlash({ audioUrl: "https://a.mp3" }, {}, RESOLVED));
    expect(out.ok).toBe(false);
    expect(out.code).toBe("BAD_REQUEST");
  });

  it("recognize_flash uses the turbo resource id and returns text", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", body: { result: { text: "flash text" } } }));

    const out = JSON.parse(
      await recognizeFlash({ audioUrl: "https://a.mp3", format: "mp3" }, {}, RESOLVED),
    );

    expect(out.mode).toBe("flash");
    expect(out.text).toBe("flash text");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash");
    expect((init.headers as Record<string, string>)["X-Api-Resource-Id"]).toBe(
      "volc.bigasr.auc_turbo",
    );
  });

  it("resolves the connection through the selector with plugin type volcano-asr", async () => {
    (resolvePluginConnections as jest.Mock).mockResolvedValue([
      { key: "c1", config: { apiKey: "k", baseUrl: "https://y" } },
    ]);
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", requestId: "t" }));

    await submitTranscription(
      { audioUrl: "https://a.mp3" },
      { configurable: { runConfig: { tenantId: "t1" } } },
      { connections: ["c1"], connectAll: false },
    );

    expect(resolvePluginConnections).toHaveBeenCalledWith(
      "volcano-asr",
      { connections: ["c1"], connectAll: false },
      { tenantId: "t1" },
    );
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("https://y/api/v3/auc/bigmodel/submit");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx jest src/agents/volcano_asr/__tests__/executors.test.ts`
Expected: FAIL — `Cannot find module '../executors'`.

- [ ] **Step 3: Implement `executors.ts`**

Create `agent/src/agents/volcano_asr/executors.ts`:

```ts
import {
  FLASH_PATH,
  FLASH_RESOURCE_ID,
  QUERY_PATH,
  SUBMIT_PATH,
  callApi,
  errorResult,
  extractText,
  extractUtterances,
  isProcessing,
  isSuccess,
  newRequestId,
  resolveConnection,
  sleep,
  type VolcanoAsrConn,
} from "./client";

const NO_API_KEY = JSON.stringify({
  ok: false,
  code: "NO_API_KEY",
  message: "未配置 API Key：请在插件连接器中填写火山引擎语音服务的 API Key。",
});

function badRequest(message: string): string {
  return JSON.stringify({ ok: false, code: "BAD_REQUEST", message });
}

type ResponseFormat = "text" | "standard" | "full";

interface AudioInput {
  audioUrl: string;
  format?: string;
  codec?: string;
  rate?: number;
  bits?: number;
  channel?: number;
  language?: string;
}

interface SubmitInput extends AudioInput {
  modelName?: string;
  enableItn?: boolean;
  enablePunc?: boolean;
  enableDdc?: boolean;
  enableSpeakerInfo?: boolean;
  showUtterances?: boolean;
  sensitiveWordsFilter?: string;
}

interface PollInput {
  timeoutSeconds?: number;
  intervalSeconds?: number;
  responseFormat?: ResponseFormat;
}

function buildAudio(input: AudioInput): Record<string, unknown> {
  const audio: Record<string, unknown> = { url: input.audioUrl };
  if (input.format) audio.format = input.format;
  if (input.codec) audio.codec = input.codec;
  if (input.rate) audio.rate = input.rate;
  if (input.bits) audio.bits = input.bits;
  if (input.channel) audio.channel = input.channel;
  if (input.language) audio.language = input.language;
  return audio;
}

function buildSubmitRequest(input: SubmitInput): Record<string, unknown> {
  return {
    model_name: input.modelName ?? "bigmodel",
    enable_itn: input.enableItn ?? true,
    enable_punc: input.enablePunc ?? false,
    enable_ddc: input.enableDdc ?? false,
    enable_speaker_info: input.enableSpeakerInfo ?? false,
    enable_channel_split: false,
    show_utterances: input.showUtterances ?? false,
    vad_segment: false,
    sensitive_words_filter: input.sensitiveWordsFilter ?? "",
  };
}

function pollTimeout(input: PollInput): number {
  return typeof input.timeoutSeconds === "number" && input.timeoutSeconds > 0
    ? input.timeoutSeconds
    : 120;
}

function pollInterval(input: PollInput): number {
  return typeof input.intervalSeconds === "number" && input.intervalSeconds >= 0
    ? input.intervalSeconds
    : 3;
}

interface PollResult {
  outcome: "completed" | "timeout" | "failed";
  statusCode: string;
  httpStatus: number;
  message: string;
  body?: unknown;
  attempts: number;
}

async function poll(
  conn: VolcanoAsrConn,
  requestId: string,
  timeoutSeconds: number,
  intervalSeconds: number,
): Promise<PollResult> {
  const deadline = Date.now() + timeoutSeconds * 1000;
  let attempts = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    attempts += 1;
    const last = await callApi({
      baseUrl: conn.baseUrl,
      apiKey: conn.apiKey,
      resourceId: conn.resourceId,
      path: QUERY_PATH,
      body: {},
      requestId,
    });
    if (isSuccess(last.statusCode)) {
      return {
        outcome: "completed",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        body: last.body,
        attempts,
      };
    }
    if (!isProcessing(last.statusCode)) {
      return {
        outcome: "failed",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        body: last.body,
        attempts,
      };
    }
    if (Date.now() + intervalSeconds * 1000 > deadline) {
      return {
        outcome: "timeout",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        attempts,
      };
    }
    await sleep(intervalSeconds * 1000);
  }
}

function formatCompleted(requestId: string, r: PollResult, responseFormat: ResponseFormat): string {
  const text = extractText(r.body);
  if (responseFormat === "text") return JSON.stringify({ text });
  const result: Record<string, unknown> = {
    requestId,
    status: "completed",
    statusCode: r.statusCode,
    text,
    attempts: r.attempts,
  };
  if (responseFormat === "full") {
    const utterances = extractUtterances(r.body);
    if (utterances) result.utterances = utterances;
    if (r.body) result.raw = r.body;
  }
  return JSON.stringify(result);
}

function formatTimeout(requestId: string, r: PollResult, timeoutSeconds: number): string {
  return JSON.stringify({
    requestId,
    status: "timeout",
    statusCode: r.statusCode,
    message: r.message || "任务仍在处理中",
    attempts: r.attempts,
    hint: `等待超过 ${timeoutSeconds} 秒仍未完成，可稍后再次调用 get_transcription 传入 requestId 继续查询。`,
  });
}

function queryFailed(requestId: string, r: PollResult): string {
  return JSON.stringify({
    ok: false,
    code: "QUERY_FAILED",
    requestId,
    statusCode: r.statusCode,
    httpStatus: r.httpStatus,
    message: r.message || "未知错误",
    attempts: r.attempts,
  });
}

async function submitTask(
  conn: VolcanoAsrConn,
  input: SubmitInput,
): Promise<{ ok: true; requestId: string } | { ok: false; payload: string }> {
  const requestId = newRequestId();
  const resp = await callApi({
    baseUrl: conn.baseUrl,
    apiKey: conn.apiKey,
    resourceId: conn.resourceId,
    path: SUBMIT_PATH,
    body: { audio: buildAudio(input), request: buildSubmitRequest(input) },
    requestId,
    sequence: -1,
  });
  if (!isSuccess(resp.statusCode)) {
    return {
      ok: false,
      payload: JSON.stringify({
        ok: false,
        code: "SUBMIT_FAILED",
        requestId: resp.requestId,
        statusCode: resp.statusCode,
        httpStatus: resp.httpStatus,
        message: resp.message || "未知错误",
      }),
    };
  }
  return { ok: true, requestId: resp.requestId };
}

export async function submitTranscription(
  input: SubmitInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    const submitted = await submitTask(conn, input);
    if (!submitted.ok) return submitted.payload;
    return JSON.stringify({
      requestId: submitted.requestId,
      status: "submitted",
      audioUrl: input.audioUrl,
      hint: "任务已提交，请用 get_transcription 传入 requestId 查询结果（该工具会自动轮询）。",
    });
  } catch (err) {
    return errorResult(err);
  }
}

export async function getTranscription(
  input: { requestId?: string } & PollInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.requestId || typeof input.requestId !== "string") {
      return badRequest("缺少必填参数 requestId（submit_transcription 返回的任务 ID）。");
    }
    const timeoutSeconds = pollTimeout(input);
    const intervalSeconds = pollInterval(input);
    const responseFormat = input.responseFormat ?? "standard";
    const result = await poll(conn, input.requestId, timeoutSeconds, intervalSeconds);
    if (result.outcome === "completed") return formatCompleted(input.requestId, result, responseFormat);
    if (result.outcome === "timeout") return formatTimeout(input.requestId, result, timeoutSeconds);
    return queryFailed(input.requestId, result);
  } catch (err) {
    return errorResult(err);
  }
}

export async function recognize(
  input: SubmitInput & PollInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    const submitted = await submitTask(conn, input);
    if (!submitted.ok) return submitted.payload;
    const timeoutSeconds = pollTimeout(input);
    const intervalSeconds = pollInterval(input);
    const responseFormat = input.responseFormat ?? "standard";
    const result = await poll(conn, submitted.requestId, timeoutSeconds, intervalSeconds);
    if (result.outcome === "completed") return formatCompleted(submitted.requestId, result, responseFormat);
    if (result.outcome === "timeout") return formatTimeout(submitted.requestId, result, timeoutSeconds);
    return queryFailed(submitted.requestId, result);
  } catch (err) {
    return errorResult(err);
  }
}

interface FlashInput extends AudioInput {
  modelName?: string;
  enableItn?: boolean;
  enablePunc?: boolean;
  enableDdc?: boolean;
  enableChannelSplit?: boolean;
  showUtterances?: boolean;
  vadSegment?: boolean;
  enableAutoLang?: boolean;
  enableLid?: boolean;
  endWindowSize?: number;
  sensitiveWordsFilter?: string;
  responseFormat?: ResponseFormat;
}

export async function recognizeFlash(
  input: FlashInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    if (!input.format || typeof input.format !== "string") {
      return badRequest("缺少必填参数 format（极速版要求显式指定音频格式，如 mp3 / wav / ogg）。");
    }

    const request: Record<string, unknown> = {
      model_name: input.modelName ?? "bigmodel",
      enable_itn: input.enableItn ?? true,
      enable_punc: input.enablePunc ?? true,
      enable_ddc: input.enableDdc ?? false,
      enable_channel_split: input.enableChannelSplit ?? false,
      show_utterances: input.showUtterances ?? false,
      vad_segment: input.vadSegment ?? false,
    };
    if (input.enableAutoLang !== undefined) request.enable_auto_lang = input.enableAutoLang;
    if (input.enableLid !== undefined) request.enable_lid = input.enableLid;
    if (input.endWindowSize) request.end_window_size = input.endWindowSize;
    if (input.sensitiveWordsFilter) request.sensitive_words_filter = input.sensitiveWordsFilter;

    const requestId = newRequestId();
    const resp = await callApi({
      baseUrl: conn.baseUrl,
      apiKey: conn.apiKey,
      resourceId: FLASH_RESOURCE_ID,
      path: FLASH_PATH,
      body: { audio: buildAudio(input), request },
      requestId,
      sequence: -1,
    });
    if (!isSuccess(resp.statusCode)) {
      return JSON.stringify({
        ok: false,
        code: "FLASH_FAILED",
        requestId: resp.requestId,
        statusCode: resp.statusCode,
        httpStatus: resp.httpStatus,
        message: resp.message || "未知错误",
      });
    }

    const text = extractText(resp.body);
    const responseFormat = input.responseFormat ?? "standard";
    if (responseFormat === "text") return JSON.stringify({ text });
    const result: Record<string, unknown> = {
      requestId: resp.requestId,
      status: "completed",
      statusCode: resp.statusCode,
      mode: "flash",
      text,
    };
    if (responseFormat === "full") {
      const utterances = extractUtterances(resp.body);
      if (utterances) result.utterances = utterances;
      if (resp.body) result.raw = resp.body;
    }
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx jest src/agents/volcano_asr/__tests__/executors.test.ts`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/src/agents/volcano_asr/executors.ts agent/src/agents/volcano_asr/__tests__/executors.test.ts
git commit -m "feat(agent): add volcano-asr executors"
```

---

## Task 3: `plugin.ts` + registration

**Files:**
- Create: `agent/src/agents/volcano_asr/plugin.ts`
- Create: `agent/src/agents/volcano_asr/index.ts`
- Modify: `agent/src/agents/index.ts`
- Test: `agent/src/agents/volcano_asr/__tests__/plugin.test.ts`

- [ ] **Step 1: Write the failing plugin tests**

Create `agent/src/agents/volcano_asr/__tests__/plugin.test.ts`:

```ts
jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  resolvePluginConnections: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
import { volcanoAsrPlugin } from "../plugin";
import { mockResponse } from "./mockResponse";

describe("volcano-asr plugin", () => {
  afterEach(() => jest.restoreAllMocks());

  it("registers a plugin with type volcano-asr", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(volcanoAsrPlugin);
    expect(volcanoAsrPlugin.meta.type).toBe("volcano-asr");
  });

  it("declares exactly the four ASR tools", async () => {
    const mw = (await volcanoAsrPlugin.middleware!({})) as {
      tools: Array<{ name: string }>;
    };
    expect(mw.tools.map((t) => t.name).sort()).toEqual([
      "get_transcription",
      "recognize",
      "recognize_flash",
      "submit_transcription",
    ]);
    expect(volcanoAsrPlugin.meta.tools?.map((t) => t.name).sort()).toEqual([
      "get_transcription",
      "recognize",
      "recognize_flash",
      "submit_transcription",
    ]);
  });

  it("openExpose names all exist among middleware tools, get_transcription readOnly", async () => {
    const mw = (await volcanoAsrPlugin.middleware!({})) as {
      tools: Array<{ name: string }>;
    };
    const toolNames = mw.tools.map((t) => t.name);
    const expose = (volcanoAsrPlugin.meta.openExpose ?? []).map((e) =>
      typeof e === "string" ? { name: e, readOnly: false } : e,
    );
    for (const e of expose) expect(toolNames).toContain(e.name);
    expect(expose.find((e) => e.name === "get_transcription")?.readOnly).toBe(true);
  });

  it("connection.test reports success for a valid key", async () => {
    jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "20000000", message: "ok" }));

    const result = await volcanoAsrPlugin.connection!.test!({
      apiKey: "k",
      baseUrl: "https://openspeech.bytedance.com",
    });

    expect(result.ok).toBe(true);
    expect(result.message).toContain("连接正常");
    expect(result.details).toBeDefined();
  });

  it("connection.test reports auth failure and masks the key", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(mockResponse({ statusCode: "45000010", message: "Invalid X-Api-Key" }));

    const result = await volcanoAsrPlugin.connection!.test!({
      apiKey: "secret-key-123456",
      baseUrl: "https://openspeech.bytedance.com",
    });

    expect(result.ok).toBe(false);
    expect(result.message).toContain("鉴权失败");
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBe("secret-key-123456");
    expect(JSON.stringify(result.details)).not.toContain("secret-key-123456");
  });

  it("connection.test reports a missing api key", async () => {
    const result = await volcanoAsrPlugin.connection!.test!({ baseUrl: "https://x" });
    expect(result.ok).toBe(false);
    expect(result.message).toContain("API Key");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx jest src/agents/volcano_asr/__tests__/plugin.test.ts`
Expected: FAIL — `Cannot find module '../plugin'`.

- [ ] **Step 3: Implement `plugin.ts`**

Create `agent/src/agents/volcano_asr/plugin.ts`:

```ts
import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import {
  DEFAULT_BASE_URL,
  DEFAULT_RESOURCE_ID,
  SUBMIT_PATH,
  callApi,
  connectionFromConfig,
  isSuccess,
  maskSecret,
  newRequestId,
} from "./client";
import {
  getTranscription,
  recognize,
  recognizeFlash,
  submitTranscription,
} from "./executors";

const TEST_AUDIO_URL =
  "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_cancan_mars_bigtts.mp3";

const audioUrlField = z
  .string()
  .url()
  .describe("Publicly accessible audio file URL (http/https)");

const submitFields = {
  audioUrl: audioUrlField,
  format: z.string().optional().describe("Audio format, e.g. mp3/wav/ogg"),
  codec: z.string().optional().describe("Codec, e.g. raw/opus"),
  rate: z.number().int().optional().describe("Sample rate in Hz"),
  bits: z.number().int().optional().describe("Bit depth"),
  channel: z.number().int().optional().describe("Channel count"),
  modelName: z.string().optional(),
  enableItn: z.boolean().optional().describe("Inverse text normalization (default true)"),
  enablePunc: z.boolean().optional().describe("Punctuation (default false; true for flash)"),
  enableDdc: z.boolean().optional(),
  enableSpeakerInfo: z.boolean().optional(),
  showUtterances: z.boolean().optional().describe("Return per-sentence utterances"),
  sensitiveWordsFilter: z.string().optional(),
};

const pollFields = {
  timeoutSeconds: z.number().int().optional().describe("Polling timeout in seconds (default 120)"),
  intervalSeconds: z.number().int().optional().describe("Polling interval in seconds (default 3)"),
  responseFormat: z
    .enum(["text", "standard", "full"])
    .optional()
    .describe("text = transcript only; standard = +metadata; full = +utterances/raw"),
};

const SCHEMAS = {
  submit: z.object(submitFields),
  get: z.object({
    requestId: z.string().describe("requestId returned by submit_transcription"),
    ...pollFields,
  }),
  recognize: z.object({ ...submitFields, ...pollFields }),
  flash: z.object({
    audioUrl: audioUrlField,
    format: z.string().describe("Required for flash: mp3/wav/ogg/..."),
    codec: z.string().optional(),
    rate: z.number().int().optional(),
    bits: z.number().int().optional(),
    channel: z.number().int().optional(),
    language: z.string().optional(),
    modelName: z.string().optional(),
    enableItn: z.boolean().optional(),
    enablePunc: z.boolean().optional(),
    enableDdc: z.boolean().optional(),
    enableChannelSplit: z.boolean().optional(),
    showUtterances: z.boolean().optional(),
    vadSegment: z.boolean().optional(),
    enableAutoLang: z.boolean().optional(),
    enableLid: z.boolean().optional(),
    endWindowSize: z.number().int().optional(),
    sensitiveWordsFilter: z.string().optional(),
    responseFormat: z.enum(["text", "standard", "full"]).optional(),
  }),
};

export const volcanoAsrPlugin: Plugin = {
  meta: {
    type: "volcano-asr",
    name: "Volcano ASR",
    category: "data",
    description:
      "火山引擎（豆包语音）录音文件识别。支持标准版异步识别（提交 + 轮询）与极速版同步识别。输入为音频文件的公网可访问 URL。",
    version: "1.0.0",
    tools: [
      { name: "submit_transcription", description: "Submit an async ASR task, returning a requestId." },
      { name: "get_transcription", description: "Poll an ASR task by requestId until it completes." },
      { name: "recognize", description: "Submit and poll an ASR task in one call." },
      { name: "recognize_flash", description: "Synchronous flash ASR (no polling)." },
    ],
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "Connect all available connections" },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
    openExpose: [
      { name: "submit_transcription" },
      { name: "get_transcription", readOnly: true },
      { name: "recognize" },
      { name: "recognize_flash" },
    ],
  },

  connection: {
    fields: [
      {
        key: "apiKey",
        type: "password",
        title: "API Key",
        widget: "password",
        required: true,
        helpText: "火山引擎语音服务的 API Key（作为 X-Api-Key），可由 VOLC_ASR_API_KEY 环境变量提供",
      },
      {
        key: "baseUrl",
        type: "string",
        title: "Base URL",
        widget: "input",
        helpText: `默认 ${DEFAULT_BASE_URL}`,
      },
      {
        key: "resourceId",
        type: "string",
        title: "Resource ID",
        widget: "input",
        helpText: `标准版资源 id，默认 ${DEFAULT_RESOURCE_ID}；极速版固定使用 volc.bigasr.auc_turbo`,
      },
    ],
    test: async (config) => {
      const conn = connectionFromConfig(config);
      const startedAt = Date.now();
      const diag: Record<string, unknown> = {
        baseUrl: conn.baseUrl,
        resourceId: conn.resourceId,
        apiKey: maskSecret(conn.apiKey),
        endpoint: conn.baseUrl + SUBMIT_PATH,
        testAudioUrl: TEST_AUDIO_URL,
      };

      if (!conn.apiKey) {
        return { ok: false, message: "缺少 API Key：连接器里没有读到 apiKey 字段。", details: diag };
      }

      let resp;
      try {
        resp = await callApi({
          baseUrl: conn.baseUrl,
          apiKey: conn.apiKey,
          resourceId: conn.resourceId,
          path: SUBMIT_PATH,
          body: {
            audio: { url: TEST_AUDIO_URL, format: "mp3" },
            request: { model_name: "bigmodel", enable_itn: true },
          },
          requestId: newRequestId(),
          sequence: -1,
        });
      } catch (err) {
        return {
          ok: false,
          message: `无法连接服务：${err instanceof Error ? err.message : String(err)}`,
          details: { ...diag, elapsedMs: Date.now() - startedAt },
        };
      }

      const elapsedMs = Date.now() - startedAt;
      const details: Record<string, unknown> = {
        ...diag,
        elapsedMs,
        httpStatus: resp.httpStatus,
        statusCode: resp.statusCode || "(none)",
        apiMessage: resp.message || "(none)",
        requestId: resp.requestId,
        responseBody: resp.rawBody ? resp.rawBody.slice(0, 500) : "(empty)",
      };

      if (isSuccess(resp.statusCode)) {
        return {
          ok: true,
          message: `连接正常：鉴权通过，任务已受理（statusCode=${resp.statusCode}，耗时 ${elapsedMs}ms）`,
          details,
        };
      }

      const authFailed =
        resp.statusCode === "45000010" ||
        /invalid x-api-key/i.test(resp.message) ||
        /grant not found/i.test(resp.message) ||
        resp.httpStatus === 401;

      if (authFailed) {
        return {
          ok: false,
          message: `鉴权失败：${resp.message || "Invalid X-Api-Key"}（statusCode=${resp.statusCode || "无"}，httpStatus=${resp.httpStatus}）`,
          details,
        };
      }

      return {
        ok: false,
        message: `连接失败：${resp.message || "未知错误"}（statusCode=${resp.statusCode || "无"}，httpStatus=${resp.httpStatus}）`,
        details,
      };
    },
  },

  middleware: (rawConfig) => {
    const pluginConfig = { ...rawConfig, connectionType: "volcano-asr" };
    return createMiddleware({
      name: "VolcanoAsr",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.submit>, exeConfig) =>
            submitTranscription(input, exeConfig, pluginConfig),
          {
            name: "submit_transcription",
            description:
              "提交一个火山引擎标准版 ASR 异步任务，立即返回 requestId。音频须为公网可访问 URL。随后用 get_transcription 查询结果。",
            schema: SCHEMAS.submit,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.get>, exeConfig) =>
            getTranscription(input, exeConfig, pluginConfig),
          {
            name: "get_transcription",
            description:
              "按 requestId 轮询火山引擎 ASR 任务，直到完成或超时。可传 timeoutSeconds/intervalSeconds/responseFormat。",
            schema: SCHEMAS.get,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.recognize>, exeConfig) =>
            recognize(input, exeConfig, pluginConfig),
          {
            name: "recognize",
            description:
              "一步完成火山引擎标准版 ASR：提交任务并轮询到结果，返回识别文本。适合单次调用。",
            schema: SCHEMAS.recognize,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.flash>, exeConfig) =>
            recognizeFlash(input, exeConfig, pluginConfig),
          {
            name: "recognize_flash",
            description:
              "火山引擎极速版录音文件识别：同步返回结果，无需轮询。必须显式指定 format。限制：最大 100MB、时长不超过 2 小时。",
            schema: SCHEMAS.flash,
          },
        ),
      ],
    });
  },
};

PluginRegistry.register(volcanoAsrPlugin);
```

- [ ] **Step 4: Create the plugin barrel**

Create `agent/src/agents/volcano_asr/index.ts`:

```ts
import "./plugin";
```

- [ ] **Step 5: Register the plugin in the agents barrel**

Modify `agent/src/agents/index.ts` — add the import at the end:

```ts
// import "./research";
// import "./data_agent";
// import "./voice_agent";
// import "./research_data_agent";
import "./sap_b1";
import "./platform_service";
import "./semantic_metrics";
import "./volcano_asr";
```

- [ ] **Step 6: Run the plugin tests to verify they pass**

Run: `npx jest src/agents/volcano_asr/__tests__/plugin.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add agent/src/agents/volcano_asr/plugin.ts agent/src/agents/volcano_asr/index.ts agent/src/agents/index.ts agent/src/agents/volcano_asr/__tests__/plugin.test.ts
git commit -m "feat(agent): register volcano-asr plugin with four ASR tools"
```

---

## Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole volcano-asr suite**

Run: `npx jest src/agents/volcano_asr`
Expected: PASS — 3 suites, 19 tests total (client 3, executors 10, plugin 6).

- [ ] **Step 2: Run the broader agent test suite to check for regressions**

Run: `npx jest`
Expected: PASS — no previously passing suite breaks (new plugin is additive; `agents/index.ts` change is a new import only).

- [ ] **Step 3: Confirm no stray files**

Run: `git status --short`
Expected: only the intended `volcano_asr/` files and the `agents/index.ts` edit are staged/committed; nothing else.

---

## Notes for the implementer

- Do NOT add comments beyond those shown; the repo favors minimal comments.
- `pluginConfig` passed to executors includes `connectionType: "volcano-asr"`; the executors fall back to that type even if omitted, so the selector test passes either way.
- The `test()` callback always returns a `PluginConnectionTestResult` and never throws.
- No network is used in tests — `global.fetch` is always mocked.
