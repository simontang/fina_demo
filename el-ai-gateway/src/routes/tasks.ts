import { randomBytes, randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";
import type { AgentRunsClient } from "../upstream/agentRuns";
import type { TaskToolClient, TaskRecord } from "../upstream/taskTools";
import type { BoTools } from "../upstream/boTools";

export type TaskRouteDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  agentRuns: AgentRunsClient;
  taskTools: TaskToolClient;
  boTools: BoTools;
};

export function renderRunMessage(
  template: string | undefined,
  vars: { uuid: string; url: string; taskId: string },
): string {
  if (template) {
    return template
      .replaceAll("{uuid}", vars.uuid)
      .replaceAll("{url}", vars.url)
      .replaceAll("{taskId}", vars.taskId);
  }
  return `Voice tagging task ${vars.taskId}. Read the task to get the associated file, then transcribe the audio and tag the text.`;
}

export type ResultTag = { tagId: string; tagKey: string; tagValue: string; evidence?: string };
export type TaskResult = { transcript: string | null; tags: ResultTag[]; like: boolean | null };

function normalizeResultTags(input: unknown): ResultTag[] {
  if (!Array.isArray(input)) return [];
  const out: ResultTag[] = [];
  for (const item of input) {
    if (!item || typeof item !== "object") continue;
    const t = item as Record<string, unknown>;
    if (typeof t.tagId !== "string") continue;
    const tag: ResultTag = {
      tagId: t.tagId,
      tagKey: String(t.tagKey ?? t.dimension ?? ""),
      tagValue: String(t.tagValue ?? t.name ?? ""),
    };
    if (typeof t.evidence === "string") tag.evidence = t.evidence;
    out.push(tag);
  }
  return out;
}

export function parseResult(result: string | undefined): TaskResult {
  const empty: TaskResult = { transcript: null, tags: [], like: null };
  if (!result) return empty;
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return empty;
  }
  if (Array.isArray(parsed)) {
    return { ...empty, tags: normalizeResultTags(parsed) };
  }
  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>;
    return {
      transcript: typeof obj.transcript === "string" ? obj.transcript : null,
      tags: normalizeResultTags(obj.tags),
      like: obj.like === true ? true : null,
    };
  }
  return empty;
}

function toDetail(task: TaskRecord) {
  const metadata = (task.metadata ?? {}) as Record<string, unknown>;
  const result = parseResult(task.result);
  return {
    taskId: task.id,
    fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
    status: task.status,
    createdAt: task.createdAt,
    title: task.title,
    durationSec: typeof metadata.durationSec === "number" ? metadata.durationSec : null,
    audioUrl: `/voice-tagging/${task.id}/audio`,
    transcript: result.transcript,
    tags: result.tags,
    like: result.like,
  };
}

// Best-effort: rewrite the customer's aggregate tags (customer_tag) to the union
// of all that customer's task tags. Never throws — a failed reconcile must not
// fail the PUT (the next edit reconciles again).
async function reconcileCustomerTags(
  deps: TaskRouteDeps,
  input: { ownerId: string; baId: string; customerId: string },
): Promise<void> {
  try {
    const tasks = await deps.taskTools.listTasks({
      ownerId: input.ownerId,
      baId: input.baId,
      customerId: input.customerId,
    });
    const union = new Map<string, string | undefined>();
    for (const task of tasks) {
      const when = task.updatedAt ?? task.createdAt;
      for (const tag of parseResult(task.result).tags) {
        if (typeof tag?.tagId !== "string") continue;
        const prev = union.get(tag.tagId);
        if (!prev || (when !== undefined && when > prev)) union.set(tag.tagId, when);
      }
    }
    const existing = await deps.boTools.queryRecords("customer_tag", [
      { field: "customer_no", op: "eq", value: input.customerId },
    ]);
    const byTagId = new Map(existing.map((row) => [String(row.tag_id), row]));
    for (const [tagId, when] of union) {
      const defRows = await deps.boTools.queryRecords("tag_definition", [
        { field: "tag_id", op: "eq", value: tagId },
      ]);
      const tagKey = String(defRows[0]?.tag_group ?? "");
      const tagValue = String(defRows[0]?.tag_name ?? "");
      const row = byTagId.get(tagId);
      if (!row) {
        await deps.boTools.createRecord("customer_tag", {
          customer_no: input.customerId,
          tag_key: tagKey,
          tag_value: tagValue,
          tag_id: tagId,
          source: "voice",
          confidence: null,
          tagged_at: when ?? new Date().toISOString(),
        });
      } else if (String(row.tag_key ?? "") !== tagKey || String(row.tag_value ?? "") !== tagValue) {
        await deps.boTools.updateRecord("customer_tag", String(row.id), { tag_key: tagKey, tag_value: tagValue });
      }
    }
    const staleIds = existing
      .filter((row) => row.source === "voice" && !union.has(String(row.tag_id)))
      .map((row) => String(row.id));
    if (staleIds.length > 0) await deps.boTools.deleteRecords("customer_tag", staleIds);
  } catch (err) {
    console.error(
      `[voice-tagging] customer tag reconcile failed for ${input.customerId}: ${(err as Error).message}`,
    );
  }
}

export function registerTaskRoutes(app: FastifyInstance, deps: TaskRouteDeps): void {
  function dispatchRun(assistantId: string, text: string, taskId: string): void {
    const timeoutMs = deps.config.agentTriggerTimeoutMs ?? deps.config.upstreamTimeoutMs;
    const threadId = randomUUID();
    // Fire-and-forget: the run message carries the task id; the agent writes
    // status/activity back to the task. We do not wait for the run outcome.
    void deps.agentRuns
      .startRun({ assistantId, threadId, text, taskId, timeoutMs })
      .catch((err: unknown) =>
        console.error(
          `[voice-tagging] agent run dispatch failed for task ${taskId}: ${(err as Error).message}`,
        ),
      );
  }

  app.post("/api/v1/voice-tagging", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const body = (request.body ?? {}) as {
      uuid?: string;
      baId?: string;
      customerId?: string;
      durationSec?: unknown;
      title?: string;
      description?: string;
      assistantId?: string;
    };
    const uuid = body.uuid ?? deps.config.voiceTaggingFileUuid;
    if (typeof uuid !== "string" || uuid.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "uuid is required (or set VOICE_TAGGING_FILE_UUID)");
    }
    if (typeof body.baId !== "string" || body.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof body.customerId !== "string" || body.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const durationSec = Number(body.durationSec);
    if (!Number.isFinite(durationSec) || durationSec <= 0) {
      throw new GatewayError(400, "BAD_REQUEST", "durationSec is required (positive seconds)");
    }

    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid });
    const title = body.title ?? `Voice tagging: ${uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: body.description,
      status: "in_progress",
      ownerId: principal.tenantId,
      metadata: { uuid, url, baId: body.baId, customerId: body.customerId, durationSec },
    });

    const assistantId = body.assistantId ?? deps.config.voiceTaggingAssistantId;
    if (!assistantId) {
      throw new GatewayError(
        400,
        "BAD_REQUEST",
        "assistantId is required (or set VOICE_TAGGING_ASSISTANT_ID)",
      );
    }
    dispatchRun(
      assistantId,
      renderRunMessage(deps.config.voiceTaggingMessageTemplate, { uuid, url, taskId }),
      taskId,
    );

    return { taskId, status: "in_progress", file: { uuid, url }, agent: { dispatched: true } };
  });

  // Upload a recording and dispatch the tagging task in one multipart request.
  app.post("/api/v1/voice-tagging/upload", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const query = request.query as {
      baId?: string;
      customerId?: string;
      durationSec?: string;
      title?: string;
      description?: string;
      assistantId?: string;
      path?: string;
      fileName?: string;
      fileCategory?: string;
      usage?: string;
    };
    if (typeof query.baId !== "string" || query.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof query.customerId !== "string" || query.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const durationSec = Number(query.durationSec);
    if (!Number.isFinite(durationSec) || durationSec <= 0) {
      throw new GatewayError(400, "BAD_REQUEST", "durationSec is required (positive seconds)");
    }
    const assistantId = query.assistantId ?? deps.config.voiceTaggingAssistantId;
    if (!assistantId) {
      throw new GatewayError(
        400,
        "BAD_REQUEST",
        "assistantId is required (or set VOICE_TAGGING_ASSISTANT_ID)",
      );
    }
    const data = await request.file();
    if (!data) throw new GatewayError(400, "BAD_REQUEST", "multipart field 'file' is required");

    const uuid = randomUUID().replace(/-/g, "");
    await deps.platformFiles.upload({
      tenantId: principal.tenantId,
      body: data.file,
      uuid,
      filename: data.filename,
      mime: data.mimetype,
      path: query.path,
      fileName: query.fileName,
      fileCategory: query.fileCategory,
      usage: query.usage,
      meta: { baId: query.baId, customerId: query.customerId, durationSec },
    });
    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid });
    const title = query.title ?? `Voice tagging: ${uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: query.description,
      status: "in_progress",
      ownerId: principal.tenantId,
      metadata: { uuid, url, baId: query.baId, customerId: query.customerId, durationSec },
    });
    dispatchRun(
      assistantId,
      renderRunMessage(deps.config.voiceTaggingMessageTemplate, { uuid, url, taskId }),
      taskId,
    );

    return { taskId, status: "in_progress", file: { uuid, url }, agent: { dispatched: true } };
  });

  // List a BA's tasks for a customer (filtered by task metadata via the task service).
  app.get("/api/v1/voice-tagging", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const query = request.query as { baId?: string; customerId?: string };
    if (typeof query.baId !== "string" || query.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof query.customerId !== "string" || query.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const records = await deps.taskTools.listTasks({
      ownerId: principal.tenantId,
      baId: query.baId,
      customerId: query.customerId,
    });
    const tasks = records.map((task) => {
      const metadata = (task.metadata ?? {}) as Record<string, unknown>;
      const result = parseResult(task.result);
      return {
        taskId: task.id,
        fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
        status: task.status,
        createdAt: task.createdAt,
        durationSec: typeof metadata.durationSec === "number" ? metadata.durationSec : null,
        audioUrl: `/voice-tagging/${task.id}/audio`,
        tags: result.tags,
        like: result.like,
      };
    });
    return { baId: query.baId, customerId: query.customerId, total: tasks.length, tasks };
  });

  app.get("/api/v1/voice-tagging/:id", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    return toDetail(task);
  });

  // Stable playback URL: presign on demand and stream the audio through the
  // gateway (Range/206 passthrough) so the client keeps one fixed URL.
  app.get("/api/v1/voice-tagging/:id/audio", async (request, reply) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const metadata = (task.metadata ?? {}) as Record<string, unknown>;
    const uuid = typeof metadata.uuid === "string" ? metadata.uuid : undefined;
    if (!uuid) throw new GatewayError(404, "NOT_FOUND", "task has no associated file");

    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid });
    const range = request.headers.range;
    const controller = new AbortController();
    request.raw.on("close", () => controller.abort());
    const doFetch = (globalThis as { fetch: (input: string, init?: unknown) => Promise<any> }).fetch;

    // HEAD: presigned URLs are GET-only, so probe with a 1-byte GET and answer with headers only.
    if (request.method === "HEAD") {
      let probe: any;
      try {
        probe = await doFetch(url, {
          method: "GET",
          headers: { Range: "bytes=0-0" },
          signal: controller.signal,
        });
      } catch (err) {
        throw new GatewayError(502, "UPSTREAM_ERROR", `audio probe failed: ${(err as Error).message}`);
      }
      if (!probe.ok && probe.status !== 206) {
        throw new GatewayError(502, "UPSTREAM_ERROR", `audio probe failed: ${probe.status}`);
      }
      const contentRange = probe.headers.get("content-range") as string | null;
      const totalMatch = contentRange ? /\/(\d+)$/.exec(contentRange) : null;
      const total = totalMatch
        ? Number(totalMatch[1])
        : Number(probe.headers.get("content-length")) || undefined;
      const headers: Record<string, string> = {
        "content-type": (probe.headers.get("content-type") as string | null) ?? "application/octet-stream",
        "accept-ranges": (probe.headers.get("accept-ranges") as string | null) ?? "bytes",
      };
      if (total) headers["content-length"] = String(total);
      try {
        await probe.body?.cancel?.();
      } catch {
        /* ignore */
      }
      reply.hijack();
      reply.raw.writeHead(200, headers);
      reply.raw.end();
      return reply;
    }

    let upstream: any;
    try {
      upstream = await doFetch(url, {
        method: "GET",
        headers: range ? { Range: range } : {},
        signal: controller.signal,
      });
    } catch (err) {
      throw new GatewayError(502, "UPSTREAM_ERROR", `audio fetch failed: ${(err as Error).message}`);
    }
    if (!upstream.ok && upstream.status !== 206) {
      throw new GatewayError(502, "UPSTREAM_ERROR", `audio fetch failed: ${upstream.status}`);
    }

    const headers: Record<string, string> = {};
    for (const name of [
      "content-type",
      "content-length",
      "content-range",
      "accept-ranges",
      "etag",
      "last-modified",
    ]) {
      const value = upstream.headers.get(name);
      if (value) headers[name] = value;
    }
    if (!headers["content-type"]) headers["content-type"] = "application/octet-stream";
    if (!headers["accept-ranges"]) headers["accept-ranges"] = "bytes";

    reply.hijack();
    reply.raw.writeHead(upstream.status, headers);
    try {
      for await (const chunk of upstream.body as AsyncIterable<Uint8Array>) {
        reply.raw.write(chunk);
      }
    } catch {
      /* client aborted or upstream stream error */
    }
    reply.raw.end();
    return reply;
  });

  // Replace a task's tags (stored in the task `result` object), preserving
  // transcript/like; then reconcile the customer's aggregate tags.
  app.put("/api/v1/voice-tagging/:id/tags", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const prev = parseResult(task.result);

    const metadata = (task.metadata ?? {}) as Record<string, unknown>;
    const baId = typeof metadata.baId === "string" ? metadata.baId : undefined;
    const customerId = typeof metadata.customerId === "string" ? metadata.customerId : undefined;
    if (!baId || !customerId) {
      throw new GatewayError(400, "BAD_REQUEST", "task is missing baId/customerId metadata");
    }

    const body = (request.body ?? {}) as { tags?: unknown };
    if (!Array.isArray(body.tags)) {
      throw new GatewayError(400, "BAD_REQUEST", "tags must be an array");
    }

    const evidenceByTag = new Map(
      prev.tags.filter((t) => t.evidence).map((t) => [t.tagId, t.evidence as string]),
    );
    const out: ResultTag[] = [];
    const seen = new Set<string>();
    for (const item of body.tags) {
      const input = (typeof item === "string" ? { tagId: item } : (item ?? {})) as {
        tagId?: unknown;
        tagValue?: unknown;
      };
      let tagId: string | undefined;
      let tagKey: string;
      let tagValue: string;
      if (typeof input.tagId === "string" && input.tagId.trim() !== "") {
        const rows = await deps.boTools.queryRecords("tag_definition", [
          { field: "tag_id", op: "eq", value: input.tagId },
        ]);
        if (rows.length === 0) {
          throw new GatewayError(400, "BAD_REQUEST", `Unknown tagId: ${input.tagId}`);
        }
        const def = rows[0];
        tagId = String(def.tag_id ?? input.tagId);
        tagKey = String(def.tag_group ?? "");
        tagValue = String(def.tag_name ?? "");
      } else {
        if (typeof input.tagValue !== "string" || input.tagValue.trim() === "") {
          throw new GatewayError(400, "BAD_REQUEST", "each tag needs a tagId or a tagValue");
        }
        tagValue = input.tagValue.trim();
        tagKey = "客户画像";
        const existing = await deps.boTools.queryRecords("tag_definition", [
          { field: "tag_group", op: "eq", value: tagKey },
          { field: "tag_name", op: "eq", value: tagValue },
        ]);
        if (existing.length > 0) {
          tagId = String(existing[0].tag_id);
        } else {
          tagId = randomBytes(16).toString("hex");
          await deps.boTools.createRecord("tag_definition", {
            category: "自定义标签",
            tag_group: tagKey,
            tag_name: tagValue,
            tag_id: tagId,
          });
        }
      }
      if (!tagId || seen.has(tagId)) continue;
      seen.add(tagId);
      const tag: ResultTag = { tagId, tagKey, tagValue };
      const evidence = evidenceByTag.get(tagId);
      if (evidence) tag.evidence = evidence;
      out.push(tag);
    }

    const next: TaskResult = { transcript: prev.transcript, tags: out, like: prev.like };
    await deps.taskTools.updateResult({ id, result: JSON.stringify(next) });
    await reconcileCustomerTags(deps, { ownerId: principal.tenantId, baId, customerId });

    const updated = await deps.taskTools.getTask({ id });
    return toDetail(updated);
  });

  // Set the user's like feedback on the task result (true | null).
  app.put("/api/v1/voice-tagging/:id/like", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const body = (request.body ?? {}) as { like?: unknown };
    if (!("like" in body) || (body.like !== true && body.like !== null)) {
      throw new GatewayError(400, "BAD_REQUEST", "like must be true or null");
    }
    const prev = parseResult(task.result);
    const next: TaskResult = {
      transcript: prev.transcript,
      tags: prev.tags,
      like: body.like === true ? true : null,
    };
    await deps.taskTools.updateResult({ id, result: JSON.stringify(next) });
    const updated = await deps.taskTools.getTask({ id });
    return toDetail(updated);
  });

  // Delete a task, then reconcile the customer's aggregate tags so the deleted
  // task's tags drop out of customer_tag immediately.
  app.delete("/api/v1/voice-tagging/:id", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const metadata = (task.metadata ?? {}) as Record<string, unknown>;
    const baId = typeof metadata.baId === "string" ? metadata.baId : undefined;
    const customerId = typeof metadata.customerId === "string" ? metadata.customerId : undefined;
    await deps.taskTools.deleteTask({ id });
    if (baId && customerId) {
      await reconcileCustomerTags(deps, { ownerId: principal.tenantId, baId, customerId });
    }
    return { taskId: id, deleted: true };
  });
}
