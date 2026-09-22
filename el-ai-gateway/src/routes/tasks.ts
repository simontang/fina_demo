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

export type TaskTag = { tagId: string; tagKey?: string; tagValue?: string; name?: string; dimension?: string };

function parseTags(result: string | undefined): TaskTag[] {
  if (!result) return [];
  try {
    const parsed = JSON.parse(result);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function normalizeTags(result: string | undefined): Array<{ tagId: string; tagKey: string; tagValue: string }> {
  return parseTags(result)
    .filter((t) => typeof t?.tagId === "string")
    .map((t) => ({
      tagId: t.tagId,
      tagKey: (t.tagKey ?? t.dimension ?? "") as string,
      tagValue: (t.tagValue ?? t.name ?? "") as string,
    }));
}

function toDetail(task: TaskRecord) {
  const metadata = (task.metadata ?? {}) as Record<string, unknown>;
  return {
    taskId: task.id,
    fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
    status: task.status,
    createdAt: task.createdAt,
    title: task.title,
    tags: normalizeTags(task.result),
  };
}

function mapActivities(activities: unknown[]) {
  return (activities as Array<Record<string, any>>).map((a) => ({
    id: a?.id,
    action: a?.action,
    markdown: a?.detail?.markdown ?? a?.markdown ?? "",
    createdAt: a?.createdAt,
  }));
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
      for (const tag of parseTags(task.result)) {
        if (typeof tag?.tagId !== "string") continue;
        const prev = union.get(tag.tagId);
        if (!prev || (when !== undefined && when > prev)) union.set(tag.tagId, when);
      }
    }
    const existing = await deps.boTools.queryRecords("customer_tag", [
      { field: "customer_no", op: "eq", value: input.customerId },
    ]);
    const existingIds = new Set(existing.map((row) => String(row.tag_id)));
    for (const [tagId, when] of union) {
      if (existingIds.has(tagId)) continue;
      const def = await deps.boTools.getRecord("tag_definition", tagId);
      await deps.boTools.createRecord("customer_tag", {
        customer_no: input.customerId,
        tag_key: String(def?.tag_group ?? ""),
        tag_value: String(def?.tag_name ?? ""),
        tag_id: tagId,
        source: "voice",
        confidence: null,
        tagged_at: when ?? new Date().toISOString(),
      });
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

    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid });
    const title = body.title ?? `Voice tagging: ${uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: body.description,
      status: "in_progress",
      ownerId: principal.tenantId,
      metadata: { uuid, url, baId: body.baId, customerId: body.customerId },
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
      return {
        taskId: task.id,
        fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
        status: task.status,
        createdAt: task.createdAt,
        tags: normalizeTags(task.result),
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

  // Activity log for a task (records tag edits etc.).
  app.get("/api/v1/voice-tagging/:id/activities", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });
    const activities = mapActivities(task.activities);
    return { taskId: id, total: activities.length, activities };
  });

  // Replace a task's tags (stored as JSON in the task `result`), then reconcile
  // the customer's aggregate tags (customer_tag) to the union of task tags.
  app.put("/api/v1/voice-tagging/:id/tags", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const task = await deps.taskTools.getTask({ id });

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

    const out: Array<{ tagId: string; tagKey: string; tagValue: string }> = [];
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
        const def = await deps.boTools.getRecord("tag_definition", input.tagId);
        if (!def) {
          throw new GatewayError(400, "BAD_REQUEST", `Unknown tagId: ${input.tagId}`);
        }
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
      out.push({ tagId, tagKey, tagValue });
    }

    await deps.taskTools.updateResult({ id, result: JSON.stringify(out) });
    await reconcileCustomerTags(deps, { ownerId: principal.tenantId, baId, customerId });

    const updated = await deps.taskTools.getTask({ id });
    const activities = mapActivities(updated.activities);
    return { ...toDetail(updated), activity: activities[0] };
  });
}
