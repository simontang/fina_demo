import { randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";
import type { AgentRunsClient } from "../upstream/agentRuns";
import type { TaskToolClient, TaskRecord } from "../upstream/taskTools";
import { resolveTags } from "../mock/customerTags";

export type TaskRouteDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  agentRuns: AgentRunsClient;
  taskTools: TaskToolClient;
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

export function renderFeedbackMessage(vars: { taskId: string; content: string }): string {
  return `Feedback for task ${vars.taskId}. Append it to this task's activity timeline (add_activity):\n${vars.content}`;
}

function parseTags(result: string | undefined): Array<{ tagId: string; name: string; dimension: string }> {
  if (!result) return [];
  try {
    const parsed = JSON.parse(result);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toDetail(task: TaskRecord) {
  const metadata = (task.metadata ?? {}) as Record<string, unknown>;
  return {
    taskId: task.id,
    fileId: typeof metadata.uuid === "string" ? metadata.uuid : undefined,
    status: task.status,
    createdAt: task.createdAt,
    title: task.title,
    tags: parseTags(task.result),
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
        tags: parseTags(task.result),
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

  // Replace a task's tags (stored as JSON in the task `result`); records an activity.
  app.put("/api/v1/voice-tagging/:id/tags", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    await deps.taskTools.getTask({ id });

    const body = (request.body ?? {}) as { tags?: unknown };
    if (!Array.isArray(body.tags)) {
      throw new GatewayError(400, "BAD_REQUEST", "tags must be an array of tagIds");
    }
    const tagIds: string[] = [];
    for (const item of body.tags) {
      const tagId = typeof item === "string" ? item : (item as { tagId?: unknown })?.tagId;
      if (typeof tagId !== "string" || !/^[0-9a-f]{32}$/.test(tagId)) {
        throw new GatewayError(400, "BAD_REQUEST", "each tag must be a 32-hex tagId");
      }
      tagIds.push(tagId);
    }
    const { tags, unknown } = resolveTags(tagIds);
    if (unknown.length > 0) {
      throw new GatewayError(400, "BAD_REQUEST", `Unknown tagId(s): ${unknown.join(", ")}`);
    }
    await deps.taskTools.updateResult({ id, result: JSON.stringify(tags) });
    const updated = await deps.taskTools.getTask({ id });
    const activities = mapActivities(updated.activities);
    return { ...toDetail(updated), activity: activities[0] };
  });

  app.post("/api/v1/voice-tagging/:id/feedback", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { id } = request.params as { id: string };
    const body = (request.body ?? {}) as { content?: string; summary?: string; assistantId?: string };
    if (typeof body.content !== "string" || body.content.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "content is required");
    }
    const assistantId = body.assistantId ?? deps.config.voiceTaggingAssistantId;
    if (!assistantId) {
      throw new GatewayError(
        400,
        "BAD_REQUEST",
        "assistantId is required (or set VOICE_TAGGING_ASSISTANT_ID)",
      );
    }
    const text = body.summary
      ? `${renderFeedbackMessage({ taskId: id, content: body.content })}\n\nSummary: ${body.summary}`
      : renderFeedbackMessage({ taskId: id, content: body.content });
    dispatchRun(assistantId, text, id);
    return { taskId: id, forwarded: true, agent: { dispatched: true } };
  });
}
