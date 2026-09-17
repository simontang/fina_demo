import { randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";
import type { AgentRunsClient } from "../upstream/agentRuns";
import type { TaskToolClient } from "../upstream/taskTools";
import { getTasks } from "../mock/tasks";

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
      title?: string;
      description?: string;
      assistantId?: string;
    };
    const uuid = body.uuid ?? deps.config.voiceTaggingFileUuid;
    if (typeof uuid !== "string" || uuid.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "uuid is required (or set VOICE_TAGGING_FILE_UUID)");
    }

    const { url } = await deps.platformFiles.presign({ tenantId: principal.tenantId, uuid });
    const title = body.title ?? `Voice tagging: ${uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: body.description,
      status: "in_progress",
      ownerId: principal.tenantId,
      metadata: { uuid, url },
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

  // List a BA's tasks for a customer (mock store): fileId / taskId / status / tags.
  app.get("/api/v1/voice-tagging", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const query = request.query as { baId?: string; customerId?: string };
    if (typeof query.baId !== "string" || query.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof query.customerId !== "string" || query.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const tasks = getTasks(query.baId, query.customerId);
    return { baId: query.baId, customerId: query.customerId, total: tasks.length, tasks };
  });

  app.get("/api/v1/voice-tagging/:id", async (request) => {
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
