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
  return `Voice tagging task ${vars.taskId}. Read the task to get the associated file, then transcribe the audio and tag the text.`;
}

export function renderFeedbackMessage(vars: { taskId: string; content: string }): string {
  return `Feedback for task ${vars.taskId}. Append it to this task's activity timeline (add_activity):\n${vars.content}`;
}

export function registerTaskRoutes(app: FastifyInstance, deps: TaskRouteDeps): void {
  app.post("/api/v1/voice-tagging", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const body = (request.body ?? {}) as {
      uuid?: string;
      title?: string;
      description?: string;
      assistantId?: string;
    };
    const uuid = body.uuid ?? deps.config.a2aVoiceTaggingFileUuid;
    if (typeof uuid !== "string" || uuid.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "uuid is required (or set A2A_VOICE_TAGGING_FILE_UUID)");
    }

    const { url } = await deps.platformFiles.presign({
      tenantId: principal.tenantId,
      uuid,
    });
    const title = body.title ?? `Voice tagging: ${uuid}`;
    const { taskId } = await deps.taskTools.createTask({
      title,
      description: body.description,
      status: "in_progress",
      ownerId: principal.tenantId,
      metadata: { uuid, url },
    });

    const assistantId = body.assistantId ?? deps.config.a2aVoiceTaggingAssistantId;
    if (!assistantId) {
      throw new GatewayError(
        400,
        "BAD_REQUEST",
        "assistantId is required (or set A2A_VOICE_TAGGING_ASSISTANT_ID)",
      );
    }
    const text = renderA2AMessage(deps.config.a2aMessageTemplate, { uuid, url, taskId });
    const timeoutMs = deps.config.a2aTriggerTimeoutMs ?? deps.config.upstreamTimeoutMs;

    // Fire-and-forget: the A2A message carries the task id; the agent writes
    // status/activity back to the task. We do not wait for the A2A task outcome.
    void deps.a2a
      .sendTask({ assistantId, text, timeoutMs })
      .catch((err: unknown) =>
        console.error(
          `[voice-tagging] A2A trigger failed for task ${taskId}: ${(err as Error).message}`,
        ),
      );

    return {
      taskId,
      status: "in_progress",
      file: { uuid, url },
      a2a: { dispatched: true },
    };
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
    const assistantId = body.assistantId ?? deps.config.a2aVoiceTaggingAssistantId;
    if (!assistantId) {
      throw new GatewayError(
        400,
        "BAD_REQUEST",
        "assistantId is required (or set A2A_VOICE_TAGGING_ASSISTANT_ID)",
      );
    }
    const text = body.summary
      ? `${renderFeedbackMessage({ taskId: id, content: body.content })}\n\nSummary: ${body.summary}`
      : renderFeedbackMessage({ taskId: id, content: body.content });
    const timeoutMs = deps.config.a2aTriggerTimeoutMs ?? deps.config.upstreamTimeoutMs;
    void deps.a2a
      .sendTask({ assistantId, text, timeoutMs })
      .catch((err: unknown) =>
        console.error(
          `[voice-tagging] A2A feedback relay failed for task ${id}: ${(err as Error).message}`,
        ),
      );
    return { taskId: id, forwarded: true, a2a: { dispatched: true } };
  });
}
