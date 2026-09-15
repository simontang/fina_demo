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
        throw new GatewayError(
          400,
          "BAD_REQUEST",
          "assistantId is required (or set A2A_VOICE_TAGGING_ASSISTANT_ID)",
        );
      }
      const text = renderA2AMessage(deps.config.a2aMessageTemplate, {
        uuid: body.uuid,
        url,
        taskId,
      });
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
