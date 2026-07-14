import type { FastifyInstance } from "fastify";
import {
  getRuntime,
  createRuntime,
  updateRuntime,
  appendEvent,
  deleteRuntime,
} from "../runtimeStore";

export function registerTaskAgentRoutes(app: FastifyInstance): void {
  app.register(
    async (apiApp) => {
      apiApp.get<{ Params: { agentId: string; threadId: string } }>(
        "/task-agents/:agentId/threads/:threadId/runtime",
        async (request, reply) => {
          const { agentId, threadId } = request.params;
          const record = getRuntime(agentId, threadId);
          if (!record) {
            return reply.status(404).send({ error: "Runtime not found" });
          }
          return { data: record };
        }
      );

      apiApp.post<{
        Params: { agentId: string; threadId: string };
        Body: {
          activeRuntime?: string;
          currentState?: string;
          selectedArtifact?: string;
          nextAction?: string;
          events?: Array<{ label: string; status: "completed" | "in_progress" | "pending" }>;
        };
      }>(
        "/task-agents/:agentId/threads/:threadId/runtime",
        async (request, reply) => {
          const { agentId, threadId } = request.params;
          const body = request.body;

          let record = getRuntime(agentId, threadId);

          if (!record) {
            if (!body.activeRuntime || !body.currentState) {
              return reply.status(400).send({
                error: "activeRuntime and currentState are required to create a new runtime",
              });
            }
            record = createRuntime(agentId, threadId, {
              activeRuntime: body.activeRuntime,
              currentState: body.currentState,
              selectedArtifact: body.selectedArtifact,
              nextAction: body.nextAction,
            });
          }

          if (body.currentState || body.selectedArtifact || body.nextAction) {
            record = updateRuntime(agentId, threadId, {
              currentState: body.currentState,
              selectedArtifact: body.selectedArtifact,
              nextAction: body.nextAction,
            })!;
          }

          if (body.events && body.events.length > 0) {
            for (const event of body.events) {
              appendEvent(agentId, threadId, {
                label: event.label,
                status: event.status,
              });
            }
            record = getRuntime(agentId, threadId);
          }

          return { data: record };
        }
      );

      apiApp.delete<{ Params: { agentId: string; threadId: string } }>(
        "/task-agents/:agentId/threads/:threadId/runtime",
        async (request, reply) => {
          const { agentId, threadId } = request.params;
          const deleted = deleteRuntime(agentId, threadId);
          if (!deleted) {
            return reply.status(404).send({ error: "Runtime not found" });
          }
          return { success: true };
        }
      );
    },
    { prefix: "/api" }
  );
}
