import type { FastifyInstance } from "fastify";

export function registerWebhookRoutes(app: FastifyInstance): void {
  app.post("/api/v1/webhooks/:topic", async (request) => {
    const { topic } = request.params as { topic: string };
    console.log(
      `[webhook] topic=${topic} headers=${JSON.stringify(request.headers)} body=${JSON.stringify(
        request.body ?? null,
      )}`,
    );
    return { ok: true, topic };
  });
}
