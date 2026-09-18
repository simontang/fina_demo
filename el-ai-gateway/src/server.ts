import Fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import type { Authenticator } from "./auth";
import { toErrorResponse } from "./lib/errors";
import type { Config } from "./types";
import type { PlatformFilesClient } from "./upstream/platformFiles";
import type { AgentRunsClient } from "./upstream/agentRuns";
import type { TaskToolClient } from "./upstream/taskTools";
import { registerFileRoutes } from "./routes/files";
import { registerTaskRoutes } from "./routes/tasks";
import { registerCustomerRoutes } from "./routes/customers";

export type ServerDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
  agentRuns: AgentRunsClient;
  taskTools: TaskToolClient;
};

export function buildServer(deps: ServerDeps): FastifyInstance {
  const app = Fastify({ logger: false, bodyLimit: deps.config.maxUploadBytes });

  app.register(multipart, { limits: { fileSize: deps.config.maxUploadBytes } });

  const rawCorsOrigins = (deps.config.corsOrigins ?? "*").trim();
  const corsOrigin =
    rawCorsOrigins === "*"
      ? "*"
      : rawCorsOrigins
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
  app.register(cors, { origin: corsOrigin });

  app.get("/health", async () => ({ status: "ok" }));

  registerFileRoutes(app, deps);
  registerTaskRoutes(app, deps);
  registerCustomerRoutes(app, deps);

  app.setErrorHandler((error, _request, reply) => {
    const { statusCode, body } = toErrorResponse(error);
    reply.status(statusCode).send(body);
  });

  return app;
}
