import { randomUUID } from "node:crypto";
import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import type { Config } from "../types";
import type { PlatformFilesClient } from "../upstream/platformFiles";

export type FileRouteDeps = {
  config: Config;
  authenticator: Authenticator;
  platformFiles: PlatformFilesClient;
};

export function registerFileRoutes(app: FastifyInstance, deps: FileRouteDeps): void {
  app.post("/api/v1/files", async (request, reply) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const data = await request.file();
    if (!data) throw new GatewayError(400, "BAD_REQUEST", "multipart field 'file' is required");

    const query = request.query as { path?: string; fileName?: string };
    const uuid = randomUUID().replace(/-/g, "");

    const receipt = await deps.platformFiles.upload({
      tenantId: principal.tenantId,
      body: data.file,
      uuid,
      filename: data.filename,
      mime: data.mimetype,
      path: query.path,
      fileName: query.fileName,
    });

    return reply.send(receipt);
  });
}
