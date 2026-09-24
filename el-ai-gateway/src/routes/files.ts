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

    const query = request.query as {
      path?: string;
      fileName?: string;
      fileCategory?: string;
      usage?: string;
      baId?: string;
      customerId?: string;
      durationSec?: string;
      meta?: string;
    };
    if (typeof query.baId !== "string" || query.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof query.customerId !== "string" || query.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const uuid = randomUUID().replace(/-/g, "");

    let meta: Record<string, unknown> = {};
    if (query.meta) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(query.meta);
      } catch {
        throw new GatewayError(400, "BAD_REQUEST", "meta must be valid URL-encoded JSON");
      }
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new GatewayError(400, "BAD_REQUEST", "meta must be a JSON object");
      }
      meta = parsed as Record<string, unknown>;
    }
    if (query.baId) meta.baId = query.baId;
    if (query.customerId) meta.customerId = query.customerId;
    if (query.durationSec !== undefined && query.durationSec !== "") {
      const durationSec = Number(query.durationSec);
      if (!Number.isFinite(durationSec) || durationSec <= 0) {
        throw new GatewayError(400, "BAD_REQUEST", "durationSec must be a positive number");
      }
      meta.durationSec = durationSec;
    }

    const receipt = await deps.platformFiles.upload({
      tenantId: principal.tenantId,
      body: data.file,
      uuid,
      filename: data.filename,
      mime: data.mimetype,
      path: query.path,
      fileName: query.fileName,
      fileCategory: query.fileCategory,
      usage: query.usage,
      meta,
    });

    return reply.send(receipt);
  });

  // List files filtered by baId + customerId (both required).
  app.get("/api/v1/files", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const query = request.query as {
      baId?: string;
      customerId?: string;
      path?: string;
      q?: string;
      recursive?: string;
      page?: string;
      size?: string;
    };
    if (typeof query.baId !== "string" || query.baId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "baId is required");
    }
    if (typeof query.customerId !== "string" || query.customerId.trim() === "") {
      throw new GatewayError(400, "BAD_REQUEST", "customerId is required");
    }
    const meta = JSON.stringify({ baId: query.baId, customerId: query.customerId });
    return deps.platformFiles.list({
      tenantId: principal.tenantId,
      meta,
      path: query.path,
      q: query.q,
      recursive: query.recursive,
      page: query.page,
      size: query.size,
    });
  });

  // Get a (presigned) URL to play/download the file — usable directly in <audio src>.
  app.get("/api/v1/files/:uuid/url", async (request) => {
    const principal = requirePrincipal(deps.authenticator, request.headers.authorization);
    const { uuid } = request.params as { uuid: string };
    if (!/^[0-9a-fA-F]{32}$/.test(uuid)) {
      throw new GatewayError(400, "BAD_REQUEST", "uuid must be 32 hex characters");
    }
    const query = request.query as { ttlSeconds?: string };
    const ttlSeconds = query.ttlSeconds ? Number(query.ttlSeconds) : undefined;
    if (ttlSeconds !== undefined && (!Number.isFinite(ttlSeconds) || ttlSeconds <= 0)) {
      throw new GatewayError(400, "BAD_REQUEST", "ttlSeconds must be a positive integer");
    }
    const link = await deps.platformFiles.presign({
      tenantId: principal.tenantId,
      uuid,
      ttlSeconds,
    });
    return { uuid, ...link };
  });
}
