import type { Readable } from "node:stream";
import type { Config } from "../types";
import { upstreamError } from "../lib/errors";
import { fetchWithTimeout, type FetchLike } from "../lib/http";

export type UploadReceipt = Record<string, unknown> & { uuid?: string };

export type PresignLink = { url: string; kind: string; expiresInSeconds: number };

export type PlatformFilesClient = {
  upload(input: {
    tenantId: string;
    body: Readable;
    uuid: string;
    filename: string;
    mime: string;
    path?: string;
    fileName?: string;
  }): Promise<UploadReceipt>;
  presign(input: { tenantId: string; uuid: string; ttlSeconds?: number }): Promise<PresignLink>;
};

export function createPlatformFilesClient(
  config: Config,
  fetchImpl: FetchLike = fetch,
): PlatformFilesClient {
  function baseHeaders(tenantId: string): Record<string, string> {
    const headers: Record<string, string> = { "X-Tenant-Id": tenantId };
    if (config.fileServiceApiKey) headers["X-Api-Key"] = config.fileServiceApiKey;
    return headers;
  }

  return {
    async upload(input) {
      const headers = baseHeaders(input.tenantId);
      headers["Content-Type"] = input.mime;
      if (input.path) headers["X-File-Path"] = input.path;
      headers["X-File-Name"] = input.fileName ?? input.filename;

      const res = await fetchWithTimeout(
        fetchImpl,
        `${config.platformFilesUrl}/${input.uuid}`,
        {
          method: "PUT",
          headers,
          body: input.body as unknown as BodyInit,
          duplex: "half",
        } as RequestInit,
        config.upstreamTimeoutMs,
      );
      if (!res.ok) throw await upstreamError(res);
      return (await res.json()) as UploadReceipt;
    },

    async presign(input) {
      const res = await fetchWithTimeout(
        fetchImpl,
        `${config.platformFilesUrl}/presign`,
        {
          method: "POST",
          headers: { ...baseHeaders(input.tenantId), "Content-Type": "application/json" },
          body: JSON.stringify({ uuid: input.uuid, ttlSeconds: input.ttlSeconds }),
        },
        config.upstreamTimeoutMs,
      );
      if (!res.ok) throw await upstreamError(res);
      return (await res.json()) as PresignLink;
    },
  };
}
