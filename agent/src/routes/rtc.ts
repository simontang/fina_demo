import type { FastifyInstance } from "fastify";

import { rtcProxyFetch, updateTrigger, voiceChatEventCallback } from "../controllers/rtcController";

/**
 * RTC / Voice Chat routes.
 *
 * Mounted under `/api/rtc/*` so the admin UI (Vite proxy) can reach it via `/api`.
 */
export function registerRtcRoutes(app: FastifyInstance): void {
  app.register(
    async (apiApp) => {
      // Volcengine voice chat event callback (optional; mainly for logging)
      apiApp.post("/rtc/voice_chat", voiceChatEventCallback as any);

      // Unified proxy endpoint used by the frontend
      apiApp.post("/rtc/proxyFetch", rtcProxyFetch as any);

      // Optional plugin hook (SendRoomUnicast)
      apiApp.post("/rtc/update-trigger", updateTrigger as any);
    },
    { prefix: "/api" }
  );
}

