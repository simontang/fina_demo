import type { FastifyReply } from "fastify";

import { DEFAULT_TOKEN_EXPIRE_TIME } from "../constants/rtc";
import type { RtcProxyRequest, UpdateTriggerRequest, VoiceChatCallbackRequest } from "../types/rtc";
import {
  formatVoiceChatLog,
  generateRtcAccessToken,
  getAgentConfig,
  getVoiceChatConfig,
  parseEventData,
  sendRoomUnicast,
  startVoiceChat,
  stopVoiceChat,
  updateVoiceChat,
} from "../utils/rtc";

export async function voiceChatEventCallback(request: VoiceChatCallbackRequest, reply: FastifyReply) {
  try {
    const eventPayload = request.body || ({} as any);
    const parsed = parseEventData(eventPayload.EventData);
    console.log(formatVoiceChatLog(eventPayload, parsed));
    return reply.send({ success: true, message: "callback received" });
  } catch (e: any) {
    console.error("VoiceChat callback failed:", e);
    return reply.status(500).send({ success: false, message: "callback failed" });
  }
}

export async function rtcProxyFetch(request: RtcProxyRequest, reply: FastifyReply) {
  try {
    const { action, params } = request.body || ({} as any);
    if (!action) return reply.status(400).send({ success: false, message: "Missing required field: action" });
    if (!params) return reply.status(400).send({ success: false, message: "Missing required field: params" });

    switch (action) {
      case "StartVoiceChat":
        return await handleStartVoiceChat(params, reply);
      case "StopVoiceChat":
        return await handleStopVoiceChat(params, reply);
      case "UpdateVoiceChat":
        return await handleUpdateVoiceChat(params, reply);
      case "GenerateRtcToken":
        return await handleGenerateRtcToken(params, reply);
      default:
        return reply.status(400).send({ success: false, message: `Unsupported action: ${action}` });
    }
  } catch (e: any) {
    console.error("rtcProxyFetch failed:", e);
    return reply.status(500).send({ success: false, message: e?.message || "Internal server error" });
  }
}

async function handleStartVoiceChat(params: any, reply: FastifyReply) {
  const { roomId, userId, taskId, welcomeSpeech, customVariables, botId } = params || {};
  if (!roomId || !userId || !taskId) {
    return reply.status(400).send({ success: false, message: "roomId, userId, taskId are required" });
  }

  const config = getVoiceChatConfig(String(userId), welcomeSpeech, customVariables, botId);
  const agentConfig = getAgentConfig(String(userId));

  const res = await startVoiceChat({
    taskId: String(taskId),
    roomId: String(roomId),
    config,
    agentConfig,
  });

  if (!res.success) return reply.status(500).send({ success: false, message: res.message || "StartVoiceChat failed" });
  return reply.send({ success: true, message: "Voice chat started", data: res.data });
}

async function handleStopVoiceChat(params: any, reply: FastifyReply) {
  const { roomId, taskId } = params || {};
  if (!roomId || !taskId) return reply.status(400).send({ success: false, message: "roomId, taskId are required" });

  const res = await stopVoiceChat({ roomId: String(roomId), taskId: String(taskId) });
  if (!res.success) return reply.status(500).send({ success: false, message: res.message || "StopVoiceChat failed" });
  return reply.send({ success: true, message: "Voice chat stopped", data: res.data });
}

async function handleUpdateVoiceChat(params: any, reply: FastifyReply) {
  const { roomId, taskId, updateFields } = params || {};
  if (!roomId || !taskId) return reply.status(400).send({ success: false, message: "roomId, taskId are required" });

  const res = await updateVoiceChat({ roomId: String(roomId), taskId: String(taskId), updateFields });
  if (!res.success) return reply.status(500).send({ success: false, message: res.message || "UpdateVoiceChat failed" });
  return reply.send({ success: true, message: "Voice chat updated", data: res.data });
}

async function handleGenerateRtcToken(params: any, reply: FastifyReply) {
  const { roomId, userId, expireTime } = params || {};
  if (!roomId || !userId) return reply.status(400).send({ success: false, message: "roomId, userId are required" });

  const appId = process.env.VOLCENGINE_APP_ID;
  const appKey = process.env.VOLCENGINE_APP_KEY;
  if (!appId || !appKey) {
    return reply.status(400).send({ success: false, message: "Missing env: VOLCENGINE_APP_ID / VOLCENGINE_APP_KEY" });
  }

  const token = generateRtcAccessToken({
    appId,
    appKey,
    roomId: String(roomId),
    userId: String(userId),
    expireSeconds: Number(expireTime || DEFAULT_TOKEN_EXPIRE_TIME),
  });

  return reply.send({
    success: true,
    message: "Token generated",
    data: { token, appId, roomId: String(roomId), userId: String(userId) },
  });
}

export async function updateTrigger(request: UpdateTriggerRequest, reply: FastifyReply) {
  try {
    const { roomId, userId, message } = request.body || ({} as any);
    if (!roomId || !userId || !message) {
      return reply.status(400).send({ success: false, message: "roomId, userId, message are required" });
    }

    const res = await sendRoomUnicast({ roomId: String(roomId), userId: String(userId), message: String(message) });
    if (!res.success) return reply.status(500).send({ success: false, message: res.message || "SendRoomUnicast failed" });
    return reply.send({ success: true, message: "Message sent", data: res.data });
  } catch (e: any) {
    console.error("updateTrigger failed:", e);
    return reply.status(500).send({ success: false, message: e?.message || "Internal server error" });
  }
}

