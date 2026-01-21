import { Signer } from "@volcengine/openapi";
import axios from "axios";
import { createHmac, randomBytes } from "crypto";

import {
  API_VERSION,
  RTC_ACCESS_TOKEN_VERSION,
  RTC_PRIVILEGES,
  RUN_STAGE_LABELS,
} from "../constants/rtc";
import type {
  AgentConfig,
  BaseApiResult,
  LocalTokenParams,
  ParsedEventData,
  PrivilegeMap,
  VoiceChatConfig,
  VoiceChatEventPayload,
} from "../types/rtc";

class ByteBufWriter {
  private chunks: Buffer[] = [];

  putUint16(value: number) {
    const buf = Buffer.allocUnsafe(2);
    buf.writeUInt16LE(value, 0);
    this.chunks.push(buf);
    return this;
  }

  putUint32(value: number) {
    const buf = Buffer.allocUnsafe(4);
    buf.writeUInt32LE(value, 0);
    this.chunks.push(buf);
    return this;
  }

  putBytes(bytes: Buffer) {
    this.putUint16(bytes.length);
    this.chunks.push(bytes);
    return this;
  }

  putString(value: string) {
    return this.putBytes(Buffer.from(value, "utf-8"));
  }

  putTreeMapUInt32(map: PrivilegeMap) {
    const entries = Object.entries(map).sort(([a], [b]) => Number(a) - Number(b));
    this.putUint16(entries.length);
    for (const [key, value] of entries) {
      this.putUint16(Number(key));
      this.putUint32(value);
    }
    return this;
  }

  pack() {
    return Buffer.concat(this.chunks);
  }
}

function buildPrivilegeMap(expireAt: number): PrivilegeMap {
  const map: PrivilegeMap = {};
  map[RTC_PRIVILEGES.PrivPublishStream] = expireAt;
  map[RTC_PRIVILEGES.PrivPublishAudioStream] = expireAt;
  map[RTC_PRIVILEGES.PrivPublishVideoStream] = expireAt;
  map[RTC_PRIVILEGES.PrivPublishDataStream] = expireAt;
  map[RTC_PRIVILEGES.PrivSubscribeStream] = expireAt;
  return map;
}

export function generateRtcAccessToken(params: LocalTokenParams): string {
  const { appId, appKey, roomId, userId, expireSeconds } = params;
  const issuedAt = Math.floor(Date.now() / 1000);
  const nonce = randomBytes(4).readUInt32LE(0);
  const expireAt = issuedAt + expireSeconds;
  const privileges = buildPrivilegeMap(expireAt);

  const msgBuf = new ByteBufWriter()
    .putUint32(nonce)
    .putUint32(issuedAt)
    .putUint32(expireAt)
    .putString(roomId)
    .putString(userId)
    .putTreeMapUInt32(privileges);

  const msg = msgBuf.pack();
  const signature = createHmac("sha256", appKey).update(msg).digest();

  const content = new ByteBufWriter().putBytes(msg).putBytes(signature).pack();
  return `${RTC_ACCESS_TOKEN_VERSION}${appId}${content.toString("base64")}`;
}

function requiredEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing environment variable: ${name}`);
  return v;
}

export async function signVolcengineRequest(params: {
  Action: string;
  requestBody: any;
  version?: string;
}): Promise<any> {
  const { Action, requestBody, version } = params;
  const apiUrl = (process.env.VOLC_RTC_API_URL || "https://rtc.volcengineapi.com").replace(/\/$/, "");
  const region = process.env.VOLC_REGION || "cn-north-1";

  const accessKeyId = requiredEnv("VOLC_ACCESSKEY");
  const secretKey = requiredEnv("VOLC_SECRETKEY");

  const queryParams = {
    Action,
    Version: version || API_VERSION,
  };

  const openApiRequestData: any = {
    region,
    method: "POST",
    params: queryParams,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  };

  const signer = new Signer(openApiRequestData, "rtc");
  signer.addAuthorization({
    accessKeyId,
    secretKey,
    sessionToken: "",
  });

  const queryString = new URLSearchParams(queryParams as any).toString();
  return axios.post(`${apiUrl}/?${queryString}`, requestBody, {
    headers: openApiRequestData.headers,
  });
}

export function parseEventData(eventData?: string): ParsedEventData | null {
  if (!eventData || typeof eventData !== "string") return null;
  try {
    return JSON.parse(eventData) as ParsedEventData;
  } catch {
    return null;
  }
}

function dateTimeFormatterUTC8(input?: number | string | Date): string {
  if (input === undefined || input === null) return "N/A";
  const date = input instanceof Date ? input : new Date(input);
  if (Number.isNaN(date.getTime())) return "N/A";

  const utc8 = new Date(date.getTime() + 8 * 60 * 60 * 1000);
  const pad = (n: number, len = 2) => n.toString().padStart(len, "0");
  const month = pad(utc8.getUTCMonth() + 1);
  const day = pad(utc8.getUTCDate());
  const hours = pad(utc8.getUTCHours());
  const minutes = pad(utc8.getUTCMinutes());
  const seconds = pad(utc8.getUTCSeconds());
  const millis = pad(utc8.getUTCMilliseconds(), 3);
  return `${month}-${day} ${hours}:${minutes}:${seconds}.${millis}`;
}

export function formatVoiceChatLog(eventPayload: VoiceChatEventPayload, parsedData: ParsedEventData | null): string {
  const logTimestamp = dateTimeFormatterUTC8(new Date());
  const roomId = parsedData?.RoomId || eventPayload.RoomId || "unknown";
  const taskId = parsedData?.TaskId || eventPayload.TaskId || "unknown";
  const runStage = parsedData?.RunStage || "unknown";
  const runStageLabel = RUN_STAGE_LABELS[runStage] || runStage;
  const userId = parsedData?.UserID || eventPayload.UserId || "unknown";
  const roundId = parsedData?.RoundID ?? "N/A";
  const eventTime = parsedData?.EventTime ? dateTimeFormatterUTC8(parsedData.EventTime) : "N/A";

  let msg = `[VoiceChat] ${logTimestamp}`;
  msg += ` | RoomId: ${roomId}`;
  msg += ` | TaskId: ${taskId}`;
  msg += ` | UserId: ${userId}`;
  msg += ` | RoundId: ${roundId}`;
  msg += ` | RunStage: ${runStage} (${runStageLabel})`;
  msg += ` | EventTime: ${eventTime}`;

  if (parsedData?.ErrorInfo?.ErrorCode && parsedData.ErrorInfo.ErrorCode !== 0) {
    msg += ` | Error: Code=${parsedData.ErrorInfo.ErrorCode}, Reason=${parsedData.ErrorInfo.Reason || "N/A"}`;
  }

  return msg;
}

export function getVoiceChatConfig(
  userId: string,
  welcomeSpeech?: string,
  customVariables?: any,
  botId?: string
): VoiceChatConfig {

  const envFirst = (...names: string[]): string => {
    for (const n of names) {
      const v = (process.env[n] || "").trim();
      if (v) return v;
    }
    return "";
  };
  const envNumber = (name: string, fallback: number): number => {
    const raw = (process.env[name] || "").trim();
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  // Coze (Bot) config (APIKey is secret -> env).
  const defaultWelcomeSpeech = welcomeSpeech || envFirst("VOICE_WELCOME_MESSAGE");
  const cozeApiKey = envFirst("COZEBOT_APIKEY", "COZE_API_KEY");
  const cozeBotId = (botId || "").trim() || envFirst("COZEBOT_BOT_ID");
  const cozeUrl = envFirst("COZEBOT_URL") || "https://api.coze.cn";

  // Speech (ASR/TTS) app id (not secret, but project-specific -> env).
  // For compatibility with fuli_survey naming, prefer VOLC_SPEECH_APP_ID.
  const speechAppId = envFirst("VOLC_SPEECH_APP_ID", "VOLC_ASR_APP_ID", "VOLC_TTS_APP_ID");

  // ASR config (smallmodel by default, matches fuli_survey).
  const asrMode = envFirst("VOLC_ASR_MODE") || "smallmodel";
  const asrCluster = envFirst("VOLC_ASR_CLUSTER") || "volcengine_streaming_common";

  // Optional bigmodel params (set via env).
  const asrAccessToken = envFirst("VOLC_SPEECH_ACCESS_TOKEN", "VOLC_ASR_ACCESS_TOKEN");
  const asrApiResourceId = envFirst("VOLC_ASR_API_RESOURCE_ID");

  // TTS config (matches fuli_survey default structure).
  const ttsCluster = envFirst("VOLC_TTS_CLUSTER") || "volcano_icl";
  const ttsVoiceType = envFirst("VOLC_TTS_VOICE_TYPE");
  const ttsSpeedRatio = envNumber("VOLC_TTS_SPEED_RATIO", 1.0);

  return {
    ASRConfig: {
      Provider: "volcano",
      ProviderParams: {
        Mode: asrMode,
        Cluster: asrCluster,
        AppId: speechAppId,
        ...(asrMode === "bigmodel" && asrAccessToken
          ? {
            AccessToken: asrAccessToken,
            ...(asrApiResourceId ? { ApiResourceId: asrApiResourceId } : {}),
          }
          : {}),
      },
    },
    TTSConfig: {
      Provider: "volcano_bidirection",
      ProviderParams: {
        app: {
          appid: speechAppId,
          token: "22Se7aE41Tb0nWaDG3ytKwqTjnUnrk5X",
        },
        audio: {
          voice_type: ttsVoiceType,
          speech_rate: 0,
        },
        ResourceId: "volc.service_type.10029",
      },
    },
    LLMConfig: {
      Mode: "CozeBot",
      WelcomeSpeech: defaultWelcomeSpeech,
      CozeBotConfig: {
        HistoryLength: Number(process.env.COZEBOT_HISTORY_LENGTH || "40"),
        BotId: cozeBotId,
        APIKey: cozeApiKey,
        Url: cozeUrl,
        UserId: userId,
        CustomVariables: customVariables ?? undefined,
      },
    },
  };
}

export function getAgentConfig(userId: string): AgentConfig {
  return {
    TargetUserId: [userId],
    UserId: `voice_chat_${userId}`,
    EnableConversationStateCallback: true,
  };
}

export async function startVoiceChat(params: {
  taskId: string;
  roomId: string;
  config: VoiceChatConfig;
  agentConfig: AgentConfig;
}): Promise<BaseApiResult<any>> {
  const appId = requiredEnv("VOLCENGINE_APP_ID");
  try {
    const response = await signVolcengineRequest({
      Action: "StartVoiceChat",
      requestBody: {
        AppId: appId,
        TaskId: params.taskId,
        RoomId: params.roomId,
        Config: params.config,
        AgentConfig: params.agentConfig,
      },
    });

    const err = response?.data?.ResponseMetadata?.Error;
    if (err) {
      return { success: false, message: err.Message || "StartVoiceChat failed" };
    }
    return { success: true, data: response?.data?.Result || response?.data };
  } catch (e: any) {
    const errMsg =
      e?.response?.data?.ResponseMetadata?.Error?.Message ||
      e?.response?.data?.message ||
      e?.message ||
      "StartVoiceChat failed";
    return { success: false, message: errMsg };
  }
}

export async function stopVoiceChat(params: { taskId: string; roomId: string }): Promise<BaseApiResult<any>> {
  const appId = requiredEnv("VOLCENGINE_APP_ID");
  try {
    const response = await signVolcengineRequest({
      Action: "StopVoiceChat",
      requestBody: {
        AppId: appId,
        TaskId: params.taskId,
        RoomId: params.roomId,
      },
    });

    const err = response?.data?.ResponseMetadata?.Error;
    if (err) {
      return { success: false, message: err.Message || "StopVoiceChat failed" };
    }
    return { success: true, data: response?.data?.Result || response?.data };
  } catch (e: any) {
    const errMsg =
      e?.response?.data?.ResponseMetadata?.Error?.Message ||
      e?.response?.data?.message ||
      e?.message ||
      "StopVoiceChat failed";
    return { success: false, message: errMsg };
  }
}

export async function updateVoiceChat(params: {
  taskId: string;
  roomId: string;
  updateFields?: any;
}): Promise<BaseApiResult<any>> {
  const appId = requiredEnv("VOLCENGINE_APP_ID");
  try {
    const requestBody = Object.assign(
      {
        AppId: appId,
        TaskId: params.taskId,
        RoomId: params.roomId,
      },
      params.updateFields || {}
    );

    const response = await signVolcengineRequest({
      Action: "UpdateVoiceChat",
      requestBody,
    });

    const err = response?.data?.ResponseMetadata?.Error;
    if (err) {
      return { success: false, message: err.Message || "UpdateVoiceChat failed" };
    }
    return { success: true, data: response?.data?.Result || response?.data };
  } catch (e: any) {
    const errMsg =
      e?.response?.data?.ResponseMetadata?.Error?.Message ||
      e?.response?.data?.message ||
      e?.message ||
      "UpdateVoiceChat failed";
    return { success: false, message: errMsg };
  }
}

export async function sendRoomUnicast(params: {
  roomId: string;
  userId: string;
  message: string;
}): Promise<BaseApiResult<any>> {
  const appId = requiredEnv("VOLCENGINE_APP_ID");
  try {
    const response = await signVolcengineRequest({
      Action: "SendRoomUnicast",
      version: "2023-07-20",
      requestBody: {
        AppId: appId,
        RoomId: params.roomId,
        From: "from_server",
        To: params.userId,
        Message: params.message,
        Binary: false,
      },
    });

    const err = response?.data?.ResponseMetadata?.Error;
    if (err) {
      return { success: false, message: err.Message || "SendRoomUnicast failed" };
    }
    return { success: true, data: response?.data?.Result || response?.data };
  } catch (e: any) {
    const errMsg =
      e?.response?.data?.ResponseMetadata?.Error?.Message ||
      e?.response?.data?.message ||
      e?.message ||
      "SendRoomUnicast failed";
    return { success: false, message: errMsg };
  }
}
