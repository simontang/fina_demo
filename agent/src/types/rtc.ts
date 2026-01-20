import type { FastifyRequest } from "fastify";

export interface BaseApiResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
}

export interface BaseApiResult<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
}

export type PrivilegeMap = Record<number, number>;

export interface LocalTokenParams {
  appId: string;
  appKey: string;
  roomId: string;
  userId: string;
  expireSeconds: number;
}

export interface RtcProxyBody {
  action: "StartVoiceChat" | "StopVoiceChat" | "GenerateRtcToken" | "UpdateVoiceChat";
  params: {
    roomId?: string;
    userId?: string;
    taskId?: string;
    expireTime?: number;
    welcomeSpeech?: string;
    customVariables?: any;
    botId?: string;
    updateFields?: any;
  };
}

export type RtcProxyRequest = FastifyRequest<{ Body: RtcProxyBody }>;

export interface VoiceChatConfig {
  ASRConfig: Record<string, any>;
  TTSConfig: Record<string, any>;
  LLMConfig: Record<string, any>;
  InterruptMode?: number;
}

export interface AgentConfig {
  TargetUserId: string[];
  EnableConversationStateCallback?: boolean;
  UserId: string;
}

export interface VoiceChatEventPayload {
  EventType: string;
  RoomId?: string;
  TaskId?: string;
  UserId?: string;
  ConversationId?: string;
  Timestamp?: number;
  Payload?: Record<string, any>;
  EventData?: string;
  [key: string]: any;
}

export interface ParsedEventData {
  AppId?: string;
  BusinessId?: string;
  RoomId?: string;
  TaskId?: string;
  UserID?: string;
  RoundID?: number;
  EventTime?: number;
  EventType?: number;
  RunStage?: string;
  ErrorInfo?: {
    ErrorCode?: number;
    Reason?: string;
  };
  [key: string]: any;
}

export type VoiceChatCallbackRequest = FastifyRequest<{
  Params: Record<string, never>;
  Body: VoiceChatEventPayload;
}>;

export type UpdateTriggerRequest = FastifyRequest<{
  Body: {
    roomId: string;
    userId: string;
    message: string;
  };
}>;

