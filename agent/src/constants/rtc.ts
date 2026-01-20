// RTC / Voice Chat constants (Volcengine ByteRTC)

export const API_VERSION = "2024-12-01";

// Token generation
export const DEFAULT_TOKEN_EXPIRE_TIME = 3600; // seconds
export const RTC_ACCESS_TOKEN_VERSION = "001";

export const RTC_PRIVILEGES = {
  PrivPublishStream: 0,
  PrivPublishAudioStream: 1,
  PrivPublishVideoStream: 2,
  PrivPublishDataStream: 3,
  PrivSubscribeStream: 4,
} as const;

export const VOICE_CHAT_DEFAULT_WELCOME_MESSAGE =
  "Hello! I'm your voice assistant. How can I help you today?";

// RunStage label mapping (for logging only)
export const RUN_STAGE_LABELS: Record<string, string> = {
  taskStart: "Task started",
  taskStop: "Task stopped",
  beginAsking: "User started speaking",
  asrFinish: "User stopped speaking",
  llmOutput: "LLM produced first token",
  answerStart: "Agent started speaking",
  answerFinish: "Agent finished speaking",
  interrupted: "Agent speech interrupted",
  reasoningStart: "LLM reasoning started",
  asr: "ASR stage",
  llm: "LLM stage",
  tts: "TTS stage",
  preParamCheck: "Parameter validation error",
};

