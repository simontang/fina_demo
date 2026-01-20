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
  "您好，我是太保的服务商—抚理健康。负责本次出院用车服务的。您现在方便接电话吗？";

// RunStage label mapping (for logging only)
export const RUN_STAGE_LABELS: Record<string, string> = {
  taskStart: "任务开始",
  taskStop: "任务结束",
  beginAsking: "用户开始说话",
  asrFinish: "用户结束说话",
  llmOutput: "大模型输出首个token",
  answerStart: "智能体开始说话",
  answerFinish: "智能体说话完成",
  interrupted: "智能体说话被打断",
  reasoningStart: "大模型开始深度思考",
  asr: "ASR处理阶段",
  llm: "LLM处理阶段",
  tts: "TTS处理阶段",
  preParamCheck: "参数校验错误",
};
