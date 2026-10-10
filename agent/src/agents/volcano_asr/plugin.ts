import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { volcanoAsrConnection } from "./connection";
import { getTranscription, recognize, recognizeFlash, submitTranscription } from "./executors";

const audioUrlField = z
  .string()
  .url()
  .describe("Publicly accessible audio file URL (http/https)");

const submitFields = {
  audioUrl: audioUrlField,
  format: z.string().optional().describe("Audio format, e.g. mp3/wav/ogg"),
  codec: z.string().optional().describe("Codec, e.g. raw/opus"),
  rate: z.number().int().optional().describe("Sample rate in Hz"),
  bits: z.number().int().optional().describe("Bit depth"),
  channel: z.number().int().optional().describe("Channel count"),
  modelName: z.string().optional(),
  enableItn: z.boolean().optional().describe("Inverse text normalization (default true)"),
  enablePunc: z.boolean().optional().describe("Punctuation (default false; true for flash)"),
  enableDdc: z.boolean().optional(),
  enableSpeakerInfo: z.boolean().optional(),
  showUtterances: z.boolean().optional().describe("Return per-sentence utterances"),
  sensitiveWordsFilter: z.string().optional(),
};

const pollFields = {
  timeoutSeconds: z.number().int().optional().describe("Polling timeout in seconds (default 120)"),
  intervalSeconds: z.number().int().optional().describe("Polling interval in seconds (default 3)"),
  responseFormat: z
    .enum(["text", "standard", "full"])
    .optional()
    .describe("text = transcript only; standard = +metadata; full = +utterances/raw"),
};

const SCHEMAS = {
  submit: z.object(submitFields),
  get: z.object({
    requestId: z.string().describe("requestId returned by submit_transcription"),
    ...pollFields,
  }),
  recognize: z.object({ ...submitFields, ...pollFields }),
  flash: z.object({
    audioUrl: audioUrlField,
    format: z.string().describe("Required for flash: mp3/wav/ogg/..."),
    codec: z.string().optional(),
    rate: z.number().int().optional(),
    bits: z.number().int().optional(),
    channel: z.number().int().optional(),
    language: z.string().optional(),
    modelName: z.string().optional(),
    enableItn: z.boolean().optional(),
    enablePunc: z.boolean().optional(),
    enableDdc: z.boolean().optional(),
    enableChannelSplit: z.boolean().optional(),
    showUtterances: z.boolean().optional(),
    vadSegment: z.boolean().optional(),
    enableAutoLang: z.boolean().optional(),
    enableLid: z.boolean().optional(),
    endWindowSize: z.number().int().optional(),
    sensitiveWordsFilter: z.string().optional(),
    responseFormat: z.enum(["text", "standard", "full"]).optional(),
  }),
};

export const volcanoAsrPlugin: Plugin = {
  meta: {
    type: "volcano-asr",
    name: "Volcano ASR",
    category: "data",
    description:
      "火山引擎（豆包语音）录音文件识别。支持标准版异步识别（提交 + 轮询）与极速版同步识别。输入为音频文件的公网可访问 URL。",
    version: "1.0.0",
    tools: [
      {
        name: "submit_transcription",
        description: "Submit an async ASR task, returning a requestId.",
      },
      {
        name: "get_transcription",
        description: "Poll an ASR task by requestId until it completes.",
      },
      { name: "recognize", description: "Submit and poll an ASR task in one call." },
      { name: "recognize_flash", description: "Synchronous flash ASR (no polling)." },
    ],
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "Connections",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "Connect all available connections" },
      },
    },
    defaultConfig: { connections: [], connectAll: false },
    openExpose: [
      { name: "submit_transcription" },
      { name: "get_transcription", readOnly: true },
      { name: "recognize" },
      { name: "recognize_flash" },
    ],
  },

  connection: volcanoAsrConnection,

  middleware: (rawConfig) => {
    const pluginConfig = { ...rawConfig, connectionType: "volcano-asr" };
    return createMiddleware({
      name: "VolcanoAsr",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.submit>, exeConfig) =>
            submitTranscription(input, exeConfig, pluginConfig),
          {
            name: "submit_transcription",
            description:
              "提交一个火山引擎标准版 ASR 异步任务，立即返回 requestId。音频须为公网可访问 URL。随后用 get_transcription 查询结果。",
            schema: SCHEMAS.submit,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.get>, exeConfig) =>
            getTranscription(input, exeConfig, pluginConfig),
          {
            name: "get_transcription",
            description:
              "按 requestId 轮询火山引擎 ASR 任务，直到完成或超时。可传 timeoutSeconds/intervalSeconds/responseFormat。",
            schema: SCHEMAS.get,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.recognize>, exeConfig) =>
            recognize(input, exeConfig, pluginConfig),
          {
            name: "recognize",
            description:
              "一步完成火山引擎标准版 ASR：提交任务并轮询到结果，返回识别文本。适合单次调用。",
            schema: SCHEMAS.recognize,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.flash>, exeConfig) =>
            recognizeFlash(input, exeConfig, pluginConfig),
          {
            name: "recognize_flash",
            description:
              "火山引擎极速版录音文件识别：同步返回结果，无需轮询。必须显式指定 format。限制：最大 100MB、时长不超过 2 小时。",
            schema: SCHEMAS.flash,
          },
        ),
      ],
    });
  },
};

PluginRegistry.register(volcanoAsrPlugin);
