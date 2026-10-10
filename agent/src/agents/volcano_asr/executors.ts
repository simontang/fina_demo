import {
  FLASH_PATH,
  FLASH_RESOURCE_ID,
  QUERY_PATH,
  SUBMIT_PATH,
  callApi,
  errorResult,
  extractText,
  extractUtterances,
  isProcessing,
  isSuccess,
  newRequestId,
  resolveConnection,
  sleep,
  type VolcanoAsrConn,
} from "./client";

const NO_API_KEY = JSON.stringify({
  ok: false,
  code: "NO_API_KEY",
  message: "未配置 API Key：请在插件连接器中填写火山引擎语音服务的 API Key。",
});

function badRequest(message: string): string {
  return JSON.stringify({ ok: false, code: "BAD_REQUEST", message });
}

type ResponseFormat = "text" | "standard" | "full";

interface AudioInput {
  audioUrl: string;
  format?: string;
  codec?: string;
  rate?: number;
  bits?: number;
  channel?: number;
  language?: string;
}

interface SubmitInput extends AudioInput {
  modelName?: string;
  enableItn?: boolean;
  enablePunc?: boolean;
  enableDdc?: boolean;
  enableSpeakerInfo?: boolean;
  showUtterances?: boolean;
  sensitiveWordsFilter?: string;
}

interface PollInput {
  timeoutSeconds?: number;
  intervalSeconds?: number;
  responseFormat?: ResponseFormat;
}

function buildAudio(input: AudioInput): Record<string, unknown> {
  const audio: Record<string, unknown> = { url: input.audioUrl };
  if (input.format) audio.format = input.format;
  if (input.codec) audio.codec = input.codec;
  if (input.rate) audio.rate = input.rate;
  if (input.bits) audio.bits = input.bits;
  if (input.channel) audio.channel = input.channel;
  if (input.language) audio.language = input.language;
  return audio;
}

function buildSubmitRequest(input: SubmitInput): Record<string, unknown> {
  return {
    model_name: input.modelName ?? "bigmodel",
    enable_itn: input.enableItn ?? true,
    enable_punc: input.enablePunc ?? false,
    enable_ddc: input.enableDdc ?? false,
    enable_speaker_info: input.enableSpeakerInfo ?? false,
    enable_channel_split: false,
    show_utterances: input.showUtterances ?? false,
    vad_segment: false,
    sensitive_words_filter: input.sensitiveWordsFilter ?? "",
  };
}

function pollTimeout(input: PollInput): number {
  return typeof input.timeoutSeconds === "number" && input.timeoutSeconds > 0
    ? input.timeoutSeconds
    : 120;
}

function pollInterval(input: PollInput): number {
  return typeof input.intervalSeconds === "number" && input.intervalSeconds >= 0
    ? input.intervalSeconds
    : 3;
}

interface PollResult {
  outcome: "completed" | "timeout" | "failed";
  statusCode: string;
  httpStatus: number;
  message: string;
  body?: unknown;
  attempts: number;
}

async function poll(
  conn: VolcanoAsrConn,
  requestId: string,
  timeoutSeconds: number,
  intervalSeconds: number,
): Promise<PollResult> {
  const deadline = Date.now() + timeoutSeconds * 1000;
  let attempts = 0;
  for (;;) {
    attempts += 1;
    const last = await callApi({
      baseUrl: conn.baseUrl,
      apiKey: conn.apiKey,
      resourceId: conn.resourceId,
      path: QUERY_PATH,
      body: {},
      requestId,
    });
    if (isSuccess(last.statusCode)) {
      return {
        outcome: "completed",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        body: last.body,
        attempts,
      };
    }
    if (!isProcessing(last.statusCode)) {
      return {
        outcome: "failed",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        body: last.body,
        attempts,
      };
    }
    if (Date.now() + intervalSeconds * 1000 > deadline) {
      return {
        outcome: "timeout",
        statusCode: last.statusCode,
        httpStatus: last.httpStatus,
        message: last.message,
        attempts,
      };
    }
    await sleep(intervalSeconds * 1000);
  }
}

function formatCompleted(requestId: string, r: PollResult, responseFormat: ResponseFormat): string {
  const text = extractText(r.body);
  if (responseFormat === "text") return JSON.stringify({ text });
  const result: Record<string, unknown> = {
    requestId,
    status: "completed",
    statusCode: r.statusCode,
    text,
    attempts: r.attempts,
  };
  if (responseFormat === "full") {
    const utterances = extractUtterances(r.body);
    if (utterances) result.utterances = utterances;
    if (r.body) result.raw = r.body;
  }
  return JSON.stringify(result);
}

function formatTimeout(requestId: string, r: PollResult, timeoutSeconds: number): string {
  return JSON.stringify({
    requestId,
    status: "timeout",
    statusCode: r.statusCode,
    message: r.message || "任务仍在处理中",
    attempts: r.attempts,
    hint: `等待超过 ${timeoutSeconds} 秒仍未完成，可稍后再次调用 get_transcription 传入 requestId 继续查询。`,
  });
}

function queryFailed(requestId: string, r: PollResult): string {
  return JSON.stringify({
    ok: false,
    code: "QUERY_FAILED",
    requestId,
    statusCode: r.statusCode,
    httpStatus: r.httpStatus,
    message: r.message || "未知错误",
    attempts: r.attempts,
  });
}

async function submitTask(
  conn: VolcanoAsrConn,
  input: SubmitInput,
): Promise<{ ok: true; requestId: string } | { ok: false; payload: string }> {
  const requestId = newRequestId();
  const resp = await callApi({
    baseUrl: conn.baseUrl,
    apiKey: conn.apiKey,
    resourceId: conn.resourceId,
    path: SUBMIT_PATH,
    body: { audio: buildAudio(input), request: buildSubmitRequest(input) },
    requestId,
    sequence: -1,
  });
  if (!isSuccess(resp.statusCode)) {
    return {
      ok: false,
      payload: JSON.stringify({
        ok: false,
        code: "SUBMIT_FAILED",
        requestId: resp.requestId,
        statusCode: resp.statusCode,
        httpStatus: resp.httpStatus,
        message: resp.message || "未知错误",
      }),
    };
  }
  return { ok: true, requestId: resp.requestId };
}

export async function submitTranscription(
  input: SubmitInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    const submitted = await submitTask(conn, input);
    if (!submitted.ok) return submitted.payload;
    return JSON.stringify({
      requestId: submitted.requestId,
      status: "submitted",
      audioUrl: input.audioUrl,
      hint: "任务已提交，请用 get_transcription 传入 requestId 查询结果（该工具会自动轮询）。",
    });
  } catch (err) {
    return errorResult(err);
  }
}

export async function getTranscription(
  input: { requestId?: string } & PollInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.requestId || typeof input.requestId !== "string") {
      return badRequest("缺少必填参数 requestId（submit_transcription 返回的任务 ID）。");
    }
    const timeoutSeconds = pollTimeout(input);
    const intervalSeconds = pollInterval(input);
    const responseFormat = input.responseFormat ?? "standard";
    const result = await poll(conn, input.requestId, timeoutSeconds, intervalSeconds);
    if (result.outcome === "completed") return formatCompleted(input.requestId, result, responseFormat);
    if (result.outcome === "timeout") return formatTimeout(input.requestId, result, timeoutSeconds);
    return queryFailed(input.requestId, result);
  } catch (err) {
    return errorResult(err);
  }
}

export async function recognize(
  input: SubmitInput & PollInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    const submitted = await submitTask(conn, input);
    if (!submitted.ok) return submitted.payload;
    const timeoutSeconds = pollTimeout(input);
    const intervalSeconds = pollInterval(input);
    const responseFormat = input.responseFormat ?? "standard";
    const result = await poll(conn, submitted.requestId, timeoutSeconds, intervalSeconds);
    if (result.outcome === "completed")
      return formatCompleted(submitted.requestId, result, responseFormat);
    if (result.outcome === "timeout")
      return formatTimeout(submitted.requestId, result, timeoutSeconds);
    return queryFailed(submitted.requestId, result);
  } catch (err) {
    return errorResult(err);
  }
}

interface FlashInput extends AudioInput {
  modelName?: string;
  enableItn?: boolean;
  enablePunc?: boolean;
  enableDdc?: boolean;
  enableChannelSplit?: boolean;
  showUtterances?: boolean;
  vadSegment?: boolean;
  enableAutoLang?: boolean;
  enableLid?: boolean;
  endWindowSize?: number;
  sensitiveWordsFilter?: string;
  responseFormat?: ResponseFormat;
}

export async function recognizeFlash(
  input: FlashInput,
  exeConfig: unknown,
  rawConfig: unknown,
): Promise<string> {
  try {
    const conn = await resolveConnection(rawConfig, exeConfig);
    if (!conn.apiKey) return NO_API_KEY;
    if (!input.audioUrl || typeof input.audioUrl !== "string") {
      return badRequest("缺少必填参数 audioUrl（音频文件的公网可访问 URL）。");
    }
    if (!input.format || typeof input.format !== "string") {
      return badRequest("缺少必填参数 format（极速版要求显式指定音频格式，如 mp3 / wav / ogg）。");
    }

    const request: Record<string, unknown> = {
      model_name: input.modelName ?? "bigmodel",
      enable_itn: input.enableItn ?? true,
      enable_punc: input.enablePunc ?? true,
      enable_ddc: input.enableDdc ?? false,
      enable_channel_split: input.enableChannelSplit ?? false,
      show_utterances: input.showUtterances ?? false,
      vad_segment: input.vadSegment ?? false,
    };
    if (input.enableAutoLang !== undefined) request.enable_auto_lang = input.enableAutoLang;
    if (input.enableLid !== undefined) request.enable_lid = input.enableLid;
    if (input.endWindowSize) request.end_window_size = input.endWindowSize;
    if (input.sensitiveWordsFilter) request.sensitive_words_filter = input.sensitiveWordsFilter;

    const requestId = newRequestId();
    const resp = await callApi({
      baseUrl: conn.baseUrl,
      apiKey: conn.apiKey,
      resourceId: FLASH_RESOURCE_ID,
      path: FLASH_PATH,
      body: { audio: buildAudio(input), request },
      requestId,
      sequence: -1,
    });
    if (!isSuccess(resp.statusCode)) {
      return JSON.stringify({
        ok: false,
        code: "FLASH_FAILED",
        requestId: resp.requestId,
        statusCode: resp.statusCode,
        httpStatus: resp.httpStatus,
        message: resp.message || "未知错误",
      });
    }

    const text = extractText(resp.body);
    const responseFormat = input.responseFormat ?? "standard";
    if (responseFormat === "text") return JSON.stringify({ text });
    const result: Record<string, unknown> = {
      requestId: resp.requestId,
      status: "completed",
      statusCode: resp.statusCode,
      mode: "flash",
      text,
    };
    if (responseFormat === "full") {
      const utterances = extractUtterances(resp.body);
      if (utterances) result.utterances = utterances;
      if (resp.body) result.raw = resp.body;
    }
    return JSON.stringify(result);
  } catch (err) {
    return errorResult(err);
  }
}
