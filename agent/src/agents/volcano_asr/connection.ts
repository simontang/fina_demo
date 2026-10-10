import type { PluginConnection } from "@axiom-lattice/protocols";
import {
  DEFAULT_BASE_URL,
  DEFAULT_RESOURCE_ID,
  SUBMIT_PATH,
  callApi,
  connectionFromConfig,
  isSuccess,
  maskSecret,
  newRequestId,
} from "./client";

const TEST_AUDIO_URL =
  "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_cancan_mars_bigtts.mp3";

export const volcanoAsrConnection: PluginConnection = {
  fields: [
    {
      key: "baseUrl",
      type: "string",
      title: "Base URL",
      widget: "input",
      required: true,
      helpText: `火山引擎语音服务根地址，默认 ${DEFAULT_BASE_URL}`,
    },
    {
      key: "apiKey",
      type: "password",
      title: "API Key",
      widget: "password",
      helpText:
        "火山引擎语音服务的 API Key（作为 X-Api-Key），可由 VOLC_ASR_API_KEY 环境变量提供",
    },
    {
      key: "resourceId",
      type: "string",
      title: "Resource ID",
      widget: "input",
      helpText: `标准版资源 id，默认 ${DEFAULT_RESOURCE_ID}；极速版固定使用 volc.bigasr.auc_turbo`,
    },
  ],
  test: async (config) => {
    const conn = connectionFromConfig(config);
    const startedAt = Date.now();
    const diag: Record<string, unknown> = {
      baseUrl: conn.baseUrl,
      resourceId: conn.resourceId,
      apiKey: maskSecret(conn.apiKey),
      endpoint: conn.baseUrl + SUBMIT_PATH,
      testAudioUrl: TEST_AUDIO_URL,
    };

    if (!conn.apiKey) {
      return { ok: false, message: "缺少 API Key：连接器里没有读到 apiKey 字段。", details: diag };
    }

    let resp;
    try {
      resp = await callApi({
        baseUrl: conn.baseUrl,
        apiKey: conn.apiKey,
        resourceId: conn.resourceId,
        path: SUBMIT_PATH,
        body: {
          audio: { url: TEST_AUDIO_URL, format: "mp3" },
          request: { model_name: "bigmodel", enable_itn: true },
        },
        requestId: newRequestId(),
        sequence: -1,
      });
    } catch (err) {
      return {
        ok: false,
        message: `无法连接服务：${err instanceof Error ? err.message : String(err)}`,
        details: { ...diag, elapsedMs: Date.now() - startedAt },
      };
    }

    const elapsedMs = Date.now() - startedAt;
    const details: Record<string, unknown> = {
      ...diag,
      elapsedMs,
      httpStatus: resp.httpStatus,
      statusCode: resp.statusCode || "(none)",
      apiMessage: resp.message || "(none)",
      requestId: resp.requestId,
      responseBody: resp.rawBody ? resp.rawBody.slice(0, 500) : "(empty)",
    };

    if (isSuccess(resp.statusCode)) {
      return {
        ok: true,
        message: `连接正常：鉴权通过，任务已受理（statusCode=${resp.statusCode}，耗时 ${elapsedMs}ms）`,
        details,
      };
    }

    const authFailed =
      resp.statusCode === "45000010" ||
      /invalid x-api-key/i.test(resp.message) ||
      /grant not found/i.test(resp.message) ||
      resp.httpStatus === 401;

    if (authFailed) {
      return {
        ok: false,
        message: `鉴权失败：${resp.message || "Invalid X-Api-Key"}（statusCode=${resp.statusCode || "无"}，httpStatus=${resp.httpStatus}）`,
        details,
      };
    }

    return {
      ok: false,
      message: `连接失败：${resp.message || "未知错误"}（statusCode=${resp.statusCode || "无"}，httpStatus=${resp.httpStatus}）`,
      details,
    };
  },
};
