import type { PluginConnection } from "@axiom-lattice/protocols";
import { connectionFromConfig } from "./client";

export const platformServiceConnection: PluginConnection = {
  fields: [
    {
      key: "baseUrl",
      type: "string",
      title: "Base URL",
      widget: "input",
      required: true,
      helpText: "可由环境变量 PLATFORM_SERVICE_URL 提供",
    },
    {
      key: "apiKey",
      type: "password",
      title: "API Key",
      widget: "password",
      helpText: "可由环境变量 FILE_SERVICE_API_KEY 提供；留空表示服务端未启用校验",
    },
  ],
  test: async (config) => {
    try {
      const base = connectionFromConfig(config).baseUrl;
      const res = await fetch(`${base}/actuator/health`);
      return { ok: res.ok, message: res.ok ? "连接成功" : `HTTP ${res.status}` };
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : String(err) };
    }
  },
};
