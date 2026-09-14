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
      helpText: "Can be provided by the PLATFORM_SERVICE_URL environment variable",
    },
    {
      key: "apiKey",
      type: "password",
      title: "API Key",
      widget: "password",
      helpText:
        "Can be provided by the FILE_SERVICE_API_KEY environment variable; leave blank if the server does not enable authentication",
    },
  ],
  test: async (config) => {
    try {
      const base = connectionFromConfig(config).baseUrl;
      const res = await fetch(`${base}/actuator/health`);
      return { ok: res.ok, message: res.ok ? "Connected" : `HTTP ${res.status}` };
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : String(err) };
    }
  },
};
