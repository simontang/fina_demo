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
    {
      key: "boStoreKey",
      type: "password",
      title: "Business Object Store Key",
      widget: "password",
      helpText:
        "Store-bound key used by Business Object APIs. One key authorizes exactly one BO store; BO object access does not send X-Tenant-Id.",
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
