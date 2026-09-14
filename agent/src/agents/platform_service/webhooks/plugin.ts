import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  MESSAGE_ID_PATTERN,
  WEBHOOK_TOPICS,
  webhooksGetDeliveryStatus,
  webhooksListDestinations,
  webhooksListRecentEvents,
  webhooksPublishEvent,
} from "./executors";

const SCHEMAS = {
  listDestinations: z.object({}),
  publish: z.object({
    topic: z.enum(WEBHOOK_TOPICS),
    data: z.record(z.unknown()),
    endpointIds: z.array(z.string()).optional(),
  }),
  listRecent: z.object({ limit: z.number().int().optional() }),
  deliveryStatus: z.object({ messageId: z.string().regex(MESSAGE_ID_PATTERN) }),
};

export const webhooksPlugin: Plugin = {
  meta: {
    type: "webhooks",
    name: "Webhooks",
    description:
      "Publish factory events to registered delivery destinations and query events and delivery status. Registration and deletion of destinations are handled by the manage_webhook admin domain.",
    version: "1.0.0",
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
    // v1: all tools are agent-path only (openExpose is not declared). On the MCP path,
    // runConfig carries no connection resolution, so the selectedEntities scope cannot take
    // effect; we will expose the Open surface once core fills this gap.
  },
  connection: {
    ...platformServiceConnection,
    // Since gateway 4.3.1 the tenant context { tenantId } is passed as the second argument
    // (protocols types are not synced yet, hence the optional parameter here).
    // If no tenant is provided at runtime, throw a missing-tenant error and never fall back to a default tenant.
    discover: async (config, context?: { tenantId?: string }) => {
      const tenantId = context?.tenantId;
      if (!tenantId) throw new Error("tenant context is missing");
      const rows = await request<Array<{ endpointId: string; url: string; topics?: string[] }>>({
        conn: connectionFromConfig(config),
        tenantId,
        method: "GET",
        path: "/api/v1/webhooks/destinations",
      });
      return rows.map((d) => ({
        id: d.endpointId,
        name: d.url,
        description: (d.topics ?? []).join(", "),
      }));
    },
  },
  middleware: (rawConfig) =>
    createMiddleware({
      name: "Webhooks",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.listDestinations>, exeConfig) =>
            webhooksListDestinations(input, exeConfig, rawConfig),
          {
            name: "webhooks_list_destinations",
            description: "List delivery destinations within the current connection scope (signing secrets are not returned).",
            schema: SCHEMAS.listDestinations,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.publish>, exeConfig) =>
            webhooksPublishEvent(input, exeConfig, rawConfig),
          {
            name: "webhooks_publish_event",
            description:
              "Publish a factory event to delivery destinations. topic must come from the fixed list; endpointIds can only narrow within the selected scope. (v1: scope is a client-side constraint; server-side targeted delivery comes in a later version. When no scope is configured, fan out to all destinations for the topic)",
            schema: SCHEMAS.publish,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.listRecent>, exeConfig) =>
            webhooksListRecentEvents(input, exeConfig, rawConfig),
          {
            name: "webhooks_list_recent_events",
            description: "List recent events for the current tenant (messageId/topic/timestamp).",
            schema: SCHEMAS.listRecent,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.deliveryStatus>, exeConfig) =>
            webhooksGetDeliveryStatus(input, exeConfig, rawConfig),
          {
            name: "webhooks_get_delivery_status",
            description: "Query the delivery status and next retry time of a message for each destination.",
            schema: SCHEMAS.deliveryStatus,
          },
        ),
      ],
    }),
};

PluginRegistry.register(webhooksPlugin);
