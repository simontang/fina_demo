import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  ENDPOINT_ID_PATTERN,
  MESSAGE_ID_PATTERN,
  WEBHOOK_TOPICS,
  deleteDestination,
  registerDestination,
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
  register: z.object({
    url: z
      .string()
      .url()
      .refine((u) => /^https?:\/\//.test(u), { message: "url must be http(s)" })
      .describe("Receiver URL (http/https)"),
    topics: z.array(z.enum(WEBHOOK_TOPICS)).min(1).describe("List of subscribed event topics"),
    description: z.string().optional(),
  }),
  delete: z.object({
    endpointId: z.string().regex(ENDPOINT_ID_PATTERN),
    confirm: z.boolean().optional().describe("Must be true to execute; otherwise a confirmation prompt is returned"),
  }),
};

export const webhooksPlugin: Plugin = {
  meta: {
    type: "webhooks",
    name: "Webhooks",
    description:
      "Register and delete delivery destinations (registration returns the whsec signing secret), publish factory events to registered destinations, and query events and delivery status.",
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
    // Exposed on the Open/MCP surface. Note: the MCP path does not resolve connection
    // config, so `selectedEntities` scope only applies on the agent path; on MCP,
    // publish fans out to every subscriber of the topic.
    openExpose: [
      { name: "list_destinations", readOnly: true },
      { name: "publish_event" },
      { name: "list_recent_events", readOnly: true },
      { name: "get_delivery_status", readOnly: true },
      { name: "register_destination" },
      { name: "delete_destination", destructive: true },
    ],
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
            name: "list_destinations",
            description: "List delivery destinations within the current connection scope (signing secrets are not returned).",
            schema: SCHEMAS.listDestinations,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.publish>, exeConfig) =>
            webhooksPublishEvent(input, exeConfig, rawConfig),
          {
            name: "publish_event",
            description:
              "Publish a factory event to delivery destinations. topic must come from the fixed list; endpointIds can only narrow within the selected scope. (v1: scope is a client-side constraint; server-side targeted delivery comes in a later version. When no scope is configured, fan out to all destinations for the topic. On the MCP path the selected scope is not applied, so publishing always fans out to every subscriber of the topic.)",
            schema: SCHEMAS.publish,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.listRecent>, exeConfig) =>
            webhooksListRecentEvents(input, exeConfig, rawConfig),
          {
            name: "list_recent_events",
            description: "List recent events for the current tenant (messageId/topic/timestamp).",
            schema: SCHEMAS.listRecent,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.deliveryStatus>, exeConfig) =>
            webhooksGetDeliveryStatus(input, exeConfig, rawConfig),
          {
            name: "get_delivery_status",
            description: "Query the delivery status and next retry time of a message for each destination.",
            schema: SCHEMAS.deliveryStatus,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.register>, exeConfig) =>
            registerDestination(input, exeConfig, rawConfig),
          {
            name: "register_destination",
            description:
              "Register a delivery destination and return the whsec signing secret. The secret enters the tool result (including conversation and audit history) and can be retrieved again via the admin API; use only in scenarios granted to administrators, and do not log or forward it.",
            schema: SCHEMAS.register,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.delete>, exeConfig) =>
            deleteDestination(input, exeConfig, rawConfig),
          {
            name: "delete_destination",
            description: "Delete a delivery destination. Requires user confirmation, then pass confirm:true.",
            schema: SCHEMAS.delete,
          },
        ),
      ],
    }),
};

PluginRegistry.register(webhooksPlugin);
