import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
import { platformServiceConnection } from "../connection";
import {
  ENDPOINT_ID_PATTERN,
  MESSAGE_ID_PATTERN,
  WEBHOOK_EVENT_TYPES,
  deleteDestination,
  registerDestination,
  webhooksGetDeliveryStatus,
  webhooksListDestinations,
  webhooksListRecentEvents,
  webhooksPublishEvent,
} from "./executors";

const CHANNELS = z
  .array(z.string().regex(/^[A-Za-z0-9._:+-]{1,128}$/))
  .max(10)
  .describe("Svix channel names used to target a subset of destinations");

const SCHEMAS = {
  listDestinations: z.object({}),
  publish: z.object({
    eventType: z.enum(WEBHOOK_EVENT_TYPES),
    payload: z.record(z.unknown()),
    channels: CHANNELS.optional(),
  }),
  listRecent: z.object({ limit: z.number().int().optional() }),
  deliveryStatus: z.object({ messageId: z.string().regex(MESSAGE_ID_PATTERN) }),
  register: z.object({
    url: z
      .string()
      .url()
      .refine((u) => /^https?:\/\//.test(u), { message: "url must be http(s)" })
      .describe("Receiver URL (http/https)"),
    filterTypes: z.array(z.enum(WEBHOOK_EVENT_TYPES)).optional(),
    channels: CHANNELS.optional(),
    description: z.string().optional(),
  }),
  delete: z.object({
    endpointId: z.string().regex(ENDPOINT_ID_PATTERN),
    confirm: z.boolean().optional().describe("Must be true to execute; otherwise a confirmation prompt is returned"),
  }),
};

function destinationSummary(d: {
  filterTypes?: string[];
  channels?: string[];
  topics?: string[];
}): string {
  const eventTypes = d.filterTypes ?? d.topics ?? [];
  const parts: string[] = [];
  if (eventTypes.length > 0) parts.push(eventTypes.join(", "));
  if (d.channels && d.channels.length > 0) parts.push(`channels: ${d.channels.join(", ")}`);
  return parts.join(" | ");
}

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
    // config, so publish targeting is caller-provided via `channels`; without channels
    // the event fans out to every subscriber of the eventType.
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
      const rows = await request<
        Array<{
          endpointId: string;
          url: string;
          filterTypes?: string[];
          channels?: string[];
          topics?: string[];
        }>
      >({
        conn: connectionFromConfig(config),
        tenantId,
        method: "GET",
        path: "/api/v1/webhooks/destinations",
      });
      return rows.map((d) => ({
        id: d.endpointId,
        name: d.url,
        description: destinationSummary(d),
      }));
    },
  },
  middleware: (rawConfig) => {
    const pluginConfig = { ...rawConfig, connectionType: "webhooks" };
    return createMiddleware({
      name: "Webhooks",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.listDestinations>, exeConfig) =>
            webhooksListDestinations(input, exeConfig, pluginConfig),
          {
            name: "list_destinations",
            description: "List the tenant's delivery destinations (signing secrets are not returned).",
            schema: SCHEMAS.listDestinations,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.publish>, exeConfig) =>
            webhooksPublishEvent(input, exeConfig, pluginConfig),
          {
            name: "publish_event",
            description:
              "Publish a factory event with {eventType, payload, channels?}. eventType must come from the fixed list. Target a subset of destinations only via channels; never send endpointIds (the server rejects them with 400). On the MCP path the connection scope is not resolved, so channels must be provided by the caller; without channels the event fans out to every subscriber of the eventType.",
            schema: SCHEMAS.publish,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.listRecent>, exeConfig) =>
            webhooksListRecentEvents(input, exeConfig, pluginConfig),
          {
            name: "list_recent_events",
            description: "List recent events for the current tenant (messageId/eventType/timestamp).",
            schema: SCHEMAS.listRecent,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.deliveryStatus>, exeConfig) =>
            webhooksGetDeliveryStatus(input, exeConfig, pluginConfig),
          {
            name: "get_delivery_status",
            description: "Query the delivery status and next retry time of a message for each destination.",
            schema: SCHEMAS.deliveryStatus,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.register>, exeConfig) =>
            registerDestination(input, exeConfig, pluginConfig),
          {
            name: "register_destination",
            description:
              "Register a delivery destination with optional filterTypes and channels, and return the whsec signing secret. filterTypes limits the eventTypes delivered; channels target channel-filtered publishes. The secret is returned by Svix only at creation time and enters the tool result (including conversation and audit history); it cannot be retrieved later, so deliver it to the receiver securely and do not log or forward it.",
            schema: SCHEMAS.register,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.delete>, exeConfig) =>
            deleteDestination(input, exeConfig, pluginConfig),
          {
            name: "delete_destination",
            description: "Delete a delivery destination. Requires user confirmation, then pass confirm:true.",
            schema: SCHEMAS.delete,
          },
        ),
      ],
    });
  },
};

PluginRegistry.register(webhooksPlugin);
