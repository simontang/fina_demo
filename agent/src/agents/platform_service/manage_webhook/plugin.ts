import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { platformServiceConnection } from "../connection";
import { WEBHOOK_TOPICS } from "../webhooks/executors";
import {
  manageWebhookDeleteDestination,
  manageWebhookListDestinations,
  manageWebhookRegisterDestination,
} from "./executors";

const SCHEMAS = {
  list: z.object({}),
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
    endpointId: z.string().regex(/^[A-Za-z0-9_-]+$/),
    confirm: z.boolean().optional().describe("Must be true to execute; otherwise a confirmation prompt is returned"),
  }),
};

export const manageWebhookPlugin: Plugin = {
  meta: {
    type: "manage_webhook",
    name: "Webhook Management",
    description:
      "Manage delivery destinations: register (returns the whsec signing secret), list, and delete. Separate from the runtime webhooks plugin to allow domain-scoped grants.",
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
    openExpose: [
      { name: "manage_webhook_list_destinations", readOnly: true },
      { name: "manage_webhook_register_destination" },
      { name: "manage_webhook_delete_destination", destructive: true },
    ],
  },
  connection: platformServiceConnection,
  middleware: (rawConfig) =>
    createMiddleware({
      name: "ManageWebhook",
      tools: [
        tool(
          (input: z.infer<typeof SCHEMAS.list>, exeConfig) =>
            manageWebhookListDestinations(input, exeConfig, rawConfig),
          {
            name: "manage_webhook_list_destinations",
            description: "List all delivery destinations for the current tenant (admin view; signing secrets are not returned).",
            schema: SCHEMAS.list,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.register>, exeConfig) =>
            manageWebhookRegisterDestination(input, exeConfig, rawConfig),
          {
            name: "manage_webhook_register_destination",
            description:
              "Register a delivery destination and return the whsec signing secret. The secret enters the tool result (including conversation and audit history) and can be retrieved again via the admin API; use only in scenarios granted to administrators, and do not log or forward it.",
            schema: SCHEMAS.register,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.delete>, exeConfig) =>
            manageWebhookDeleteDestination(input, exeConfig, rawConfig),
          {
            name: "manage_webhook_delete_destination",
            description: "Delete a delivery destination. Requires user confirmation, then pass confirm:true.",
            schema: SCHEMAS.delete,
          },
        ),
      ],
    }),
};

PluginRegistry.register(manageWebhookPlugin);
