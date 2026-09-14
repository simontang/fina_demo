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
      .describe("接收端 URL（http/https）"),
    topics: z.array(z.enum(WEBHOOK_TOPICS)).min(1).describe("订阅的事件 topic 列表"),
    description: z.string().optional(),
  }),
  delete: z.object({
    endpointId: z.string().regex(/^[A-Za-z0-9_-]+$/),
    confirm: z.boolean().optional().describe("必须为 true 才执行；否则返回确认提示"),
  }),
};

export const manageWebhookPlugin: Plugin = {
  meta: {
    type: "manage_webhook",
    name: "Webhook 管理",
    description:
      "管理投递目标：注册（返回 whsec 签名密钥）、列出、删除。与运行时 webhooks 插件分离，便于按域授权。",
    version: "1.0.0",
    configSchema: {
      type: "object",
      properties: {
        connections: {
          type: "array",
          title: "连接",
          widget: "connectionSelect",
          items: { type: "string" },
        },
        connectAll: { type: "boolean", title: "连接所有可用连接" },
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
            description: "列出当前租户的全部投递目标（管理视图，不返回签名密钥）。",
            schema: SCHEMAS.list,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.register>, exeConfig) =>
            manageWebhookRegisterDestination(input, exeConfig, rawConfig),
          {
            name: "manage_webhook_register_destination",
            description:
              "注册投递目标并返回 whsec 签名密钥。该密钥会进入工具结果（含对话与审计历史），且可由管理接口重新取回；仅应在授权给管理员的场景使用，勿记录/转发。",
            schema: SCHEMAS.register,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.delete>, exeConfig) =>
            manageWebhookDeleteDestination(input, exeConfig, rawConfig),
          {
            name: "manage_webhook_delete_destination",
            description: "删除投递目标。必须用户确认后传 confirm:true。",
            schema: SCHEMAS.delete,
          },
        ),
      ],
    }),
};

PluginRegistry.register(manageWebhookPlugin);
