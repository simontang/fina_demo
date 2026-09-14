import { PluginRegistry } from "@axiom-lattice/core";
import type { Plugin } from "@axiom-lattice/protocols";
import { createMiddleware, tool } from "langchain";
import { z } from "zod";
import { connectionFromConfig, request } from "../client";
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
    name: "Webhook 事件",
    description:
      "向已注册的投递目标发布工厂事件，并查询事件与投递状态。目标的增删在控制台完成（含 whsec 一次性展示），不在对话中进行。",
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
    // v1：全部工具仅 agent 路径（不声明 openExpose）。MCP 路径 runConfig 不带连接解析，
    // selectedEntities scope 无法生效，待 core 补齐后再上 Open 面。
  },
  connection: {
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
    // gateway 4.3.1 起以第二参传入租户上下文 { tenantId }（protocols 的类型尚未同步，故此处用可选参数）。
    // 若运行时未提供租户，则抛租户缺失错误，绝不回退到默认租户。
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
            description: "列出当前连接 scope 内的投递目标（不返回签名密钥）。",
            schema: SCHEMAS.listDestinations,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.publish>, exeConfig) =>
            webhooksPublishEvent(input, exeConfig, rawConfig),
          {
            name: "webhooks_publish_event",
            description:
              "向投递目标发布一个工厂事件。topic 必须来自固定清单；endpointIds 只能在已选 scope 内收窄。（v1：scope 为客户端约束；服务端定向发送待后续版本，未配置 scope 时按 topic 全量扇出）",
            schema: SCHEMAS.publish,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.listRecent>, exeConfig) =>
            webhooksListRecentEvents(input, exeConfig, rawConfig),
          {
            name: "webhooks_list_recent_events",
            description: "列出当前租户最近的事件（messageId/topic/timestamp）。",
            schema: SCHEMAS.listRecent,
          },
        ),
        tool(
          (input: z.infer<typeof SCHEMAS.deliveryStatus>, exeConfig) =>
            webhooksGetDeliveryStatus(input, exeConfig, rawConfig),
          {
            name: "webhooks_get_delivery_status",
            description: "查询某条消息在各目标的投递状态与下次重试时间。",
            schema: SCHEMAS.deliveryStatus,
          },
        ),
      ],
    }),
};

PluginRegistry.register(webhooksPlugin);
