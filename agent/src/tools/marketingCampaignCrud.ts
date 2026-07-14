import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";
import { cdpFetch, getTenantIdFromExecutionConfig } from "./cdpToolClient";

const jsonStrategySchema = z.union([z.record(z.unknown()), z.array(z.unknown())]);

const campaignMutationSchema = z.object({
  name: z.string().describe("活动名称"),
  description: z.string().optional().describe("活动描述"),
  type: z.string().describe("活动类型，如 reactivation, retention, acquisition, promotion"),
  status: z.enum(["draft", "scheduled", "running", "stopped", "completed"]).optional().describe("活动状态"),
  goal: z.string().describe("活动目标"),
  startTime: z.string().describe("开始时间，格式 yyyy-MM-dd HH:mm:ss"),
  endTime: z.string().describe("结束时间，格式 yyyy-MM-dd HH:mm:ss，必须晚于 startTime"),
  mainSegmentDataId: z.number().optional().describe("主人群包，对应 segment_data id"),
  segmentationStrategy: jsonStrategySchema.optional().describe("人群分群策略 JSON"),
  controlGroupStrategy: jsonStrategySchema.optional().describe("control group 策略 JSON"),
  contentChannelStrategy: jsonStrategySchema.optional().describe("内容渠道/策略 JSON"),
  offerStrategy: jsonStrategySchema.optional().describe("offer 策略 JSON"),
  waveStrategy: jsonStrategySchema.optional().describe("多波段策略 JSON"),
  abTestStrategy: jsonStrategySchema.optional().describe("A/B 测试策略 JSON"),
  statistics: jsonStrategySchema.optional().describe("统计 JSON"),
});

registerToolLattice(
  "marketing_campaign_list",
  {
    name: "marketing_campaign_list",
    description:
      "分页查询营销活动列表。可按 type 和 status 过滤，返回活动基础信息、主人群包、策略 JSON、生命周期时间和分页信息。",
    schema: z.object({
      type: z.string().optional().describe("按活动类型过滤"),
      status: z.enum(["draft", "scheduled", "running", "stopped", "completed"]).optional().describe("按活动状态过滤"),
      page: z.number().default(1).describe("页码，默认 1"),
      pageSize: z.number().default(20).describe("每页条数，默认 20"),
    }),
  },
  async (input, exeConfig) => {
    const params = new URLSearchParams();
    if (input.type) params.set("type", input.type);
    if (input.status) params.set("status", input.status);
    params.set("page", String(input.page));
    params.set("pageSize", String(input.pageSize));
    const result = await cdpFetch(
      `/api/v1/marketing-campaigns?${params.toString()}`,
      undefined,
      getTenantIdFromExecutionConfig(exeConfig),
    );
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_get",
  {
    name: "marketing_campaign_get",
    description: "按 ID 查询单个营销活动，自动按当前 tenant 隔离。",
    schema: z.object({
      id: z.number().describe("营销活动 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(
      `/api/v1/marketing-campaigns/${input.id}`,
      undefined,
      getTenantIdFromExecutionConfig(exeConfig),
    );
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_create",
  {
    name: "marketing_campaign_create",
    description:
      "创建营销活动。复杂策略字段使用 JSON object/array。tenant 从 runConfig.tenantId 注入，body 中不要传 tenantId。",
    schema: campaignMutationSchema,
  },
  async (input, exeConfig) => {
    const result = await cdpFetch("/api/v1/marketing-campaigns", {
      method: "POST",
      body: JSON.stringify(input),
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_update",
  {
    name: "marketing_campaign_update",
    description:
      "更新营销活动。请求体字段同创建接口，复杂策略字段使用 JSON object/array。",
    schema: campaignMutationSchema.extend({
      id: z.number().describe("营销活动 ID"),
    }),
  },
  async (input, exeConfig) => {
    const { id, ...body } = input;
    const result = await cdpFetch(`/api/v1/marketing-campaigns/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_delete",
  {
    name: "marketing_campaign_delete",
    description: "逻辑删除营销活动。",
    schema: z.object({
      id: z.number().describe("营销活动 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(`/api/v1/marketing-campaigns/${input.id}`, {
      method: "DELETE",
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_start",
  {
    name: "marketing_campaign_start",
    description: "启动 draft/scheduled 状态的营销活动，成功后状态变为 running。",
    schema: z.object({
      id: z.number().describe("营销活动 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(`/api/v1/marketing-campaigns/${input.id}/start`, {
      method: "POST",
      body: "{}",
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_stop",
  {
    name: "marketing_campaign_stop",
    description: "停止 scheduled/running 状态的营销活动，成功后状态变为 stopped。",
    schema: z.object({
      id: z.number().describe("营销活动 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(`/api/v1/marketing-campaigns/${input.id}/stop`, {
      method: "POST",
      body: "{}",
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

registerToolLattice(
  "marketing_campaign_schedule",
  {
    name: "marketing_campaign_schedule",
    description:
      "将营销活动设置为 scheduled。可选覆盖 startTime/endTime；未传则使用活动已有时间。",
    schema: z.object({
      id: z.number().describe("营销活动 ID"),
      startTime: z.string().optional().describe("新的开始时间，格式 yyyy-MM-dd HH:mm:ss"),
      endTime: z.string().optional().describe("新的结束时间，格式 yyyy-MM-dd HH:mm:ss"),
    }),
  },
  async (input, exeConfig) => {
    const { id, ...body } = input;
    const result = await cdpFetch(`/api/v1/marketing-campaigns/${id}/schedule`, {
      method: "POST",
      body: JSON.stringify(body),
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);
