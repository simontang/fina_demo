import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";
import { cdpFetch, getTenantIdFromExecutionConfig } from "./cdpToolClient";

const conditionSchema = z.object({
  field: z.string().optional(),
  metric: z.string().optional(),
  operator: z.string().optional(),
  value: z.unknown().optional(),
}).passthrough();

const segmentationStrategySchema = z.union([
  z.object({
    version: z.string().optional(),
    audienceKey: z.string().optional(),
    source: z.object({
      segmentDataId: z.number().optional(),
      segmentDefinitionId: z.number().optional(),
      runId: z.string().optional(),
      description: z.string().optional(),
    }).passthrough().optional(),
    subSegments: z.array(z.object({
      subSegmentKey: z.string(),
      name: z.string(),
      priority: z.number().optional(),
      criteria: z.array(conditionSchema).optional(),
      tags: z.array(z.string()).optional(),
    }).passthrough()).optional(),
    assignment: z.object({
      mode: z.string().optional(),
      fallbackSubSegmentKey: z.string().optional(),
    }).passthrough().optional(),
    exclusions: z.array(conditionSchema).optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("人群策略。reactivation 推荐包含 source、subSegments、assignment、exclusions");

const controlGroupStrategySchema = z.union([
  z.object({
    enabled: z.boolean().optional(),
    method: z.string().optional(),
    unit: z.string().optional(),
    ratio: z.number().optional(),
    seed: z.string().optional(),
    stratifyBy: z.array(z.string()).optional(),
    excludeFromWaves: z.boolean().optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("对照组策略，ratio 使用 0-1 小数");

const contentChannelStrategySchema = z.union([
  z.object({
    version: z.string().optional(),
    defaultLocale: z.string().optional(),
    channels: z.array(z.object({
      channelKey: z.string(),
      channel: z.string(),
      templateKey: z.string().optional(),
      eligibleSubSegmentKeys: z.array(z.string()).optional(),
      fallbackForChannelKeys: z.array(z.string()).optional(),
      sendWindow: z.object({
        timezone: z.string().optional(),
        start: z.string().optional(),
        end: z.string().optional(),
      }).passthrough().optional(),
      frequencyCap: z.object({
        maxMessages: z.number().optional(),
        windowDays: z.number().optional(),
      }).passthrough().optional(),
      variables: z.array(z.string()).optional(),
    }).passthrough()).optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("渠道策略。channelKey 供 wave 和 A/B variant 引用");

const offerStrategySchema = z.union([
  z.object({
    version: z.string().optional(),
    budget: z.object({
      currency: z.string().optional(),
      maxTotalCost: z.number().optional(),
    }).passthrough().optional(),
    offers: z.array(z.object({
      offerCode: z.string(),
      type: z.string().optional(),
      value: z.number().optional(),
      currency: z.string().optional(),
      validDays: z.number().optional(),
      eligibleSubSegmentKeys: z.array(z.string()).optional(),
      perCustomerLimit: z.number().optional(),
    }).passthrough()).optional(),
    allocation: z.object({
      method: z.string().optional(),
      rules: z.array(z.object({
        subSegmentKey: z.string().optional(),
        offerCode: z.string().optional(),
      }).passthrough()).optional(),
    }).passthrough().optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("权益策略。offerCode 供 wave 和 A/B variant 引用");

const waveStrategySchema = z.union([
  z.object({
    enabled: z.boolean().optional(),
    timezone: z.string().optional(),
    waves: z.array(z.object({
      waveId: z.string(),
      name: z.string(),
      scheduledAt: z.string().optional(),
      eligibleSubSegmentKeys: z.array(z.string()).optional(),
      channelKeys: z.array(z.string()).optional(),
      offerCodes: z.array(z.string()).optional(),
      entryRule: z.object({
        fromWaveIds: z.array(z.string()).optional(),
        excludeGroups: z.array(z.string()).optional(),
        includeIf: z.array(conditionSchema).optional(),
      }).passthrough().optional(),
    }).passthrough()).optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("波次策略。waveId 必须稳定，后续 wave 通过 fromWaveIds 引用前序 wave");

const abTestStrategySchema = z.union([
  z.object({
    enabled: z.boolean().optional(),
    unit: z.string().optional(),
    primaryMetric: z.string().optional(),
    scope: z.object({
      subSegmentKeys: z.array(z.string()).optional(),
      waveIds: z.array(z.string()).optional(),
    }).passthrough().optional(),
    variants: z.array(z.object({
      variantId: z.string(),
      name: z.string().optional(),
      trafficRatio: z.number().optional(),
      channelKey: z.string().optional(),
      templateKey: z.string().optional(),
      offerCode: z.string().optional(),
    }).passthrough()).optional(),
    winnerPolicy: z.object({
      method: z.string().optional(),
      minSampleSizePerVariant: z.number().optional(),
      confidence: z.number().optional(),
    }).passthrough().optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("A/B 测试策略。scope 和 variant 通过稳定 key 引用 wave、channel 和 offer");

const statisticsSchema = z.union([
  z.object({
    version: z.string().optional(),
    lastComputedAt: z.string().nullable().optional(),
    audience: z.object({
      targetCount: z.number().optional(),
      controlCount: z.number().optional(),
      treatmentCount: z.number().optional(),
    }).passthrough().optional(),
    delivery: z.object({
      sent: z.number().optional(),
      delivered: z.number().optional(),
      opened: z.number().optional(),
      clicked: z.number().optional(),
      failed: z.number().optional(),
    }).passthrough().optional(),
    conversion: z.object({
      converted: z.number().optional(),
      conversionRate: z.number().nullable().optional(),
      incrementalLift: z.number().nullable().optional(),
    }).passthrough().optional(),
    revenue: z.object({
      currency: z.string().optional(),
      grossRevenue: z.number().optional(),
      incrementalRevenue: z.number().nullable().optional(),
      offerCost: z.number().optional(),
      grossMargin: z.number().nullable().optional(),
    }).passthrough().optional(),
  }).passthrough(),
  z.array(z.unknown()),
]).describe("统计快照，只填写已计算的实际结果，不作为目标值");

const campaignMutationBaseSchema = z.object({
  name: z.string().describe("活动名称"),
  description: z.string().optional().describe("活动描述"),
  type: z.string().describe("活动类型，如 reactivation, retention, acquisition, promotion"),
  status: z.enum(["draft", "scheduled", "running", "stopped", "completed"]).optional().describe("活动状态"),
  goal: z.string().describe("活动目标"),
  startTime: z.string().describe("开始时间，格式 yyyy-MM-dd HH:mm:ss"),
  endTime: z.string().describe("结束时间，格式 yyyy-MM-dd HH:mm:ss，必须晚于 startTime"),
  mainSegmentDataId: z.number().optional().describe("主人群包，对应 segment_data id"),
  segmentationStrategy: segmentationStrategySchema.optional(),
  controlGroupStrategy: controlGroupStrategySchema.optional(),
  contentChannelStrategy: contentChannelStrategySchema.optional(),
  offerStrategy: offerStrategySchema.optional(),
  waveStrategy: waveStrategySchema.optional(),
  abTestStrategy: abTestStrategySchema.optional(),
  statistics: statisticsSchema.optional(),
});

function requireMatchingSegmentDataId(
  value: z.infer<typeof campaignMutationBaseSchema>,
  context: z.RefinementCtx,
): void {
  const strategy = value.segmentationStrategy;
  if (!strategy || Array.isArray(strategy)) return;
  const sourceId = strategy.source?.segmentDataId;
  if (value.mainSegmentDataId != null && sourceId != null && value.mainSegmentDataId !== sourceId) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["segmentationStrategy", "source", "segmentDataId"],
      message: "must match mainSegmentDataId",
    });
  }
}

const campaignMutationSchema = campaignMutationBaseSchema.superRefine(requireMatchingSegmentDataId);
const campaignUpdateSchema = campaignMutationBaseSchema.extend({
  id: z.number().describe("营销活动 ID"),
}).superRefine(requireMatchingSegmentDataId);

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
      "创建营销活动。reactivation 策略应使用推荐 v1 结构，并保持 subSegmentKey、channelKey、offerCode、waveId、variantId 引用一致。" +
      "mainSegmentDataId、source.segmentDataId、segmentDefinitionId 和 runId 必须来自 Segment 工具结果，不要编造。" +
      "tenant 从 runConfig.tenantId 注入，body 中不要传 tenantId。",
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
      "更新营销活动。请求体字段同创建接口；保留推荐 v1 的稳定 key 引用，并使用 Segment 工具返回的真实 source 标识。",
    schema: campaignUpdateSchema,
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
