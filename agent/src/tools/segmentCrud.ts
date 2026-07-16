import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";
import { cdpFetch, getTenantIdFromExecutionConfig, getThreadIdFromExecutionConfig } from "./cdpToolClient";

export { getTenantIdFromExecutionConfig } from "./cdpToolClient";

// ============================================================
// segment_definition_list
// ============================================================

registerToolLattice(
  "segment_definition_list",
  {
    name: "segment_definition_list",
    description:
      "查询所有已注册的分群定义列表。当你需要了解当前有哪些分群、查找分群定义 ID 或 " +
      "在创建新分群前检查是否已存在相似分群时使用。返回分群的 id, name, description, datasourceId, " +
      "querySql, status(1=启用/0=禁用), createdAt, updatedAt。",
    schema: z.object({}),
  },
  async (_input, exeConfig) => {
    const result = await cdpFetch(
      "/api/v1/segment-definitions",
      undefined,
      getTenantIdFromExecutionConfig(exeConfig),
    );
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_definition_create
// ============================================================

registerToolLattice(
  "segment_definition_create",
  {
    name: "segment_definition_create",
    description:
      "创建新的分群定义。你需要提供分群名称(name)、关联数据源 ID(datasourceId)和分群 SQL(querySql)。" +
      "datasourceId 可以通过 util-segment-ops assistant 获取可用的数据源列表。" +
      "SQL 必须以 SELECT 或 WITH 开头，禁止包含 INSERT/DELETE/DROP 等写操作关键字。" +
      "可选填写 description 和 status(1=启用/0=禁用，默认 1)。",
    schema: z.object({
      name: z.string().describe("分群名称，如 '高价值客户'"),
      description: z.string().optional().describe("分群描述"),
      datasourceId: z.number().describe("关联数据源 ID，对应 t_datasource_config.id"),
      querySql: z.string().describe("分群 SQL，必须以 SELECT 或 WITH 开头"),
      status: z.number().default(1).describe("状态：1=启用，0=禁用"),
    }),
  },
  async (input, exeConfig) => {
    const threadId = getThreadIdFromExecutionConfig(exeConfig);
    const body: Record<string, unknown> = { ...input };
    if (threadId) body.threadId = threadId;
    const result = await cdpFetch("/api/v1/segment-definitions", {
      method: "POST",
      body: JSON.stringify(body),
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_definition_update
// ============================================================

registerToolLattice(
  "segment_definition_update",
  {
    name: "segment_definition_update",
    description:
      "更新已有分群定义。提供分群 ID 和需要修改的字段。未提供的字段保持不变。" +
      "可更新字段：name, description, datasourceId, querySql, status。" +
      "SQL 必须以 SELECT 或 WITH 开头，禁止写操作关键字。",
    schema: z.object({
      id: z.number().describe("要更新的分群定义 ID"),
      name: z.string().optional().describe("新的分群名称"),
      description: z.string().optional().describe("新的分群描述"),
      datasourceId: z.number().optional().describe("新的数据源 ID"),
      querySql: z.string().optional().describe("新的分群 SQL"),
      status: z.number().optional().describe("状态：1=启用，0=禁用"),
    }),
  },
  async (input, exeConfig) => {
    const { id, ...body } = input;
    const requestBody: Record<string, unknown> = { ...body };
    const threadId = getThreadIdFromExecutionConfig(exeConfig);
    if (threadId) requestBody.threadId = threadId;
    const result = await cdpFetch(`/api/v1/segment-definitions/${id}`, {
      method: "PUT",
      body: JSON.stringify(requestBody),
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_definition_delete
// ============================================================

registerToolLattice(
  "segment_definition_delete",
  {
    name: "segment_definition_delete",
    description:
      "删除分群定义（逻辑删除）。关联的分群数据不会被级联删除。",
    schema: z.object({
      id: z.number().describe("要删除的分群定义 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(`/api/v1/segment-definitions/${input.id}`, {
      method: "DELETE",
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_definition_process
// ============================================================

registerToolLattice(
  "segment_definition_process",
  {
    name: "segment_definition_process",
    description:
      "执行分群处理：连接该分群关联的 CDP 数据源，执行 querySql，将结果集保存到 t_segment_data 表。" +
      "这是 CDP 服务最核心的操作。执行后可通过 segment_data_list 查看结果。" +
      "可选传递 params（SQL 命名参数，用于替换 SQL 中 :paramName 占位符）。",
    schema: z.object({
      id: z.number().describe("要执行处理的分群定义 ID"),
      params: z.record(z.unknown()).optional().describe("SQL 命名参数，如 { min_amount: 5000 }"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(`/api/v1/segment-definitions/${input.id}/process`, {
      method: "POST",
      body: input.params ? JSON.stringify({ params: input.params }) : "{}",
    }, getTenantIdFromExecutionConfig(exeConfig));
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_data_get
// ============================================================

registerToolLattice(
  "segment_data_get",
  {
    name: "segment_data_get",
    description:
      "按 segment_data ID 查询单个已物化分群快照。用于校验用户指定的 Artifact 是否仍存在，" +
      "并获取 definitionId、runId、rowCount 和创建时间；不要用最新分页结果替代指定快照。",
    schema: z.object({
      id: z.number().describe("segment_data 快照 ID"),
    }),
  },
  async (input, exeConfig) => {
    const result = await cdpFetch(
      `/api/v1/segment-data/${input.id}`,
      undefined,
      getTenantIdFromExecutionConfig(exeConfig),
    );
    return JSON.stringify(result);
  }
);

// ============================================================
// segment_data_list
// ============================================================

registerToolLattice(
  "segment_data_list",
  {
    name: "segment_data_list",
    description:
      "查询分群数据（分页）。可按 definitionId 过滤查看某个分群定义的执行结果。" +
      "返回 items（分群数据列表）、total、page、pageSize。" +
      "每条数据包含 id, definitionId, runId, dataJson(分群结果JSON), rowCount, createdAt。",
    schema: z.object({
      definitionId: z.number().optional().describe("按分群定义 ID 过滤"),
      page: z.number().default(1).describe("页码，默认 1"),
      pageSize: z.number().default(20).describe("每页条数，默认 20"),
    }),
  },
  async (input, exeConfig) => {
    const params = new URLSearchParams();
    if (input.definitionId) params.set("definitionId", String(input.definitionId));
    params.set("page", String(input.page));
    params.set("pageSize", String(input.pageSize));
    const qs = params.toString();
    const result = await cdpFetch(
      `/api/v1/segment-data?${qs}`,
      undefined,
      getTenantIdFromExecutionConfig(exeConfig),
    );
    return JSON.stringify(result);
  }
);
