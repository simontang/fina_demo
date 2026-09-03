import { registerToolLattice } from "@axiom-lattice/core";
import { z } from "zod";
import {
  JsonValue,
  assertDatasourceAllowed,
  metricsFetch,
  resolveAllowedDatasourceId,
  resolveSemanticMetricsServer,
  serializeMetricParameters,
  validateReadOnlySql,
} from "./metricsToolClient";

const datasourceIdSchema = z.union([z.string(), z.number()]);
const jsonValueSchema: z.ZodType<JsonValue> = z.lazy(() => z.union([
  z.string(),
  z.number(),
  z.boolean(),
  z.null(),
  z.array(jsonValueSchema),
  z.record(jsonValueSchema),
]));
const jsonPayloadSchema = z.union([z.record(jsonValueSchema), z.array(jsonValueSchema)]);

const serverInput = {
  serverKey: z.string().optional().describe("Metrics server key. Optional when the tenant has one semantic metrics server or runConfig.metricsDataSource is set."),
};

const filterSchema = z.object({
  dimension: z.string(),
  operator: z.string(),
  values: z.array(z.unknown()).optional(),
});

const orderBySchema = z.object({
  field: z.string(),
  direction: z.enum(["ASC", "DESC"]).optional(),
});

const tableMetaTypeSchema = z.enum(["table_catalog", "table_view_detail"]);
const metricMetaTypeSchema = z.enum(["metric_index", "metric_detail"]);
const tableAccessGrantSchema = z.object({
  schemaName: z.string().optional(),
  tablePattern: z.string(),
  patternType: z.enum(["PREFIX", "EXACT"]).default("EXACT"),
  caseSensitive: z.boolean().default(false),
  status: z.number().default(1),
});

registerToolLattice(
  "metrics_datasource_list",
  {
    name: "metrics_datasource_list",
    description:
      "列出当前租户可用的 Metrics Server datasource。只返回 lattice_metrics_configs.selectedDataSources 允许的 datasource。",
    schema: z.object(serverInput),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const result = await metricsFetch(server, "/datasources");
    const selected = new Set((server.config.selectedDataSources || []).map(String));
    const dataSources = Array.isArray(result)
      ? result.filter((item) => selected.has(String((item as { id?: unknown }).id)))
      : result;
    return JSON.stringify({ tenantId: server.tenantId, serverKey: server.serverKey, dataSources }, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_get",
  {
    name: "metrics_meta_get",
    description:
      "获取某个 datasource 的完整 Metrics meta，包括 metrics index、metric details、tables details。用于 Agent 一次性发现语义模型。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/meta`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_index",
  {
    name: "metrics_metric_index",
    description:
      "获取某个 datasource 的 metric/table 轻量索引。用于先发现有哪些指标和 table/view 可用。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/metrics/index`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_detail",
  {
    name: "metrics_metric_detail",
    description:
      "读取单个 metric 的完整语义定义，包括 time context、dimensions、AI agent context 和 SQL 注册状态。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      metricName: z.string(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/metrics/${encodeURIComponent(input.metricName)}/detail`,
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_query",
  {
    name: "metrics_metric_query",
    description:
      "执行 semantic metric query 或只读 customSql query。customSql 仅允许 SELECT/WITH，禁止写操作和多语句。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      metrics: z.array(z.string()).optional(),
      groupBy: z.array(z.string()).optional(),
      filters: z.array(filterSchema).optional(),
      orderBy: z.array(orderBySchema).optional(),
      limit: z.number().optional(),
      debug: z.boolean().optional(),
      customSql: z.string().optional(),
      params: z.record(z.unknown()).optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    if (input.customSql) {
      validateReadOnlySql(input.customSql);
    }
    const body = {
      datasourceId,
      metrics: input.metrics,
      groupBy: input.groupBy,
      filters: input.filters,
      orderBy: input.orderBy,
      limit: input.limit,
      debug: input.debug,
      customSql: input.customSql,
      params: input.params,
    };
    const result = await metricsFetch(server, "/metrics/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_table_grant_list",
  {
    name: "metrics_table_grant_list",
    description:
      "列出当前租户在某个 datasource 上的 table grants。租户来自 runConfig.tenantId。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/table-grants`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_table_grant_create",
  {
    name: "metrics_table_grant_create",
    description:
      "为当前租户创建 datasource table grant。patternType 支持 PREFIX/EXACT，body 不传 tenantId。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      schemaName: z.string().optional(),
      tablePattern: z.string(),
      patternType: z.enum(["PREFIX", "EXACT"]).default("PREFIX"),
      caseSensitive: z.boolean().default(false),
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/table-grants`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schemaName: input.schemaName,
        tablePattern: input.tablePattern,
        patternType: input.patternType,
        caseSensitive: input.caseSensitive,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_table_grant_update",
  {
    name: "metrics_table_grant_update",
    description:
      "更新当前租户在某个 datasource 上的 table grant。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      grantId: z.number(),
      schemaName: z.string().optional(),
      tablePattern: z.string(),
      patternType: z.enum(["PREFIX", "EXACT"]).default("PREFIX"),
      caseSensitive: z.boolean().default(false),
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/table-grants/${input.grantId}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          schemaName: input.schemaName,
          tablePattern: input.tablePattern,
          patternType: input.patternType,
          caseSensitive: input.caseSensitive,
          status: input.status,
        }),
      },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_table_grant_delete",
  {
    name: "metrics_table_grant_delete",
    description:
      "删除当前租户在某个 datasource 上的 table grant。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      grantId: z.number(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/table-grants/${input.grantId}`,
      { method: "DELETE" },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_list",
  {
    name: "metrics_datasource_table_list",
    description:
      "列出 datasource 中可被 Builder/Admin 建模探查的真实表/视图。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      schemaName: z.string().optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const params = new URLSearchParams();
    if (input.schemaName) params.set("schemaName", input.schemaName);
    const suffix = params.toString() ? `?${params}` : "";
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/schema/tables${suffix}`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_query",
  {
    name: "metrics_datasource_query",
    description:
      "执行 datasource 建模探查 SQL。仅允许 SELECT/WITH，禁止写操作和多语句；不按 table grants 过滤。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      sql: z.string(),
      params: z.record(z.unknown()).optional(),
      maxRows: z.number().optional(),
      debug: z.boolean().optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    validateReadOnlySql(input.sql);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sql: input.sql,
        params: input.params,
        maxRows: input.maxRows,
        debug: input.debug,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_sql_probe",
  {
    name: "metrics_datasource_sql_probe",
    description:
      "兼容旧工具名：执行 runtime customSql 查询。仅允许 SELECT/WITH；服务端会按已发布 table meta/table grants 校验。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      sql: z.string(),
      params: z.record(z.unknown()).optional(),
      maxRows: z.number().optional(),
      debug: z.boolean().optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    validateReadOnlySql(input.sql);
    const result = await metricsFetch(server, "/metrics/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasourceId,
        customSql: input.sql,
        params: input.params,
        limit: input.maxRows,
        debug: input.debug,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_meta_list",
  {
    name: "metrics_datasource_table_meta_list",
    description:
      "列出 datasource 上已发布给 Agent runtime 使用的 table meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      objectType: tableMetaTypeSchema.optional(),
      objectKey: z.string().optional(),
      page: z.number().default(1),
      pageSize: z.number().default(20),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    if (input.objectKey) params.set("objectKey", input.objectKey);
    params.set("page", String(input.page));
    params.set("pageSize", String(input.pageSize));
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/meta/tables?${params}`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_meta_get",
  {
    name: "metrics_datasource_table_meta_get",
    description:
      "按 tableKey 读取 datasource 上已发布的 table meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      tableKey: z.string(),
      objectType: tableMetaTypeSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    const suffix = params.toString() ? `?${params}` : "";
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(input.tableKey)}${suffix}`,
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_meta_create",
  {
    name: "metrics_datasource_table_meta_create",
    description:
      "创建 datasource table meta，并创建或复用对应 table grant，使该表可供 Agent runtime 使用。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      objectType: tableMetaTypeSchema.default("table_view_detail"),
      objectKey: z.string().optional(),
      payload: jsonPayloadSchema,
      status: z.number().default(1),
      accessGrant: tableAccessGrantSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/meta/tables`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        objectType: input.objectType,
        objectKey: input.objectKey,
        payload: input.payload,
        status: input.status,
        accessGrant: input.accessGrant,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_meta_update",
  {
    name: "metrics_datasource_table_meta_update",
    description:
      "更新 datasource table meta；如传 accessGrant，会同步创建或复用 runtime table grant。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      tableKey: z.string(),
      objectType: tableMetaTypeSchema.default("table_view_detail"),
      payload: jsonPayloadSchema,
      status: z.number().default(1),
      accessGrant: tableAccessGrantSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(input.tableKey)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objectType: input.objectType,
          payload: input.payload,
          status: input.status,
          accessGrant: input.accessGrant,
        }),
      },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_table_meta_delete",
  {
    name: "metrics_datasource_table_meta_delete",
    description:
      "删除 datasource table meta，并移除同名 runtime table grant。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      tableKey: z.string(),
      objectType: tableMetaTypeSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    const suffix = params.toString() ? `?${params}` : "";
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/tables/${encodeURIComponent(input.tableKey)}${suffix}`,
      { method: "DELETE" },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_metric_meta_list",
  {
    name: "metrics_datasource_metric_meta_list",
    description:
      "列出 datasource 上已发布给 Agent runtime 使用的 metric meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      objectType: metricMetaTypeSchema.optional(),
      objectKey: z.string().optional(),
      page: z.number().default(1),
      pageSize: z.number().default(20),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    if (input.objectKey) params.set("objectKey", input.objectKey);
    params.set("page", String(input.page));
    params.set("pageSize", String(input.pageSize));
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics?${params}`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_metric_meta_get",
  {
    name: "metrics_datasource_metric_meta_get",
    description:
      "按 metricKey 读取 datasource 上已发布的 metric meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      metricKey: z.string(),
      objectType: metricMetaTypeSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    const suffix = params.toString() ? `?${params}` : "";
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(input.metricKey)}${suffix}`,
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_metric_meta_create",
  {
    name: "metrics_datasource_metric_meta_create",
    description:
      "创建 datasource metric meta，支持 metric_index 或 metric_detail。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      objectType: metricMetaTypeSchema.default("metric_detail"),
      objectKey: z.string().optional(),
      payload: jsonPayloadSchema,
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        objectType: input.objectType,
        objectKey: input.objectKey,
        payload: input.payload,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_metric_meta_update",
  {
    name: "metrics_datasource_metric_meta_update",
    description:
      "更新 datasource metric meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      metricKey: z.string(),
      objectType: metricMetaTypeSchema.default("metric_detail"),
      payload: jsonPayloadSchema,
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(input.metricKey)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objectType: input.objectType,
          payload: input.payload,
          status: input.status,
        }),
      },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_datasource_metric_meta_delete",
  {
    name: "metrics_datasource_metric_meta_delete",
    description:
      "删除 datasource metric meta。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      metricKey: z.string(),
      objectType: metricMetaTypeSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const params = new URLSearchParams();
    if (input.objectType) params.set("objectType", input.objectType);
    const suffix = params.toString() ? `?${params}` : "";
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/meta/metrics/${encodeURIComponent(input.metricKey)}${suffix}`,
      { method: "DELETE" },
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_list",
  {
    name: "metrics_metric_list",
    description:
      "列出某个 datasource 中已注册在 t_metrics_meta 的 SQL-level metric 定义。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/metrics`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_get",
  {
    name: "metrics_metric_get",
    description:
      "读取某个 datasource 中一个 SQL-level metric 定义。metricCode 通常等于 metricName。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      metricCode: z.string(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const datasourceId = await resolveAllowedDatasourceId(server, exeConfig, input.datasourceId);
    const result = await metricsFetch(
      server,
      `/datasources/${encodeURIComponent(datasourceId)}/metrics/${encodeURIComponent(input.metricCode)}`,
    );
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_create",
  {
    name: "metrics_metric_create",
    description:
      "创建 SQL-level metric 定义，绑定到当前租户允许的 datasource。parameters 可传 JSON array 或 JSON string。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema,
      metricCode: z.string(),
      metricName: z.string(),
      description: z.string().optional(),
      querySql: z.string(),
      parameters: z.union([z.string(), z.array(z.record(z.unknown()))]).optional(),
      valueColumn: z.string().optional(),
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasourceId,
        metricCode: input.metricCode,
        metricName: input.metricName,
        description: input.description,
        querySql: input.querySql,
        parameters: serializeMetricParameters(input.parameters),
        valueColumn: input.valueColumn,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_update",
  {
    name: "metrics_metric_update",
    description:
      "更新 SQL-level metric 定义。为了防止跨 datasource 修改，必须提供 datasourceId，且 id 必须已属于该 datasource。",
    schema: z.object({
      ...serverInput,
      id: z.number(),
      datasourceId: datasourceIdSchema,
      metricCode: z.string(),
      metricName: z.string(),
      description: z.string().optional(),
      querySql: z.string(),
      parameters: z.union([z.string(), z.array(z.record(z.unknown()))]).optional(),
      valueColumn: z.string().optional(),
      status: z.number(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    await assertMetricIdInDatasource(server, datasourceId, input.id);
    const result = await metricsFetch(server, `/metrics/${input.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasourceId,
        metricCode: input.metricCode,
        metricName: input.metricName,
        description: input.description,
        querySql: input.querySql,
        parameters: serializeMetricParameters(input.parameters),
        valueColumn: input.valueColumn,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_metric_delete",
  {
    name: "metrics_metric_delete",
    description:
      "删除 SQL-level metric 定义。为了防止跨 datasource 删除，必须提供 datasourceId，且 id 必须已属于该 datasource。",
    schema: z.object({
      ...serverInput,
      id: z.number(),
      datasourceId: datasourceIdSchema,
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    assertDatasourceAllowed(server.config, input.datasourceId);
    const datasourceId = String(input.datasourceId);
    await assertMetricIdInDatasource(server, datasourceId, input.id);
    const result = await metricsFetch(server, `/metrics/${input.id}`, { method: "DELETE" });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_object_list",
  {
    name: "metrics_meta_object_list",
    description:
      "分页列出 DB-backed metrics meta object，可按 datasourceId、objectType、objectKey 过滤。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional(),
      objectType: z.enum(["catalog_config", "metric_index", "metric_detail", "table_catalog", "table_view_detail"]).optional(),
      objectKey: z.string().optional(),
      page: z.number().default(1),
      pageSize: z.number().default(20),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    if (input.datasourceId !== undefined) {
      assertDatasourceAllowed(server.config, input.datasourceId);
    }
    const params = new URLSearchParams();
    if (input.datasourceId !== undefined) params.set("datasourceId", String(input.datasourceId));
    if (input.objectType) params.set("objectType", input.objectType);
    if (input.objectKey) params.set("objectKey", input.objectKey);
    params.set("page", String(input.page));
    params.set("pageSize", String(input.pageSize));
    const result = await metricsFetch(server, `/meta/objects?${params}`);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_object_get",
  {
    name: "metrics_meta_object_get",
    description: "读取单个 DB-backed metrics meta object。",
    schema: z.object({
      ...serverInput,
      id: z.number(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const result = await metricsFetch(server, `/meta/objects/${input.id}`);
    assertMetaObjectAllowed(server, result);
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_object_create",
  {
    name: "metrics_meta_object_create",
    description:
      "创建 DB-backed metrics meta object。payload 必须沿用现有 JSON meta 结构；datasourceId 为空表示全局 overlay。",
    schema: z.object({
      ...serverInput,
      datasourceId: datasourceIdSchema.optional().nullable(),
      objectType: z.enum(["catalog_config", "metric_index", "metric_detail", "table_catalog", "table_view_detail"]),
      objectKey: z.string(),
      payload: jsonPayloadSchema,
      status: z.number().default(1),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    if (input.datasourceId !== undefined && input.datasourceId !== null) {
      assertDatasourceAllowed(server.config, input.datasourceId);
    }
    const result = await metricsFetch(server, "/meta/objects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasourceId: input.datasourceId,
        objectType: input.objectType,
        objectKey: input.objectKey,
        payload: input.payload,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_object_update",
  {
    name: "metrics_meta_object_update",
    description:
      "更新 DB-backed metrics meta object。若对象或请求包含 datasourceId，必须在当前租户 allow-list 内。",
    schema: z.object({
      ...serverInput,
      id: z.number(),
      datasourceId: datasourceIdSchema.optional().nullable(),
      objectType: z.enum(["catalog_config", "metric_index", "metric_detail", "table_catalog", "table_view_detail"]),
      objectKey: z.string(),
      payload: jsonPayloadSchema,
      status: z.number(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const existing = await metricsFetch(server, `/meta/objects/${input.id}`);
    assertMetaObjectAllowed(server, existing);
    if (input.datasourceId !== undefined && input.datasourceId !== null) {
      assertDatasourceAllowed(server.config, input.datasourceId);
    }
    const result = await metricsFetch(server, `/meta/objects/${input.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        datasourceId: input.datasourceId,
        objectType: input.objectType,
        objectKey: input.objectKey,
        payload: input.payload,
        status: input.status,
      }),
    });
    return JSON.stringify(result, null, 2);
  },
);

registerToolLattice(
  "metrics_meta_object_delete",
  {
    name: "metrics_meta_object_delete",
    description:
      "删除 DB-backed metrics meta object。若对象绑定 datasourceId，必须在当前租户 allow-list 内。",
    schema: z.object({
      ...serverInput,
      id: z.number(),
    }),
  },
  async (input, exeConfig) => {
    const server = await resolveSemanticMetricsServer(exeConfig, input.serverKey);
    const existing = await metricsFetch(server, `/meta/objects/${input.id}`);
    assertMetaObjectAllowed(server, existing);
    const result = await metricsFetch(server, `/meta/objects/${input.id}`, { method: "DELETE" });
    return JSON.stringify(result, null, 2);
  },
);

async function assertMetricIdInDatasource(
  server: Awaited<ReturnType<typeof resolveSemanticMetricsServer>>,
  datasourceId: string,
  id: number,
): Promise<void> {
  const result = await metricsFetch(server, `/datasources/${encodeURIComponent(datasourceId)}/metrics`);
  if (!Array.isArray(result) || !result.some((item) => String((item as { id?: unknown }).id) === String(id))) {
    throw new Error(`metric id "${id}" does not belong to allowed datasource "${datasourceId}"`);
  }
}

function assertMetaObjectAllowed(
  server: Awaited<ReturnType<typeof resolveSemanticMetricsServer>>,
  object: unknown,
): void {
  const datasourceId = (object as { datasourceId?: unknown })?.datasourceId;
  if (datasourceId !== undefined && datasourceId !== null) {
    assertDatasourceAllowed(server.config, String(datasourceId));
  }
}
