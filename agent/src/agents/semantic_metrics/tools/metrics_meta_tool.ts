import z from "zod";
import { tool } from "langchain";
import { resolveMetricsClientFromSelector } from "./SemanticMetricsV2Client";
import { assertDatasourceSelected, effectiveDatasourceScope, readToolRunConfig, resolveDatasourceId, type SemanticMetricsToolParams } from "./types";

const DESCRIPTION = `SEMANTIC ASSET REGISTRY (Builder/Admin only) — the only tool for reading and publishing the runtime's semantic layer: table meta (meta/tables) and metric meta (meta/metrics).

OWNS: listing published tables and metrics, reading one by objectKey, publishing a semantic table (create_table: columns tagged with role dimension|measure), publishing a metric (create_metric: a metric_index payload then a metric_detail payload using the SQL-free calculation DSL), and correcting published assets (update_table / update_metric).

DOES NOT OWN:
- Running queries or reading the aggregated runtime view → metrics_runtime_tool (get_meta / query / custom_sql).
- Probing physical tables, columns, grants, or samples → metrics_datasource_tool.

USE WHEN: physical exploration is done and the model has been proposed and confirmed by the user — to publish the semantic table and metrics, or to correct published assets afterwards.

COMMON MISTAKES (do not):
- Do not call get_meta here — that action belongs to metrics_runtime_tool, not this tool.
- Do not try to "query numbers" through this tool — list/read actions return definitions only; use metrics_runtime_tool query for data.
- Do not publish a metric whose sourceTable has no published table meta, or whose dimensions are not published columns with role dimension — the server will reject it.
- The builder assistant must load the semantic-metrics-modeling skill and pass modelingSkillLoaded: true on write actions; no removal actions are exposed on purpose.`;

const MODELING_SKILL_NAME = "semantic-metrics-modeling";
const BUILDER_ASSISTANT_ID = "semantic-metrics-builder";

/**
 * Harness guard: a builder assistant must declare that it loaded the modeling
 * skill before publishing meta. Returns an error string to short-circuit, or
 * undefined to allow the write action. Non-builder assistants are not gated.
 */
function requireModelingSkillLoaded(
  action: string,
  input: { modelingSkillLoaded?: true },
  exeConfig: unknown,
): string | undefined {
  const isWriteAction = action === "create_table" || action === "update_table"
    || action === "create_metric" || action === "update_metric";
  if (!isWriteAction) return undefined;
  const runConfig = readToolRunConfig(exeConfig);
  const assistantId = (runConfig as { assistant_id?: string } | undefined)?.assistant_id;
  if (assistantId === BUILDER_ASSISTANT_ID && input.modelingSkillLoaded !== true) {
    return JSON.stringify({
      success: false,
      error: `Load the '${MODELING_SKILL_NAME}' skill before publishing meta.`,
    });
  }
  return undefined;
}

/** Whether an action requires a payload argument. */
function actionRequiresPayload(action: string): boolean {
  return action === "create_table" || action === "update_table"
    || action === "create_metric" || action === "update_metric";
}

/**
 * Create the `metrics_meta_tool` for the Semantic Metrics V2 plugin.
 *
 * The tool dispatches on `action` and delegates to the matching
 * {@link SemanticMetricsV2Client} method for the resolved connection. Each
 * handler returns a pretty-printed JSON string (or an `Error: ...` string).
 * Only read and publish actions are exposed: list/get/create/update for
 * semantic tables and metrics. Delete actions are intentionally not part of the
 * input schema. Explicit action inputs win over request-scoped
 * `runConfig.metricsDataSource` defaults.
 *
 * @example
 * ```ts
 * const metaTool = createMetricsMetaTool({
 *   connectionType: "semantic-metrics",
 *   connections: ["primary"],
 * });
 * const out = await metaTool.invoke(
 *   {
 *     action: "create_table",
 *     connectionKey: "primary",
 *     datasourceId: "15",
 *     payload: { objectType: "table_view_detail", objectKey: "hankel_distr_sell_in", status: 1, payload: {} },
 *   },
 *   { configurable: { runConfig: { tenantId: "tenant-1" } } },
 * );
 * ```
 *
 * @param params - Build-time configuration ({@link SemanticMetricsToolParams}).
 * @returns A LangChain dynamic structured tool named `metrics_meta_tool`.
 *
 * @remarks
 * - The plugin captures its `connections`/`connectAll` selector, not resolved
 *   credentials; the connection is resolved from the tenant-scoped Connection
 *   Store on every invocation.
 * - A missing `connectionKey` or missing `datasourceId` surfaces as an
 *   `Error: ...` result string rather than a thrown exception.
 * - `delete_table` / `delete_metric` inputs fail zod schema validation and are
 *   rejected before the action handler runs.
 */
export function createMetricsMetaTool(params: SemanticMetricsToolParams) {
  const { connectionType, connections, connectAll } = params;

  return tool(
    async (
      input: {
        action:
          | "list_tables" | "read_table_meta" | "create_table" | "update_table"
          | "list_metrics" | "read_metric_meta" | "create_metric" | "update_metric";
        connectionKey?: string;
        datasourceId?: number;
        objectKey?: string;
        payload?: Record<string, unknown>;
        modelingSkillLoaded?: true;
      },
      exeConfig: unknown,
    ): Promise<string> => {
      try {
        const guardError = requireModelingSkillLoaded(input.action, input, exeConfig);
        if (guardError !== undefined) return guardError;
        const client = await resolveMetricsClientFromSelector(
          { connectionType, connections, connectAll },
          readToolRunConfig(exeConfig) ?? {},
          input.connectionKey,
        );
        const ds = resolveDatasourceId(input, exeConfig);
        assertDatasourceSelected(effectiveDatasourceScope(client.getSelectedEntities()), ds);
        if (actionRequiresPayload(input.action) && input.payload === undefined) {
          throw new Error(`payload is required for ${input.action}`);
        }
        const payload = input.payload ?? {};

        switch (input.action) {
          case "list_tables": {
            return JSON.stringify(await client.listTables(ds), null, 2);
          }
          case "read_table_meta": {
            if (!input.objectKey) throw new Error("objectKey is required for read_table_meta");
            return JSON.stringify(await client.getTable(ds, input.objectKey), null, 2);
          }
          case "create_table": {
            return JSON.stringify(await client.createTable(ds, payload), null, 2);
          }
          case "update_table": {
            if (!input.objectKey) throw new Error("objectKey is required for update_table");
            return JSON.stringify(await client.updateTable(ds, input.objectKey, payload), null, 2);
          }
          case "list_metrics": {
            return JSON.stringify(await client.listMetrics(ds), null, 2);
          }
          case "read_metric_meta": {
            if (!input.objectKey) throw new Error("objectKey is required for read_metric_meta");
            return JSON.stringify(await client.getMetric(ds, input.objectKey), null, 2);
          }
          case "create_metric": {
            return JSON.stringify(await client.createMetric(ds, payload), null, 2);
          }
          case "update_metric": {
            if (!input.objectKey) throw new Error("objectKey is required for update_metric");
            return JSON.stringify(await client.updateMetric(ds, input.objectKey, payload), null, 2);
          }
        }
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
    {
      name: "metrics_meta_tool",
      description: DESCRIPTION,
      schema: z.object({
        action: z.enum([
          "list_tables", "read_table_meta", "create_table", "update_table",
          "list_metrics", "read_metric_meta", "create_metric", "update_metric",
        ]).describe(
          "Actions ONLY in this meta tool (runtime query actions like get_meta/query belong to metrics_runtime_tool): list_tables/list_metrics (browse published assets), read_table_meta/read_metric_meta (details by objectKey), create_table/create_metric (publish), update_table/update_metric (correct)",
        ),
        connectionKey: z.string().optional().describe("Connection key. Omit when only one semantic-metrics connection exists"),
        datasourceId: z.coerce.number().optional().describe("Numeric datasource id, e.g. 15 (optional if set in runConfig.metricsDataSource)"),
        objectKey: z.string().optional().describe("Semantic table/metric key (required for read_table_meta/update_table/read_metric_meta/update_metric)"),
        payload: z.record(z.unknown()).optional().describe("Meta payload (required for create/update actions; objectType/objectKey/status/payload/accessGrant)"),
        modelingSkillLoaded: z.literal(true).optional().describe("Set true after loading the semantic-metrics-modeling skill (required for the builder assistant on write actions)"),
      }),
    },
  );
}
