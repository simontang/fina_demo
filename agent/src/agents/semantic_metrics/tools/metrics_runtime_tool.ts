import z from "zod";
import { tool } from "langchain";
import { resolveMetricsClientFromSelector } from "./SemanticMetricsV2Client";
import { assertDatasourceSelected, effectiveDatasourceScope, readToolRunConfig, resolveDatasourceId, type SemanticMetricsToolParams } from "./types";

const DESCRIPTION = `RUNTIME QUERY SURFACE — read what is published and answer business questions with numbers. Business agents use ONLY this tool; Builder agents also use it to verify freshly published metrics.

OWNS: list_datasources (returns the single datasource authorized by the current connection key — redacted output, the first step before anything else), read_semantic_catalog (INDEX ONLY: the list of published metrics and tables with short descriptions — NOT the full definitions), read_metric / read_table (the full definition of ONE metric/table on demand), and query_metrics (the semantic metric query: metrics / groupBy / filters / orderBy / limit — the server generates the SQL, so there is no free-form SQL here).

DOES NOT OWN:
- Publishing or correcting semantic tables/metrics → metrics_meta_tool (create_table/update_table/create_metric/update_metric/list_tables/read_table_meta/list_metrics/read_metric_meta).
- Physical exploration of raw tables, grants, or samples → metrics_datasource_tool (query_sql).
- Free-form custom SQL — intentionally not provided; use query against published metrics.

USE WHEN: read_semantic_catalog to see the index of what is published; read_metric to get ONE metric's dimensions/filters before querying it; query_metrics to answer a business question from already-published metrics (Business) or to verify a just-published metric against its acceptance evidence (Builder).

COMMON MISTAKES (do not):
- Do not call meta-tool actions here (create_table / update_table / create_metric / update_metric / list_tables / read_table_meta / list_metrics / read_metric_meta) — they live in metrics_meta_tool and will fail schema validation here.
- Do not look for a custom-SQL action — there is none; explore physical data with metrics_datasource_tool query_sql, and query business numbers with the semantic query action.
- Metrics and dimensions in query_metrics must already be published; querying an unpublished metric fails with "not found in catalog".`;

/**
 * Create the `metrics_runtime_tool` for the Semantic Metrics V2 plugin.
 *
 * The tool dispatches on `action` and delegates to the matching
 * {@link SemanticMetricsV2Client} method for the resolved connection. Each
 * handler returns a pretty-printed JSON string (or an `Error: ...` string).
 * Explicit action inputs win over request-scoped `runConfig.metricsDataSource`
 * defaults.
 *
 * @example
 * ```ts
 * const runtimeTool = createMetricsRuntimeTool({
 *   connectionType: "semantic-metrics",
 *   connections: ["primary"],
 * });
 * const meta = await runtimeTool.invoke(
 *   { action: "read_semantic_catalog", connectionKey: "primary", datasourceId: "15" },
 *   { configurable: { runConfig: { tenantId: "tenant-1" } } },
 * );
 * const rows = await runtimeTool.invoke(
 *   {
 *     action: "query_metrics",
 *     connectionKey: "primary",
 *     query: { datasourceId: "15", metrics: ["hankel_sell_in_nes"], groupBy: ["sales_team"], limit: 10 },
 *   },
 *   { configurable: { runConfig: { tenantId: "tenant-1" } } },
 * );
 * ```
 *
 * @param params - Build-time configuration ({@link SemanticMetricsToolParams}).
 * @returns A LangChain dynamic structured tool named `metrics_runtime_tool`.
 *
 * @remarks
 * - The plugin captures its `connections`/`connectAll` selector, not resolved
 *   credentials; the connection is resolved from the tenant-scoped Connection
 *   Store on every invocation.
 * - The `query_metrics` action carries the full semantic query request as a nested
 *   `query` object (its `datasourceId` may be a string or a number), so no
 *   flat `datasourceId`/`sql` fallback applies to it.
 * - A missing `connectionKey` or missing `datasourceId` surfaces as an
 *   `Error: ...` result string rather than a thrown exception.
 */
export function createMetricsRuntimeTool(params: SemanticMetricsToolParams) {
  const { connectionType, connections, connectAll } = params;

  return tool(
    async (
      input: {
        action: "list_datasources" | "read_semantic_catalog" | "read_metric" | "read_table" | "query_metrics";
        connectionKey?: string;
        datasourceId?: number;
        metricName?: string;
        tableName?: string;
        query?: Record<string, unknown>;
      },
      exeConfig: unknown,
    ): Promise<string> => {
      try {
        const client = await resolveMetricsClientFromSelector(
          { connectionType, connections, connectAll },
          readToolRunConfig(exeConfig) ?? {},
          input.connectionKey,
        );
        const scope = effectiveDatasourceScope(client.getSelectedEntities());

        switch (input.action) {
          case "list_datasources": {
            const result = (await client.listDatasources()) as Array<Record<string, unknown>>;
            const filtered = scope.unrestricted
              ? result
              : result.filter((d) => scope.ids.includes(Number(d.id)));
            // Whitelist non-sensitive fields only — raw entries carry JDBC urls and usernames.
            const safe = filtered.map((d) => ({
              id: d.id,
              name: d.name,
              sourceType: d.sourceType,
              description: d.description,
              statusLabel: d.statusLabel,
            }));
            return JSON.stringify(safe, null, 2);
          }
          case "read_semantic_catalog": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            // Index only: the full meta response carries a complete definition for
            // every metric and table, which explodes the model context. Details are
            // fetched one at a time via read_metric / read_table.
            const result = (await client.getRuntimeMeta(ds)) as { index?: unknown } | null;
            return JSON.stringify({ index: result?.index ?? null }, null, 2);
          }
          case "read_metric": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            if (!input.metricName) {
              throw new Error("metricName is required for the 'read_metric' action");
            }
            return JSON.stringify(await client.getMetric(ds, input.metricName), null, 2);
          }
          case "read_table": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            if (!input.tableName) {
              throw new Error("tableName is required for the 'read_table' action");
            }
            return JSON.stringify(await client.getTable(ds, input.tableName), null, 2);
          }
          case "query_metrics": {
            if (!input.query) throw new Error("query is required for the 'query_metrics' action");
            const bodyDatasourceId = (input.query as { datasourceId?: unknown }).datasourceId;
            if (bodyDatasourceId !== undefined) {
              assertDatasourceSelected(scope, Number(bodyDatasourceId));
            }
            return JSON.stringify(await client.queryMetrics(input.query), null, 2);
          }
        }
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
    {
      name: "metrics_runtime_tool",
      description: DESCRIPTION,
      schema: z.object({
        action: z.enum(["list_datasources", "read_semantic_catalog", "read_metric", "read_table", "query_metrics"]).describe(
          "Actions ONLY in this runtime tool: list_datasources (return the datasource authorized by this connection key — ALWAYS call this first to obtain the datasourceId; output is redacted to non-sensitive fields), read_semantic_catalog (INDEX ONLY: the list of published metrics and tables with short descriptions — call this before querying), read_metric (full definition of ONE metric by metricName — use after the catalog to get its dimensions/filters), read_table (full definition of ONE table by tableName), query_metrics (run a semantic metric query — the only way to get business numbers). Do NOT put meta-tool actions (list_tables/read_table_meta/create_table/update_table/list_metrics/read_metric_meta/create_metric/update_metric) here",
        ),
        connectionKey: z.string().optional().describe("Connection key. Omit when only one semantic-metrics connection exists"),
        datasourceId: z.coerce.number().optional().describe("Numeric datasource id, e.g. 15 (required for read_semantic_catalog; optional if set in runConfig.metricsDataSource)"),
        metricName: z.string().optional().describe("Metric name from read_semantic_catalog, required for the read_metric action"),
        tableName: z.string().optional().describe("Table name from read_semantic_catalog, required for the read_table action"),
        query: z.record(z.unknown()).optional().describe("Semantic metric query request. Required for the query_metrics action, e.g. { datasourceId: 15, metrics: [\"sell_in_nes\"], groupBy: [\"sales_team\"], filters: [{ dimension: \"posting_year\", operator: \"GTE\", values: [2024] }], limit: 10 }"),
      }),
    },
  );
}
