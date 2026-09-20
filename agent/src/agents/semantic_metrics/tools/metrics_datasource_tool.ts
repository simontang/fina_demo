import z from "zod";
import { tool } from "langchain";
import { resolveMetricsClientFromSelector } from "./SemanticMetricsV2Client";
import { assertDatasourceSelected, effectiveDatasourceScope, readToolRunConfig, resolveDatasourceId, type SemanticMetricsToolParams } from "./types";

const DESCRIPTION = `PHYSICAL LAYER EXPLORATION — the only tool for probing raw datasource structure (Builder/Admin only; business agents do not need it).

OWNS: reading the datasource authorized by the current connection key, reading table-grants / visible scope, and running read-only SQL (SELECT/WITH only) to inspect visible physical tables, columns, sample rows, and value distributions. Also: per-datasource connection test and pool status.

DOES NOT OWN:
- Published semantic assets (table meta / metric meta) → use metrics_meta_tool.
- Runtime meta and business metric queries → use metrics_runtime_tool.

USE WHEN: starting a modeling task — \`list_datasources\` is always the FIRST step: it returns the single datasource authorized by this connection key and the datasource id that every other action and tool requires (get_grants, query_sql here; read_semantic_catalog, query_metrics in metrics_runtime_tool; the meta actions in metrics_meta_tool). Read grants next, then probe tables, columns, samples, and distributions within the grant. If a field's business meaning is unclear, probe sample values and distributions here before publishing anything.

COMMON MISTAKES (do not):
- Do not answer business metric questions here (e.g. "monthly sales by region") — that is metrics_runtime_tool \`query\` against published metrics, NOT this tool's \`query_sql\` (physical-table SQL).
- Do not read or publish semantic meta through this tool — metrics_meta_tool owns the semantic layer.
- Do not probe tables outside the tenant's table-grants; un-granted probes are rejected and break the exploration boundary.
- Do not attempt writes — SQL is validated read-only and rejected before sending.`;

/**
 * Create the `metrics_datasource_tool` for the Semantic Metrics V2 plugin.
 *
 * The tool dispatches on `action` and delegates to the matching
 * {@link SemanticMetricsV2Client} method for the resolved connection. Each
 * handler returns a pretty-printed JSON string (or an `Error: ...` string).
 * Explicit action inputs win over request-scoped `runConfig.metricsDataSource`
 * defaults.
 *
 * @example
 * ```ts
 * const dsTool = createMetricsDatasourceTool({
 *   connectionType: "semantic-metrics",
 *   connections: ["primary"],
 * });
 * const out = await dsTool.invoke(
 *   { action: "get_grants", connectionKey: "primary", datasourceId: "15" },
 *   { configurable: { runConfig: { tenantId: "tenant-1" } } },
 * );
 * ```
 *
 * @param params - Build-time configuration ({@link SemanticMetricsToolParams}).
 * @returns A LangChain dynamic structured tool named `metrics_datasource_tool`.
 *
 * @remarks
 * - The plugin captures its `connections`/`connectAll` selector, not resolved
 *   credentials; the connection is resolved from the tenant-scoped Connection
 *   Store on every invocation.
 * - A missing `connectionKey` or missing `datasourceId` surfaces as an
 *   `Error: ...` result string rather than a thrown exception.
 */
export function createMetricsDatasourceTool(params: SemanticMetricsToolParams) {
  const { connectionType, connections, connectAll } = params;

  return tool(
    async (
      input: {
        action: "list_datasources" | "get_grants" | "query_sql" | "test_connection" | "pool_status";
        connectionKey?: string;
        datasourceId?: number;
        sql?: string;
        params?: Record<string, string | number | boolean>;
        maxRows?: number;
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
          case "get_grants": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            const result = await client.getTableGrants(ds);
            return JSON.stringify(result, null, 2);
          }
          case "query_sql": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            if (!input.sql) throw new Error("sql is required for the 'query_sql' action");
            const result = await client.queryDatasource(ds, {
              sql: input.sql,
              params: input.params,
              maxRows: input.maxRows,
            });
            return JSON.stringify(result, null, 2);
          }
          case "test_connection": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            const result = await client.testDatasource(ds);
            return JSON.stringify(result, null, 2);
          }
          case "pool_status": {
            const ds = resolveDatasourceId(input, exeConfig);
            assertDatasourceSelected(scope, ds);
            const result = await client.getPoolStatus(ds);
            return JSON.stringify(result, null, 2);
          }
        }
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
    {
      name: "metrics_datasource_tool",
      description: DESCRIPTION,
      schema: z.object({
        action: z.enum(["list_datasources", "get_grants", "query_sql", "test_connection", "pool_status"]).describe(
          "list_datasources: return the datasource authorized by this connection key — ALWAYS call this first to obtain the datasource id used by every other action and tool; get_grants: read table-grants; query_sql: run read-only SQL against PHYSICAL tables (note: this is query_sql, NOT the semantic 'query' action of metrics_runtime_tool); test_connection: test the datasource; pool_status: connection-pool status",
        ),
        connectionKey: z.string().optional().describe("Connection key. Omit when only one semantic-metrics connection exists"),
        datasourceId: z.coerce.number().optional().describe("Numeric datasource id, e.g. 15 (required for all actions except list_datasources; optional if set in runConfig.metricsDataSource)"),
        sql: z.string().optional().describe("Read-only SQL (SELECT/WITH only). Required for the query action. Metadata templates — PostgreSQL/SQL Server columns: select column_name, data_type from information_schema.columns where table_schema = :schemaName and table_name = :tableName order by ordinal_position; HANA columns: select column_name, data_type_name from table_columns where schema_name = :schemaName and table_name = :tableName order by position; distributions: select <dim>, count(*) as row_count from <table> group by <dim> order by row_count desc limit :maxRows"),
        params: z.record(z.union([z.string(), z.number(), z.boolean()])).optional().describe("Named :param values for the SQL"),
        maxRows: z.number().optional().describe("Max rows to return"),
      }),
    },
  );
}
