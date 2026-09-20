/**
 * Semantic Metrics V2 shared types and SQL validation.
 */

/** Connection config read from the tenant-scoped Connection Store. */
export interface SemanticMetricsV2Config {
  serverUrl: string;
  apiKey?: string;
  headers?: Record<string, string>;
  /**
   * Resource ids this connection exposes to agents; empty/absent means all.
   * Read from `selectedEntities` (the standard connection resource-selection
   * field written by the connection UI) by
   * {@link SemanticMetricsV2Client.getSelectedEntities}.
   */
  selectedEntities?: Array<number | string>;
}

/**
 * Request-scoped execution-config `runConfig` object read by semantic-metrics tools.
 *
 * Carries run identity and an optional preset datasource selection used when a
 * tool input omits `datasourceId`. Connections are resolved from the plugin's
 * own `connections` / `connectAll` config at invocation time (see
 * {@link resolveMetricsClientFromSelector}), not injected here.
 */
export interface SemanticMetricsRunConfig {
  /** Optional preset datasource selection used when a tool input omits `datasourceId`. */
  metricsDataSource?: { datasourceId?: string };
  /** Tenant id used to resolve this plugin's connections from the Connection Store. */
  tenantId?: string;
}

/**
 * Read the execution-config runConfig object for a tool invocation.
 *
 * @param exeConfig - LangChain execution config (may carry `configurable.runConfig`).
 * @returns The runConfig object, or `undefined` when absent.
 */
export function readToolRunConfig(exeConfig: unknown): SemanticMetricsRunConfig | undefined {
  return (exeConfig as { configurable?: { runConfig?: SemanticMetricsRunConfig } })
    ?.configurable?.runConfig;
}

/**
 * Shared params for semantic-metrics tool factories.
 *
 * The plugin captures its connection selector (not resolved credentials) at
 * build time and resolves against the tenant-scoped Connection Store on every
 * invocation, so config-layer changes apply to an already-compiled agent.
 */
export interface SemanticMetricsToolParams {
  /** Plugin type used as `ConnectionEntry.type` when resolving connections. */
  connectionType: string;
  /** Explicit connection keys from the middleware config. */
  connections?: string[];
  /** When true, resolve every connection of `connectionType` for the tenant. */
  connectAll?: boolean;
}

/**
 * Effective datasource authorization for a resolved connection.
 *
 * Datasource scope is owned by the connection itself (`selectedEntities`); the
 * agent config only selects connections, so there is no agent-level narrowing.
 */
export interface DatasourceScope {
  /**
   * True when the connection selects no specific resources and every datasource
   * is allowed. When false, `ids` is authoritative.
   */
  unrestricted: boolean;
  /** Allowed numeric datasource ids; only meaningful when `unrestricted` is false. */
  ids: number[];
}

/**
 * Resolve the effective datasource id from an explicit tool input, falling back
 * to `runConfig.metricsDataSource.datasourceId`.
 *
 * @param input - Tool input carrying an optional explicit datasource id (string is coerced).
 * @param exeConfig - LangChain execution config whose runConfig may preset a datasource id.
 * @returns The effective numeric datasource id.
 * @throws When neither an explicit id nor a runConfig preset is available.
 */
export function resolveDatasourceId(
  input: { datasourceId?: string | number },  exeConfig: unknown,
): number {
  const id = input.datasourceId ?? readToolRunConfig(exeConfig)?.metricsDataSource?.datasourceId;
  if (id === undefined || id === null || id === "") {
    throw new Error("datasourceId is required (provide it or set runConfig.metricsDataSource.datasourceId)");
  }
  return Number(id);
}

/**
 * Enforce the effective datasource scope on a resolved datasource id.
 *
 * @param scope - Effective scope from {@link effectiveDatasourceScope}.
 * @param datasourceId - The datasource id a tool action is about to operate on.
 * @throws DATASOURCE_NOT_SELECTED when the scope is narrowed and the id is outside it.
 */
export function assertDatasourceSelected(scope: DatasourceScope, datasourceId: number): void {
  if (scope.unrestricted) return;
  if (!scope.ids.includes(datasourceId)) {
    throw new Error(
      `DATASOURCE_NOT_SELECTED: datasource ${datasourceId} is not in the effective scope [${scope.ids.join(", ")}]`,
    );
  }
}

/**
 * Compute the effective datasource scope from the connection's own resource
 * selection (`selectedEntities`). Agent config selects connections only, so
 * this is the single constraint layer.
 *
 * @param connectionScope - Connection-layer selected resource ids (empty means the connection exposes all).
 * @returns The effective {@link DatasourceScope}.
 */
export function effectiveDatasourceScope(connectionScope: number[]): DatasourceScope {
  return connectionScope.length > 0
    ? { unrestricted: false, ids: [...connectionScope] }
    : { unrestricted: true, ids: [] };
}

/** SQL validation outcome for read-only datasource / custom SQL queries. */
export interface SqlValidationResult {
  ok: boolean;
  reason?: string;
}

const DDL_DML_KEYWORDS = [
  "insert", "update", "delete", "drop", "alter", "create",
  "truncate", "grant", "revoke", "merge", "replace", "upsert",
];

// Matches any DML/DDL keyword as a whole word, case-insensitively.
const DDL_DML_RE = new RegExp(`\\b(?:${DDL_DML_KEYWORDS.join("|")})\\b`, "i");

// SELECT INTO creates a table — not read-only, and not covered by the keyword list.
const SELECT_INTO_RE = /\bselect\b[\s\S]*?\binto\b/i;

/**
 * Validate that a SQL string is a single read-only statement (SELECT or WITH).
 *
 * This is a fail-closed heuristic gate that runs before a query is sent to a
 * Metrics Server datasource. It is deliberately conservative: anything it
 * cannot positively classify as a single read-only statement is rejected.
 *
 * @param sql - Raw SQL string.
 * @returns `{ ok: true }` when allowed, otherwise `{ ok: false, reason }`.
 *
 * @remarks
 * Known heuristic limitations (this is not a full SQL parser):
 * - String literals are not parsed, so a quoted literal that merely contains a
 *   keyword (e.g. `select 'delete' as x`) is conservatively rejected.
 * - A `;` inside a string literal likewise causes a conservative rejection as
 *   multiple statements.
 * - The `SELECT INTO` guard also conservatively rejects a quoted literal or
 *   `LIKE '%into%'` that merely contains the word `into` (e.g. `select 'into' as v`,
 *   `select * from t where msg like '%into%'`).
 * - The Metrics Server remains the final authority on whether a query runs.
 */
export function validateSql(sql: string): SqlValidationResult {
  const trimmed = sql.trim();
  if (!trimmed) {
    return { ok: false, reason: "SQL is empty" };
  }

  const withoutComments = trimmed
    .replace(/--[^\n]*/g, " ")
    .replace(/\/\*[\s\S]*?\*\//g, " ");

  const statements = withoutComments
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean);
  if (statements.length > 1) {
    return { ok: false, reason: "Multiple statements are not allowed" };
  }

  const firstKeyword = withoutComments.trim().split(/\s+/)[0].toLowerCase();
  if (firstKeyword !== "select" && firstKeyword !== "with") {
    return { ok: false, reason: `Only SELECT or WITH queries are allowed (got "${firstKeyword}")` };
  }

  if (SELECT_INTO_RE.test(withoutComments)) {
    return { ok: false, reason: "SELECT INTO is not allowed (creates a table)" };
  }

  if (DDL_DML_RE.test(withoutComments)) {
    return { ok: false, reason: "Only read-only queries are allowed (DML/DDL detected)" };
  }

  return { ok: true };
}
