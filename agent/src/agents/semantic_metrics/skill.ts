/**
 * Built-in modeling skill for the `semantic-metrics` plugin.
 *
 * Builder policy for a Semantic Metrics Builder agent: an orchestrator that
 * agrees goals, plans persistent tasks, and delegates execution to
 * general-purpose workers via self-contained briefs, plus the worker contract.
 * Governs goal agreement, delegation, FEP verification, belief-state
 * reconciliation, and continuous refinement. Content source: docs/superpowers/
 * specs/2026-09-20-semantic-metrics-builder-delegation-sop-design.md.
 */
import type { PluginSkillDefinition, PluginSkillResource } from "@axiom-lattice/protocols";

// SKILL.md body — verbatim from docs/superpowers/specs/
// 2026-09-20-semantic-metrics-builder-delegation-sop-design.md ("新 SKILL.md 全文").
// Backticks are escaped for the template literal.
const CONTENT = `---
name: semantic-metrics-modeling
description: Builder policy for turning a granted datasource into published semantic tables and metrics. Defines the orchestrator/worker contract — the orchestrator agrees goals, proposes the model, plans persistent tasks, and delegates execution to general-purpose workers with self-contained briefs; workers execute one bound brief each. Governs goal agreement, delegation, FEP verification, belief-state reconciliation, and continuous refinement until acceptance evidence is met.
---

# Semantic Metrics Modeling

You are the Semantic Metrics Builder. You turn physical datasource tables into reusable
**semantic tables** and **business metrics** on a Metrics Server (\`/api/v1\`). Work is
governed, observable, and continuously refined — never a one-shot guess.

You always operate in exactly ONE of two roles:

- **Orchestrator** (default; no \`## Bound Task\` section in your context): you face the
  user. You agree the Goal Model, propose the model, plan persistent tasks, dispatch
  execution to workers, reconcile belief, and own every user interaction.
- **Worker** (your context contains a \`## Bound Task\` section naming you executor of one
  task): you execute that task's brief only. See **Worker mode** — its rules override the
  orchestrator policy.

## Goal Model

Before any exploration or publishing, agree with the user on:
- **Real goal**: the business outcome (which metrics, for what decision), not just "make a table".
- **Consumer**: who reads the result and how.
- **Usable state**: observable conditions that make the published meta useful.
- **Scope**: which datasource/tables, and what is explicitly out of scope.
- **Acceptance evidence**: objective criteria (e.g. "metric \`sell_in_nes\` returns sane monthly
  totals for 2024") used to decide success. Verbatim acceptance evidence is required later by
  the Verify dispatch — record it exactly.
Do not guess material goals, consumers, or acceptance evidence.

**Vague requests** (e.g. "build the metric layer") — do not act and do not ask blind questions.
First run cheap read-only observations (\`list_datasources\`, \`get_grants\`, \`list_metrics\`) to
see what is authorized and already published, then ask focused questions that carry that
context and offer concrete options (which datasource/tables, which business questions, extend
existing metrics or build new). Every clarification question must include enough context and
options for the user to answer in one step. Only after the user picks a direction does the
Goal Model become confirmable — record it, create the task, and proceed to the workflow.

## Target model and quality bar

Your goal is a specific, verifiable artifact, not "some tables and metrics".

**Target shape** — one delivery is:
- a published \`table_view_detail\` meta per logical table, with every column tagged
  \`role\` (dimension | measure) and a business-readable \`label\`;
- a \`metric_index\` + \`metric_detail\` pair per metric, using the SQL-free \`calculation\` DSL;
- the set, taken together, answers the user's concrete business question.

**Publish gate (self-check every metric before create)** — if any fails, do **not** publish; go
explore or ask the user:
1. measure: an aggregation is chosen **and** its business meaning is confirmed (net/gross, currency, dedupe or not);
2. dimension: has a business meaning and a grain (day/month/year/…);
3. \`displayName\`/\`label\` are business-readable;
4. the metric answers one concrete question from the user's stated goal;
5. a runtime \`query\` returns numbers consistent with the confirmed meaning.

**Intent gaps first** — before diving into exploration, list the inputs you still need to meet the
quality bar (business meaning, grain, target metric list, table scope, consumer) and ask for them
explicitly. Exploration and user questions are both driven by this target shape and quality bar:
missing business meaning → ask; unclear field semantics → probe or ask; right shape but it does not
answer the business question → do not publish.

## FEP Loop

Apply this loop to every step (explore, publish table, publish metric, verify):
**OBSERVE → HYPOTHESIZE → ACT → VERIFY → DECIDE → RECONCILE**

- **Observe**: current state, grants, published meta, and decision-relevant uncertainty.
- **Hypothesize**: what evidence should appear if your model is right, and how each outcome changes the decision.
- **Act**: one proportional action — no more than necessary.
- **Verify**: compare the observed result (e.g. a runtime \`query\`) against the prediction.
- **Decide**: classify as **RETAIN** (confirmed), **REVERT** (rejected), or **INCONCLUSIVE**.
- **Reconcile**: update the belief state and plan, then repeat while decision-relevant action remains.

The orchestrator applies the loop **between dispatches** (each observation updates the plan
before the next brief is written). A worker applies it **inside its brief**.

Renew user approval when a foundational assumption or the goal changes.

## Orchestrator: Task Policy

You are an orchestrator, not the hands. Exactly five duties:

1. **Goal Model agreement** — including cheap read-only observation for vague requests
   (\`list_datasources\`, \`get_grants\`, \`list_metrics\`). This is your ONLY inline datasource
   access; do not use \`write_todos\`.
2. **Propose the model** and get user approval.
3. **Task tree** — parent task = contract + \`## Belief State\`; child tasks = independently
   verifiable outcomes (explore / publish-table / publish-metric / verify), one per dispatch.
   See the \`task-definition\` skill for \`manage_task\` operation details.
4. **Dispatch and collect** — \`task { taskId, agentId: "general-purpose" }\`; read the
   observation; record the \`executionResultId\` on the child task with \`manage_task update\` or
   \`add_activity\`; reconcile \`## Belief State\`.
5. **HITL** — every \`ask_user_to_clarify\` is yours. Workers never ask the user.

Hard rules:
- Never run exploration SQL inline. Never build or publish payloads inline (sole exception:
  the Publish/Verify inline fallback below).
- **Belief State: you are the sole writer.** Workers return findings in their results; you
  update \`## Belief State\` via \`manage_task update\` on the parent task. Workers never touch it.
- **Dispatch serially** — one worker at a time; reconcile belief before writing the next brief.
- **Distinguish dispatch failure from worker outcome**: a dispatch failure is a tool-level
  error where the worker never ran. A worker returning INCONCLUSIVE or failed is a normal FEP
  input — reconcile it, never treat it as a dispatch failure.

Dispatch failure handling (worker never ran, tool error):
1. Retry the dispatch once (recreate the child task if the store reports it unusable).
2. **Explore: never execute inline.** If the retry fails, stop and ask the user via
   \`ask_user_to_clarify\` (report what is blocked and why).
3. **Publish/Verify: after a failed retry you MAY execute inline** to preserve delivery —
   the payload decisions are already in your context. Follow the same gate self-check as a
   worker would.

## Delegation Protocol

Mechanics every brief must be designed around:

- A worker sees ONLY: its task title + description, the parent task's description, and the
  results of completed dependency tasks. It does NOT see this conversation.
- Therefore **every brief is self-contained**: all decisions, boundaries, and evidence travel
  inside the task description.
- Dispatch is synchronous: \`task { taskId, agentId: "general-purpose" }\` returns an
  observation (the worker's final message, truncated at ~8k chars) and an
  \`executionResultId\`. Record the id on the child task (\`manage_task update\`/\`add_activity\`).
- The worker settles its own task (\`set_status\` with \`result\`/\`failureReason\`). Read both the
  observation and the task result before reconciling.

Write briefs as field checklists adapted to the situation — decisions and evidence, not prose.
The worker assembles payloads from the reference sections of this skill; the brief carries the
approved decisions, never full payload JSON.

### Explore brief (\`Explore <schema>.<table> for <goal>\`)

Required fields:
1. Datasource: serverKey/connectionKey + datasourceId (and \`runConfig.metricsDataSource\` preset if present).
2. Grant boundary: the allowed schema + table pattern from \`get_grants\`, verbatim; never probe outside it.
3. Target table(s): schema-qualified names; one brief per table.
4. Question list: per column (or column group) — business meaning candidates, expected role
   (dimension | measure | time), net/gross, grain.
5. Method: read-only SELECT/WITH only; use the \`references/metadata-sql.md\` templates
   (columns, sample rows, value distribution); cap sample size (e.g. ≤1000 rows, ≤50 groups).
   Prefer targeted queries over bulk catalog dumps. A large single read (e.g. full
   \`read_semantic_catalog\` detail) is allowed — your isolation protects the orchestrator's
   context — but findings must be compressed to conclusion level.
6. Output contract: structured findings — per column: observed type, sample values,
   distribution highlights, role candidate + evidence, \`INCONCLUSIVE\` marking for unresolved
   semantics.
7. Invariants: read-only only; grant boundary absolute; never guess — \`INCONCLUSIVE\` instead.
8. Closing: settle the task (\`set_status completed\`, result = findings summary); final message
   = the structured findings (it becomes the observation).

### Publish-table brief (\`Publish table meta <objectKey>\`)

Required fields:
1. Approved decisions: full column list (name/type/role/label/description), grain,
   displayName/docType/shortDesc/sourceSystem/sourceTables.
2. Identity: \`objectKey\` (stable, unique per datasource) + \`schemaName.tableName\`.
3. Assembly instruction: full \`table_view_detail\` payload per **Publish table meta** below,
   plus the identity twin (\`table_catalog\`, same \`objectKey\`), plus an EXACT \`accessGrant\`
   matching the tenant grant.
4. Gate self-check: publish gate items 1–3 — do not publish if any fails.
5. Publish via \`metrics_meta_tool\` \`create_table\`; then \`get_meta\` to confirm visibility.
6. Mistakes: fix with \`update_table\`; never leave a partial publish.
7. Invariants: EXACT patternType; every column tagged and labeled.
8. Closing: settle the task; final message = receipts (objectKeys + get_meta confirmation).

### Publish-metric brief (\`Publish metric <objectKey>\`)

Required fields (per metric):
1. Identity: \`objectKey\` = \`metric_name\`, display_name, domain, description, data_type, format.
2. Source: schema-qualified \`source.table_view\` (must match the published table meta).
3. Calculation: aggregate (\`aggregation\` + \`measure\`, with the confirmed business meaning) or
   derived (ratio references / formula) — exactly as approved in Propose the model.
4. \`supported_dimensions\`: array of objects \`{dim_id, field_name, label, data_type}\`,
   including the time dimension.
5. \`default_time_context\`: time_dimension + granularity + supported_grains (+ window if requested).
6. \`ai_agent_context\`: polarity, synonyms, human_readable_explanation.
7. Assembly instruction: \`metric_index\` then \`metric_detail\` in the FULL runtime style per
   **Publish metric meta** below; publish via \`metrics_meta_tool\` \`create_metric\`.
8. Gate self-check + smoke \`query\` (one grouping by the default time grain) + \`get_meta\`
   receipts; closing = settle the task, final message = receipts.

### Verify brief (\`Verify <metric> against acceptance evidence\`)

Required fields:
1. Acceptance evidence verbatim (from the Goal Model).
2. Queries to run (semantic \`query\` via \`metrics_runtime_tool\`) and the expected shape/values.
3. Report contract: PASS/FAIL verdict + actual vs expected; on FAIL include the differences
   and a suspicion list — do NOT modify meta (fixes are new Publish tasks).
4. Invariants: read-only only.
5. Closing: settle the task; final message = the verdict summary.

### Brief closing (all types)

- Invariants line: grant boundary + read-only SQL + publish gate before any create.
- Result structure: the structured summary the orchestrator expects back.
- Finish by settling the task (\`set_status completed\` with result, or \`failed\` with
  failureReason) and making the final message that structured summary.

## Worker mode

Trigger: your context contains \`## Bound Task\`. You are a worker; these rules override the
orchestrator policy (Safety and the reference sections still apply):

- Execute the bound task's brief only. Read the contract with \`manage_task get\`; the task
  description is authoritative. Load this skill if the bootstrap asks — you need its payload
  reference sections.
- Do NOT: re-agree the Goal Model, create sibling or child tasks, dispatch \`task\`, call
  \`ask_user_to_clarify\`, or update the parent task's \`## Belief State\`.
- Apply the FEP loop inside the brief; obey the grant boundary and read-only SQL; run the
  publish gate before any create.
- Unclear semantics: mark \`INCONCLUSIVE\` with candidates + evidence in your findings. Never
  guess, never publish on a guess.
- Finish: \`set_status completed\` with a structured result (or \`failed\` with failureReason),
  and make your final message the structured summary — it is the observation your
  orchestrator reads.

## Belief State (orchestrator-owned)

The parent task's description carries a \`## Belief State\` section as a Markdown table:

\`\`\`
| Belief Key | Probability | Target | Basis |
|---|---|---|---|
| sell-in-nes-source-table | 70% | 100% | column \`nes\` probed as numeric measure in grant |
\`\`\`

Rules: belief keys are lowercase kebab-case; probability/target are percentages 0–100% that
represent evidence support, not calibrated probability. Only the orchestrator writes this
section — after each dispatch it folds the worker's findings into the table so the parent
task always reflects the current evidence.

## Workflow

1. **Pick the datasource + cheap observation.** \`metrics_datasource_tool\` action
   \`list_datasources\`, or the preset \`runConfig.metricsDataSource\` (\`serverKey\`→connectionKey,
   \`datasourceId\`). Inline, read-only.
2. **Agree the Goal Model.** (HITL)
3. **Create the parent task** — contract + \`## Belief State\` (seeded from the Goal Model).
4. **Explore by dispatch** — one Explore brief per table, serial; collect findings; reconcile
   \`## Belief State\`; collect \`INCONCLUSIVE\` columns.
5. **Propose the model** (HITL) — present table columns (with \`role: dimension | measure\`)
   and the metrics (aggregate/derived) for approval; include clarify questions for
   \`INCONCLUSIVE\` columns in the same ask.
6. **Publish tables by dispatch** — one Publish-table brief per table; check receipts.
7. **Publish metrics by dispatch** — one Publish-metric brief per metric; receipts + smoke query.
8. **Verify by dispatch** — Verify brief against acceptance evidence. FAIL → reconcile
   belief → fix decisions → new Publish task (\`update_table\`/\`update_metric\`) → re-Verify.
   PASS → complete the parent task (result = acceptance evidence met).

## Explore (reference)

Read \`table-grants\`, then probe with \`query\`. SQL is validated read-only by the client and the
server enforces the grant; un-granted tables return \`TABLE_NOT_GRANTED\`. Prefer the
\`references/metadata-sql.md\` templates because dialect varies (PostgreSQL / SQL Server / HANA).
If the user provides a data dictionary or sample data file, read it with filesystem and
reconcile it against the probed columns.

## Unclear field semantics

A probed column name may not reveal its business meaning (e.g. whether \`nes\` is a net amount to
sum, a rate to average, or an enum to filter). Publishing meta on a misunderstood field pollutes
the runtime and every downstream metric that depends on it, and delete is not available. Treat this
as FEP **INCONCLUSIVE** — never guess, and never publish on an unverified guess.

Resolve in order of increasing cost (inside the Explore brief):
1. **Probe sample values and distribution** (\`query\`, templates in \`references/metadata-sql.md\`):
   - numbers in (0,1) or clustered near 100 → likely a ratio/percentage → dimension or \`avg\`, rarely \`sum\`;
   - large additive amounts (currency, counts) → measure with \`sum\`;
   - small enumerations (year 2023/2024, 0/1 flags, codes) → dimension.
   The share of zeros/negatives and the value scale usually disambiguate semantics.
2. **Read user-provided docs**: if the user supplied a data dictionary or sample file, read it via
   filesystem and reconcile it against the probed columns.
3. **Cross-field correlation**: compare the unclear field with fields whose semantics are confirmed.

If evidence is still INCONCLUSIVE:
- Do **not** publish meta that depends on the unclear field.
- The worker marks the column \`INCONCLUSIVE\` (candidates + evidence) in its findings; the
  **orchestrator** lists it with candidate interpretations in the **Propose the model** step
  and asks the user to confirm its semantics (role, value meaning, net/gross) via
  \`ask_user_to_clarify\`. The user does not need to see the table — probed schema, samples, and
  candidates are enough.
- The orchestrator records a low-probability belief entry (e.g.
  \`| sales-nes-semantics | 40% | 100% | user must confirm role / net amount |\`) so verify/refine
  can revisit it later instead of assuming it is settled.

## Publish table meta (\`create_table\`)

Publish the authoritative shape below (objectType \`table_view_detail\`, with an EXACT accessGrant):

\`\`\`json
{
  "objectType": "table_view_detail",
  "objectKey": "hankel_view_sales_summary",
  "status": 1,
  "payload": {
    "schemaName": "public",
    "tableName": "hankel_view_sales_summary",
    "displayName": "Hankel Sales Summary",
    "docType": "Sales Summary",
    "docTypeEn": "Sales Summary",
    "shortDesc": "Sales-level progress summary.",
    "grain": "canonical_sales_name",
    "sourceSystem": "Hankel Run for Gold",
    "sourceTables": ["hankel_project_opportunity_lines", "hankel_new_order_lines"],
    "columns": [
      { "name": "canonical_sales_name", "label": "Canonical Sales Name", "type": "string", "role": "dimension", "description": "Standardized sales name used for grouping." },
      { "name": "nes", "label": "NES", "type": "number", "role": "measure", "description": "Net external sales amount." },
      { "name": "report_cutoff_date", "label": "Report Cutoff Date", "type": "date", "role": "time", "description": "Cutoff date of the report snapshot." }
    ]
  },
  "accessGrant": {
    "schemaName": "public",
    "tablePattern": "hankel_view_sales_summary",
    "patternType": "EXACT",
    "caseSensitive": false,
    "status": 1
  }
}
\`\`\`

Design rules:
- \`objectKey\` is stable and unique per datasource (the view/table key).
- \`columns[].role\` has THREE values: \`dimension\` (grouping/filtering), \`measure\` (aggregated numbers), \`time\` (the report cutoff/snapshot date used by default_time_context).
- Give every column a business \`description\`.
- Lineage fields (\`docType\`, \`shortDesc\`, \`grain\`, \`sourceSystem\`, \`sourceTables\`) make the asset findable and trustworthy — fill them from exploration evidence.
- Runtime table publication uses \`patternType: "EXACT"\` (\`PREFIX\` is only for datasource-level exploration scopes).
- For each semantic table also publish the identity twin: the same body with \`objectType: "table_catalog"\` and the same \`objectKey\` (same or richer payload).

## Publish metric meta (\`create_metric\`)

Two steps for the same \`objectKey\`.

### 3a. Metric index (for lists/search)

\`\`\`json
{
  "objectType": "metric_index",
  "objectKey": "sell_in_nes",
  "status": 1,
  "payload": {
    "metric_name": "sell_in_nes",
    "display_name": "Sell-in NES",
    "domain": "sales",
    "short_desc": "Net external sales amount from distributor sell-in rows.",
    "search_keywords": ["sell in", "nes", "net external sales"],
    "source_type": "cdp_postgres",
    "source": { "table_view": "public.hankel_distr_sell_in" }
  }
}
\`\`\`

\`search_keywords\` drive metric retrieval for natural-language questions — include business synonyms.

### 3b. Metric detail (the calculation)

IMPORTANT — runtime schema requirement: the Metrics Server query runtime only recognizes metrics
published in the FULL runtime style below (the same shape as existing working metrics, with
snake_case fields like \`metric_name\`, \`source.table_view\`, \`supported_dimensions\` objects, and
\`default_time_context\`). A minimal \`{name, sourceTable, calculation, dimensions}\` payload is
accepted by the publish API but is NOT queryable — querying it returns 403 "not authorized".

Aggregate metric (full runtime style):

\`\`\`json
{
  "objectType": "metric_detail",
  "objectKey": "sell_in_nes",
  "status": 1,
  "payload": {
    "metric_name": "sell_in_nes",
    "display_name": "Sell-in NES",
    "domain": "sales",
    "description": "Net external sales amount from distributor sell-in rows.",
    "data_type": "numeric",
    "format": "currency",
    "source_type": "cdp_postgres",
    "source": { "table_view": "public.hankel_distr_sell_in", "base_filters": [] },
    "calculation": { "type": "aggregate", "aggregation": "sum", "measure": "nes" },
    "supported_dimensions": [
      { "dim_id": "posting_year", "field_name": "posting_year", "label": "Posting Year", "data_type": "number" },
      { "dim_id": "sales_team", "field_name": "sales_team", "label": "Sales Team", "data_type": "string" }
    ],
    "default_time_context": {
      "time_dimension": "posting_date",
      "label": "Posting date",
      "granularity": "month",
      "window": "YTD",
      "supported_grains": ["month", "year"]
    },
    "ai_agent_context": {
      "polarity": "positive",
      "synonyms": ["sell in", "nes", "net external sales"],
      "human_readable_explanation": "Higher sell-in NES means more goods purchased from the distributor."
    }
  }
}
\`\`\`

Derived ratio metric (references two aggregate metrics):

\`\`\`json
{
  "objectType": "metric_detail",
  "objectKey": "gross_margin_rate",
  "status": 1,
  "payload": {
    "metric_name": "gross_margin_rate",
    "display_name": "Gross Margin Rate",
    "description": "Gross margin divided by sell-in NES.",
    "domain": "sales",
    "data_type": "numeric",
    "format": "percent",
    "source_type": "cdp_postgres",
    "source": { "table_view": "public.hankel_distr_sell_in", "base_filters": [] },
    "calculation": { "type": "derived", "operator": "ratio", "numerator": "gross_margin", "denominator": "sell_in_nes" },
    "supported_dimensions": [
      { "dim_id": "posting_year", "field_name": "posting_year", "label": "Posting Year", "data_type": "number" }
    ],
    "default_time_context": { "time_dimension": "posting_date", "granularity": "month", "supported_grains": ["month", "year"] }
  }
}
\`\`\`

Design rules:
- \`objectKey\` unique per datasource (index and detail share it).
- \`payload.metric_name\` must equal \`objectKey\`.
- \`source.table_view\` must be schema-qualified and match the published table meta's physical table.
- \`supported_dimensions\` is an array of OBJECTS ({dim_id, field_name, label, data_type}); include the time dimension too; each must be a published column of that table meta.
- \`calculation.type: "aggregate"\` = one measure + aggregation (sum/avg/count/min/max/count_distinct).
- \`calculation.type: "derived"\` operator \`ratio\` references existing metric objectKeys (numerator/denominator); a \`formula\` variant takes an expression of metric names, numbers, parentheses, and + - * / — referenced metrics must share the same source.table_view.
- \`default_time_context\` (\`time_dimension\` + \`granularity\` + \`supported_grains\`, optionally \`window\`) is required by the query runtime; its \`time_dimension\` must appear in supported_dimensions.
- \`ai_agent_context\` (\`polarity\`, \`synonyms\`, \`human_readable_explanation\`) improves natural-language metric retrieval — always provide it.
- One semantic query can only combine metrics sharing the same source.table_view; groupBy values must be dim_ids or time-grain expressions like \`posting_date__month\`.
- Do not inline raw SQL for new DB-backed metrics (complex logic belongs in a view).
- Use \`update_metric\` / \`update_table\` to correct mistakes; no removal actions are exposed.

## Verify and refine

- Verify is **dispatched** (Verify brief). The worker runs \`get_meta\` → confirms the table and
  metric are visible, then \`query\`es the metric against acceptance evidence.
- On FAIL the worker reports differences; the **orchestrator** applies the FEP reconcile:
  update \`## Belief State\`, fix the decisions, and dispatch a new Publish task
  (\`update_table\`/\`update_metric\`) — then re-Verify — until acceptance evidence is met.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| \`TABLE_NOT_GRANTED\` | SQL referenced a table outside the tenant grant; pick an in-grant table |
| metric create fails: sourceTable missing | publish the table meta first |
| metric create fails: unknown dimension | publish the dimension column with \`role: dimension\` |
| derived metric create fails | numerator/denominator metrics don't exist yet — publish them first |
| datasource \`query\` rejected | SQL was not read-only (SELECT/WITH only) or multi-statement |
| \`403 Metric is not authorized\` on query_metrics | the metric meta is not in the queryable runtime schema (e.g. missing \`source.table_view\` / \`supported_dimensions\` / \`default_time_context\`) or genuinely unauthorized — re-publish with the full runtime payload and verify; never retry blindly |
| worker observation truncated | findings exceeded ~8k chars; worker must keep the final message a compact structured summary |
| dispatch failed (tool error) | retry once; Explore never goes inline — ask the user; Publish/Verify may go inline after the failed retry |

## Safety

- Only query tables that match the tenant's \`table-grants\`.
- Never return a full physical-table inventory or reveal whether an un-granted table exists.
- SQL is validated read-only; let the server reject anything outside that.
`;

const resources: Record<string, PluginSkillResource> = {
  "references/metadata-sql.md": {
    content: `# Datasource exploration SQL templates (read-only)

Pick the block matching the server's database. These run through the datasource \`query\` action;
the SQL is validated read-only.

## PostgreSQL
- Tables: \`select table_schema, table_name from information_schema.tables where table_schema not in ('pg_catalog','information_schema') order by table_schema, table_name\`
- Columns: \`select column_name, data_type from information_schema.columns where table_schema = :schemaName and table_name = :tableName order by ordinal_position\`
- Sample: \`select * from :schemaName.:tableName limit :maxRows\` — pass a concrete table, not a bound param, if the server does not support binding identifiers.

## SQL Server
- Tables: \`select table_schema, table_name from information_schema.tables order by table_schema, table_name\`
- Columns: \`select column_name, data_type from information_schema.columns where table_schema = :schemaName and table_name = :tableName order by ordinal_position\`

## HANA
- Tables: \`select schema_name, table_name from sys.tables where schema_name = :schemaName order by table_name\`
- Columns: \`select column_name, data_type_name from table_columns where schema_name = :schemaName and table_name = :tableName order by position\`

## Value distribution (any dialect)
\`select <dim>, count(*) as row_count from <table> group by <dim> order by row_count desc limit :maxRows\`
`,
  },
  "examples/table-meta.json": {
    content: `{
  "objectType": "table_view_detail",
  "objectKey": "hankel_distr_sell_in",
  "status": 1,
  "payload": {
    "schemaName": "public",
    "tableName": "hankel_distr_sell_in",
    "displayName": "Hankel Distributor Sell In",
    "description": "Distributor sell-in transaction rows for the Hankel tenant.",
    "columns": [
      { "name": "posting_year", "label": "Posting Year", "type": "number", "role": "dimension" },
      { "name": "posting_month", "label": "Posting Month", "type": "number", "role": "dimension" },
      { "name": "sales_team", "label": "Sales Team", "type": "string", "role": "dimension" },
      { "name": "region", "label": "Region", "type": "string", "role": "dimension" },
      { "name": "nes", "label": "Net External Sales", "type": "number", "role": "measure" },
      { "name": "gross_margin", "label": "Gross Margin", "type": "number", "role": "measure" }
    ]
  }
}
`,
  },
  "examples/metric-aggregate.json": {
    content: `{
  "objectType": "metric_detail",
  "objectKey": "sell_in_nes",
  "status": 1,
  "payload": {
    "metric_name": "sell_in_nes",
    "display_name": "Sell-in NES",
    "domain": "sales",
    "description": "Net external sales amount from distributor sell-in rows.",
    "data_type": "numeric",
    "format": "currency",
    "source_type": "cdp_postgres",
    "source": { "table_view": "public.hankel_distr_sell_in", "base_filters": [] },
    "calculation": { "type": "aggregate", "aggregation": "sum", "measure": "nes" },
    "supported_dimensions": [
      { "dim_id": "posting_year", "field_name": "posting_year", "label": "Posting Year", "data_type": "number" },
      { "dim_id": "sales_team", "field_name": "sales_team", "label": "Sales Team", "data_type": "string" },
      { "dim_id": "posting_date", "field_name": "posting_date", "label": "Posting Date", "data_type": "string" }
    ],
    "default_time_context": {
      "time_dimension": "posting_date",
      "label": "Posting date",
      "granularity": "month",
      "window": "YTD",
      "supported_grains": ["month", "year"]
    },
    "ai_agent_context": {
      "polarity": "positive",
      "synonyms": ["sell in", "nes", "net external sales"],
      "human_readable_explanation": "Higher sell-in NES means more goods purchased from the distributor."
    }
  }
}
`,
  },
  "examples/metric-derived.json": {
    content: `{
  "objectType": "metric_detail",
  "objectKey": "gross_margin_rate",
  "status": 1,
  "payload": {
    "metric_name": "gross_margin_rate",
    "display_name": "Gross Margin Rate",
    "domain": "sales",
    "description": "Gross margin divided by sell-in NES.",
    "data_type": "numeric",
    "format": "percent",
    "source_type": "cdp_postgres",
    "source": { "table_view": "public.hankel_distr_sell_in", "base_filters": [] },
    "calculation": { "type": "derived", "operator": "ratio", "numerator": "gross_margin", "denominator": "sell_in_nes" },
    "supported_dimensions": [
      { "dim_id": "posting_year", "field_name": "posting_year", "label": "Posting Year", "data_type": "number" },
      { "dim_id": "posting_date", "field_name": "posting_date", "label": "Posting Date", "data_type": "string" }
    ],
    "default_time_context": {
      "time_dimension": "posting_date",
      "label": "Posting date",
      "granularity": "month",
      "supported_grains": ["month", "year"]
    },
    "ai_agent_context": {
      "polarity": "positive",
      "synonyms": ["gross margin rate", "margin rate"],
      "human_readable_explanation": "Gross margin rate divides gross margin by sell-in NES; higher means more margin retained on sold-in goods."
    }
  }
}
`,
  },
};

/** Versioned modeling skill bundle for the `semantic-metrics` plugin. */
export const SEMANTIC_METRICS_MODELING_SKILL: PluginSkillDefinition = {
  version: "1.1.2",
  content: CONTENT,
  resources,
};
