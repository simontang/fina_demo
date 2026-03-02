# Agent Skill: How to Query Metrics via the Metrics API

This document teaches an AI agent how to read metric metadata and translate it into correct API calls. Follow the steps in order.

---

## Overview: The Three-Step Loop

```
1. /index   → discover what metrics exist
2. /detail  → understand one metric before querying it
3. /query   → execute the query with the right parameters
```

Repeat steps 2–3 for each metric the user asks about. Drill-down means running step 3 multiple times with progressively narrower filters.

---

## Step 1 — Discover Available Metrics

```
GET /api/v1/datasources/{datasource_id}/metrics/index
```

**Key fields in the response:**

| Field | How to use it |
|---|---|
| `metricName` | The identifier you pass to `/detail` and to the `metrics` array in a query |
| `domain` | Group related metrics (e.g. `sales_performance`, `pricing_and_margin`) |
| `shortDesc` | One-line description to present to the user |
| `searchKeywords` | Match against user's natural-language request |

**Example response excerpt:**
```json
{
  "metrics": [
    { "metricName": "order_amt_tax_inc", "domain": "sales_performance", "shortDesc": "订单金额（含税）" },
    { "metricName": "avg_discount_rate",  "domain": "pricing_and_margin", "shortDesc": "折扣率" }
  ]
}
```

---

## Step 2 — Read the Metric Detail

```
GET /api/v1/datasources/{datasource_id}/metrics/{metric_name}/detail
```

This response is self-contained. Everything you need to write a correct query is here.

### 2.1 Identify the time axis

`defaultTimeContext` tells you the primary date field and exactly how to use it:

```json
"defaultTimeContext": {
  "timeDimension": "DocDate",
  "label": "单据日期",
  "granularity": "month",
  "window": "current_month",
  "supportedGrains": ["day", "week", "month", "year"],
  "queryUsage": {
    "filter": {
      "dimensionKey": "DocDate",
      "supportedOperators": ["BETWEEN", "GT", "GTE", "LT", "LTE", "EQ"],
      "valueFormat": "YYYY-MM-DD",
      "example": { "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-12-31"] }
    },
    "groupBy": {
      "pattern": "{timeDimension}__{grain}",
      "examples": ["DocDate__year", "DocDate__month", "DocDate__week", "DocDate__day"]
    }
  }
}
```

Rules:
- To **filter by date range**: copy `queryUsage.filter.example` and replace the values.
- To **group by time grain**: pick one string from `queryUsage.groupBy.examples` (e.g. `"DocDate__month"`).
- When the user gives no date range: apply `granularity` + `window` as the implicit default in your explanation, but still ask the server with an explicit date filter for clarity.

### 2.2 Identify available dimensions

`supportedDimensions` lists every axis available for GROUP BY and filtering. Each entry is self-documenting:

```json
"supportedDimensions": [
  {
    "dim_id": "org_region", "field_name": "Region", "type": "categorical",
    "filter_operators": ["IN", "EQ", "NEQ"],
    "filter_example": { "dimension": "org_region", "operator": "IN", "values": ["华东", "华南"] },
    "group_by_example": "org_region"
  },
  {
    "dim_id": "ship_date", "field_name": "ShipDate", "type": "datetime",
    "filter_operators": ["BETWEEN", "GT", "GTE", "LT", "LTE", "EQ"],
    "filter_example": { "dimension": "ship_date", "operator": "BETWEEN", "values": ["2025-01-01", "2025-01-31"] },
    "group_by_example": "ship_date__day"
  }
]
```

Rules:
- **`group_by_example`**: copy-paste directly into `groupBy[]`.
- **`filter_example`**: copy-paste directly into `filters[]`, replace the `values` with the user's actual values.
- Use `dim_id` (never `field_name`) as the `dimension` key in filters.
- `type: "categorical"` → values are strings; use `IN` for multiple, `EQ` / `NEQ` for single, `LIKE` for fuzzy.
- `type: "datetime"` → values are `YYYY-MM-DD`; use `BETWEEN` for ranges, `GT`/`LT` for open-ended.

### 2.3 Understand result polarity

`aiAgentContext.polarity`:
- `"positive"` — higher value is better (e.g. order amount, delivery qty).
- `"negative"` — lower value is better (e.g. discount rate, return amount).

Use this when summarising results to the user ("the discount rate **increased**, which is a negative sign").

---

## Step 3 — Execute a Query

```
POST /api/v1/metrics/query
Content-Type: application/json
```

### Full request schema

```json
{
  "datasourceId": 1,
  "metrics":  ["<metric_name>", "..."],
  "groupBy":  ["<dim_id or DocDate__grain>", "..."],
  "filters":  [{ "dimension": "<dim_id or DocDate>", "operator": "<OP>", "values": ["..."] }],
  "orderBy":  [{ "field": "<same as groupBy item>", "direction": "ASC|DESC" }],
  "limit":    100,
  "debug":    false
}
```

### Filter operator reference

| Operator | Applies to | `values` array |
|---|---|---|
| `BETWEEN` | datetime, number | `[min, max]` (inclusive) |
| `IN` | categorical | `["v1", "v2", ...]` |
| `EQ` | any | `["v"]` |
| `NEQ` | any | `["v"]` |
| `GT` / `GTE` | datetime, number | `["v"]` |
| `LT` / `LTE` | datetime, number | `["v"]` |
| `LIKE` | string | `["%pattern%"]` |
| `NOT_NULL` | any | `[]` |

---

## Workflow Patterns

### Pattern A — Time Trend (most common first query)

**Goal:** Show how a metric changes over time within a period.

```json
{
  "datasourceId": 1,
  "metrics":  ["order_amt_tax_inc"],
  "groupBy":  ["DocDate__month"],
  "filters":  [{ "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-12-31"] }],
  "orderBy":  [{ "field": "DocDate__month", "direction": "ASC" }]
}
```

Use `DocDate__week` or `DocDate__day` for shorter time windows.

---

### Pattern B — Dimension Breakdown (who/what is top or bottom)

**Goal:** Rank performance across a categorical dimension in a fixed period.

```json
{
  "datasourceId": 1,
  "metrics":  ["order_amt_tax_inc"],
  "groupBy":  ["org_region"],
  "filters":  [{ "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-03-31"] }],
  "orderBy":  [{ "field": "value", "direction": "DESC" }],
  "limit":    10
}
```

Pick `dim_id` values from `supportedDimensions` in the detail response.

---

### Pattern C — Drill Down (zoom into an anomaly)

**Goal:** After Pattern A reveals a bad month, find which segment caused it.

Step 1 — find the anomaly month:
```json
{ "groupBy": ["DocDate__month"], "filters": [{ "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-12-31"] }] }
```

Step 2 — drill into that month by adding a categorical dimension:
```json
{
  "groupBy":  ["sales_person"],
  "filters":  [
    { "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-06-01", "2025-06-30"] },
    { "dimension": "org_region", "operator": "EQ", "values": ["华东"] }
  ],
  "orderBy":  [{ "field": "value", "direction": "ASC" }]
}
```

Step 3 — optionally add a second groupBy dimension to cross-analyse:
```json
{ "groupBy": ["sales_person", "product_category"], ... }
```

---

### Pattern D — Multi-Metric Comparison

**Goal:** Query two or more metrics in one call to compare them side-by-side.

```json
{
  "datasourceId": 1,
  "metrics":  ["order_amt_tax_inc", "net_sales_amt"],
  "groupBy":  ["DocDate__month"],
  "filters":  [{ "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-12-31"] }],
  "orderBy":  [{ "field": "DocDate__month", "direction": "ASC" }]
}
```

The response contains one `results[]` entry per metric. Each entry has its own `rows[]` and `metricName` label.

---

### Pattern E — Time × Dimension Cross-Analysis

**Goal:** See how a dimension behaves across time (e.g. monthly sales by region).

```json
{
  "datasourceId": 1,
  "metrics":  ["order_amt_tax_inc"],
  "groupBy":  ["DocDate__month", "org_region"],
  "filters":  [{ "dimension": "DocDate", "operator": "BETWEEN", "values": ["2025-01-01", "2025-06-30"] }],
  "orderBy":  [
    { "field": "DocDate__month", "direction": "ASC" },
    { "field": "org_region",     "direction": "ASC" }
  ]
}
```

---

## Recommended Analysis Flow

```
User question
      │
      ▼
Step 1: /index — find matching metric(s) by domain or keyword
      │
      ▼
Step 2: /detail — read defaultTimeContext + supportedDimensions
      │
      ▼
Step 3a: Pattern A (time trend) — get the big picture
      │
      ├─ anomaly found? ──► Step 3c: Pattern C (drill down by region/person/category)
      │
      ├─ user asks "who"? ──► Step 3b: Pattern B (dimension breakdown)
      │
      └─ user asks "compare X and Y"? ──► Step 3d: Pattern D (multi-metric)
```

Always start broad (monthly trend over a year), then narrow (weekly in the problem month), then pinpoint (drill by dimension in the problem week).

---

## Debug Mode

Set `"debug": true` to receive `executedSqls` in the response. Use this to verify that the generated SQL matches your intent before presenting results to the user.

```json
{ ..., "debug": true }
```

Response:
```json
{
  "results": [...],
  "executedSqls": ["SELECT TO_NVARCHAR(\"DocDate\", 'YYYY-MM') AS \"DocDate__month\", SUM(\"GTotal\") AS \"value\" FROM MTC_VW_AI_ORDR WHERE ..."]
}
```

---

## Common Mistakes to Avoid

| Mistake | Correct approach |
|---|---|
| Using `field_name` (e.g. `"SlpName"`) in `groupBy` | Use `dim_id` (e.g. `"sales_person"`) |
| Writing `"DocDate__Month"` (capital M) | Always lowercase grain: `"DocDate__month"` |
| Omitting a date filter entirely | Always include at least one `DocDate` filter to avoid full-table scans |
| Querying a metric not in `supportedDimensions` by an unsupported dim | Only use `dim_id` values listed in the detail response |
| Putting a `groupBy` field in `orderBy` using its display name | `orderBy.field` must match exactly what you put in `groupBy` |
