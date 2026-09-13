# Semantic Layer Architecture Research

Last refreshed: 2026-06-01

## Executive Takeaways

1. A useful AI semantic layer is not just a set of SQL views. It needs governed business objects, metrics, dimensions, relationships, access policies, query APIs, and metadata introspection.
2. Cube's architecture is useful because it treats the semantic layer as a runtime: model definitions, access control, caching, APIs, and metadata discovery all live behind a deterministic query boundary.
3. dbt's architecture is useful because it separates authored semantic definitions, MetricFlow SQL generation, cloud service dispatch, and downstream APIs/CLI.
4. For B1, the most important split is:
   - `semantic/read`: stable analytical facts such as `SalesOrderLine`, `DocumentFlow`, `InventoryPosition`, `Customer360`, `PriceResolution`, and `AR/AP Aging`.
   - `service/action`: command APIs above Service Layer such as `search_business_partner`, `resolve_price`, `validate_document`, and `create_draft`.

## Cube: Architecture Pattern

Sources:

- Cube documentation: https://cube.dev/docs
- Cube local clone: `docs/architecture-discussions/semantic-service-layer/cubejs`
- Local files inspected:
  - `cubejs/README.md`
  - `cubejs/docs/content/product/introduction.mdx`
  - `cubejs/docs/content/product/data-modeling/overview.mdx`
  - `cubejs/docs/content/product/apis-integrations/core-data-apis/index.mdx`
  - `cubejs/docs/content/product/apis-integrations/mcp-server.mdx`
  - `cubejs/docs/content/product/auth/data-access-policies.mdx`

### What Cube Makes Explicit

Cube positions the semantic layer as infrastructure between AI/BI consumers and the warehouse. The important pieces for our design discussion are:

| Area | Cube pattern | B1 implication |
| --- | --- | --- |
| Data model | Code-first model with cubes, measures, dimensions, joins, and views. | Model B1 business facts as named semantic objects instead of asking AI to understand `ORDR/RDR1/OINV/INV1` directly. |
| Consumer facade | Views act as exposed data products over the underlying cube graph. | Expose curated views like `SalesOrderLine` and `DocumentFlow`, not every internal join surface. |
| Query runtime | AI and BI query the semantic layer; SQL is generated/validated by runtime. | AI should call `/semantic/query` or Semantic SQL-like contracts, not raw database SQL by default. |
| Access control | Row-level, member-level, and masking policies are model-level concerns. | B1 permissions, company DB scope, branch/warehouse/user restrictions should be enforced centrally. |
| Caching | Pre-aggregations reduce latency and warehouse pressure. | Metrics such as monthly sales, stock position, AR aging, and customer 360 should have cache/materialization strategy. |
| APIs | REST, GraphQL, SQL, DAX, MCP and metadata APIs are consumption protocols over one model. | B1 should keep one semantic contract and expose it through AI tools, BI, REST, and maybe SQL-compatible interfaces. |
| Introspection | Meta API lets agents discover available metrics, dimensions, and relationships. | AI needs discoverable model metadata with labels, synonyms, allowed filters, lineage, and examples. |

### Local Cube Code Areas Worth Inspecting Later

| Path | Why it matters |
| --- | --- |
| `docs/content/product/data-modeling/reference/` | DSL shape for cubes, measures, dimensions, joins, views, segments, hierarchies, and pre-aggregations. |
| `docs/content/product/apis-integrations/core-data-apis/` | Query surfaces: REST, GraphQL, SQL, DAX, common query concepts. |
| `docs/content/product/auth/` | Centralized policy model for row/member security and masking. |
| `docs/content/product/caching/` | Pre-aggregation model, matching, refresh, production operation. |
| `docs/content/product/apis-integrations/mcp-server.mdx` | How Cube exposes an AI-facing MCP endpoint above the semantic layer. |
| `packages/cubejs-server-core/` | Runtime/query orchestration code. |
| `rust/cubesql/` | SQL interface implementation. |
| `rust/cube/cubesqlplanner/` | Query planning internals. |

## dbt Semantic Layer: Architecture Pattern

Sources:

- dbt Semantic Layer architecture: https://docs.getdbt.com/docs/use-dbt-semantic-layer/sl-architecture
- dbt semantic models: https://docs.getdbt.com/docs/build/semantic-models
- dbt Semantic Layer APIs: https://docs.getdbt.com/docs/dbt-apis/sl-api-overview

### What dbt Makes Explicit

dbt's useful pattern is separation of concerns:

| Component | Role | B1 implication |
| --- | --- | --- |
| Semantic model spec | YAML definitions for entities, measures, dimensions, defaults, and relationships. | Treat B1 semantic definitions as versioned configuration, not hardcoded prompt text. |
| MetricFlow | Generates SQL from semantic model + metric query. | Keep SQL generation deterministic and testable; AI supplies intent, runtime generates SQL. |
| Service layer | Dispatches metric query requests to target engines and coordinates execution. | Introduce a B1 query service between agents and SQL Server/HANA. |
| APIs | GraphQL/JDBC and integrations for downstream tools. | Do not make the agent the only consumer; BI and apps should share the same contract. |
| CLI | Supports querying metrics/dimensions and metadata through local workflows. | Add local/debug CLI for model validation, metadata search, and query replay. |

dbt's feature comparison is especially relevant: local/open components can define models and generate SQL, while cloud/service components provide APIs and managed integrations. This suggests a practical B1 split:

- Open/local: semantic manifest, tests, SQL generation templates, mock metadata, CLI.
- Runtime/service: authorization, company DB routing, query execution, caching, audit logs, API keys/OAuth.

## Semantic Layer Shape For B1

### Minimum Object Contract

Each semantic object should define:

| Field | Example |
| --- | --- |
| `name` | `SalesOrderLine` |
| `description` | Sales order line with header, customer, item, warehouse, tax, currency, and status context. |
| `grain` | One row per `DocEntry + LineNum`. |
| `source_lineage` | `ORDR`, `RDR1`, `OCRD`, `OITM`, `OSLP`, `OWHS`. |
| `dimensions` | `doc_date`, `customer_code`, `customer_name`, `item_code`, `salesperson`, `warehouse`, `status`. |
| `measures` | `order_amount_net`, `order_amount_gross`, `quantity`, `open_quantity`, `open_amount`. |
| `joins` | To customer, item, salesperson, warehouse, document flow. |
| `filters` | Date, customer, item, salesperson, status, branch, warehouse. |
| `security` | Company DB, user/role, branch, warehouse, field masking. |
| `dialect_support` | SQL Server and HANA SQL generation paths. |
| `examples` | "Monthly sales by customer", "Open order amount by warehouse". |
| `quality_rules` | Currency handling, canceled document exclusion, tax inclusion policy. |

### Priority Semantic Objects

| Priority | Semantic object | Reason |
| ---: | --- | --- |
| 1 | `SalesOrderLine` | Foundation for sales order metrics and order status. |
| 1 | `ARInvoiceLine` | Foundation for realized revenue and customer profitability. |
| 1 | `DocumentFlow` | Required for order -> delivery -> invoice lineage and fulfillment analysis. |
| 1 | `InventoryPosition` | Required for availability, committed quantity, and ATP-like checks. |
| 1 | `BusinessPartner` / `Customer360` | Required for customer lookup, lifecycle, credit, AR, and segmentation. |
| 1 | `Item` / `Item360` | Required for item lookup, sales/purchase/inventory constraints, and pricing. |
| 2 | `PriceResolution` | Required to explain price list, special price, historical price, currency, tax, and UOM sources. |
| 2 | `AR_Aging` / `AP_Aging` | Required for risk, overdue, and credit exposure use cases. |
| 2 | `PurchaseFlow` | Required for PO -> receipt -> AP invoice analysis. |
| 3 | `ProductionFlow` | Required for production progress and variance analysis. |

## Proposed B1 Semantic APIs

These endpoints are discussion candidates, not implementation commitments.

| Endpoint | Purpose |
| --- | --- |
| `GET /semantic/catalog` | List semantic objects, metrics, dimensions, joins, examples, and permission scopes. |
| `GET /semantic/catalog/{object}` | Return full object metadata, lineage, allowed filters/group-bys, and examples. |
| `POST /semantic/query` | Execute governed analytical query by object/metrics/dimensions/filters. |
| `POST /semantic/explain` | Explain generated SQL, source lineage, metric formula, and access filters. |
| `POST /semantic/validate-query` | Validate query shape without execution. |
| `GET /semantic/lineage/{object}` | Show physical table/view lineage and dependent metrics. |
| `POST /semantic/suggest` | Convert natural-language intent into candidate semantic query plans, without execution by default. |

## Design Principle

For AI, the semantic layer should be the default read boundary. Raw SQL and raw Service Layer should remain escape hatches for expert/debug workflows, not the primary tool surface.
