# B1 Semantic and Service Layer Design Questions

Last refreshed: 2026-06-01

## Decisions To Make

### 1. Semantic Layer Runtime Boundary

Should B1 provide:

- official physical SQL/HANA views,
- a metadata manifest that partners generate into views,
- a hosted semantic query API,
- or all three?

Working recommendation: expose a versioned semantic manifest and a hosted/runtime query API. Physical views can be implementation artifacts, but the contract should be semantic object + metric metadata + query API.

### 2. Fact Model Grain

For each core fact, what is the grain?

| Fact | Proposed grain |
| --- | --- |
| `SalesOrderLine` | `ORDR.DocEntry + RDR1.LineNum` |
| `DeliveryLine` | `ODLN.DocEntry + DLN1.LineNum` |
| `ARInvoiceLine` | `OINV.DocEntry + INV1.LineNum` |
| `PurchaseOrderLine` | `OPOR.DocEntry + POR1.LineNum` |
| `InventoryPosition` | `ItemCode + WhsCode + Batch/Serial optional` |
| `DocumentFlow` | source document line -> target document line edge |
| `PriceResolution` | BP/item/date/quantity/UOM/currency price resolution event |

### 3. Cross-Dialect Contract

Should semantic queries compile to SQL Server and HANA from one logical contract?

Working recommendation: yes. The current `b1s` evidence already shows why AI should not own dialect translation. Dialect-specific SQL belongs in the runtime/compiler.

### 4. Metadata Needed By AI

For each object/member, do we have:

- business label and description,
- synonyms,
- examples,
- allowed filters and group-bys,
- default time dimension,
- source table lineage,
- currency/tax/UOM semantics,
- null/unknown handling,
- access scope,
- common mistakes?

Working recommendation: treat this as product metadata, not prompt-only documentation.

### 5. Service Layer Command Boundary

Which operations must become first-class commands rather than raw OData calls?

Recommended first wave:

| Command | Why first |
| --- | --- |
| `search_business_partner` | Prevents brittle BP lookup and ambiguous matches. |
| `search_item` | Handles item name/spec/barcode/customer item code and sales/purchase flags. |
| `resolve_price` | Explains price list/special price/history/currency/UOM. |
| `validate_sales_order_request` | Converts raw errors into preflight warnings/errors. |
| `create_sales_order_draft` | Safe write path after validation. |
| `trace_document_flow` | Common cross-document question; hard to do with raw API calls. |

### 6. Human Approval Model

Which command outputs require a human confirmation token?

Working recommendation:

- Read/search/validate: no confirmation.
- Create draft: configurable; allowed automatically for low-risk scenarios after validation.
- Submit/post/convert to official document: require explicit confirmation unless workflow policy says otherwise.
- Master data mutation: require approval workflow by default.

### 7. Tool Surface Size

How many MCP tools should be visible at once?

Working recommendation: do not expose all raw Service Layer EntitySets. Use toolsets and dynamic loading:

- default: `semantic`, `master-data`, `sales-documents`
- optional: `inventory`, `pricing`, `purchasing`, `finance`
- restricted: `raw-service-layer`, `admin`

### 8. Audit and Debug

What should every command log?

- user identity,
- company DB,
- tool name,
- request ID / idempotency key,
- resolved entities,
- Service Layer calls made,
- semantic queries made,
- warnings/errors returned,
- final write object IDs,
- raw error hash, not necessarily full sensitive payload.

## Open Questions For SAP B1 / Product Discussion

1. Can B1 publish official semantic object definitions for both SQL Server and HANA?
2. Can Service Layer publish compact AI metadata packs by domain instead of only full OData metadata?
3. Can B1 provide validate/simulate APIs for documents before write?
4. Can document creation support idempotency keys?
5. Can errors be mapped to stable domain codes with fix hints?
6. Can B1 expose price resolution as an explainable service?
7. Can custom UDF/UDT/addon metadata be described in the same manifest?
8. Can permission scopes be surfaced in tool metadata so AI sees only usable operations?
9. Can semantic lineage expose physical source tables for audit without making AI query them directly?
10. Can remote MCP use per-user OAuth while local/dev MCP uses explicit environment credentials?
