# AI-Friendly Service Layer, MCP, and CLI Practices

Last refreshed: 2026-06-01

## Executive Takeaways

1. AI-facing APIs are moving away from "give the model all raw endpoints" toward curated tools with schemas, descriptions, permission boundaries, and user-scoped authentication.
2. MCP is becoming the common adapter layer for AI clients. The server decides which tools/resources/prompts to expose; the AI client discovers them dynamically.
3. Salesforce is a useful benchmark because it exposes SObjects, custom business logic, flows, query interfaces, and prompt templates through governed MCP servers rather than asking every AI client to hand-roll REST integrations.
4. CLI-backed MCP servers are useful for developer/operator workflows, but they need toolset gating and dynamic tool loading to avoid overwhelming the model context.
5. For B1, the best service layer shape is a command layer over raw Service Layer: validate/simulate first, create/update second, with evidence and structured domain errors.

## MCP Baseline

Sources:

- MCP introduction: https://modelcontextprotocol.io/docs/getting-started/intro
- MCP architecture: https://modelcontextprotocol.io/docs/learn/architecture

### Protocol Concepts That Matter

| MCP concept | Practical meaning for B1 |
| --- | --- |
| Host | The AI application, IDE, chat UI, or agent runtime. |
| Client | The connector inside the host that maintains one connection to a B1 MCP server. |
| Server | The B1 adapter exposing tools/resources/prompts. |
| Tools | Executable actions such as `search_business_partner`, `validate_document`, `create_sales_order_draft`. |
| Resources | Context data such as semantic catalog, B1 object metadata, field dictionary, current company DB profile. |
| Prompts | Reusable workflow prompts such as "validate incoming PO email" or "explain price mismatch". |
| Discovery | AI clients call list methods to discover tools/resources/prompts and their schemas. |
| Transport | Local stdio for developer machines; Streamable HTTP + OAuth for remote enterprise use. |
| Capability negotiation | Client/server agree what primitives and notifications are supported. |

The key architectural lesson: MCP is not a business API by itself. It is the adapter protocol. We still need carefully designed B1 business commands behind the MCP tool surface.

## Salesforce Hosted MCP Servers

Sources:

- Salesforce Hosted MCP Servers overview: https://developer.salesforce.com/docs/platform/hosted-mcp-servers/guide/hosted-mcp-servers-overview.html

### What Salesforce Wrapped For AI

Salesforce positions hosted MCP servers as a governed way for external AI agents to interact with Salesforce data and automation. The important AI-facing adjustments are:

| Pattern | Salesforce example | B1 design implication |
| --- | --- | --- |
| Standard connector | One MCP-compatible server can be used by Claude, ChatGPT, Cursor, custom agents, etc. | Build one B1 MCP surface instead of one connector per AI client. |
| Per-user OAuth | Agents act with the user's existing permissions. | B1 tools should bind to user/company DB/session permissions, not global service credentials. |
| Security perimeter | Data and logic remain under platform governance policies. | Service Layer access, field restrictions, branch/warehouse permissions, and audit logs stay server-side. |
| Raw data operations | SObject read/create/update/delete tools respect field-level security and sharing rules. | Raw B1 object operations can exist, but should be gated and scoped. |
| Custom tools | Apex Invocable Actions, Aura methods, Apex REST methods, Flows, Named Queries can be exposed as tools. | Allow partner/customer-specific B1 workflows to be registered as tool definitions without editing the AI client. |
| Product integrations | Data 360 SQL, Tableau analytics, and other Salesforce assets can be exposed. | Semantic queries, reports, dashboards, and B1 analytics should be tools/resources too. |
| Prompt templates | Prompt Builder templates can be used from MCP clients. | Package business workflows as prompts alongside tools. |

### B1 Equivalent

Instead of exposing `BusinessPartners`, `Items`, `Orders`, `Drafts`, and hundreds of OData fields directly, expose:

| AI intent | B1 MCP tool candidate |
| --- | --- |
| Find customer | `search_business_partner` |
| Resolve item from free text/customer SKU/barcode | `search_item` |
| Determine applicable price | `resolve_price` |
| Check if order can be created | `validate_sales_order_request` |
| Create draft after validation | `create_sales_order_draft` |
| Explain why create failed | `explain_service_layer_error` |
| Trace order lifecycle | `trace_document_flow` |
| Query governed analytics | `run_semantic_query` |

## Salesforce DX MCP Server

Sources:

- Salesforce CLI MCP repository: https://github.com/salesforcecli/mcp

### Useful Practices

Salesforce DX MCP is CLI-backed and aimed at developer/operator workflows. Important patterns:

| Practice | Why it matters |
| --- | --- |
| `npx -y @salesforce/mcp` packaging | Easy local install and consistent MCP config across clients. |
| Required `--orgs` scope | Forces explicit org authorization before tools can touch Salesforce. |
| `--toolsets` | Tools are grouped by capability (`orgs`, `metadata`, `data`, `users`, testing, etc.). |
| `--tools` | Individual tool allowlisting can be combined with toolsets. |
| Avoid "all tools" by default | Salesforce notes that too many tools can overwhelm LLM context. |
| `--dynamic-tools` | Starts with a smaller core and loads tools as needed, reducing initial context. |
| `--allow-non-ga-tools` | Experimental/non-GA tools are explicitly gated. |
| `--no-telemetry` / `--debug` | Operational controls are visible at configuration time. |

### B1 Equivalent

Suggested local B1 MCP CLI:

```json
{
  "mcpServers": {
    "B1": {
      "command": "npx",
      "args": [
        "-y",
        "@fina/b1-mcp",
        "--company-db",
        "SBODemoUS",
        "--toolsets",
        "semantic,master-data,sales-documents,inventory",
        "--tools",
        "search_business_partner,search_item,resolve_price,validate_sales_order_request,create_sales_order_draft",
        "--dynamic-tools"
      ]
    }
  }
}
```

Toolset candidates:

| Toolset | Tools |
| --- | --- |
| `semantic` | `catalog`, `run_semantic_query`, `explain_metric`, `trace_lineage`. |
| `master-data` | `search_business_partner`, `get_business_partner`, `search_item`, `get_item`. |
| `pricing` | `resolve_price`, `compare_price_sources`, `explain_price`. |
| `sales-documents` | `validate_sales_order_request`, `create_sales_order_draft`, `submit_draft`, `explain_document_status`. |
| `inventory` | `check_inventory_availability`, `suggest_alternative_warehouse`. |
| `raw-service-layer` | Raw OData escape-hatch tools, disabled by default. |
| `admin` | Metadata refresh, connection health, session diagnostics. |

## dbt MCP Server

Sources:

- dbt MCP server: https://docs.getdbt.com/docs/dbt-ai/about-mcp

### What dbt Wrapped For AI

dbt exposes project metadata, CLI runs, platform APIs, Discovery API, Semantic Layer queries, text-to-SQL, and SQL execution through an MCP server. It supports both local and remote server modes.

B1 implication:

- Local mode is appropriate for developers diagnosing metadata, SQL generation, and local test fixtures.
- Remote mode is appropriate for business users and production agents where OAuth, audit, and centralized policy enforcement are required.
- MCP tools should expose not only actions, but also metadata and lineage so the agent can reason before acting.

## Cube MCP Server

Sources:

- Local Cube doc: `cubejs/docs/content/product/apis-integrations/mcp-server.mdx`
- Cube docs: https://cube.dev/docs

Cube's MCP design is useful because it sits above the analytics/semantic layer, not beside it. The remote MCP server uses HTTPS and OAuth; the local server can use an API key for self-hosted or development use.

B1 implication:

- The B1 MCP server should not bypass the semantic layer for analytics.
- `run_semantic_query` should call the same `/semantic/query` service used by UI/BI consumers.
- Remote MCP should support OAuth/user identity; local MCP can support environment-variable credentials for development only.

## GitHub MCP Server

Sources:

- GitHub MCP server documentation: https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp-in-your-ide/use-the-github-mcp-server
- GitHub MCP server repository: https://github.com/github/github-mcp-server

Relevant pattern: GitHub documents MCP as a way to provide structured repository/platform context and tools to IDE agents. For B1, the comparable idea is to make the tool surface discoverable, scoped, and client-independent. The AI client should not need one-off prompt instructions for every B1 API nuance.

## AI-Friendly Service Layer Contract For B1

### Command Response Envelope

All business commands should return a consistent envelope:

```json
{
  "ok": true,
  "command": "validate_sales_order_request",
  "companyDb": "SBODemoUS",
  "inputEcho": {},
  "resolved": {},
  "data": {},
  "warnings": [],
  "blockingErrors": [],
  "evidence": [
    {
      "source": "BusinessPartners",
      "key": "C20000",
      "fields": ["CardCode", "CardName", "Frozen"]
    }
  ],
  "suggestedNextActions": [
    "create_sales_order_draft"
  ],
  "debug": {
    "serviceLayerCalls": [],
    "semanticQueries": []
  }
}
```

### Domain Errors

Map raw SQL/OData/Service Layer failures into stable domain errors:

| Error code | Meaning |
| --- | --- |
| `BP_NOT_FOUND` | Customer/vendor could not be resolved. |
| `BP_AMBIGUOUS` | Multiple business partners match. |
| `BP_FROZEN` | Business partner is blocked/frozen. |
| `ITEM_NOT_FOUND` | Item could not be resolved. |
| `ITEM_NOT_SALES_ITEM` | Item exists but cannot be sold. |
| `PRICE_NOT_RESOLVED` | No applicable price was found. |
| `CURRENCY_MISMATCH` | Requested/document currency conflicts with price/customer settings. |
| `INSUFFICIENT_STOCK` | Inventory cannot satisfy requested quantity. |
| `CONTACT_NOT_FOUND` | Required contact person could not be resolved. |
| `ADDRESS_NOT_FOUND` | Ship-to/bill-to address could not be resolved. |
| `SERVICE_LAYER_REJECTED` | Service Layer rejected the final payload after preflight. |

### Validate/Simulate Before Write

For AI action safety, write workflows should be two-step:

1. `validate_*` or `simulate_*` returns resolved entities, payload preview, warnings, and blocking errors.
2. `create_*` or `submit_*` accepts either the validated request ID or an explicit confirmation token.

This protects against ambiguous customer/item matches and turns Service Layer errors into user-understandable fixes.

## Design Principle

The AI-facing service layer should hide protocol mechanics and expose business intent. Service Layer remains the system of record API underneath, but AI should normally see curated, permission-aware, evidence-bearing command tools.
