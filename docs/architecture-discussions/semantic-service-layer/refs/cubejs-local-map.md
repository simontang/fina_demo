# Cube.js Local Clone Map

Last refreshed: 2026-06-01

Clone location:

```text
docs/architecture-discussions/semantic-service-layer/cubejs
```

Remote:

```text
https://github.com/cube-js/cube.git
```

Observed revision:

```text
b2c02a4
```

## Files Inspected

| Path | Notes |
| --- | --- |
| `README.md` | Positions Cube Core as an open-source semantic layer exposing metrics, dimensions, joins, access rules through SQL/REST/GraphQL for BI, apps, and AI agents. |
| `docs/content/product/introduction.mdx` | High-level architecture: data modeling, access control, caching, APIs, meta API, Semantic SQL, agentic analytics. |
| `docs/content/product/data-modeling/overview.mdx` | Data modeling tutorial: cubes, measures, dimensions, reusable SQL generation. |
| `docs/content/product/apis-integrations/core-data-apis/index.mdx` | API selection: SQL, DAX, REST, GraphQL, MCP, semantic layer sync, auth methods. |
| `docs/content/product/apis-integrations/mcp-server.mdx` | Remote and local MCP setup, OAuth, Codex/Claude/Cursor examples, available workflows. |
| `docs/content/product/auth/data-access-policies.mdx` | Access policies for row-level, member-level, and masking behavior. |

## Directories To Inspect Next

| Path | Why |
| --- | --- |
| `docs/content/product/data-modeling/reference/` | Full DSL reference for B1 semantic object schema design. |
| `docs/content/product/data-modeling/concepts/` | Joins, multi-fact queries, data blending, multi-stage calculations. |
| `docs/content/product/caching/` | Pre-aggregation design, refresh, matching, production operations. |
| `docs/content/product/apis-integrations/core-data-apis/` | Query protocol design and common query concepts. |
| `packages/cubejs-server-core/` | Server runtime and orchestration implementation. |
| `rust/cubesql/` | SQL API/runtime implementation. |
| `rust/cube/cubesqlplanner/` | Query planning internals. |

## Git Hygiene

`cubejs/` is ignored by:

```text
docs/architecture-discussions/semantic-service-layer/.gitignore
```

Do not stage or commit the cloned source tree.
