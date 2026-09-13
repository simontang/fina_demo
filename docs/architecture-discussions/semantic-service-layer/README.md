# Semantic Layer and Service Layer Architecture Discussion

Last refreshed: 2026-06-01

This directory is for architecture discussion around two related but different layers:

- Semantic layer for governed read/query/analytics use cases.
- AI-friendly service layer for business actions, command APIs, MCP tools, and CLI workflows.

## Directory Map

| Path | Purpose |
| --- | --- |
| `B1_AI_SEMANTIC_LAYER_ARCHITECTURE.md` | Semantic layer architecture overview: model, runtime, governance, overlays, and metadata APIs. |
| `B1_AI_SEMANTIC_LAYER_CAPABILITY_REQUIREMENTS.md` | Standalone outbound capability and architecture requirements for a B1 AI-ready semantic layer. |
| `B1_AI_SEMANTIC_LAYER_CAPABILITY_REQUIREMENTS.html` | HTML rendering of the semantic layer requirements, including architecture diagrams. |
| `research/semantic-layer-architecture.md` | Notes from Cube, dbt Semantic Layer, and local B1 requirements. |
| `research/ai-service-layer-mcp-cli-practices.md` | MCP, CLI, Salesforce, dbt, Cube, and GitHub patterns for AI-facing tool/API wrappers. |
| `notes/b1-semantic-service-layer-design-questions.md` | Concrete design questions for the B1 semantic/service layer discussion. |
| `refs/cubejs-local-map.md` | Local map of the cloned Cube.js repository. |
| `cubejs/` | Shallow clone of `https://github.com/cube-js/cube.git`; intentionally ignored by git. |

## Local Reference Clone

Cube.js was cloned locally for code and documentation inspection:

```bash
git clone --depth 1 https://github.com/cube-js/cube.git docs/architecture-discussions/semantic-service-layer/cubejs
```

The `cubejs/` folder is excluded by this directory's `.gitignore` and should not be submitted with this repository.

## How To Use This Folder

Use `research/` for external architecture/practice notes, `notes/` for our design decisions and open questions, and `refs/` for lightweight indexes into cloned or downloaded source material.

The existing broader B1 requirement document remains at `docs/B1_AI_SEMANTIC_LAYER_TARGET_ARCHITECTURE.md`; this folder is a more focused workspace for discussion and source-backed design comparison.
