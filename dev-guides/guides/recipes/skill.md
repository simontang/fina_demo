# Recipe: Creating a SKILL.md

Skills provide specialized instructions and tools to agents through markdown files with frontmatter.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | `skills/my-skill/SKILL.md` | Create skill definition |
| 2 | `skills/my-skill/` | Add optional resource files |
| 3 | Agent config or store | Register skill with agent |

## Step 1: Create SKILL.md

```markdown
---
name: my-data-analyzer
description: >
  Data analysis skill for CSV and JSON files. Use when the user
  asks about data analysis, statistics, or chart generation.
version: 1.0.0
author: Your Name
license: MIT
compatibility: [REACT, DEEP_AGENT, TEAM]
tags: [data, analysis, statistics, visualization]
---

# Data Analyzer Skill

## Purpose
Analyze structured data files (CSV, JSON) and generate insights.

## Capabilities
- Read and parse CSV and JSON files
- Calculate descriptive statistics (mean, median, std deviation)
- Generate data visualizations (charts, graphs)
- Identify trends and outliers
- Export results to CSV or markdown tables

## Instructions
When the user provides a data file:
1. First, examine the file structure (columns, data types, row count)
2. Ask the user what analysis they need if not specified
3. Run the analysis
4. Present results in a clear format with visualizations when helpful

## Tools
- `read_file` — to read the data file
- `execute_python` — to run analysis scripts

## Example Usage
User: "Analyze this sales data for trends"
→ Read the file, check structure, run statistical analysis, generate charts.
```

## Step 2: Place Skill Files

```
skills/
  my-data-analyzer/
    SKILL.md           ← Skill definition (required)
    analyze.py         ← Optional: bundled scripts
    chart_template.py  ← Optional: reusable templates
    README.md          ← Optional: human documentation
```

## Step 3: Register Skill

### Option A: FileSystem Skill Store (development)

```typescript
import { FileSystemSkillStore } from "@axiom-lattice/core";

const skillStore = new FileSystemSkillStore({
  rootPath: "./skills",
});

configureStores({ skill: skillStore });
```

### Option B: Sandbox Skill Store (production — skills stored in sandbox volume)

```typescript
import { SandboxSkillStore } from "@axiom-lattice/core";

const skillStore = new SandboxSkillStore({
  volumeClient: volumeFsClient,
});

configureStores({ skill: skillStore });
```

### Option C: PostgreSQL Skill Store (production)

```typescript
import { createPgStoreConfig } from "@axiom-lattice/pg-stores";

const stores = createPgStoreConfig(process.env.DATABASE_URL!);
configureStores(stores);
// Skills are managed via API: POST /api/skills
```

## Step 4: Attach to Agent

```typescript
const config: ReactAgentConfig = {
  type: AgentType.REACT,
  key: "data-analyst",
  name: "Data Analyst",
  modelKey: "azure-gpt-4o",
  prompt: "...",
  skillCategories: ["my-data-analyzer"],  // Skill categories to load
  // Skills matching these categories are injected into agent context
};
registerAgentLattice(config);
```

## Frontmatter Reference

| Field | Required | Type | Description |
|---|---|---|---|
| `name` | Yes | string | Unique skill identifier |
| `description` | Yes | string | When to use this skill |
| `version` | No | string | Semver version |
| `author` | No | string | Skill author |
| `license` | No | string | License identifier |
| `compatibility` | No | string[] | Compatible agent types |
| `tags` | No | string[] | Searchable tags |
| `requires` | No | string[] | Required tools or dependencies |

## Skill API

```bash
# Upload skill
curl -X POST http://localhost:4001/api/skills \
  -F "file=@SKILL.md" \
  -F "name=my-data-analyzer"

# List skills
curl http://localhost:4001/api/skills

# Get skill by ID
curl http://localhost:4001/api/skills/:id

# Update skill
curl -X PUT http://localhost:4001/api/skills/:id \
  -F "file=@SKILL.md"

# Delete skill
curl -X DELETE http://localhost:4001/api/skills/:id
```

## Gotchas

- `skillParser` handles parsing SKILL.md frontmatter at `packages/core/src/skill_lattice/`
- Skills are loaded into agent context — keep SKILL.md concise (under 2000 words)
- Skill `description` field is critical — the LLM uses it to decide when to activate the skill
- Skills with the same `name` will overwrite at registration time
