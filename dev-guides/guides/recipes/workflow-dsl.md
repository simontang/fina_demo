# Recipe: Workflow DSL

Orchestrate multiple agent steps using the concise Workflow DSL.

## Overview

The Workflow DSL (`packages/protocols/src/WorkflowDSL.ts`) defines multi-step agent pipelines as declarative JSON. Steps are compiled into a LangGraph StateGraph automatically.

Template syntax: `{{input}}` = user input, `{{id}}` = output of step with that id, `{{item}}` = current element in map iterations.

## Step Types

| Type | Purpose | Key Fields |
|---|---|---|
| `AgentStep` | Invoke the workflow's agent | `prompt` (template with `{{id}}` refs), `schema?` |
| `ConditionStep` | Branch on state field/expression | `if` (expression), `then?`, `else?`, `branches?` |
| `HumanStep` | Pause for human input | `prompt`, `title?`, `schema?` |
| `MapStep` | Iterate over array, optionally reduce | `source` (step id), `each` (AgentStep), `reduce?` |
| `ParallelStep` | Run steps in parallel | `steps: WorkflowStep[]` |
| `EndStep` | Terminal state | `status?: "success" \| "failed"` |

## Configuration

```typescript
import { AgentType } from "@axiom-lattice/protocols";
import type { WorkflowAgentConfig } from "@axiom-lattice/protocols";
import { registerAgentLattice } from "@axiom-lattice/core";

const config: WorkflowAgentConfig = {
  type: AgentType.WORKFLOW,
  key: "my-workflow",
  name: "My Workflow",
  description: "A workflow agent",
  prompt: "Default prompt for workflow agents",
  modelKey: "azure-gpt-4o",
  workflow: {
    name: "my-workflow",
    steps: [ /* ... */ ],
  },
};

registerAgentLattice(config);
```

## Example 1: Linear Pipeline

```json
{
  "name": "knowledge-qa",
  "steps": [
    { "id": "researcher", "prompt": "Research: {{input}}" },
    { "id": "writer", "prompt": "Write based on {{researcher}}" },
    { "type": "end" }
  ]
}
```

## Example 2: Condition (if/else)

```json
{
  "name": "customer-routing",
  "steps": [
    { "id": "intent", "prompt": "Classify: {{input}}" },
    {
      "type": "condition", "if": "intent",
      "then": { "id": "support", "prompt": "Support: {{input}}" },
      "else": { "id": "sales", "prompt": "Sales: {{input}}" }
    },
    { "type": "end" }
  ]
}
```

## Example 3: Switch (multi-branch)

```json
{
  "name": "support-routing",
  "steps": [
    { "id": "intent", "prompt": "Classify intent: {{input}}" },
    {
      "type": "condition", "if": "intent",
      "branches": {
        "support": { "id": "support", "prompt": "Tech support: {{input}}" },
        "sales":   { "id": "sales",   "prompt": "Sales: {{input}}" },
        "billing": { "id": "billing", "prompt": "Billing: {{input}}" },
        "default": { "id": "fallback", "prompt": "Escalate to human: {{input}}" }
      }
    },
    { "type": "end" }
  ]
}
```

## Example 4: Condition with Expression

```json
{
  "name": "score-check",
  "steps": [
    { "id": "score", "prompt": "Score: {{input}}" },
    {
      "type": "condition", "if": "score >= 60",
      "then": [
        { "id": "congrats", "prompt": "Congratulations!" },
        { "type": "end" }
      ],
      "else": { "type": "end", "status": "failed" }
    }
  ]
}
```

## Example 5: Human Approval

```json
{
  "name": "approval-process",
  "steps": [
    { "id": "draft", "prompt": "Draft: {{input}}" },
    {
      "id": "review", "type": "human",
      "title": "Review",
      "prompt": "Review this draft and approve or reject:\\n{{draft}}",
      "schema": {
        "type": "object",
        "properties": {
          "approved": { "type": "boolean" },
          "comments": { "type": "string" }
        }
      }
    },
    {
      "type": "condition", "if": "review.approved",
      "then": { "prompt": "Publish: {{draft}}" },
      "else": { "prompt": "Revise: {{review.comments}}" }
    },
    { "type": "end" }
  ]
}
```

## Example 6: Parallel Fan-Out

```json
{
  "name": "due-diligence",
  "steps": [
    { "id": "info", "prompt": "Gather: {{input}}" },
    {
      "type": "parallel", "steps": [
        { "id": "legal",   "prompt": "Legal: {{info}}" },
        { "id": "finance", "prompt": "Finance: {{info}}" },
        { "id": "market",  "prompt": "Market: {{info}}" }
      ]
    },
    { "id": "report", "prompt": "Summary: {{legal}} {{finance}} {{market}}" },
    { "type": "end" }
  ]
}
```

## Example 7: Map-Reduce

```json
{
  "name": "batch-review",
  "steps": [
    { "id": "items", "prompt": "Extract items: {{input}}" },
    {
      "id": "results", "type": "map", "source": "items",
      "each": { "id": "auditor", "prompt": "Review: {{item}}" },
      "batch": 10, "concurrency": 3
    },
    { "id": "summary", "prompt": "Summarize: {{results}}" },
    { "type": "end" }
  ]
}
```

## Example 8: Full Orchestration

```json
{
  "name": "smart-support",
  "steps": [
    { "id": "intent", "prompt": "Classify: {{input}}" },
    {
      "type": "condition", "if": "intent",
      "then": [
        { "id": "kb", "prompt": "Search KB: {{input}}" },
        {
          "type": "condition", "if": "kb.confidence > 0.8",
          "then": { "id": "reply", "prompt": "Direct reply: {{kb}}" },
          "else": [
            { "id": "pre_merge", "prompt": "Prepare merge" },
            {
              "type": "parallel", "steps": [
                { "id": "faq",  "prompt": "FAQ: {{input}}" },
                { "id": "hist", "prompt": "History: {{input}}" }
              ]
            },
            { "id": "merged", "prompt": "Merge: FAQ={{faq}} History={{hist}}" },
            {
              "id": "review", "type": "human",
              "title": "Escalate",
              "prompt": "Handle:\\n{{merged}}",
              "schema": { "type": "object", "properties": { "action": { "type": "string" } } }
            },
            { "id": "response", "prompt": "Reply: {{review}}" }
          ]
        }
      ],
      "else": { "id": "clarify", "prompt": "Ask user to clarify" }
    },
    { "type": "end" }
  ]
}
```

## Gotchas

- `WorkflowDSL` has `name` and `steps` (not `version` and `steps`)
- Use `{{id}}` template syntax in `prompt` strings, NOT `{ $ref: "id.output" }`
- `ConditionStep` uses `if`/`then`/`else`/`branches`, NOT `input.$ref`/`conditions[].when`/`goto`
- `MapStep` uses `source` (step id string) and `each` (AgentStep), NOT `items.$ref`
- `AgentStep` has `prompt` and optional `schema`, NOT `agent`/`input`/`description`
- `AgentConfig.type` is `AgentType.WORKFLOW` = `"workflow"` (lowercase)
- The agent field is `modelKey`, not `llm`
- All these examples are from the framework source: `packages/protocols/src/WorkflowDSL.ts`
