jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { describe, expect, it } from "@jest/globals";
import { semanticMetricsPlugin } from "../plugin";
import { SEMANTIC_METRICS_MODELING_SKILL } from "../skill";

describe("semantic-metrics-modeling skill (delegation SOP)", () => {
  const { content, version, resources } = SEMANTIC_METRICS_MODELING_SKILL;

  it("is version 1.1.2 and declared on the plugin", () => {
    expect(version).toBe("1.1.2");
    const skill = semanticMetricsPlugin.skills?.["semantic-metrics-modeling"];
    expect(skill).toBeDefined();
    expect(skill?.version).toBe("1.1.2");
    expect(content.length).toBeGreaterThan(1000);
  });

  it("declares frontmatter name and description matching the skill key", () => {
    expect(content.startsWith("---")).toBe(true);
    expect(content).toContain("name: semantic-metrics-modeling");
    expect(content).toContain("description:");
  });

  it("defines the orchestrator/worker delegation contract", () => {
    expect(content).toContain("Delegation Protocol");
    expect(content).toContain('agentId: "general-purpose"');
    expect(content).toContain("## Bound Task");
  });

  it("makes the orchestrator the sole belief writer and bans inline exploration", () => {
    expect(content).toContain("you are the sole writer");
    expect(content).toContain("Never run exploration SQL inline");
  });

  it("tells explore workers to prefer targeted reads and compress bulk dumps", () => {
    expect(content).toContain("Prefer targeted queries over bulk catalog dumps");
    expect(content).toContain("findings must be compressed to conclusion level");
  });

  it("keeps the payload reference sections unnumbered", () => {
    expect(content).toContain("## Publish table meta (`create_table`)");
    expect(content).toContain("## Publish metric meta (`create_metric`)");
    expect(content).not.toContain("## 2. Publish");
    expect(content).not.toContain("## 3. Publish");
    expect(content).not.toContain("## 4.");
  });

  it("replaces the old inline-activity task policy and leaks no placeholder", () => {
    expect(content).not.toContain("activities inside the active task");
    expect(content).not.toContain("保持原文不变");
  });

  it("keeps content aligned with the spec workflow and DSL sections", () => {
    expect(content).toContain("## Workflow");
    expect(content).toContain('"role": "dimension"');
    expect(content).toContain('"calculation"');
    expect(content).toContain("aggregate");
    expect(content).toContain("derived");
    expect(content).toContain("TABLE_NOT_GRANTED");
    expect(content).toContain("## Goal Model");
    expect(content).toContain("Vague requests");
    expect(content).toContain("cheap read-only observations");
    expect(content).toContain('"role": "time"');
    expect(content).toContain("table_catalog");
    expect(content).toContain("search_keywords");
    expect(content).toContain("ai_agent_context");
    expect(content).toContain("## Target model and quality bar");
    expect(content).toContain("## FEP Loop");
    expect(content).toContain("## Orchestrator: Task Policy");
    expect(content).toContain("## Belief State");
    expect(content).toContain("## Unclear field semantics");
    expect(content).toContain("INCONCLUSIVE");
  });

  it("ships four resources under safe relative paths", () => {
    const paths = Object.keys(resources ?? {});
    expect([...paths].sort()).toEqual([
      "examples/metric-aggregate.json",
      "examples/metric-derived.json",
      "examples/table-meta.json",
      "references/metadata-sql.md",
    ]);
    for (const p of paths) {
      expect(p.startsWith("/")).toBe(false);
      expect(p.includes("..")).toBe(false);
    }
    for (const r of Object.values(resources)) {
      expect(r.content.length).toBeGreaterThan(0);
    }
    expect(resources["examples/metric-aggregate.json"].content).toContain("aggregate");
    expect(resources["examples/metric-derived.json"].content).toContain("derived");
    expect(resources["examples/table-meta.json"].content).toContain("table_view_detail");
    expect(resources["references/metadata-sql.md"].content).toContain("PostgreSQL");
  });

  it("ships metric examples in the queryable full runtime style", () => {
    const res = resources ?? {};
    const metricExamples = [
      "examples/metric-aggregate.json",
      "examples/metric-derived.json",
    ] as const;
    for (const key of metricExamples) {
      const example = JSON.parse(res[key].content) as {
        objectType?: unknown;
        objectKey?: unknown;
        status?: unknown;
        payload?: Record<string, unknown>;
      };
      const payload = example.payload ?? {};
      expect(example.objectType).toBe("metric_detail");
      expect(example.status).toBe(1);
      expect(payload.metric_name).toBe(example.objectKey);
      for (const required of [
        "metric_name", "display_name", "domain", "description", "data_type", "format",
        "source_type", "source", "calculation", "supported_dimensions",
        "default_time_context", "ai_agent_context",
      ]) {
        expect(payload).toHaveProperty(required);
      }
      for (const forbidden of ["name", "displayName", "sourceTable", "dimensions"]) {
        expect(payload).not.toHaveProperty(forbidden);
      }
      const source = payload.source as { table_view?: unknown };
      expect(typeof source?.table_view).toBe("string");
      expect(String(source.table_view)).toContain(".");
      const dims = payload.supported_dimensions as Array<Record<string, unknown>>;
      expect(Array.isArray(dims) && dims.length > 0).toBe(true);
      for (const dim of dims) {
        for (const field of ["dim_id", "field_name", "label", "data_type"]) {
          expect(typeof dim[field]).toBe("string");
        }
      }
      const time = payload.default_time_context as {
        time_dimension?: unknown;
        granularity?: unknown;
        supported_grains?: unknown;
      };
      expect(typeof time?.time_dimension).toBe("string");
      expect(dims.some((dim) => dim.field_name === time.time_dimension)).toBe(true);
      expect(Array.isArray(time?.supported_grains)).toBe(true);
      const ai = payload.ai_agent_context as Record<string, unknown>;
      expect(typeof ai?.polarity).toBe("string");
      expect(Array.isArray(ai?.synonyms)).toBe(true);
      expect(typeof ai?.human_readable_explanation).toBe("string");
    }
    const aggregate = JSON.parse(res["examples/metric-aggregate.json"].content) as {
      payload?: { calculation?: Record<string, unknown> };
    };
    expect(aggregate.payload?.calculation).toMatchObject({ type: "aggregate" });
    expect(aggregate.payload?.calculation?.aggregation).toBeDefined();
    expect(aggregate.payload?.calculation?.measure).toBeDefined();
    const derived = JSON.parse(res["examples/metric-derived.json"].content) as {
      payload?: { calculation?: Record<string, unknown> };
    };
    expect(derived.payload?.calculation).toMatchObject({ type: "derived", operator: "ratio" });
    expect(derived.payload?.calculation?.numerator).toBeDefined();
    expect(derived.payload?.calculation?.denominator).toBeDefined();
  });
});
