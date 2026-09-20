jest.mock("@axiom-lattice/core", () => require("./coreMock").createCoreMock());

import { describe, expect, it } from "@jest/globals";
import { AgentType } from "@axiom-lattice/protocols";
import { semanticMetricsPlugin } from "../plugin";
import { SEMANTIC_METRICS_BUILDER_PROMPT } from "../prompt";

describe("semantic-metrics-builder agent", () => {
  it("is declared on the plugin as a deep agent with a modeling bootstrap prompt", () => {
    const config = semanticMetricsPlugin.agents?.["semantic-metrics-builder"];
    expect(config).toBeDefined();
    expect(config?.type).toBe(AgentType.DEEP_AGENT);
    expect(config?.prompt).toBe(SEMANTIC_METRICS_BUILDER_PROMPT);
    expect(SEMANTIC_METRICS_BUILDER_PROMPT).toContain("skill_name: \"semantic-metrics-modeling\"");
  });

  it("preconfigures the modeling middleware surface", () => {
    const config = semanticMetricsPlugin.agents?.["semantic-metrics-builder"];
    const middleware = config?.middleware ?? [];
    expect(middleware.map((m) => m.type)).toEqual([
      "semantic-metrics", "skill", "task", "ask_user_to_clarify", "filesystem",
    ]);

    const metrics = middleware.find((m) => m.type === "semantic-metrics");
    expect(metrics?.config).toMatchObject({ connections: [], connectAll: true });

    const skill = middleware.find((m) => m.type === "skill");
    expect(skill?.config).toEqual({ readAll: false, skills: ["semantic-metrics-modeling", "task-definition"] });
  });
});
