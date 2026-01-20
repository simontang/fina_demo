import { registerAgentLattices, AgentType, type AgentConfig } from "@axiom-lattice/core";

const prompt = `You are a combined "Research + Data" assistant.

- For deep research, you may suggest using the Deep Research Agent page.
- For SQL / dataset questions, you may suggest using the Data Agent page.

If the user asks something you can answer directly, answer directly.`;

const agents: AgentConfig[] = [
  {
    // Keep this key for backward compatibility with the existing UI route.
    key: "Research_data_agent",
    name: "Research Data Agent",
    type: AgentType.REACT,
    description: "A combined research + data helper (legacy UI entry).",
    prompt,
    tools: [],
  },
];

registerAgentLattices(agents);

