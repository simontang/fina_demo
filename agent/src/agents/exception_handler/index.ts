import { registerAgentLattices, AgentType, type AgentConfig } from "@axiom-lattice/core";

const prompt = `You are an Exception Handler Agent.

When the user provides an error, you:
- Ask for the minimal missing context (logs, endpoint, payload, env vars)
- Propose likely root causes ranked by probability
- Provide step-by-step fixes and verification commands

Be concise, and prefer actionable troubleshooting steps.`;

const agents: AgentConfig[] = [
  {
    key: "exception_handler_agent",
    name: "Exception Handler Agent",
    type: AgentType.REACT,
    description: "Helps debug errors and provides troubleshooting steps.",
    prompt,
    tools: [],
  },
];

registerAgentLattices(agents);

