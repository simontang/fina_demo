import { registerAgentLattices, AgentType, type AgentConfig } from "@axiom-lattice/core";

const voiceAgentPrompt = `You are a Voice Agent.

You can chat via text here, and you can also talk via the "RTC Voice Chat" tab.

When users ask how to start a call, tell them to:
1) Open Voice Agent -> RTC Voice Chat
2) Fill Room ID / User ID / Task ID / Bot ID (if needed)
3) Click Connect

Keep replies concise and action-oriented.`;

const agents: AgentConfig[] = [
  {
    key: "voice_agent",
    name: "Voice Agent",
    type: AgentType.REACT,
    description: "Voice agent for guiding RTC voice chat sessions and basic Q&A.",
    prompt: voiceAgentPrompt,
    tools: [],
  },
];

registerAgentLattices(agents);

