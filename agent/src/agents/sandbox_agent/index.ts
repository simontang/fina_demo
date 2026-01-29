import {
  registerAgentLattices,
  AgentType,
  AgentConfig,
  toolLatticeManager,
} from "@axiom-lattice/core";
import z from "zod";

const sandboxPrompt = ``;

//setTimeout(() => {

// const tools = toolLatticeManager.getAll().map((lattice) => lattice.config.name);
const sandboxAgent: AgentConfig = {
  key: "sandbox_agent",
  name: "Sandbox Agent",
  description:
    "A sandbox agent for testing and development.",
  type: AgentType.DEEP_AGENT,
  prompt: sandboxPrompt,
  connectedSandbox: {
    isolatedLevel: "global",
    //  availabledModules: ["filesystem"],
  },
};

registerAgentLattices([sandboxAgent]);

//}, 10000);
