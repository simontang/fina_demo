import { FastifyRequest, FastifyReply } from "fastify";
import { AgentConfig, getAllAgentConfigs } from "@axiom-lattice/core";
/**
 * Agent Controller
 * Handles agent-related operations for admin panel
 */

/**
 * Agent list response interface
 */
interface AgentListResponse {
  success: boolean;
  message: string;
  data: {
    records: AgentConfig[];
    total: number;
  };
}

/**
 * Agent response interface
 */
interface AgentResponse {
  success: boolean;
  message: string;
  data?: AgentConfig;
}

/**
 * Get list of all available agents
 * This endpoint returns all registered agents in the system
 * Fetches agents dynamically from the Agent Service using AgentLattices manager
 */
export async function getAgentList(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<AgentListResponse> {
  const agentConfigs = await getAllAgentConfigs();
  return {
    success: true,
    message: "Successfully retrieved agent list",
    data: {
      records: agentConfigs,
      total: agentConfigs.length,
    },
  };
}

/**
 * Get a single agent by ID
 */
export async function getAgent(
  request: FastifyRequest<{ Params: { id: string } }>,
  reply: FastifyReply
): Promise<AgentResponse> {
  const { id } = request.params;
  const agentConfigs = await getAllAgentConfigs();
  const agent = agentConfigs.find((a) => a.key === id);

  if (!agent) {
    return reply.status(404).send({
      success: false,
      message: "Agent not found",
    });
  }

  return {
    success: true,
    message: "Successfully retrieved agent",
    data: agent,
  };
}
