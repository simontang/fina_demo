import z from "zod";
import { registerToolLattice } from "@axiom-lattice/core";
import {
  createRuntime,
  updateRuntime,
  appendEvent,
} from "../runtimeStore";

const schema = z.object({
  agentId: z.string().describe("当前 agent 的 key"),
  threadId: z.string().describe("当前对话 thread ID"),
  activeRuntime: z.string().optional().describe("运行时标识，首次调用必填"),
  currentState: z.string().optional().describe("当前运行状态描述"),
  selectedArtifact: z.string().optional().describe("当前选中的产物名称"),
  nextAction: z.string().optional().describe("建议的下一步操作"),
  events: z
    .array(
      z.object({
        label: z.string().describe("事件描述"),
        status: z.enum(["completed", "in_progress", "pending"]).describe("事件状态"),
      })
    )
    .optional()
    .describe("追加到时间线的事件列表"),
});

type TaskAgentRuntimeUpdateInput = z.infer<typeof schema>;

registerToolLattice(
  "task_agent_runtime_update",
  {
    name: "task_agent_runtime_update",
    description:
      "写入或更新当前 task agent 的运行态，包括当前状态、选中的产物、建议下一步操作和时间线事件。" +
      "首次调用时需要提供 activeRuntime 和 currentState。后续调用只需提供变更的字段。",
    schema,
  },
  async (input: TaskAgentRuntimeUpdateInput) => {
    const { agentId, threadId, activeRuntime, currentState, selectedArtifact, nextAction, events } = input;

    const existing = updateRuntime(agentId, threadId, {});

    if (!existing) {
      if (!activeRuntime || !currentState) {
        return JSON.stringify({
          error: "首次调用需要提供 activeRuntime 和 currentState",
        });
      }
      createRuntime(agentId, threadId, {
        activeRuntime,
        currentState,
        selectedArtifact,
        nextAction,
      });
    } else {
      updateRuntime(agentId, threadId, {
        currentState,
        selectedArtifact,
        nextAction,
      });
    }

    if (events && events.length > 0) {
      for (const ev of events) {
        appendEvent(agentId, threadId, { label: ev.label, status: ev.status });
      }
    }

    return JSON.stringify({
      success: true,
      message: "Runtime state updated",
    });
  }
);
