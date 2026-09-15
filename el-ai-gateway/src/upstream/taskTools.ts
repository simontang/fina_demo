import { GatewayError } from "../lib/errors";
import type { McpCaller } from "./mcp";

const TOOL = "task_manage_task";

export type TaskRecord = {
  id: string;
  status?: string;
  title?: string;
  result?: string;
  activities: unknown[];
  raw: unknown;
};

export type TaskToolClient = {
  createTask(input: {
    title: string;
    description?: string;
    status?: string;
    metadata?: Record<string, unknown>;
  }): Promise<{ taskId: string; raw: unknown }>;
  getTask(input: { id: string }): Promise<TaskRecord>;
  addActivity(input: { id: string; content: string; summary?: string }): Promise<{ raw: unknown }>;
};

function parseResult(text: string, structured: unknown): any {
  if (structured && typeof structured === "object") return structured;
  try {
    return JSON.parse(text);
  } catch {
    return { rawText: text };
  }
}

export function createTaskToolClient(mcp: McpCaller): TaskToolClient {
  async function invoke(args: Record<string, unknown>): Promise<any> {
    const { text, structured } = await mcp.callTool(TOOL, args);
    const data = parseResult(text, structured);
    if (data && (data.success === false || data.error)) {
      const message = String(data.error ?? "task tool failed");
      if (/not found|does not exist/i.test(message)) {
        throw new GatewayError(404, "NOT_FOUND", message);
      }
      throw new GatewayError(502, "MCP_ERROR", message);
    }
    return data;
  }

  return {
    async createTask(input) {
      const data = await invoke({
        action: "create",
        title: input.title,
        description: input.description,
        status: input.status ?? "in_progress",
        metadata: input.metadata,
      });
      const taskId = (data?.taskId ?? data?.task?.id ?? data?.id) as string | undefined;
      if (!taskId) throw new GatewayError(502, "MCP_ERROR", "create task returned no id");
      return { taskId, raw: data };
    },

    async getTask({ id }) {
      const data = await invoke({ action: "get", id });
      const task = data?.task ?? data ?? {};
      return {
        id: task.id ?? id,
        status: task.status,
        title: task.title,
        result: task.result,
        activities: data?.activities ?? task.activities ?? [],
        raw: data,
      };
    },

    async addActivity({ id, content, summary }) {
      return { raw: await invoke({ action: "add_activity", id, content, summary }) };
    },
  };
}
