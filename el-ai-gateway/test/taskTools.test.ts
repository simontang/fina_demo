import { describe, expect, it, vi } from "vitest";
import { createTaskToolClient } from "../src/upstream/taskTools";
import type { McpCaller } from "../src/upstream/mcp";

function callerReturning(text: string): McpCaller {
  return {
    callTool: vi.fn(async () => ({ text, isError: false })),
  };
}

describe("taskTools", () => {
  it("creates a task and returns its id", async () => {
    const caller = callerReturning(
      JSON.stringify({ success: true, taskId: "task-1", task: { id: "task-1" } }),
    );
    const tools = createTaskToolClient(caller);
    const out = await tools.createTask({
      title: "Voice tagging",
      status: "in_progress",
      metadata: { uuid: "u" },
    });
    expect(out.taskId).toBe("task-1");
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "create",
      title: "Voice tagging",
      description: undefined,
      status: "in_progress",
      metadata: { uuid: "u" },
    });
  });

  it("reads a task and its activities", async () => {
    const caller = callerReturning(
      JSON.stringify({
        success: true,
        task: { id: "t1", status: "completed", title: "T" },
        activities: [{ id: "a1" }],
      }),
    );
    const tools = createTaskToolClient(caller);
    const task = await tools.getTask({ id: "t1" });
    expect(task).toMatchObject({ id: "t1", status: "completed", title: "T" });
    expect(task.activities).toEqual([{ id: "a1" }]);
  });

  it("maps a not-found task to 404", async () => {
    const caller = callerReturning(JSON.stringify({ success: false, error: "Task not found" }));
    const tools = createTaskToolClient(caller);
    await expect(tools.getTask({ id: "missing" })).rejects.toMatchObject({
      statusCode: 404,
      code: "NOT_FOUND",
    });
  });

  it("adds an activity", async () => {
    const caller = callerReturning(JSON.stringify({ success: true }));
    const tools = createTaskToolClient(caller);
    await tools.addActivity({ id: "t1", content: "tag is wrong", summary: "feedback" });
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "add_activity",
      id: "t1",
      content: "tag is wrong",
      summary: "feedback",
    });
  });
});
