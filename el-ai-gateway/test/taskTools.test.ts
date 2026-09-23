import { describe, expect, it, vi } from "vitest";
import { createTaskToolClient } from "../src/upstream/taskTools";
import type { McpCaller } from "../src/upstream/mcp";

function callerReturning(text: string): McpCaller {
  return {
    callTool: vi.fn(async () => ({ text, isError: false })),
  };
}

describe("taskTools", () => {
  it("creates a task with an explicit owner and returns its id", async () => {
    const caller = callerReturning(
      JSON.stringify({ success: true, data: { id: "task-1", status: "in_progress" } }),
    );
    const tools = createTaskToolClient(caller);
    const out = await tools.createTask({
      title: "Voice tagging",
      status: "in_progress",
      ownerId: "tenant_a",
      metadata: { uuid: "u" },
    });
    expect(out.taskId).toBe("task-1");
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "create",
      title: "Voice tagging",
      description: undefined,
      status: "in_progress",
      ownerType: "user",
      ownerId: "tenant_a",
      metadata: { uuid: "u" },
    });
  });

  it("fails when create returns no id", async () => {
    const caller = callerReturning(JSON.stringify({ success: true, data: {} }));
    const tools = createTaskToolClient(caller);
    await expect(tools.createTask({ title: "x", ownerId: "t" })).rejects.toMatchObject({
      statusCode: 502,
      code: "UPSTREAM_ERROR",
    });
  });

  it("reads a task and its activities", async () => {
    const caller = callerReturning(
      JSON.stringify({
        success: true,
        data: { task: { id: "t1", status: "completed", title: "T" }, activities: [{ id: "a1" }] },
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

  it("lists tasks by metadata filter (baId + customerId)", async () => {
    const caller = callerReturning(
      JSON.stringify({
        success: true,
        data: [
          {
            id: "t1",
            status: "completed",
            title: "T",
            metadata: { uuid: "u1", baId: "ba_001", customerId: "cus_8899" },
            result: "[]",
            createdAt: "2026-09-15T06:13:00Z",
          },
        ],
        count: 1,
      }),
    );
    const tools = createTaskToolClient(caller);
    const tasks = await tools.listTasks({ ownerId: "tenant_a", baId: "ba_001", customerId: "cus_8899" });
    expect(tasks).toHaveLength(1);
    expect(tasks[0]).toMatchObject({ id: "t1", status: "completed", metadata: { uuid: "u1" } });
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", {
      action: "list",
      ownerType: "user",
      ownerId: "tenant_a",
      metadataFilter: { baId: "ba_001", customerId: "cus_8899" },
    });
  });

  it("returns an empty array when list has no tasks", async () => {
    const caller = callerReturning(JSON.stringify({ success: true, data: [], count: 0 }));
    const tools = createTaskToolClient(caller);
    await expect(
      tools.listTasks({ ownerId: "tenant_a", baId: "ba_x", customerId: "cus_y" }),
    ).resolves.toEqual([]);
  });

  it("deletes a task by id", async () => {
    const caller = callerReturning(JSON.stringify({ success: true, data: { deleted: ["t1"] } }));
    const tools = createTaskToolClient(caller);
    await expect(tools.deleteTask({ id: "t1" })).resolves.toEqual({ raw: { deleted: ["t1"] } });
    expect(caller.callTool).toHaveBeenCalledWith("task_manage_task", { action: "delete", id: "t1" });
  });
});
