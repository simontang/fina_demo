import { getTag } from "./customerTags";

export type TaskTag = { tagId: string; name: string; dimension: string };

export type TaskSummary = {
  taskId: string;
  fileId: string;
  status: string;
  createdAt: string;
  tags: TaskTag[];
};

function tags(ids: string[]): TaskTag[] {
  return ids.map((id) => {
    const def = getTag(id);
    return { tagId: id, name: def?.name ?? id, dimension: def?.dimension ?? "unknown" };
  });
}

/**
 * Mock task store keyed by `baId|customerId`. Replace with the real task/tag
 * store later — the API contract stays.
 */
const TASKS: Record<string, TaskSummary[]> = {
  "ba_001|cus_8899": [
    {
      taskId: "c3915a5a-85ed-4e31-a09e-492b3c11e938",
      fileId: "471c20082b524316accc1b23cba8a4de",
      status: "completed",
      createdAt: "2026-09-15T06:13:00Z",
      tags: tags([
        "9ce355bfacca49c4a9e9322a9317c196",
        "4a1b2c3d5e6f47089a0b1c2d3e4f5061",
        "a1b2c3d4e5f6470899aabbccddeeff00",
      ]),
    },
    {
      taskId: "b3ca978f-3832-4a2c-959b-40fa48c43352",
      fileId: "2ccf6fef88b64a16b62fe491a8f7a132",
      status: "in_progress",
      createdAt: "2026-09-15T05:18:00Z",
      tags: tags(["2f0a7d1c6b4e48a2b3c5d6e7f8091a2b"]),
    },
  ],
  "ba_002|cus_8899": [
    {
      taskId: "8e1e06e6-c704-4d56-8170-862f25aeca4d",
      fileId: "e72d3ea86f9e4d90b13600d6fcda7475",
      status: "completed",
      createdAt: "2026-09-14T09:30:00Z",
      tags: tags(["6f5e4d3c2b1a40799887aabbccddeeff"]),
    },
  ],
  "ba_001|cus_1001": [
    {
      taskId: "03f5636d-db6b-4ed6-bd97-41664d4205ee",
      fileId: "438620af520b4ef7a51806fc14ccea32",
      status: "failed",
      createdAt: "2026-09-14T08:00:00Z",
      tags: [],
    },
  ],
};

/** Tasks for a BA + customer. Unknown combinations return an empty list. */
export function getTasks(baId: string, customerId: string): TaskSummary[] {
  return TASKS[`${baId}|${customerId}`] ?? [];
}

export type TaskDetail = TaskSummary & { title: string };

/** Task detail by id. Returns undefined when the task is unknown. */
export function getTaskById(taskId: string): TaskDetail | undefined {
  for (const list of Object.values(TASKS)) {
    const task = list.find((t) => t.taskId === taskId);
    if (task) return { ...task, title: `Voice tagging: ${task.fileId}` };
  }
  return undefined;
}
