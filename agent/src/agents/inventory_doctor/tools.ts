/**
 * Inventory Doctor Agent Tools
 * 
 * Tools for diagnosing and fixing WMS pick-shortage incidents
 */

import z from "zod";
import { registerToolLattice } from "@axiom-lattice/core";

/**
 * Get WMS movement tasks for a SKU and location
 * Returns pending tasks that might affect inventory visibility
 */
registerToolLattice(
  "get_wms_movement_tasks",
  {
    name: "get_wms_movement_tasks",
    description:
      "Retrieve in-flight movement tasks (putaway/move/wave) for a specific SKU and location. Use this to check for tasks in middle states that might cause inventory discrepancies.",
    needUserApprove: false,
    schema: z.object({
      skuId: z.string().describe("SKU identifier"),
      locationId: z.string().describe("Location identifier"),
    }),
  },
  async (input: { skuId: string; locationId: string }) => {
    // Mock data with random values
    const taskTypes = ["putaway", "move", "wave", "replenishment"];
    const statuses = ["in_progress", "pending", "queued", "processing"];
    const hasPendingTasks = Math.random() > 0.3; // 70% chance of having pending tasks
    const taskCount = hasPendingTasks ? Math.floor(Math.random() * 3) + 1 : 0;

    const tasks = Array.from({ length: taskCount }, (_, i) => ({
      id: `move-${Math.floor(Math.random() * 9000) + 1000}`,
      type: taskTypes[Math.floor(Math.random() * taskTypes.length)],
      status: statuses[Math.floor(Math.random() * statuses.length)],
      etaMin: Math.floor(Math.random() * 60) + 5, // 5-65 minutes
    }));

    return {
      pending: hasPendingTasks,
      tasks,
    };
  }
);

/**
 * Get location activity logs
 * Returns recent activity logs for a location to identify unconfirmed moves or cancellations
 */
registerToolLattice(
  "get_location_logs",
  {
    name: "get_location_logs",
    description:
      "Retrieve activity logs for a location within a specified lookback period. Use this to identify unconfirmed moves, cancellations, or other activities that might explain inventory discrepancies.",
    needUserApprove: false,
    schema: z.object({
      locationId: z.string().describe("Location identifier"),
      lookbackHours: z.number().describe("Number of hours to look back"),
    }),
  },
  async (input: { locationId: string; lookbackHours: number }) => {
    // Mock data with random values
    const actions = ["move", "putaway", "pick", "adjustment", "cycle_count"];
    const statuses = [
      "pending_confirm",
      "completed",
      "cancelled",
      "in_progress",
      "failed",
    ];
    const operators = [
      "op_x",
      "op_y",
      "op_z",
      "worker_01",
      "worker_05",
      "worker_12",
    ];
    const locations = ["A-01", "A-02", "B-01", "B-02", "C-03", "D-05"];

    const logCount = Math.floor(Math.random() * 5) + 1; // 1-5 logs
    const now = new Date();
    const logs = Array.from({ length: logCount }, (_, i) => {
      const hoursAgo = Math.floor(Math.random() * input.lookbackHours);
      const timestamp = new Date(now.getTime() - hoursAgo * 60 * 60 * 1000);
      const fromLoc = locations[Math.floor(Math.random() * locations.length)];
      const toLoc = locations[Math.floor(Math.random() * locations.length)];

      return {
        ts: timestamp.toISOString(),
        action: actions[Math.floor(Math.random() * actions.length)],
        from: fromLoc,
        to: toLoc !== fromLoc ? toLoc : locations[Math.floor(Math.random() * locations.length)],
        status: statuses[Math.floor(Math.random() * statuses.length)],
        operator: operators[Math.floor(Math.random() * operators.length)],
      };
    });

    // Sort by timestamp descending
    logs.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());

    return logs;
  }
);

/**
 * Retry synchronization for a stuck task
 * Attempts to refresh the status of a task that might be in a middle state
 */
registerToolLattice(
  "retry_sync",
  {
    name: "retry_sync",
    description:
      "Retry synchronization for a task that appears to be stuck in a middle state. This will attempt to refresh the task status and resolve data delays.",
    needUserApprove: false,
    schema: z.object({
      taskId: z.string().describe("Task identifier to retry"),
    }),
  },
  async (input: { taskId: string }) => {
    // Mock data with random outcomes
    const fixed = Math.random() > 0.2; // 80% success rate
    const messages = [
      "status refreshed to completed",
      "task synchronized successfully",
      "sync completed, inventory updated",
      "task status updated to completed",
      "synchronization failed, task still in progress",
      "unable to sync, task may require manual intervention",
    ];

    return {
      fixed,
      message: fixed
        ? messages[Math.floor(Math.random() * 4)]
        : messages[Math.floor(Math.random() * 2) + 4],
    };
  }
);

/**
 * Dispatch a cycle count task for physical verification
 * Creates a cycle count task and assigns it to an operator
 */
registerToolLattice(
  "dispatch_cycle_count",
  {
    name: "dispatch_cycle_count",
    description:
      "Dispatch a cycle count task to physically verify inventory at specified locations. Use this when physical verification is needed to resolve inventory discrepancies.",
    needUserApprove: false,
    schema: z.object({
      skuId: z.string().describe("SKU identifier to verify"),
      locations: z.array(z.string()).describe("List of location identifiers to check"),
      priority: z.string().describe("Priority level (e.g., 'high', 'medium', 'low')"),
    }),
  },
  async (input: {
    skuId: string;
    locations: string[];
    priority: string;
  }) => {
    // Mock data with random values
    const workers = [
      "worker_01",
      "worker_05",
      "worker_07",
      "worker_12",
      "worker_15",
      "worker_20",
    ];
    const taskId = `cc-${Math.floor(Math.random() * 9000) + 1000}`;
    const assignee = workers[Math.floor(Math.random() * workers.length)];
    const etaMinutes = Math.floor(Math.random() * 30) + 5; // 5-35 minutes
    const eta = `${etaMinutes}m`;

    return {
      taskId,
      assignee,
      eta,
    };
  }
);

/**
 * Notify picker about inventory status
 * Sends a notification to the picker about the current status
 */
registerToolLattice(
  "notify_picker",
  {
    name: "notify_picker",
    description:
      "Send a notification message to the picker about inventory status, retry instructions, or other relevant information.",
    needUserApprove: false,
    schema: z.object({
      message: z.string().describe("Message to send to the picker"),
    }),
  },
  async (input: { message: string }) => {
    // Mock data with random delivery status
    const delivered = Math.random() > 0.1; // 90% success rate

    return {
      delivered,
    };
  }
);

/**
 * Write case report to file system
 * Saves the diagnostic case report as a markdown file
 */
registerToolLattice(
  "write_case_report",
  {
    name: "write_case_report",
    description:
      "Save the diagnostic case report as a markdown file. Use this to record the diagnosis, actions taken, and recommendations for audit purposes.",
    needUserApprove: false,
    schema: z.object({
      markdown: z.string().describe("Markdown content of the case report"),
    }),
  },
  async (input: { markdown: string }) => {
    // Mock data with random path and timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
    const reportId = Math.floor(Math.random() * 10000);
    const paths = [
      `/reports/inventory-case-latest.md`,
      `/reports/inventory-case-${reportId}.md`,
      `/reports/case-${timestamp}.md`,
      `/reports/inventory-diagnosis-${reportId}.md`,
    ];

    return {
      saved: true,
      path: paths[Math.floor(Math.random() * paths.length)],
    };
  }
);
