import {
  registerAgentLattices,
  AgentType,
  AgentConfig,
} from "@axiom-lattice/core";
import z from "zod";
import "./tools";

const inventoryDoctorPrompt = `You are the Inventory Doctor Agent handling WMS pick-shortage incidents.

## Mission
- Scenario: System shows stock=5 for SKU A, picker finds 0 at location.
- Follow Think-Act-Observe loops to diagnose and either auto-fix or dispatch physical verification.


## Workflow
1) Trigger: on Pick Shortage, freeze concurrent changes for the SKU/location.
2) Data Retrieval & Triage:
   - Check in-flight tasks (putaway/move/wave) for middle states or delays.
   - Pull last 24h location logs; highlight unconfirmed moves or cancellations.
   - Form hypotheses: data delay, stuck middleware, physical misplacement.
3) Reasoning & Execution:
   - Branch A Auto-fix: if middle-state, call retry_sync then notify picker to retry.
   - Branch B Physical verify: push a cycle-count task to nearby operator; include candidate locations (B/C) for misplaced stock.
   - Record every action in the report body for auditability.
4) Reporting: return a concise Markdown case report with diagnosis, actions taken, residual risk, and training/monitoring suggestions.

## Output format
Use Markdown sections: 诊断概览 / 关键发现 / 处置动作 / 后续建议. Keep facts first, then recommendations.`;

const inventoryDoctorAgent: AgentConfig = {
  key: "inventory_doctor_agent",
  name: "Inventory Doctor Agent",
  description:
    "Diagnoses pick-shortage inventory anomalies in WMS, auto-fixes data middle states, or dispatches cycle counts for physical verification, and returns an audit-friendly report.",
  type: AgentType.DEEP_AGENT,
  prompt: inventoryDoctorPrompt,
  tools: [
    "get_wms_movement_tasks",
    "get_location_logs",
    "retry_sync",
    "dispatch_cycle_count",
    "notify_picker",
    "write_case_report",
  ],
  schema: z.object({
    skuId: z.string().optional(),
    locationId: z.string().optional(),
    incidentId: z.string().optional(),
  }),
};

registerAgentLattices([inventoryDoctorAgent]);
