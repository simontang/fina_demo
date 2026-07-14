export interface TaskAgentRuntimeEvent {
  label: string;
  status: "completed" | "in_progress" | "pending";
  time: string;
}

export interface TaskAgentRuntime {
  agentId: string;
  threadId: string;
  activeRuntime: string;
  currentState: string;
  selectedArtifact: string;
  nextAction: string;
  events: TaskAgentRuntimeEvent[];
  updatedAt: string;
}

const runtimeMap = new Map<string, TaskAgentRuntime>();

function key(agentId: string, threadId: string): string {
  return `${agentId}:${threadId}`;
}

export function getRuntime(agentId: string, threadId: string): TaskAgentRuntime | undefined {
  return runtimeMap.get(key(agentId, threadId));
}

export function createRuntime(
  agentId: string,
  threadId: string,
  runtime: {
    activeRuntime: string;
    currentState: string;
    selectedArtifact?: string;
    nextAction?: string;
  }
): TaskAgentRuntime {
  const now = new Date().toISOString();
  const record: TaskAgentRuntime = {
    agentId,
    threadId,
    activeRuntime: runtime.activeRuntime,
    currentState: runtime.currentState,
    selectedArtifact: runtime.selectedArtifact || "",
    nextAction: runtime.nextAction || "",
    events: [],
    updatedAt: now,
  };
  runtimeMap.set(key(agentId, threadId), record);
  return record;
}

export function updateRuntime(
  agentId: string,
  threadId: string,
  updates: Partial<{
    currentState: string;
    selectedArtifact: string;
    nextAction: string;
  }>
): TaskAgentRuntime | undefined {
  const record = runtimeMap.get(key(agentId, threadId));
  if (!record) return undefined;
  if (updates.currentState !== undefined) record.currentState = updates.currentState;
  if (updates.selectedArtifact !== undefined) record.selectedArtifact = updates.selectedArtifact;
  if (updates.nextAction !== undefined) record.nextAction = updates.nextAction;
  record.updatedAt = new Date().toISOString();
  return record;
}

export function appendEvent(
  agentId: string,
  threadId: string,
  event: { label: string; status: "completed" | "in_progress" | "pending" }
): TaskAgentRuntime | undefined {
  const record = runtimeMap.get(key(agentId, threadId));
  if (!record) return undefined;
  record.events.push({
    label: event.label,
    status: event.status,
    time: new Date().toISOString(),
  });
  record.updatedAt = new Date().toISOString();
  return record;
}

export function deleteRuntime(agentId: string, threadId: string): boolean {
  return runtimeMap.delete(key(agentId, threadId));
}
