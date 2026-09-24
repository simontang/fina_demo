import type { McpCaller } from "./mcp";

export type EventPublisher = {
  customerTagUpdated(customerId: string): Promise<void>;
};

/**
 * Publishes the platform webhook event that a customer's tag aggregate changed.
 * Envelope: eventType `job.completed` with `payload.type = "customer_tag.updated"`
 * (no tag data — receivers refetch via the query API). Best-effort: never throws.
 */
export function createEventPublisher(mcp: McpCaller): EventPublisher {
  return {
    async customerTagUpdated(customerId: string): Promise<void> {
      try {
        await mcp.callTool("webhooks_publish_event", {
          eventType: "job.completed",
          payload: { type: "customer_tag.updated", customerId },
        });
      } catch {
        /* notification is best-effort; never fail the request */
      }
    },
  };
}
