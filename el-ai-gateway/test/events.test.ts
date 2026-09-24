import { describe, expect, it, vi } from "vitest";
import { createEventPublisher } from "../src/upstream/events";
import type { McpCaller } from "../src/upstream/mcp";

describe("event publisher", () => {
  it("publishes job.completed with payload.type=customer_tag.updated", async () => {
    const mcp: McpCaller = { callTool: vi.fn(async () => ({ text: "{}", isError: false })) };
    await createEventPublisher(mcp).customerTagUpdated("cus_8899");
    expect(mcp.callTool).toHaveBeenCalledWith("webhooks_publish_event", {
      eventType: "job.completed",
      payload: { type: "customer_tag.updated", customerId: "cus_8899" },
    });
  });

  it("swallows upstream errors (best-effort)", async () => {
    const mcp: McpCaller = {
      callTool: vi.fn(async () => {
        throw new Error("down");
      }),
    };
    await expect(createEventPublisher(mcp).customerTagUpdated("c")).resolves.toBeUndefined();
  });
});
