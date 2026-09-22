import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import type { BoTools } from "../upstream/boTools";

export type CustomerRouteDeps = {
  authenticator: Authenticator;
  boTools: BoTools;
};

export function registerCustomerRoutes(app: FastifyInstance, deps: CustomerRouteDeps): void {
  // All business tags of a customer, read from the customer_tag aggregate.
  app.get("/api/v1/customers/:customerId/tags", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { customerId } = request.params as { customerId: string };
    const rows = await deps.boTools.queryRecords("customer_tag", [
      { field: "customer_no", op: "eq", value: customerId },
    ]);
    const tags = rows.map((row) => ({
      tagId: row.tag_id,
      tagKey: row.tag_key,
      tagValue: row.tag_value,
      source: row.source,
      confidence: row.confidence ?? null,
      taggedAt: row.tagged_at,
    }));
    return { customerId, total: tags.length, tags };
  });
}
