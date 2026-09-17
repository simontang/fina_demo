import type { FastifyInstance } from "fastify";
import type { Authenticator } from "../auth";
import { requirePrincipal } from "../auth";
import { GatewayError } from "../lib/errors";
import { getCustomerTags } from "../mock/customerTags";

export type CustomerRouteDeps = {
  authenticator: Authenticator;
};

export function registerCustomerRoutes(app: FastifyInstance, deps: CustomerRouteDeps): void {
  // All business tags of a customer: name + tag uuid (+ dimension/evidence).
  app.get("/api/v1/customers/:customerId/tags", async (request) => {
    requirePrincipal(deps.authenticator, request.headers.authorization);
    const { customerId } = request.params as { customerId: string };
    const tags = getCustomerTags(customerId);
    if (!tags) {
      throw new GatewayError(404, "NOT_FOUND", `Customer '${customerId}' not found`);
    }
    return { customerId, total: tags.length, tags };
  });
}
