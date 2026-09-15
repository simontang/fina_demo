import { GatewayError } from "./lib/errors";
import type { Config, Principal } from "./types";

export type Authenticator = (authorization: string | undefined) => Principal | null;

export function createAuthenticator(config: Config): Authenticator {
  return (authorization) => {
    if (config.authDisabled) {
      return { tenantId: config.authDevTenant, keyLabel: "dev" };
    }
    if (!authorization?.startsWith("Bearer ")) return null;
    const key = authorization.slice("Bearer ".length).trim();
    const tenantId = config.gatewayApiKeys.get(key);
    if (!tenantId) return null;
    return { tenantId, keyLabel: key };
  };
}

export function requirePrincipal(auth: Authenticator, authorization: string | undefined): Principal {
  const principal = auth(authorization);
  if (!principal) throw new GatewayError(401, "UNAUTHORIZED", "Missing or invalid API key");
  return principal;
}
