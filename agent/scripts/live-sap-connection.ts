/**
 * Live integration check for sap-b1 connection resolution + pagination against a
 * real tenant store and real Service Layer.
 *
 * Verifies the two reported bugs end-to-end:
 *   1. the selected sap-b1 connection resolves to the tenant's own endpoint
 *      (not the default host), and a real Banks call returns that tenant's data;
 *   2. $top > server page size reports hasMore + odata.nextLink, and the next
 *      page is reachable via $skip.
 *
 * Run from the agent directory (Node 22.6+ / 26 strips TS types natively):
 *
 *   node scripts/live-sap-connection.ts
 *
 * Requires DATABASE_URL (from .env or the environment) and network access.
 * Overridable (defaults match the fina_demo store):
 *   SAP_B1_LIVE_TENANT / SAP_B1_LIVE_KEY / SAP_B1_LIVE_EXPECTED
 *   SAP_B1_ISOLATION_TENANT / SAP_B1_ISOLATION_EXPECTED
 *   SAP_B1_EXPECTED_BANK / SAP_B1_FORBIDDEN_BANK
 */
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

const here = path.dirname(fileURLToPath(import.meta.url));
const agentDir = path.resolve(here, "..");
dotenv.config({ path: path.join(agentDir, ".env") });

const TENANT = process.env.SAP_B1_LIVE_TENANT ?? "tenant_5";
const KEY = process.env.SAP_B1_LIVE_KEY ?? "b1s";
const EXPECTED = process.env.SAP_B1_LIVE_EXPECTED ?? "https://argoincorp.evario.ai/b1s/v1";
const ISO_TENANT = process.env.SAP_B1_ISOLATION_TENANT ?? "tenant_2";
const ISO_EXPECTED = process.env.SAP_B1_ISOLATION_EXPECTED ?? "https://b1s.alphafina.cn/b1s/v1";
const EXPECTED_BANK = process.env.SAP_B1_EXPECTED_BANK ?? "CIBB";
const FORBIDDEN_BANK = process.env.SAP_B1_FORBIDDEN_BANK ?? "BOA";

const { createPgStoreConfig } = await import("@axiom-lattice/pg-stores");
const { ConnectionRegistry } = await import("@axiom-lattice/core");
const { resolveSapBaseUrl, sapApiCallExecutor } = await import(
  "../src/agents/sap_b1/plugin.ts"
);

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
}

if (!process.env.DATABASE_URL) {
  console.error("DATABASE_URL is required (set it or provide agent/.env).");
  process.exit(2);
}

const stores = await createPgStoreConfig(process.env.DATABASE_URL);
ConnectionRegistry.setStore(stores.connection);

const runConfig = (tenantId: string) => ({ configurable: { runConfig: { tenantId } } });

// 1) Real selected connection through the real tenant store.
const baseUrl = await resolveSapBaseUrl({ connections: [KEY] }, runConfig(TENANT));
console.log(`${TENANT} baseUrl =`, baseUrl);
assert(baseUrl === EXPECTED, `${TENANT} should resolve to ${EXPECTED}`);

// 2) Per-tenant isolation.
const isoBaseUrl = await resolveSapBaseUrl({ connections: [KEY] }, runConfig(ISO_TENANT));
console.log(`${ISO_TENANT} baseUrl =`, isoBaseUrl);
assert(isoBaseUrl === ISO_EXPECTED, `${ISO_TENANT} should resolve to ${ISO_EXPECTED}`);

// 3) Real call through the plugin executor with the resolved baseUrl.
const result = await sapApiCallExecutor(
  { entitySet: "Banks", method: "GET", queryOptions: "$top=50&$select=BankCode,BankName" },
  { baseUrl }
);
assert(result.ok === true, `Banks call should succeed, got status ${result.status}`);
const codes = ((result.data as { value?: Array<{ BankCode: string }> }).value ?? []).map(
  (r) => r.BankCode
);
console.log("Banks codes =", codes.join(", "));
assert(codes.includes(EXPECTED_BANK), `Banks should contain ${EXPECTED_BANK}`);
assert(!codes.includes(FORBIDDEN_BANK), `Banks should NOT contain ${FORBIDDEN_BANK}`);

// 4) Real pagination: $top > server page size must report hasMore + nextLink.
const page1 = await sapApiCallExecutor(
  { entitySet: "BusinessPartners", method: "GET", queryOptions: "$top=50&$select=CardCode" },
  { baseUrl }
);
const rows1 = ((page1.data as { value?: unknown[] }).value ?? []) as unknown[];
const nextLink = (page1.data as Record<string, unknown>)["odata.nextLink"] as string | undefined;
console.log(
  `BusinessPartners page1 rows=${rows1.length} hasMore=${page1.hasMore} nextLink=${nextLink ?? "<none>"}`
);
assert(rows1.length === 20, `page1 should be server page size 20, got ${rows1.length}`);
assert(page1.hasMore === true, "page1 must report hasMore=true when a nextLink exists");
assert(
  typeof nextLink === "string" && nextLink.includes("skip="),
  "page1 must expose odata.nextLink carrying $skip"
);

const skip = nextLink!.match(/\$skip=(\d+)/)![1];
const page2 = await sapApiCallExecutor(
  {
    entitySet: "BusinessPartners",
    method: "GET",
    queryOptions: `$top=50&$select=CardCode&$skip=${skip}`,
  },
  { baseUrl }
);
const rows2 = ((page2.data as { value?: unknown[] }).value ?? []) as unknown[];
console.log(`BusinessPartners page2 (skip=${skip}) rows=${rows2.length} hasMore=${page2.hasMore}`);
assert(rows2.length > 0, "page2 must return the next page of rows");

console.log("\nLIVE PASS: connection resolves to the tenant endpoint and returns its data.");
console.log("LIVE PASS: pagination exposes nextLink/$skip and pages forward.");
