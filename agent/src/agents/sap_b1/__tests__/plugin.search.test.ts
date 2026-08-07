/**
 * sap_api_search / sap_api_call 语义测试（迁移自 tools.test.ts，测 plugin.ts）
 * search 测试为纯函数断言；call 测试使用 mock fetch（不再依赖真实网络）。
 */
jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), get: jest.fn(), list: jest.fn(() => []), listMeta: jest.fn(() => []) },
}));
jest.mock("@langchain/core/tools", () => ({
  tool: (fn: any, cfg: any) => ({ ...cfg, invoke: fn }),
}));
jest.mock("langchain", () => ({
  createMiddleware: (opts: any) => opts,
}));

import { sapApiSearchExecutor, sapApiCallExecutor } from "../plugin";

// ============================================================
// sap_api_search
// ============================================================

describe("sap_api_search", () => {
  const search = (input: any) => sapApiSearchExecutor(input);

  it("finds BusinessPartners by exact name", async () => {
    const result = await search({ query: "BusinessPartners" });
    expect(result.totalMatches).toBeGreaterThanOrEqual(1);
    expect(result.results.some((r: any) => r.name === "BusinessPartners")).toBe(true);
  });

  it("finds Items by exact name", async () => {
    const result = await search({ query: "Items" });
    expect(result.results.some((r: any) => r.name === "Items")).toBe(true);
    const items = result.results.find((r: any) => r.name === "Items");
    expect(items.primaryKey).toBe("ItemCode");
    expect(items.fields).toContain("ItemCode");
    expect(items.fields).toContain("ItemName");
  });

  it("finds Orders by exact name", async () => {
    const result = await search({ query: "Orders" });
    expect(result.results.some((r: any) => r.name === "Orders")).toBe(true);
  });

  it("finds PurchaseOrders by exact name", async () => {
    const result = await search({ query: "PurchaseOrders" });
    expect(result.results.some((r: any) => r.name === "PurchaseOrders")).toBe(true);
  });

  it("finds Warehouses by exact name", async () => {
    const result = await search({ query: "Warehouses" });
    expect(result.results.some((r: any) => r.name === "Warehouses")).toBe(true);
  });

  it("finds by partial/substring match", async () => {
    const result = await search({ query: "business" });
    expect(result.totalMatches).toBeGreaterThan(1);
    result.results.forEach((r: any) => {
      expect(r.name.toLowerCase()).toContain("business");
    });
  });

  it("finds by partial match — Item", async () => {
    const result = await search({ query: "Item" });
    expect(result.totalMatches).toBeGreaterThan(1);
    expect(result.results.some((r: any) => r.name === "Items")).toBe(true);
    expect(result.results.some((r: any) => r.name === "ItemGroups")).toBe(true);
  });

  it("finds by partial match — Order", async () => {
    const result = await search({ query: "Order" });
    expect(result.totalMatches).toBeGreaterThan(0);
    expect(result.results.some((r: any) => r.name === "Orders")).toBe(true);
  });

  it("finds by partial match — Purchase", async () => {
    const result = await search({ query: "Purchase" });
    expect(result.totalMatches).toBeGreaterThan(1);
  });

  it("finds by partial match — Inventory", async () => {
    const result = await search({ query: "Inventory" });
    expect(result.totalMatches).toBeGreaterThan(1);
  });

  it("finds by Chinese keyword — 客户 (BP)", async () => {
    const result = await search({ query: "客户" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("BusinessPartner");
    });
  });

  it("finds by Chinese keyword — 物料 (Item)", async () => {
    const result = await search({ query: "物料" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Item / Product");
    });
  });

  it("finds by Chinese keyword — 订单 (Document)", async () => {
    const result = await search({ query: "订单" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Document");
    });
  });

  it("finds by Chinese keyword — 库存 (Inventory)", async () => {
    const result = await search({ query: "库存" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Inventory / Warehouse");
    });
  });

  it("filters by domain", async () => {
    const result = await search({ query: "", domain: "BusinessPartner" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("BusinessPartner");
    });
  });

  it("filters by domain — Item / Product", async () => {
    const result = await search({ query: "", domain: "Item / Product" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Item / Product");
    });
  });

  it("filters by domain — Inventory / Warehouse", async () => {
    const result = await search({ query: "", domain: "Inventory / Warehouse" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Inventory / Warehouse");
    });
  });

  it("returns EntitySet with primaryKey in results", async () => {
    const result = await search({ query: "BusinessPartners" });
    const bp = result.results.find((r: any) => r.name === "BusinessPartners");
    expect(bp.primaryKey).toBe("CardCode");
    expect(bp.kind).toBe("EntitySet");
  });

  it("returns FunctionImport in results", async () => {
    const result = await search({ query: "InitData", maxResults: 20 });
    expect(result.results.some((r: any) => r.kind === "FunctionImport")).toBe(true);
  });

  it("returns suggestion when no results found", async () => {
    const result = await search({ query: "xyzwqkxyz" });
    expect(result.totalMatches).toBe(0);
    expect(result.results).toHaveLength(0);
    expect(result.suggestion).toBeDefined();
  });

  it("respects maxResults limit", async () => {
    const result = await search({ query: "", maxResults: 3 });
    expect(result.results.length).toBeLessThanOrEqual(3);
  });

  it("returns domain distribution", async () => {
    const result = await search({ query: "" });
    expect(typeof result.domainsFound).toBe("object");
    if (result.results.length > 0) {
      const total = Object.values(result.domainsFound).reduce(
        (s: number, v: any) => s + v,
        0
      ) as number;
      expect(total).toBe(result.results.length);
    }
  });

  it("returns hint in each result", async () => {
    const result = await search({ query: "BusinessPartners" });
    expect(result.results[0].hint).toBeDefined();
    expect(result.results[0].hint).toContain("sap_api_call");
  });
});

// ============================================================
// sap_api_call（mock fetch，语义断言）
// ============================================================

describe("sap_api_call", () => {
  const realFetch = global.fetch;

  function mockFetchOnce(handler: (url: string, opts: RequestInit) => Response | Promise<Response>) {
    global.fetch = jest.fn(async (url: any, opts: any) => handler(String(url), opts)) as any;
  }

  afterEach(() => {
    global.fetch = realFetch;
  });

  const cfg = { baseUrl: "https://x/b1s/v1" };

  it("returns HTTP response with unified format", async () => {
    mockFetchOnce(() =>
      new Response(JSON.stringify({ value: [{ CardCode: "C001", CardName: "Acme" }] }), {
        status: 200,
      })
    );
    const result = await sapApiCallExecutor(
      { entitySet: "BusinessPartners", method: "GET", queryOptions: "$top=1" },
      cfg
    );
    expect(result.ok).toBe(true);
    expect(result.status).toBe(200);
    expect(result.data.value).toHaveLength(1);
    expect(result.data.value[0].CardCode).toBe("C001");
  });

  it("respects $top limit", async () => {
    mockFetchOnce(() =>
      new Response(JSON.stringify({ value: [{ CardCode: "1" }, { CardCode: "2" }, { CardCode: "3" }] }), {
        status: 200,
      })
    );
    const result = await sapApiCallExecutor(
      { entitySet: "BusinessPartners", method: "GET", queryOptions: "$top=3&$select=CardCode" },
      cfg
    );
    expect(result.data.value.length).toBeLessThanOrEqual(3);
  });

  it("returns error for POST with 401 (无有效认证)", async () => {
    mockFetchOnce(() =>
      new Response(
        JSON.stringify({ error: { code: "2", message: { value: "Invalid session" } } }),
        { status: 401 }
      )
    );
    const result = await sapApiCallExecutor(
      { entitySet: "BusinessPartners", method: "POST", body: { CardCode: "TEST001", CardName: "Test" } },
      cfg
    );
    expect(result.ok).toBe(false);
    expect(result.error).toBeDefined();
    expect(result.hint).toContain("认证");
  });

  it("returns SL error in hint for 400", async () => {
    mockFetchOnce(() =>
      new Response(
        JSON.stringify({ error: { code: "1", message: { value: "Property 'Foo' does not exist" } } }),
        { status: 400 }
      )
    );
    const result = await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET", queryOptions: "$top=1" },
      cfg
    );
    expect(result.ok).toBe(false);
    expect(result.hint).toContain("Foo");
  });
});
