type ToolExecutor = (input: any) => Promise<any>;

const registry = new Map<
  string,
  { config: { name: string; schema?: { parse: (v: any) => any } }; executor: ToolExecutor }
>();

jest.mock("@axiom-lattice/core", () => ({
  registerToolLattice: (
    key: string,
    config: { name: string; schema?: { parse: (v: any) => any } },
    executor: ToolExecutor
  ) => {
    registry.set(key, { config, executor });
  },
  getToolLattice: (key: string) => registry.get(key) ?? null,
}));

function callTool(key: string, input: any): Promise<any> {
  const entry = registry.get(key);
  if (!entry) throw new Error(`Tool "${key}" not registered`);
  const parsed = entry.config.schema ? entry.config.schema.parse(input) : input;
  return entry.executor(parsed);
}

// ============================================================
// sap_api_search
// ============================================================

describe("sap_api_search", () => {
  beforeAll(async () => {
    await import("../tools");
  });

  it("finds BusinessPartners by exact name", async () => {
    const result = await callTool("sap_api_search", { query: "BusinessPartners" });
    expect(result.totalMatches).toBeGreaterThanOrEqual(1);
    expect(result.results.some((r: any) => r.name === "BusinessPartners")).toBe(true);
  });

  it("finds Items by exact name", async () => {
    const result = await callTool("sap_api_search", { query: "Items" });
    expect(result.results.some((r: any) => r.name === "Items")).toBe(true);
    const items = result.results.find((r: any) => r.name === "Items");
    expect(items.primaryKey).toBe("ItemCode");
    expect(items.fields).toContain("ItemCode");
    expect(items.fields).toContain("ItemName");
  });

  it("finds Orders by exact name", async () => {
    const result = await callTool("sap_api_search", { query: "Orders" });
    expect(result.results.some((r: any) => r.name === "Orders")).toBe(true);
  });

  it("finds PurchaseOrders by exact name", async () => {
    const result = await callTool("sap_api_search", { query: "PurchaseOrders" });
    expect(result.results.some((r: any) => r.name === "PurchaseOrders")).toBe(true);
  });

  it("finds Warehouses by exact name", async () => {
    const result = await callTool("sap_api_search", { query: "Warehouses" });
    expect(result.results.some((r: any) => r.name === "Warehouses")).toBe(true);
  });

  it("finds by partial/substring match", async () => {
    const result = await callTool("sap_api_search", { query: "business" });
    expect(result.totalMatches).toBeGreaterThan(1);
    result.results.forEach((r: any) => {
      expect(r.name.toLowerCase()).toContain("business");
    });
  });

  it("finds by partial match — Item", async () => {
    const result = await callTool("sap_api_search", { query: "Item" });
    expect(result.totalMatches).toBeGreaterThan(1);
    expect(result.results.some((r: any) => r.name === "Items")).toBe(true);
    expect(result.results.some((r: any) => r.name === "ItemGroups")).toBe(true);
  });

  it("finds by partial match — Order", async () => {
    const result = await callTool("sap_api_search", { query: "Order" });
    expect(result.totalMatches).toBeGreaterThan(0);
    expect(result.results.some((r: any) => r.name === "Orders")).toBe(true);
  });

  it("finds by partial match — Purchase", async () => {
    const result = await callTool("sap_api_search", { query: "Purchase" });
    expect(result.totalMatches).toBeGreaterThan(1);
  });

  it("finds by partial match — Inventory", async () => {
    const result = await callTool("sap_api_search", { query: "Inventory" });
    expect(result.totalMatches).toBeGreaterThan(1);
  });

  it("finds by Chinese keyword — 客户 (BP)", async () => {
    const result = await callTool("sap_api_search", { query: "客户" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("BusinessPartner");
    });
  });

  it("finds by Chinese keyword — 物料 (Item)", async () => {
    const result = await callTool("sap_api_search", { query: "物料" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Item / Product");
    });
  });

  it("finds by Chinese keyword — 订单 (Document)", async () => {
    const result = await callTool("sap_api_search", { query: "订单" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Document");
    });
  });

  it("finds by Chinese keyword — 库存 (Inventory)", async () => {
    const result = await callTool("sap_api_search", { query: "库存" });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Inventory / Warehouse");
    });
  });

  it("filters by domain", async () => {
    const result = await callTool("sap_api_search", {
      query: "",
      domain: "BusinessPartner",
    });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("BusinessPartner");
    });
  });

  it("filters by domain — Item / Product", async () => {
    const result = await callTool("sap_api_search", {
      query: "",
      domain: "Item / Product",
    });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Item / Product");
    });
  });

  it("filters by domain — Inventory / Warehouse", async () => {
    const result = await callTool("sap_api_search", {
      query: "",
      domain: "Inventory / Warehouse",
    });
    expect(result.totalMatches).toBeGreaterThan(0);
    result.results.forEach((r: any) => {
      expect(r.domain).toBe("Inventory / Warehouse");
    });
  });

  it("returns EntitySet with primaryKey in results", async () => {
    const result = await callTool("sap_api_search", { query: "BusinessPartners" });
    const bp = result.results.find((r: any) => r.name === "BusinessPartners");
    expect(bp.primaryKey).toBe("CardCode");
    expect(bp.kind).toBe("EntitySet");
  });

  it("returns FunctionImport in results", async () => {
    const result = await callTool("sap_api_search", {
      query: "InitData",
      maxResults: 20,
    });
    expect(result.results.some((r: any) => r.kind === "FunctionImport")).toBe(true);
  });

  it("returns suggestion when no results found", async () => {
    const result = await callTool("sap_api_search", { query: "xyzwqkxyz" });
    expect(result.totalMatches).toBe(0);
    expect(result.results).toHaveLength(0);
    expect(result.suggestion).toBeDefined();
  });

  it("respects maxResults limit", async () => {
    const result = await callTool("sap_api_search", { query: "", maxResults: 3 });
    expect(result.results.length).toBeLessThanOrEqual(3);
  });

  it("returns domain distribution", async () => {
    const result = await callTool("sap_api_search", { query: "" });
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
    const result = await callTool("sap_api_search", { query: "BusinessPartners" });
    expect(result.results[0].hint).toBeDefined();
    expect(result.results[0].hint).toContain("sap_api_call");
  });
});

// ============================================================
// sap_api_call
// ============================================================

describe("sap_api_call", () => {
  beforeAll(async () => {
    await import("../tools");
  });

  it("returns HTTP response with unified format", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "BusinessPartners",
      method: "GET",
      queryOptions: "$top=1&$select=CardCode,CardName",
    });

    expect(result.ok).toBe(true);
    expect(result.status).toBe(200);
    expect(result.data).toBeDefined();
    expect(result.data.value).toBeDefined();
    expect(result.data.value.length).toBeLessThanOrEqual(1);
  });

  it("passes filter query correctly", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "BusinessPartners",
      method: "GET",
      queryOptions: "$top=1&$select=CardCode,CardName",
    });

    const bp = result.data.value[0];
    expect(bp.CardCode).toBeDefined();
    expect(bp.CardName).toBeDefined();
  });

  it("returns 200 for Items query", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "Items",
      method: "GET",
      queryOptions: "$top=1&$select=ItemCode,ItemName",
    });

    expect(result.ok).toBe(true);
    expect(result.status).toBe(200);
    expect(result.data.value[0].ItemCode).toBeDefined();
  });

  it("returns 200 for Orders query", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "Orders",
      method: "GET",
      queryOptions: "$top=1&$select=DocEntry,DocTotal",
    });

    expect(result.ok).toBe(true);
    expect(result.data.value[0].DocEntry).toBeDefined();
  });

  it("returns 200 for Warehouses query", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "Warehouses",
      method: "GET",
      queryOptions: "$top=1&$select=WarehouseCode,WarehouseName",
    });

    expect(result.ok).toBe(true);
    expect(result.data.value[0].WarehouseCode).toBeDefined();
  });

  it("returns 200 for InventoryGenEntries query", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "InventoryGenEntries",
      method: "GET",
      queryOptions: "$top=1&$select=DocEntry,DocDate",
    });

    expect(result.ok).toBe(true);
  });

  it("respects $top limit", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "BusinessPartners",
      method: "GET",
      queryOptions: "$top=3&$select=CardCode",
    });

    expect(result.data.value.length).toBeLessThanOrEqual(3);
  });

  it("returns error for POST without auth", async () => {
    const result = await callTool("sap_api_call", {
      entitySet: "BusinessPartners",
      method: "POST",
      body: { CardCode: "TEST001", CardName: "Test" },
    });

    expect(result.ok).toBe(false);
    expect(result.error).toBeDefined();
  });
});
