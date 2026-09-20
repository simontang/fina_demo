import {
  boObjectCreate,
  boObjectDelete,
  boRecordCreate,
  boRecordDelete,
  boRecordQuery,
  boRecordUpdate,
  boStoreCreate,
  boStoreList,
} from "../business_objects/executors";

const rawConfig = {
  _resolvedConnections: [{
    config: {
      baseUrl: "http://svc:5707",
      apiKey: "k",
      boStoreKey: "bos_secret",
      selectedEntities: ["customer"],
    },
  }],
};

const exeConfig = { configurable: { runConfig: { tenantId: "t1" } } };

describe("business object executors", () => {
  beforeEach(() => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
    } as Response);
  });

  afterEach(() => jest.restoreAllMocks());

  it("creates stores through the schema tool without forwarding tenant identity", async () => {
    await boStoreCreate({
      storeKey: "crm_store",
      name: "CRM",
      jdbcUrl: "jdbc:postgresql://pg:5432/bo_crm",
      username: "bo",
      password: "secret",
    }, exeConfig, rawConfig);

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/bo/stores");
    expect(init.method).toBe("POST");
    expect(init.headers["X-Tenant-Id"]).toBeUndefined();
    expect(init.headers["X-Api-Key"]).toBe("k");
    expect(JSON.parse(init.body)).toMatchObject({
      storeKey: "crm_store",
      jdbcUrl: "jdbc:postgresql://pg:5432/bo_crm",
    });
  });

  it("creates object definitions and lets platform-service own DDL synchronization", async () => {
    await boObjectCreate({
      storeKey: "crm_store",
      objectKey: "customer",
      displayName: "Customer",
      fields: [{ key: "name", type: "string", required: true }],
    }, exeConfig, rawConfig);

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/bo/objects");
    expect(init.method).toBe("POST");
    expect(init.headers["X-BO-Connection-Key"]).toBe("bos_secret");
    expect(init.headers["X-Tenant-Id"]).toBeUndefined();
    expect(JSON.parse(init.body)).toMatchObject({
      storeKey: "crm_store",
      objectKey: "customer",
      fields: [{ key: "name", type: "string", required: true }],
    });
  });

  it("does not let record runtime callers pass a storeKey", async () => {
    await boRecordQuery({
      objectKey: "customer",
      filters: [{ field: "name", op: "contains", value: "ACME" }],
      page: 1,
      pageSize: 10,
    }, exeConfig, rawConfig);

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/bo/objects/customer/records/query");
    expect(init.method).toBe("POST");
    expect(init.headers["X-BO-Connection-Key"]).toBe("bos_secret");
    expect(init.headers["X-Tenant-Id"]).toBeUndefined();
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      filters: [{ field: "name", op: "contains", value: "ACME" }],
      page: 1,
      pageSize: 10,
    });
    expect(body).not.toHaveProperty("storeKey");
    expect(body).not.toHaveProperty("objectKey");
  });

  it("uses PATCH for record updates", async () => {
    await boRecordUpdate({
      objectKey: "customer",
      id: "r1",
      data: { tier: "gold" },
    }, exeConfig, rawConfig);

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/bo/objects/customer/records/r1");
    expect(init.method).toBe("PATCH");
    expect(init.headers["X-BO-Connection-Key"]).toBe("bos_secret");
    expect(init.headers["X-Tenant-Id"]).toBeUndefined();
    expect(JSON.parse(init.body)).toEqual({ data: { tier: "gold" } });
  });

  it("checks selectedEntities before record access", async () => {
    const out = await boRecordCreate({
      objectKey: "invoice",
      data: { amount: 1 },
    }, exeConfig, rawConfig);

    expect(JSON.parse(out)).toMatchObject({
      ok: false,
      code: "ERROR",
    });
    expect(JSON.parse(out).message).toContain("not allowed");
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("strips connection credentials from store listings", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ([{
        id: 1,
        storeKey: "elc",
        name: "ELC",
        description: "d",
        jdbcUrl: "jdbc:postgresql://host:5432/elc",
        schemaName: "bo",
        username: "elc_bo_runtime",
        status: 1,
      }]),
    } as Response);

    const out = JSON.parse(await boStoreList({}, exeConfig, rawConfig));
    expect(out).toEqual([
      { storeKey: "elc", name: "ELC", description: "d", schemaName: "bo", status: 1 },
    ]);
    const text = JSON.stringify(out);
    expect(text).not.toContain("jdbc");
    expect(text).not.toContain("username");
    expect(text).not.toContain("elc_bo_runtime");
  });

  it("passes through store errors unchanged", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 403,
      text: async () => JSON.stringify({ code: "FORBIDDEN", message: "nope" }),
    } as Response);

    const out = JSON.parse(await boStoreList({}, exeConfig, rawConfig));
    expect(out).toMatchObject({ ok: false, status: 403, code: "FORBIDDEN", message: "nope" });
  });

  it("delete operations require explicit confirmation", async () => {
    const objectOut = await boObjectDelete({ objectKey: "customer" }, exeConfig, rawConfig);
    const recordOut = await boRecordDelete({ objectKey: "customer", id: "r1" }, exeConfig, rawConfig);

    expect(JSON.parse(objectOut).code).toBe("CONFIRM_REQUIRED");
    expect(JSON.parse(recordOut).code).toBe("CONFIRM_REQUIRED");
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
