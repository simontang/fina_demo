jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
import { businessObjectRecordsPlugin } from "../business_objects/recordsPlugin";
import { businessObjectSchemaPlugin } from "../business_objects/schemaPlugin";
import { storagePlugin } from "../storage/plugin";
import { webhooksPlugin } from "../webhooks/plugin";

describe("storage plugin", () => {
  it("registers a plugin with type storage", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(storagePlugin);
    expect(storagePlugin.meta.type).toBe("storage");
  });

  it("exposes exactly the non-upload tools to Open, readOnly/destructive annotated", () => {
    const expose = storagePlugin.meta.openExpose as Array<
      string | { name: string; readOnly?: boolean; destructive?: boolean }
    >;
    const names = expose.map((e) => (typeof e === "string" ? e : e.name));
    expect(names.sort()).toEqual([
      "delete",
      "get_download_url",
      "get_metadata",
      "list",
    ]);
    expect(expose).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "delete", destructive: true }),
        expect.objectContaining({ name: "list", readOnly: true }),
        expect.objectContaining({ name: "get_metadata", readOnly: true }),
        expect.objectContaining({ name: "get_download_url", readOnly: true }),
      ]),
    );
  });

  it("openExpose names all exist among middleware tools (invariant)", async () => {
    const mw = await storagePlugin.middleware!({});
    const toolNames = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    const expose = storagePlugin.meta.openExpose as Array<string | { name: string }>;
    for (const e of expose) {
      const name = typeof e === "string" ? e : e.name;
      expect(toolNames).toContain(name);
    }
    expect(toolNames).toContain("upload");
  });
});

describe("webhooks plugin", () => {
  it("registers a plugin with type webhooks and exposes all tools to Open", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(webhooksPlugin);
    expect(webhooksPlugin.meta.type).toBe("webhooks");
    const expose = (
      webhooksPlugin.meta.openExpose as Array<
        string | { name: string; readOnly?: boolean; destructive?: boolean }
      >
    ).map((e) => (typeof e === "string" ? e : e.name));
    expect(expose.sort()).toEqual([
      "delete_destination",
      "get_delivery_status",
      "list_destinations",
      "list_recent_events",
      "publish_event",
      "register_destination",
    ]);
  });

  it("provides the six webhook tools", async () => {
    const mw = await webhooksPlugin.middleware!({});
    const names = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    expect(names.sort()).toEqual([
      "delete_destination",
      "get_delivery_status",
      "list_destinations",
      "list_recent_events",
      "publish_event",
      "register_destination",
    ]);
  });
});

describe("webhooks plugin connection", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  const discoverWithContext = (
    config: Record<string, unknown>,
    context?: { tenantId?: string },
  ) =>
    (
      webhooksPlugin.connection!.discover as unknown as (
        config: Record<string, unknown>,
        context?: { tenantId?: string },
      ) => Promise<Array<{ id: string; name: string; description?: string }>>
    )(config, context);

  it("connection.test returns ok on HTTP 200 and failure with message otherwise", async () => {
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200 } as unknown as Response);

    const okResult = await webhooksPlugin.connection!.test!({ baseUrl: "http://svc:5707" });
    expect(okResult).toEqual({ ok: true, message: "Connected" });
    expect(fetchMock).toHaveBeenCalledWith("http://svc:5707/actuator/health");

    fetchMock.mockResolvedValue({ ok: false, status: 503 } as unknown as Response);
    const failResult = await webhooksPlugin.connection!.test!({ baseUrl: "http://svc:5707" });
    expect(failResult).toEqual({ ok: false, message: "HTTP 503" });
  });

  it("discover rejects when tenant context is missing", async () => {
    await expect(
      webhooksPlugin.connection!.discover!({ baseUrl: "http://svc:5707" }),
    ).rejects.toThrow("tenant context is missing");
  });

  it("discover maps destinations and forwards the tenant header", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [{ endpointId: "ep_1", url: "http://a", topics: ["gate.passed"] }],
    } as unknown as Response);

    const result = await discoverWithContext(
      { baseUrl: "http://svc:5707" },
      { tenantId: "t1" },
    );

    expect(result).toEqual([{ id: "ep_1", name: "http://a", description: "gate.passed" }]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://svc:5707/api/v1/webhooks/destinations",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-Tenant-Id": "t1" }),
      }),
    );
  });

  it("discover summarizes filterTypes and channels for the new protocol", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [
        {
          endpointId: "ep_2",
          url: "http://b",
          filterTypes: ["gate.passed", "job.completed"],
          channels: ["vip", "beta"],
        },
      ],
    } as unknown as Response);

    const result = await discoverWithContext(
      { baseUrl: "http://svc:5707" },
      { tenantId: "t1" },
    );

    expect(result).toEqual([
      {
        id: "ep_2",
        name: "http://b",
        description: "gate.passed, job.completed | channels: vip, beta",
      },
    ]);
  });
});

describe("business object plugins", () => {
  it("registers schema and records plugins", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(businessObjectSchemaPlugin);
    expect(PluginRegistry.register).toHaveBeenCalledWith(businessObjectRecordsPlugin);
    expect(businessObjectSchemaPlugin.meta.type).toBe("business-object-schema");
    expect(businessObjectRecordsPlugin.meta.type).toBe("business-object-records");
  });

  it("exposes schema management tools separately from record runtime tools", async () => {
    const schemaMw = await businessObjectSchemaPlugin.middleware!({});
    const schemaTools = ((schemaMw as { tools: Array<{ name: string }> }).tools ?? [])
      .map((t) => t.name)
      .sort();
    expect(schemaTools).toEqual([
      "create_object",
      "create_store",
      "delete_object",
      "get_object",
      "grant_store",
      "list_objects",
      "list_store_grants",
      "list_stores",
      "test_store",
      "update_object",
    ]);

    const recordsMw = await businessObjectRecordsPlugin.middleware!({});
    const recordTools = ((recordsMw as { tools: Array<{ name: string }> }).tools ?? [])
      .map((t) => t.name)
      .sort();
    expect(recordTools).toEqual([
      "create_record",
      "delete_record",
      "get_record",
      "query_records",
      "update_record",
    ]);
  });

  it("BO connection discovery maps object definitions into selectable entities", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [
        { objectKey: "customer", displayName: "Customer", storeKey: "crm_store" },
      ],
    } as unknown as Response);

    const discover = businessObjectRecordsPlugin.connection!.discover as unknown as (
      config: Record<string, unknown>,
    ) => Promise<Array<{ id: string; name: string; description?: string }>>;

    const result = await discover({ baseUrl: "http://svc:5707", boConnectionKey: "tenant" });

    expect(result).toEqual([{ id: "customer", name: "Customer", description: "store: crm_store" }]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://svc:5707/api/v1/bo/objects",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-BO-Connection-Key": "tenant" }),
      }),
    );
    const [, init] = fetchMock.mock.calls[0];
    expect((init?.headers as Record<string, string>)["X-Tenant-Id"]).toBeUndefined();
  });
});
