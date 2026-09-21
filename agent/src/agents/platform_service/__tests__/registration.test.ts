jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
  resolvePluginConnections: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry, resolvePluginConnections } from "@axiom-lattice/core";
import { AgentType } from "@axiom-lattice/protocols";
import { businessObjectPlugin } from "../business_objects/plugin";
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

describe("business objects plugin", () => {
  it("registers one business-objects plugin in the data category", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(businessObjectPlugin);
    expect(businessObjectPlugin.meta.type).toBe("business-objects");
    expect(businessObjectPlugin.meta.category).toBe("data");
  });

  it("provides the full store/object/record tool set", async () => {
    const mw = await businessObjectPlugin.middleware!({});
    const names = ((mw as { tools: Array<{ name: string }> }).tools ?? [])
      .map((t) => t.name)
      .sort();
    expect(names).toEqual([
      "create_object",
      "create_record",
      "create_records",
      "create_store",
      "create_store_key",
      "delete_object",
      "delete_record",
      "delete_records",
      "delete_store_key",
      "get_object",
      "get_record",
      "list_objects",
      "list_store_keys",
      "list_stores",
      "query_records",
      "test_store",
      "update_object",
      "update_record",
      "update_store_key",
    ]);
  });

  it("resolves its connection dynamically, passing its own plugin type", async () => {
    (resolvePluginConnections as jest.Mock).mockReset();
    (resolvePluginConnections as jest.Mock).mockResolvedValue([
      { key: "demo", config: { baseUrl: "https://ada.alphafina.cn", boStoreKey: "bos_x" } },
    ]);
    const fetchMock = jest
      .spyOn(global, "fetch")
      .mockResolvedValue({ ok: true, status: 200, json: async () => [] } as Response);

    const mw = await businessObjectPlugin.middleware!({ connections: ["demo"], connectAll: false });
    const tool = ((mw as { tools: Array<{ name: string; invoke: Function }> }).tools).find(
      (t) => t.name === "list_stores",
    )!;
    await tool.invoke({}, { configurable: { runConfig: { tenantId: "estee_lauder" } } });

    expect(resolvePluginConnections).toHaveBeenCalledWith(
      "business-objects",
      { connections: ["demo"], connectAll: false },
      { tenantId: "estee_lauder" },
    );
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toBe("https://ada.alphafina.cn/api/v1/bo/stores");
    jest.restoreAllMocks();
  });

  it("exposes only read-only tools to Open, all present in the middleware", async () => {
    const expose = (businessObjectPlugin.meta.openExpose ?? []).map((e) =>
      typeof e === "string" ? { name: e, readOnly: false } : e,
    );
    expect(expose.map((e) => e.name).sort()).toEqual([
      "get_object",
      "get_record",
      "list_objects",
      "list_store_keys",
      "list_stores",
      "query_records",
      "test_store",
    ]);
    for (const e of expose) expect(e.readOnly).toBe(true);

    const mw = await businessObjectPlugin.middleware!({});
    const toolNames = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    for (const e of expose) expect(toolNames).toContain(e.name);
  });

  it("ships a business-objects-builder agent wired to the plugin and modeling skill", () => {
    const agent = businessObjectPlugin.agents?.["business-objects-builder"];
    expect(agent).toBeDefined();
    expect(agent!.type).toBe(AgentType.DEEP_AGENT);
    const types = (agent!.middleware ?? []).map((m) => m.type).sort();
    expect(types).toEqual([
      "ask_user_to_clarify",
      "business-objects",
      "code_eval",
      "filesystem",
      "skill",
      "task",
    ]);
    const skillMw = (agent!.middleware ?? []).find((m) => m.type === "skill");
    expect((skillMw!.config as { skills: string[] }).skills).toContain("business-objects-modeling");
  });

  it("names the modeling skill with the plugin prefix", () => {
    for (const key of Object.keys(businessObjectPlugin.skills ?? {})) {
      expect(key.startsWith("business-objects-")).toBe(true);
    }
  });

  it("BO connection discovery maps object definitions into selectable entities", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [
        { objectKey: "customer", displayName: "Customer", storeKey: "crm_store" },
      ],
    } as unknown as Response);

    const discover = businessObjectPlugin.connection!.discover as unknown as (
      config: Record<string, unknown>,
    ) => Promise<Array<{ id: string; name: string; description?: string }>>;

    const result = await discover({ baseUrl: "http://svc:5707", boStoreKey: "bos_secret" });

    expect(result).toEqual([{ id: "customer", name: "Customer", description: "store: crm_store" }]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://svc:5707/api/v1/bo/objects",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-BO-Connection-Key": "bos_secret" }),
      }),
    );
    const [, init] = fetchMock.mock.calls[0];
    expect((init?.headers as Record<string, string>)["X-Tenant-Id"]).toBeUndefined();
  });

  it("BO connection test verifies the single store authorized by boStoreKey", async () => {
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ storeKey: "crm_store", name: "CRM" }),
    } as unknown as Response);

    const result = await businessObjectPlugin.connection!.test!({
      baseUrl: "http://svc:5707",
      boStoreKey: "bos_secret",
    });

    expect(result.ok).toBe(true);
    expect(result.message).toContain("CRM authorized");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://svc:5707/api/v1/bo/stores/current",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-BO-Connection-Key": "bos_secret" }),
      }),
    );
  });
});
