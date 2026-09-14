jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), list: jest.fn(() => []), get: jest.fn() },
  getSandBoxManager: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (o: unknown) => o,
  tool: (fn: unknown, cfg: Record<string, unknown>) => ({ ...cfg, invoke: fn }),
}));

import { PluginRegistry } from "@axiom-lattice/core";
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
      "storage_delete",
      "storage_get_download_url",
      "storage_get_metadata",
      "storage_list",
    ]);
    expect(expose).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "storage_delete", destructive: true }),
        expect.objectContaining({ name: "storage_list", readOnly: true }),
        expect.objectContaining({ name: "storage_get_metadata", readOnly: true }),
        expect.objectContaining({ name: "storage_get_download_url", readOnly: true }),
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
    expect(toolNames).toContain("storage_upload");
  });
});

describe("webhooks plugin", () => {
  it("registers a plugin with type webhooks and no Open exposure in v1", () => {
    expect(PluginRegistry.register).toHaveBeenCalledWith(webhooksPlugin);
    expect(webhooksPlugin.meta.type).toBe("webhooks");
    expect(webhooksPlugin.meta.openExpose ?? []).toEqual([]);
  });

  it("provides the four webhook tools", async () => {
    const mw = await webhooksPlugin.middleware!({});
    const names = ((mw as { tools: Array<{ name: string }> }).tools ?? []).map((t) => t.name);
    expect(names.sort()).toEqual([
      "webhooks_get_delivery_status",
      "webhooks_list_destinations",
      "webhooks_list_recent_events",
      "webhooks_publish_event",
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
    expect(okResult).toEqual({ ok: true, message: "连接成功" });
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
});

