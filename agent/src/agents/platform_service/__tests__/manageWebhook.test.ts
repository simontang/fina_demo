import {
  manageWebhookDeleteDestination,
  manageWebhookListDestinations,
  manageWebhookRegisterDestination,
} from "../manage_webhook/executors";

const rawConfig = {
  _resolvedConnections: [{ config: { baseUrl: "http://svc:5707", apiKey: "k" } }],
};
const exeConfig = { configurable: { runConfig: { tenantId: "t1" } } };

describe("manageWebhookListDestinations", () => {
  afterEach(() => jest.restoreAllMocks());

  it("GETs destinations with tenant header and returns JSON", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => [{ endpointId: "ep_1" }],
    } as Response);
    const out = await manageWebhookListDestinations({}, exeConfig, rawConfig);
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/destinations");
    expect(init.method).toBe("GET");
    expect(init.headers["X-Tenant-Id"]).toBe("t1");
    expect(JSON.parse(out)).toEqual([{ endpointId: "ep_1" }]);
  });

  it("maps HTTP errors to error results", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      status: 404,
      text: async () => JSON.stringify({ code: "NOT_FOUND", message: "x" }),
    } as Response);
    const out = await manageWebhookListDestinations({}, exeConfig, rawConfig);
    expect(JSON.parse(out)).toMatchObject({ ok: false, code: "NOT_FOUND", status: 404 });
  });
});

describe("manageWebhookRegisterDestination", () => {
  afterEach(() => jest.restoreAllMocks());

  it("POSTs the destination and returns the signing secret", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ endpointId: "ep_1", secret: "whsec_x" }),
    } as Response);
    const out = await manageWebhookRegisterDestination(
      { url: "http://a", topics: ["gate.passed"], description: "d" },
      exeConfig,
      rawConfig,
    );
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/destinations");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      url: "http://a",
      topics: ["gate.passed"],
      description: "d",
    });
    expect(JSON.parse(out).secret).toBe("whsec_x");
  });
});

describe("manageWebhookDeleteDestination", () => {
  afterEach(() => jest.restoreAllMocks());

  it("requires confirm before fetching", async () => {
    const fetchSpy = jest.spyOn(global, "fetch");
    const out = await manageWebhookDeleteDestination({ endpointId: "ep_abc" }, exeConfig, rawConfig);
    expect(JSON.parse(out).code).toBe("CONFIRM_REQUIRED");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects an endpointId that could alter the path before fetching", async () => {
    const fetchSpy = jest.spyOn(global, "fetch");
    const out = await manageWebhookDeleteDestination(
      { endpointId: "../x", confirm: true },
      exeConfig,
      rawConfig,
    );
    expect(JSON.parse(out).code).toBe("BAD_REQUEST");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("DELETEs the destination when confirmed", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
    } as Response);
    await manageWebhookDeleteDestination(
      { endpointId: "ep_abc", confirm: true },
      exeConfig,
      rawConfig,
    );
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/destinations/ep_abc");
    expect(init.method).toBe("DELETE");
  });
});
