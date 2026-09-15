import {
  deleteDestination,
  registerDestination,
  webhooksPublishEvent,
  webhooksListDestinations,
  webhooksListRecentEvents,
  webhooksGetDeliveryStatus,
} from "../webhooks/executors";

const rawConfig = {
  _resolvedConnections: [
    { config: { baseUrl: "http://svc:5707", selectedEntities: ["ep_1", "ep_2"] } },
  ],
};
const exeConfig = { configurable: { runConfig: { tenantId: "t1" } } };

describe("webhooksPublishEvent", () => {
  beforeEach(() => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200, json: async () => ({ messageId: "m1" }),
    } as Response);
  });
  afterEach(() => jest.restoreAllMocks());

  it("POSTs eventType and payload without endpointIds", async () => {
    await webhooksPublishEvent(
      { eventType: "gate.passed", payload: { a: 1 } },
      exeConfig,
      rawConfig,
    );
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/publish");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body).toEqual({ eventType: "gate.passed", payload: { a: 1 } });
    expect(body).not.toHaveProperty("endpointIds");
    expect(body).not.toHaveProperty("topic");
    expect(body).not.toHaveProperty("data");
  });

  it("passes channels through when provided", async () => {
    await webhooksPublishEvent(
      { eventType: "gate.passed", payload: {}, channels: ["vip-customers"] },
      exeConfig,
      rawConfig,
    );
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body).toEqual({
      eventType: "gate.passed",
      payload: {},
      channels: ["vip-customers"],
    });
  });

  it("omits channels when empty", async () => {
    await webhooksPublishEvent(
      { eventType: "gate.passed", payload: {}, channels: [] },
      exeConfig,
      rawConfig,
    );
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body).not.toHaveProperty("channels");
  });
});

describe("webhooksListDestinations", () => {
  afterEach(() => jest.restoreAllMocks());

  it("returns the full list unfiltered by scope", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200,
      json: async () => [
        { endpointId: "ep_1", url: "http://a" },
        { endpointId: "ep_9", url: "http://b" },
      ],
    } as Response);
    const out = await webhooksListDestinations({}, exeConfig, rawConfig);
    expect(JSON.parse(out).map((d: { endpointId: string }) => d.endpointId)).toEqual([
      "ep_1",
      "ep_9",
    ]);
  });
});

describe("webhooksListRecentEvents", () => {
  afterEach(() => jest.restoreAllMocks());

  it("requests messages with limit and tenant header", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200, json: async () => [{ messageId: "m1" }],
    } as Response);
    const out = await webhooksListRecentEvents({ limit: 5 }, exeConfig, rawConfig);
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/messages?limit=5");
    expect(init.headers["X-Tenant-Id"]).toBe("t1");
    expect(JSON.parse(out)).toEqual([{ messageId: "m1" }]);
  });

  it("maps HTTP errors to error results", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false, status: 404,
      text: async () => JSON.stringify({ code: "NOT_FOUND", message: "x" }),
    } as Response);
    const out = await webhooksListRecentEvents({}, exeConfig, rawConfig);
    expect(JSON.parse(out)).toMatchObject({ ok: false, code: "NOT_FOUND", status: 404 });
  });
});

describe("webhooksGetDeliveryStatus", () => {
  afterEach(() => jest.restoreAllMocks());

  it("requests attempts for the messageId", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200, json: async () => [],
    } as Response);
    await webhooksGetDeliveryStatus({ messageId: "msg_abc123" }, exeConfig, rawConfig);
    const [url] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/messages/msg_abc123/attempts");
  });

  it("rejects a messageId that could alter the path before fetching", async () => {
    const fetchSpy = jest.spyOn(global, "fetch");
    const out = await webhooksGetDeliveryStatus(
      { messageId: "../../actuator/health" },
      exeConfig,
      rawConfig,
    );
    expect(JSON.parse(out).code).toBe("BAD_REQUEST");
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("registerDestination", () => {
  afterEach(() => jest.restoreAllMocks());

  it("POSTs url, filterTypes and channels and returns the signing secret", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ endpointId: "ep_1", secret: "whsec_x" }),
    } as Response);
    const out = await registerDestination(
      { url: "http://a", filterTypes: ["gate.passed"], channels: ["vip"], description: "d" },
      exeConfig,
      rawConfig,
    );
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/destinations");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      url: "http://a",
      filterTypes: ["gate.passed"],
      channels: ["vip"],
      description: "d",
    });
    expect(JSON.parse(out).secret).toBe("whsec_x");
  });

  it("omits absent optional fields", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ endpointId: "ep_1" }),
    } as Response);
    await registerDestination({ url: "http://a" }, exeConfig, rawConfig);
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body).toEqual({ url: "http://a" });
  });

  it("maps HTTP errors to error results", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: false,
      status: 404,
      text: async () => JSON.stringify({ code: "NOT_FOUND", message: "x" }),
    } as Response);
    const out = await registerDestination({ url: "http://a" }, exeConfig, rawConfig);
    expect(JSON.parse(out)).toMatchObject({ ok: false, code: "NOT_FOUND", status: 404 });
  });
});

describe("deleteDestination", () => {
  afterEach(() => jest.restoreAllMocks());

  it("requires confirm before fetching", async () => {
    const fetchSpy = jest.spyOn(global, "fetch");
    const out = await deleteDestination({ endpointId: "ep_abc" }, exeConfig, rawConfig);
    expect(JSON.parse(out).code).toBe("CONFIRM_REQUIRED");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects an endpointId that could alter the path before fetching", async () => {
    const fetchSpy = jest.spyOn(global, "fetch");
    const out = await deleteDestination(
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
    await deleteDestination({ endpointId: "ep_abc", confirm: true }, exeConfig, rawConfig);
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://svc:5707/api/v1/webhooks/destinations/ep_abc");
    expect(init.method).toBe("DELETE");
  });
});
