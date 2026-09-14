import {
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

  it("defaults targets to selectedEntities", async () => {
    await webhooksPublishEvent({ topic: "gate.passed", data: { a: 1 } }, exeConfig, rawConfig);
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body.endpointIds.sort()).toEqual(["ep_1", "ep_2"]);
  });

  it("rejects endpointIds outside the scope", async () => {
    const out = await webhooksPublishEvent(
      { topic: "gate.passed", data: {}, endpointIds: ["ep_9"] },
      exeConfig,
      rawConfig,
    );
    expect(JSON.parse(out).code).toBe("OUT_OF_SCOPE");
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("narrows within scope", async () => {
    await webhooksPublishEvent(
      { topic: "gate.passed", data: {}, endpointIds: ["ep_2"] },
      exeConfig,
      rawConfig,
    );
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body.endpointIds).toEqual(["ep_2"]);
  });
});

describe("webhooksListDestinations", () => {
  afterEach(() => jest.restoreAllMocks());

  it("filters returned destinations by scope", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200,
      json: async () => [
        { endpointId: "ep_1", url: "http://a" },
        { endpointId: "ep_9", url: "http://b" },
      ],
    } as Response);
    const out = await webhooksListDestinations({}, exeConfig, rawConfig);
    expect(JSON.parse(out).map((d: { endpointId: string }) => d.endpointId)).toEqual(["ep_1"]);
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

describe("webhooksPublishEvent empty scope", () => {
  afterEach(() => jest.restoreAllMocks());

  it("omits endpointIds for topic fan-out", async () => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true, status: 200, json: async () => ({ messageId: "m1" }),
    } as Response);
    const emptyConfig = {
      _resolvedConnections: [{ config: { baseUrl: "http://svc:5707", selectedEntities: [] } }],
    };
    await webhooksPublishEvent({ topic: "gate.passed", data: {} }, exeConfig, emptyConfig);
    const body = JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body);
    expect(body).not.toHaveProperty("endpointIds");
  });
});
