import { webhooksPublishEvent, webhooksListDestinations } from "../webhooks/executors";

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
