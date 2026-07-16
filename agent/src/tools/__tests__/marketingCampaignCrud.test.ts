import { registerToolLattice } from "@axiom-lattice/core";
import fs from "node:fs";
import path from "node:path";
import "../marketingCampaignCrud";

jest.mock("@axiom-lattice/core", () => ({
  registerToolLattice: jest.fn(),
}));

const registerToolLatticeMock = registerToolLattice as jest.Mock;

const campaignInput = {
  name: "Dormant reactivation",
  type: "reactivation",
  goal: "Reactivate dormant members",
  startTime: "2026-07-14 10:00:00",
  endTime: "2026-07-21 10:00:00",
};

const inputByTool: Record<string, Record<string, unknown>> = {
  marketing_campaign_list: {
    type: "reactivation",
    status: "scheduled",
    page: 2,
    pageSize: 30,
  },
  marketing_campaign_get: { id: 1 },
  marketing_campaign_create: campaignInput,
  marketing_campaign_update: { id: 1, ...campaignInput },
  marketing_campaign_delete: { id: 1 },
  marketing_campaign_start: { id: 1 },
  marketing_campaign_stop: { id: 1 },
  marketing_campaign_schedule: {
    id: 1,
    startTime: "2026-07-15 10:00:00",
    endTime: "2026-07-22 10:00:00",
  },
};

describe("marketingCampaignCrud tenant routing", () => {
  beforeEach(() => {
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({}),
    } as Response);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each(Object.keys(inputByTool))(
    "passes the execution tenant to %s",
    async (toolName) => {
      const registration = registerToolLatticeMock.mock.calls.find(
        ([key]) => key === toolName,
      );
      expect(registration).toBeDefined();

      const executor = registration[2] as (
        input: Record<string, unknown>,
        config: unknown,
      ) => Promise<string>;
      await executor(inputByTool[toolName], {
        configurable: { runConfig: { tenantId: "retail_cdp" } },
      });

      const [, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
      expect(requestInit.headers["X-Tenant-Id"]).toBe("retail_cdp");
    },
  );

  it("builds list query parameters", async () => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === "marketing_campaign_list",
    );
    const executor = registration[2] as (
      input: Record<string, unknown>,
      config: unknown,
    ) => Promise<string>;

    await executor(inputByTool.marketing_campaign_list, {
      configurable: { runConfig: { tenantId: "retail_cdp" } },
    });

    const [url] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe(
      "http://127.0.0.1:5706/api/v1/marketing-campaigns?type=reactivation&status=scheduled&page=2&pageSize=30",
    );
  });

  it.each([
    ["marketing_campaign_start", "/api/v1/marketing-campaigns/1/start"],
    ["marketing_campaign_stop", "/api/v1/marketing-campaigns/1/stop"],
    ["marketing_campaign_schedule", "/api/v1/marketing-campaigns/1/schedule"],
  ])("calls the expected action endpoint for %s", async (toolName, path) => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === toolName,
    );
    const executor = registration[2] as (
      input: Record<string, unknown>,
      config: unknown,
    ) => Promise<string>;

    await executor(inputByTool[toolName], {
      configurable: { runConfig: { tenantId: "retail_cdp" } },
    });

    const [url, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe(`http://127.0.0.1:5706${path}`);
    expect(requestInit.method).toBe("POST");
  });

  it("accepts the documented reactivation v1 strategy", () => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === "marketing_campaign_create",
    );
    const schema = registration[1].schema as { parse: (input: unknown) => unknown };
    const fixture = JSON.parse(fs.readFileSync(
      path.resolve(process.cwd(), "../docs/api/marketing-campaign-reactivation-example.json"),
      "utf8",
    ));

    expect(() => schema.parse(fixture)).not.toThrow();
  });

  it("rejects conflicting main and source segment data ids", () => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === "marketing_campaign_create",
    );
    const schema = registration[1].schema as { parse: (input: unknown) => unknown };

    expect(() => schema.parse({
      ...campaignInput,
      mainSegmentDataId: 10,
      segmentationStrategy: { source: { segmentDataId: 11 } },
    })).toThrow("must match mainSegmentDataId");
  });

  it("keeps legacy object and array strategies compatible", () => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === "marketing_campaign_create",
    );
    const schema = registration[1].schema as { parse: (input: unknown) => unknown };

    expect(() => schema.parse({
      ...campaignInput,
      segmentationStrategy: [{ legacy: true }],
      offerStrategy: { customPolicy: "legacy" },
    })).not.toThrow();
  });
});
