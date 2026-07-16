import { registerToolLattice } from "@axiom-lattice/core";
import { getTenantIdFromExecutionConfig } from "../segmentCrud";

jest.mock("@axiom-lattice/core", () => ({
  registerToolLattice: jest.fn(),
}));

const registerToolLatticeMock = registerToolLattice as jest.Mock;

const inputByTool: Record<string, Record<string, unknown>> = {
  segment_definition_list: {},
  segment_definition_create: {
    name: "test",
    datasourceId: 1,
    querySql: "SELECT 1",
  },
  segment_definition_update: { id: 1 },
  segment_definition_delete: { id: 1 },
  segment_definition_process: { id: 1 },
  segment_data_get: { id: 42 },
  segment_data_list: { page: 1, pageSize: 20 },
};

describe("segmentCrud tenant routing", () => {
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

  it("falls back to the configured default tenant when runConfig is absent", () => {
    expect(getTenantIdFromExecutionConfig()).toBe(
      process.env.TENANT_ID || "default",
    );
  });

  it("gets the exact segment data snapshot by id", async () => {
    const registration = registerToolLatticeMock.mock.calls.find(
      ([key]) => key === "segment_data_get",
    );
    const executor = registration[2] as (
      input: Record<string, unknown>,
      config: unknown,
    ) => Promise<string>;

    await executor({ id: 42 }, {
      configurable: { runConfig: { tenantId: "retail_cdp" } },
    });

    expect((global.fetch as jest.Mock).mock.calls[0][0]).toBe(
      "http://127.0.0.1:5706/api/v1/segment-data/42",
    );
  });
});
