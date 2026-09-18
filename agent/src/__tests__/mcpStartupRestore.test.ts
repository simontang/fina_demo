import {
  getStoreLattice,
  mcpManager,
} from "@axiom-lattice/core";
import type { McpServerConfigEntry } from "@axiom-lattice/protocols";
import {
  convertConfigToMcpConnection,
  restoreConnectedMcpServersAcrossTenants,
  selectConnectedMcpConfigs,
} from "../mcpStartupRestore";

jest.mock("@axiom-lattice/core", () => ({
  getStoreLattice: jest.fn(),
  mcpManager: {
    addServer: jest.fn(),
    connect: jest.fn(),
    registerToolsToToolLattice: jest.fn(),
  },
}));

const getStoreLatticeMock = getStoreLattice as jest.Mock;
const mcpManagerMock = mcpManager as unknown as {
  addServer: jest.Mock;
  connect: jest.Mock;
  registerToolsToToolLattice: jest.Mock;
};

describe("MCP startup restoration", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mcpManagerMock.connect.mockResolvedValue(undefined);
    mcpManagerMock.registerToolsToToolLattice.mockResolvedValue(undefined);
  });

  it("restores connected MCP configs across all tenants", async () => {
    const getAllConfigsWithoutTenant = jest.fn().mockResolvedValue([
      mcpConfig({
        tenantId: "shenzhiyuan",
        key: "qcc",
        selectedTools: ["get_company_by_query"],
      }),
      mcpConfig({
        tenantId: "default",
        key: "draft",
        status: "disconnected",
        selectedTools: ["ignored"],
      }),
    ]);
    getStoreLatticeMock.mockReturnValue({
      store: { getAllConfigsWithoutTenant },
    });

    const summary = await restoreConnectedMcpServersAcrossTenants();

    expect(getStoreLatticeMock).toHaveBeenCalledWith("default", "mcp");
    expect(getAllConfigsWithoutTenant).toHaveBeenCalledTimes(1);
    expect(mcpManagerMock.addServer).toHaveBeenCalledWith("qcc", {
      env: { QCC_API_KEY: "secret" },
      transport: "http",
      url: "https://mcp.example/qcc",
    });
    expect(mcpManagerMock.connect).toHaveBeenCalledTimes(1);
    expect(mcpManagerMock.registerToolsToToolLattice).toHaveBeenCalledWith(
      "qcc",
      ["get_company_by_query"],
    );
    expect(summary).toMatchObject({
      totalConfigs: 2,
      connectedConfigs: 1,
      restoredServers: 1,
    });
  });

  it("deduplicates global MCP server keys and prefers default tenant configs", () => {
    const { selected, duplicateKeys } = selectConnectedMcpConfigs([
      mcpConfig({ tenantId: "tenant_a", key: "qcc", selectedTools: ["a"] }),
      mcpConfig({ tenantId: "default", key: "qcc", selectedTools: ["default"] }),
      mcpConfig({ tenantId: "tenant_b", key: "qcc-risk", selectedTools: ["risk"] }),
    ]);

    expect(duplicateKeys).toEqual(["qcc"]);
    expect(selected).toEqual([
      expect.objectContaining({ tenantId: "default", key: "qcc", selectedTools: ["default"] }),
      expect.objectContaining({ tenantId: "tenant_b", key: "qcc-risk", selectedTools: ["risk"] }),
    ]);
  });

  it("converts streamable_http configs to the MCP client http transport", () => {
    expect(convertConfigToMcpConnection({
      transport: "streamable_http",
      url: "https://mcp.example/stream",
      env: { TOKEN: "secret" },
    })).toEqual({
      transport: "http",
      url: "https://mcp.example/stream",
      env: { TOKEN: "secret" },
    });
  });
});

function mcpConfig(overrides: Partial<McpServerConfigEntry>): McpServerConfigEntry {
  return {
    id: `${overrides.tenantId || "tenant"}-${overrides.key || "qcc"}`,
    tenantId: "tenant",
    key: "qcc",
    name: "QCC",
    description: undefined,
    config: {
      transport: "streamable_http",
      url: "https://mcp.example/qcc",
      env: { QCC_API_KEY: "secret" },
    },
    selectedTools: [],
    isEnvEncrypted: false,
    status: "connected",
    createdAt: new Date("2026-01-01T00:00:00.000Z"),
    updatedAt: new Date("2026-01-01T00:00:00.000Z"),
    ...overrides,
  };
}
