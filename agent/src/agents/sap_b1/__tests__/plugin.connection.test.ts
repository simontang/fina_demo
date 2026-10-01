/**
 * sap-b1 连接解析测试。
 *
 * 回归 bug：Agent 选中的是 Argo 的 sap-b1 connection，但工具运行时始终查默认的
 * b1s.alphafina.cn，因为在 invoke 期从未解析 `connections` 选择器。
 */
jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), get: jest.fn(), list: jest.fn(() => []), listMeta: jest.fn(() => []) },
  resolvePluginConnections: jest.fn(),
}));
jest.mock("langchain", () => ({
  createMiddleware: (opts: any) => opts,
  tool: (fn: any, cfg: any) => ({ ...cfg, invoke: fn }),
}));

import { resolvePluginConnections } from "@axiom-lattice/core";
import { sapB1Plugin, resolveSapBaseUrl } from "../plugin";

const mockResolve = resolvePluginConnections as jest.Mock;

describe("resolveSapBaseUrl", () => {
  beforeEach(() => mockResolve.mockReset());

  it("优先使用 invoke 期 _resolvedConnections 的 baseUrl", async () => {
    const url = await resolveSapBaseUrl(
      { _resolvedConnections: [{ config: { baseUrl: "https://build.example/b1s/v1" } }] },
      {
        configurable: {
          runConfig: {
            _resolvedConnections: [{ config: { baseUrl: "https://argoincorp.evario.ai/b1s/v1" } }],
          },
        },
      }
    );
    expect(url).toBe("https://argoincorp.evario.ai/b1s/v1");
    expect(mockResolve).not.toHaveBeenCalled();
  });

  it("从租户 Connection Store 动态解析选中的 sap-b1 连接", async () => {
    mockResolve.mockResolvedValue([
      { key: "argo", config: { baseUrl: "https://argoincorp.evario.ai/b1s/v1" } },
    ]);
    const url = await resolveSapBaseUrl(
      { connections: ["argo"] },
      { configurable: { runConfig: { tenantId: "argo" } } }
    );
    expect(url).toBe("https://argoincorp.evario.ai/b1s/v1");
    expect(mockResolve).toHaveBeenCalledWith(
      "sap-b1",
      { connections: ["argo"], connectAll: false },
      { tenantId: "argo" }
    );
  });

  it("未配置选择器时返回 undefined（交由 env/default 兜底）", async () => {
    expect(await resolveSapBaseUrl({}, { configurable: { runConfig: {} } })).toBeUndefined();
    expect(mockResolve).not.toHaveBeenCalled();
  });

  it("选择器配置了但解析不到连接 → 抛出可操作错误", async () => {
    mockResolve.mockResolvedValue([]);
    await expect(
      resolveSapBaseUrl(
        { connections: ["missing"] },
        { configurable: { runConfig: { tenantId: "argo" } } }
      )
    ).rejects.toThrow(/sap-b1/);
  });
});

describe("sap_b1 middleware 连接接线", () => {
  const realFetch = global.fetch;
  afterEach(() => {
    global.fetch = realFetch;
    mockResolve.mockReset();
  });

  function toolNamed(name: string) {
    const mw = sapB1Plugin.middleware!({ connections: ["argo"] }, {} as any) as any;
    return mw.tools.find((t: any) => t.name === name);
  }

  it("sap_api_call 命中选中的 connection，而非默认端点", async () => {
    mockResolve.mockResolvedValue([
      { key: "argo", config: { baseUrl: "https://argoincorp.evario.ai/b1s/v1" } },
    ]);
    let captured = "";
    global.fetch = jest.fn(async (url: any) => {
      captured = String(url);
      return new Response(JSON.stringify({ value: [] }), { status: 200 });
    }) as any;

    await toolNamed("sap_api_call").invoke(
      { entitySet: "BusinessPartners", method: "GET", queryOptions: "$top=1" },
      { configurable: { runConfig: { tenantId: "argo" } } }
    );

    expect(captured).toContain("https://argoincorp.evario.ai/b1s/v1/BusinessPartners");
    expect(captured).not.toContain("b1s.alphafina.cn");
  });

  it("sap_api_call 描述包含分页指引（$skip / nextLink），否则 agent 不会翻页", () => {
    const desc = toolNamed("sap_api_call").description as string;
    expect(desc).toContain("$skip");
    expect(desc).toContain("nextLink");
  });
});
