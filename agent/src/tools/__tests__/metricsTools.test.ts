import { registerToolLattice, metricsServerManager } from "@axiom-lattice/core";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import ts from "typescript";
import "../metricsTools";
import { metricsFetch, validateReadOnlySql } from "../metricsToolClient";

jest.mock("@axiom-lattice/core", () => ({
  registerToolLattice: jest.fn(),
  metricsServerManager: {
    getServerKeys: jest.fn(),
    getConfig: jest.fn(),
  },
}));

const registerToolLatticeMock = registerToolLattice as jest.Mock;
const metricsServerManagerMock = metricsServerManager as unknown as {
  getServerKeys: jest.Mock;
  getConfig: jest.Mock;
};

const serverConfig = {
  type: "semantic",
  serverUrl: "http://metrics.example/api/v1",
  selectedDataSources: ["15"],
  headers: { "X-Metrics-Test": "yes" },
};

describe("metrics MCP-style tools", () => {
  it("loads the metrics registration module from the production entrypoint", () => {
    const filename = resolve(__dirname, "../../index.ts");
    const source = ts.createSourceFile(filename, readFileSync(filename, "utf8"), ts.ScriptTarget.Latest, true);
    const imports = source.statements
      .filter(ts.isImportDeclaration)
      .map((statement) => (statement.moduleSpecifier as ts.StringLiteral).text);

    expect(imports).toContain("./tools/metricsTools");
  });

  beforeEach(() => {
    metricsServerManagerMock.getServerKeys.mockResolvedValue([
      { key: "argo", type: "semantic" },
    ]);
    metricsServerManagerMock.getConfig.mockResolvedValue(serverConfig);
    jest.spyOn(global, "fetch").mockResolvedValue(jsonResponse({ code: 200, data: {} }) as Response);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    metricsServerManagerMock.getServerKeys.mockReset();
    metricsServerManagerMock.getConfig.mockReset();
  });

  it("registers the expected tool surface", () => {
    const toolNames = registerToolLatticeMock.mock.calls.map(([name]) => name);

    expect(toolNames).toEqual(expect.arrayContaining([
      "metrics_datasource_list",
      "metrics_meta_get",
      "metrics_metric_index",
      "metrics_metric_detail",
      "metrics_metric_query",
      "metrics_metric_list",
      "metrics_metric_get",
      "metrics_metric_create",
      "metrics_metric_update",
      "metrics_metric_delete",
      "metrics_visible_scope_list",
      "metrics_visible_scope_create",
      "metrics_visible_scope_update",
      "metrics_visible_scope_delete",
      "metrics_table_grant_list",
      "metrics_table_grant_create",
      "metrics_table_grant_update",
      "metrics_table_grant_delete",
      "metrics_datasource_table_list",
      "metrics_datasource_query",
      "metrics_datasource_sql_probe",
      "metrics_datasource_table_meta_list",
      "metrics_datasource_table_meta_get",
      "metrics_datasource_table_meta_create",
      "metrics_datasource_table_meta_update",
      "metrics_datasource_table_meta_delete",
      "metrics_datasource_metric_meta_list",
      "metrics_datasource_metric_meta_get",
      "metrics_datasource_metric_meta_create",
      "metrics_datasource_metric_meta_update",
      "metrics_datasource_metric_meta_delete",
      "metrics_meta_object_list",
      "metrics_meta_object_get",
      "metrics_meta_object_create",
      "metrics_meta_object_update",
      "metrics_meta_object_delete",
    ]));
  });

  it("uses runConfig tenant when resolving metrics config", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: [{ id: 15, name: "Argo" }],
    }));

    await execute("metrics_datasource_list", {}, tenantConfig());

    expect(metricsServerManagerMock.getServerKeys).toHaveBeenCalledWith("tenant_5");
    expect(metricsServerManagerMock.getConfig).toHaveBeenCalledWith("tenant_5", "argo");
  });

  it("filters datasource list by selectedDataSources", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: [
        { id: 15, name: "Allowed" },
        { id: 99, name: "Blocked" },
      ],
    }));

    const result = JSON.parse(await execute("metrics_datasource_list", {}, tenantConfig()));

    expect(result.dataSources).toEqual([{ id: 15, name: "Allowed" }]);
  });

  it("rejects a datasource outside the tenant allow-list before fetch", async () => {
    await expect(execute("metrics_metric_index", { datasourceId: 99 }, tenantConfig()))
      .rejects
      .toThrow("not allowed");

    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("validates customSql as read-only before querying", async () => {
    await expect(execute("metrics_metric_query", {
      datasourceId: 15,
      customSql: "DELETE FROM orders",
    }, tenantConfig())).rejects.toThrow("Only SELECT or WITH");

    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("posts semantic query to the metrics server", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: { semanticModel: "sales", columns: [], rows: [] },
    }));

    await execute("metrics_metric_query", {
      datasourceId: 15,
      metrics: ["sales_amount"],
      groupBy: ["DocDate__month"],
      limit: 50,
    }, tenantConfig());

    const [url, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://metrics.example/api/v1/metrics/query");
    expect(new Headers(requestInit.headers).get("X-Metrics-Test")).toBe("yes");
    expect(new Headers(requestInit.headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(JSON.parse(requestInit.body)).toMatchObject({
      datasourceId: "15",
      metrics: ["sales_amount"],
      groupBy: ["DocDate__month"],
      limit: 50,
    });
  });

  it("routes customSql metric query through the compatible metrics query API", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: { semanticModel: "adhoc", columns: [], rows: [] },
    }));

    await execute("metrics_metric_query", {
      datasourceId: 15,
      customSql: "SELECT * FROM hankel_sales",
      limit: 25,
      debug: true,
    }, tenantConfig());

    const [url, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://metrics.example/api/v1/metrics/query");
    expect(new Headers(requestInit.headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(JSON.parse(requestInit.body)).toMatchObject({
      datasourceId: "15",
      customSql: "SELECT * FROM hankel_sales",
      limit: 25,
      debug: true,
    });
  });

  describe.each([
    ["metrics_visible_scope", "scopeId"],
    ["metrics_table_grant", "grantId"],
  ])("%s datasource-owned scope tools", (toolPrefix, idField) => {
    it.each([
      ["list", "GET", ""],
      ["create", "POST", ""],
      ["update", "PUT", "/7"],
      ["delete", "DELETE", "/7"],
    ])("routes %s without tenant identity", async (action, method, suffix) => {
      metricsServerManagerMock.getConfig.mockResolvedValue({
        ...serverConfig,
        apiKey: "metrics-test-key",
        headers: {
          ...serverConfig.headers,
          "X-Tenant-Id": "configured-tenant",
          "x-tenant-id": "lowercase-tenant",
        },
      });
      const response = { datasourceId: 15, id: 7 };
      (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({ code: 200, data: response }));

      const result = await execute(`${toolPrefix}_${action}`, {
        datasourceId: 15,
        [idField]: 7,
        schemaName: "public",
        tablePattern: "hankel_",
        tenantId: "input-tenant",
      }, tenantConfig());

      const [url, request] = (global.fetch as jest.Mock).mock.calls[0];
      const headers = new Headers(request.headers);
      expect(url).toBe(`http://metrics.example/api/v1/datasources/15/visible-scopes${suffix}`);
      expect(request.method || "GET").toBe(method);
      expect(headers.has("X-Tenant-Id")).toBe(false);
      expect(headers.get("X-Metrics-Test")).toBe("yes");
      expect(headers.get("Authorization")).toBe("Bearer metrics-test-key");
      expect(request).not.toHaveProperty("includeTenantHeader");
      expect(metricsServerManagerMock.getConfig).toHaveBeenCalledWith("tenant_5", "argo");
      expect(JSON.parse(result)).toEqual(response);
      if (method === "POST" || method === "PUT") {
        expect(headers.get("Content-Type")).toBe("application/json");
        expect(JSON.parse(request.body)).toEqual({
          schemaName: "public",
          tablePattern: "hankel_",
          patternType: "PREFIX",
          caseSensitive: false,
          status: 1,
        });
      } else {
        expect(request.body).toBeUndefined();
      }
    });

    it.each(["list", "create", "update", "delete"])("rejects %s outside selectedDataSources", async (action) => {
      await expect(execute(`${toolPrefix}_${action}`, {
        datasourceId: 99,
        [idField]: 7,
        tablePattern: "hankel_",
      }, tenantConfig())).rejects.toThrow("not allowed");

      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("resolves the default datasource without a runConfig tenant", async () => {
      await execute(`${toolPrefix}_list`, {}, {});

      const [url, request] = (global.fetch as jest.Mock).mock.calls[0];
      expect(url).toBe("http://metrics.example/api/v1/datasources/15/visible-scopes");
      expect(new Headers(request.headers).has("X-Tenant-Id")).toBe(false);
    });

    it("preserves runConfig server and datasource selection for reads", async () => {
      metricsServerManagerMock.getConfig.mockResolvedValue({
        ...serverConfig,
        selectedDataSources: ["15", "20"],
      });

      await execute(`${toolPrefix}_list`, { serverKey: "input-server", datasourceId: 20 }, {
        configurable: {
          runConfig: {
            tenantId: "tenant_5",
            metricsDataSource: { serverKey: "run-server", datasourceId: 15 },
          },
        },
      });

      expect(metricsServerManagerMock.getConfig).toHaveBeenCalledWith("tenant_5", "run-server");
      expect((global.fetch as jest.Mock).mock.calls[0][0])
        .toBe("http://metrics.example/api/v1/datasources/15/visible-scopes");
    });

    it.each(["update", "delete"])("requires the compatible id field for %s", async (action) => {
      await expect(execute(`${toolPrefix}_${action}`, {
        datasourceId: 15,
        tablePattern: "hankel_",
      }, tenantConfig())).rejects.toThrow(idField);

      expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  it.each([
    ["record", { "x-TeNaNt-Id": "request-tenant", "X-Request-Test": "yes" }],
    ["Headers", new Headers({ "x-TeNaNt-Id": "request-tenant", "X-Request-Test": "yes" })],
    ["tuples", [["x-TeNaNt-Id", "request-tenant"], ["X-Request-Test", "yes"]]],
  ] as [string, RequestInit["headers"]][])("strips request-level tenant headers from %s when metricsFetch explicitly opts out", async (_, requestHeaders) => {
    await metricsFetch({ tenantId: "tenant_5", serverKey: "argo", config: { ...serverConfig, type: "semantic" } },
      "/datasources/15/visible-scopes", {
        includeTenantHeader: false,
        headers: requestHeaders,
      });

    const headers = new Headers((global.fetch as jest.Mock).mock.calls[0][1].headers);
    expect(headers.has("X-Tenant-Id")).toBe(false);
    expect(headers.get("X-Request-Test")).toBe("yes");
  });

  it("calls datasource schema table and query endpoints with compatible tenant context", async () => {
    await execute("metrics_datasource_table_list", { datasourceId: 15, schemaName: "public" }, tenantConfig());
    await execute("metrics_datasource_query", {
      datasourceId: 15,
      sql: "SELECT column_name FROM information_schema.columns WHERE table_name = :tableName",
      params: { tableName: "hankel_sales" },
      maxRows: 20,
    }, tenantConfig());

    expect((global.fetch as jest.Mock).mock.calls[0][0])
      .toBe("http://metrics.example/api/v1/datasources/15/schema/tables?schemaName=public");
    expect((global.fetch as jest.Mock).mock.calls[1][0])
      .toBe("http://metrics.example/api/v1/datasources/15/query");
    expect(new Headers((global.fetch as jest.Mock).mock.calls[0][1].headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(new Headers((global.fetch as jest.Mock).mock.calls[1][1].headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(JSON.parse((global.fetch as jest.Mock).mock.calls[1][1].body)).toMatchObject({
      sql: "SELECT column_name FROM information_schema.columns WHERE table_name = :tableName",
      maxRows: 20,
    });
  });

  it("rejects sql probe for datasource outside the tenant allow-list before fetch", async () => {
    await expect(execute("metrics_datasource_sql_probe", {
      datasourceId: 99,
      sql: "SELECT * FROM hankel_sales",
    }, tenantConfig())).rejects.toThrow("not allowed");

    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("keeps datasource sql probe as a runtime metrics query alias", async () => {
    await execute("metrics_datasource_sql_probe", {
      datasourceId: 15,
      sql: "SELECT * FROM hankel_sales",
      maxRows: 5,
    }, tenantConfig());

    const [url, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://metrics.example/api/v1/metrics/query");
    expect(JSON.parse(requestInit.body)).toMatchObject({
      datasourceId: "15",
      customSql: "SELECT * FROM hankel_sales",
      limit: 5,
    });
  });

  it("calls datasource published table and metric meta endpoints", async () => {
    await execute("metrics_datasource_table_meta_create", {
      datasourceId: 15,
      objectKey: "hankel_sales",
      payload: { tableName: "hankel_sales", displayName: "Hankel Sales" },
      accessGrant: { schemaName: "public", tablePattern: "hankel_sales", patternType: "EXACT" },
    }, tenantConfig());
    await execute("metrics_datasource_metric_meta_create", {
      datasourceId: 15,
      objectKey: "hankel_sales_amount",
      payload: { metric_name: "hankel_sales_amount", source: { table_view: "hankel_sales" } },
    }, tenantConfig());

    expect((global.fetch as jest.Mock).mock.calls[0][0])
      .toBe("http://metrics.example/api/v1/datasources/15/meta/tables");
    expect((global.fetch as jest.Mock).mock.calls[1][0])
      .toBe("http://metrics.example/api/v1/datasources/15/meta/metrics");
    expect(new Headers((global.fetch as jest.Mock).mock.calls[0][1].headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(new Headers((global.fetch as jest.Mock).mock.calls[1][1].headers).get("X-Tenant-Id")).toBe("tenant_5");
    expect(JSON.parse((global.fetch as jest.Mock).mock.calls[0][1].body)).not.toHaveProperty("accessGrant");
  });

  it("updates and deletes published table meta without changing visible scopes", async () => {
    await execute("metrics_datasource_table_meta_update", {
      datasourceId: 15,
      tableKey: "hankel_sales",
      payload: { tableName: "hankel_sales" },
      accessGrant: { tablePattern: "hankel_", patternType: "PREFIX" },
    }, tenantConfig());
    await execute("metrics_datasource_table_meta_delete", {
      datasourceId: 15,
      tableKey: "hankel_sales",
    }, tenantConfig());

    expect(global.fetch).toHaveBeenCalledTimes(2);
    const [[updateUrl, updateRequest], [deleteUrl, deleteRequest]] = (global.fetch as jest.Mock).mock.calls;
    expect(updateUrl).toBe("http://metrics.example/api/v1/datasources/15/meta/tables/hankel_sales");
    expect(JSON.parse(updateRequest.body)).toEqual({
      objectType: "table_view_detail",
      payload: { tableName: "hankel_sales" },
      status: 1,
    });
    expect(deleteUrl).toBe(updateUrl);
    expect(deleteRequest.method).toBe("DELETE");
    expect(deleteRequest.body).toBeUndefined();
  });

  it("checks metric id belongs to the allowed datasource before update", async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce(jsonResponse({ code: 200, data: [{ id: 7, metricCode: "sales_amount" }] }))
      .mockResolvedValueOnce(jsonResponse({ code: 200, data: { id: 7 } }));

    await execute("metrics_metric_update", {
      id: 7,
      datasourceId: 15,
      metricCode: "sales_amount",
      metricName: "sales_amount",
      querySql: "SELECT 1",
      parameters: [{ name: "startDate", type: "STRING" }],
      status: 1,
    }, tenantConfig());

    const [listUrl] = (global.fetch as jest.Mock).mock.calls[0];
    const [updateUrl, updateInit] = (global.fetch as jest.Mock).mock.calls[1];
    expect(listUrl).toBe("http://metrics.example/api/v1/datasources/15/metrics");
    expect(updateUrl).toBe("http://metrics.example/api/v1/metrics/7");
    expect(JSON.parse(updateInit.body).parameters).toBe('[{"name":"startDate","type":"STRING"}]');
  });

  it("creates global meta objects without datasource allow-list checks", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: { id: 1, datasourceId: null },
    }));

    await execute("metrics_meta_object_create", {
      objectType: "catalog_config",
      objectKey: "default",
      payload: { metric_catalog_version: "db-v1" },
      status: 1,
    }, tenantConfig());

    const [url, requestInit] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toBe("http://metrics.example/api/v1/meta/objects");
    expect(JSON.parse(requestInit.body)).toMatchObject({
      objectType: "catalog_config",
      objectKey: "default",
      payload: { metric_catalog_version: "db-v1" },
    });
  });

  it("blocks reading a meta object bound to a disallowed datasource", async () => {
    (global.fetch as jest.Mock).mockResolvedValue(jsonResponse({
      code: 200,
      data: { id: 1, datasourceId: 99 },
    }));

    await expect(execute("metrics_meta_object_get", { id: 1 }, tenantConfig()))
      .rejects
      .toThrow("not allowed");
  });

  it("allows SELECT and WITH SQL but rejects multi statements", () => {
    expect(() => validateReadOnlySql("SELECT 'delete' AS text FROM orders")).not.toThrow();
    expect(() => validateReadOnlySql("WITH base AS (SELECT 1) SELECT * FROM base")).not.toThrow();
    expect(() => validateReadOnlySql("SELECT TOP 10 [delete], [Order Date] FROM [Sales Order]")).not.toThrow();
    expect(() => validateReadOnlySql("SELECT 1; SELECT 2")).toThrow("single read-only SQL");
  });
});

async function execute(toolName: string, input: Record<string, unknown>, config: unknown): Promise<string> {
  const registration = registerToolLatticeMock.mock.calls.find(([key]) => key === toolName);
  expect(registration).toBeDefined();
  const executor = registration[2] as (
    input: Record<string, unknown>,
    config: unknown,
  ) => Promise<string>;
  return executor(registration[1].schema.parse(input), config);
}

function tenantConfig(): unknown {
  return { configurable: { runConfig: { tenantId: "tenant_5" } } };
}

function jsonResponse(body: unknown): Partial<Response> {
  return {
    ok: true,
    status: 200,
    text: async () => JSON.stringify(body),
  };
}
