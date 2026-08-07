/**
 * sap_api_call / sap_api_search 行为测试（针对 plugin.ts 生产代码）
 *
 * 覆盖目标：工具发出的请求与 curl 等价 —— curl 能成功的查询，工具必须成功。
 * 测试当前版本应为 RED（暴露已知 bug），修复后全绿。
 */
jest.mock("@axiom-lattice/core", () => ({
  PluginRegistry: { register: jest.fn(), get: jest.fn(), list: jest.fn(() => []), listMeta: jest.fn(() => []) },
}));
jest.mock("@langchain/core/tools", () => ({
  tool: (fn: any, cfg: any) => ({ ...cfg, invoke: fn }),
}));
jest.mock("langchain", () => ({
  createMiddleware: (opts: any) => opts,
}));

import {
  API_LIST,
  applyDefaultSelect,
  buildUrl,
  encodeQueryOptions,
  cleanODataNoise,
  trimNestedCollections,
  sapApiCallExecutor,
} from "../plugin";

import sapFixture from "../__fixtures__/sap_metadata.json";

type Fixture = {
  fetched_at: string;
  companies: Record<string, Record<string, { properties: string[]; navigation: string[] }>>;
};
const fixture = sapFixture as Fixture;

const typeName = (et: string) => et.split(".").pop()!;

function entityTypesByCompany(entitySet: string, entityType?: string): { company: string; type: string }[] {
  const out: { company: string; type: string }[] = [];
  const type = typeName(entityType || entitySet);
  for (const [company, entities] of Object.entries(fixture.companies)) {
    if (entities[type]) out.push({ company, type });
  }
  return out;
}

// ============================================================
// 1. 字段校验：API_LIST 每个字段必须真实存在于对应 EntityType
// ============================================================

describe("字段校验（对照真实 $metadata fixture）", () => {
  const entries = API_LIST.filter((e) => e.kind === "EntitySet");

  it.each(entries.map((e) => [e.name, e] as const))(
    "字段表无错字段 — %s",
    (_name, entry) => {
      const matches = entityTypesByCompany(entry.name, entry.entityType);
      expect(matches.length).toBeGreaterThan(0);
      for (const { company, type } of matches) {
        const props = fixture.companies[company][type];
        const known = new Set([...props.properties, ...props.navigation]);
        for (const field of entry.fields) {
          expect(known.has(field)).toBe(true);
        }
      }
    }
  );
});

// ============================================================
// 2. URL 构造：applyDefaultSelect + buildUrl
// ============================================================

describe("applyDefaultSelect（默认 $select 白名单注入）", () => {
  it("Orders 无 queryOptions → 注入 Document 类白名单 + $top=20，且不含错误字段 RoundDif", () => {
    const q = applyDefaultSelect("Orders", "GET", undefined, undefined)!;
    expect(q).toContain("$select=");
    expect(q).toContain("$top=20");
    expect(q).not.toContain("RoundDif");
    expect(q).not.toContain("DocumentLines");
  });

  it("Warehouses → 白名单只含 WarehouseCode,WarehouseName", () => {
    const q = applyDefaultSelect("Warehouses", "GET", undefined, undefined)!;
    expect(q).toContain("$select=WarehouseCode,WarehouseName");
    expect(q).not.toContain("DocEntry");
  });

  it("无类别映射的实体（SalesPersons）→ 只注入 $top，不注入 $select", () => {
    const q = applyDefaultSelect("SalesPersons", "GET", undefined, undefined)!;
    expect(q).toContain("$top=20");
    expect(q).not.toContain("$select=");
  });

  it("FunctionImport GET → 不注入任何查询参数", () => {
    const q = applyDefaultSelect("SBOBobService_GetCurrencyRate", "GET", undefined, undefined);
    expect(q).toBeUndefined();
  });

  it("LLM 显式传 $select → 不覆盖", () => {
    const q = applyDefaultSelect("Orders", "GET", undefined, "$select=DocEntry,DocNum&$top=5")!;
    expect(q).toContain("$select=DocEntry,DocNum");
    expect(q).not.toContain("CardCode");
  });

  it("GET 带 id → id 被忽略，仍按集合查询注入 $top", () => {
    const q = applyDefaultSelect("Orders", "GET", "1173", undefined)!;
    expect(q).toContain("$top=20");
  });
});

describe("buildUrl", () => {
  it("PATCH/DELETE 使用主键路径", () => {
    expect(buildUrl("https://x/b1s/v1", "Orders", "PATCH", "1173")).toBe(
      "https://x/b1s/v1/Orders('1173')"
    );
    expect(buildUrl("https://x/b1s/v1", "BusinessPartners", "DELETE", "C001")).toBe(
      "https://x/b1s/v1/BusinessPartners('C001')"
    );
  });

  it("GET 带 id → 不再拼主键路径（避免 SL 500）", () => {
    const url = buildUrl("https://x/b1s/v1", "Orders", "GET", "1173", "$top=20");
    expect(url).toBe("https://x/b1s/v1/Orders?$top=20");
  });
});

// ============================================================
// 3. 编码：encodeQueryOptions（幂等、引号内编码）
// ============================================================

describe("encodeQueryOptions", () => {
  it("引号内 & → %26（不拆参数）", () => {
    expect(encodeQueryOptions("$filter=ItemCode eq 'A&B'")).toBe(
      "$filter=ItemCode%20eq%20%27A%26B%27"
    );
  });

  it("引号外 & 保留为参数分隔符", () => {
    expect(encodeQueryOptions("$filter=CardName eq 'A' and CardType eq 'c'&$top=5")).toBe(
      "$filter=CardName%20eq%20%27A%27%20and%20CardType%20eq%20%27c%27&$top=5"
    );
  });

  it("+ → %2B（防止 SL 按空格解码）", () => {
    expect(encodeQueryOptions("$filter=Phone1 eq '+8613800000000'")).toContain("%27%2B86");
  });

  it("# → %23", () => {
    expect(encodeQueryOptions("$filter=CardName eq 'A#B'")).toContain("A%23B");
  });

  it("单引号 → %27", () => {
    expect(encodeQueryOptions("$filter=CardName eq 'Acme'")).toContain("%27Acme%27");
  });

  it("中文 → 百分号编码", () => {
    expect(encodeQueryOptions("$filter=CardName eq '张三'")).toContain(
      "%E5%BC%A0%E4%B8%89"
    );
  });

  it("幂等：已编码的 %xx 不重复编码", () => {
    expect(encodeQueryOptions("$filter=CardName eq %27Acme%27")).toBe(
      "$filter=CardName%20eq%20%27Acme%27"
    );
  });

  it("无引号的纯参数原样保留", () => {
    expect(encodeQueryOptions("$top=5&$orderby=DocDate desc")).toBe(
      "$top=5&$orderby=DocDate%20desc"
    );
  });
});

// ============================================================
// 4. 端到端：sapApiCallExecutor（mock fetch，抓取实际请求）
// ============================================================

describe("sapApiCallExecutor（与 curl 等价）", () => {
  const realFetch = global.fetch;

  function mockFetchOnce(handler: (url: string, opts: RequestInit) => Response | Promise<Response>) {
    global.fetch = jest.fn(async (url: any, opts: any) => handler(String(url), opts)) as any;
  }

  afterEach(() => {
    global.fetch = realFetch;
  });

  it("GET $filter 带空格和单引号 → 编码与 curl 等价（另注入设计内 $top+白名单）", async () => {
    let captured = "";
    mockFetchOnce((url) => {
      captured = url;
      return new Response(JSON.stringify({ value: [] }), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "BusinessPartners", method: "GET", queryOptions: "$filter=CardName eq 'Acme Corp' and CardType eq 'c'" },
      { baseUrl: "https://b1s.alphafina.cn/b1s/v1" }
    );
    expect(captured).toBe(
      "https://b1s.alphafina.cn/b1s/v1/BusinessPartners?$top=20&$select=CardCode,CardName,CardType,GroupCode,Phone1,Valid&$filter=CardName%20eq%20%27Acme%20Corp%27%20and%20CardType%20eq%20%27c%27"
    );
  });

  it("GET $filter 值含 & → %26 编码，不拆参数", async () => {
    let captured = "";
    mockFetchOnce((url) => {
      captured = url;
      return new Response(JSON.stringify({ value: [] }), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "Items", method: "GET", queryOptions: "$filter=ItemCode eq 'A&B'" },
      { baseUrl: "https://b1s.alphafina.cn/b1s/v1" }
    );
    expect(captured).toContain("A%26B");
    expect(captured.split("?").length).toBe(2);
  });

  it("GET Orders 不带参数 → 注入白名单 $select + $top，不含错误字段", async () => {
    let captured = "";
    mockFetchOnce((url) => {
      captured = url;
      return new Response(JSON.stringify({ value: [] }), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET" },
      { baseUrl: "https://b1s.alphafina.cn/b1s/v1" }
    );
    expect(captured).toContain("$top=20");
    expect(captured).toContain("$select=");
    expect(captured).not.toContain("RoundDif");
    expect(captured).not.toContain("DocumentLines");
  });

  it("FunctionImport GET → 不带 $top/$select", async () => {
    let captured = "";
    mockFetchOnce((url) => {
      captured = url;
      return new Response(JSON.stringify(0), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "SBOBobService_GetCurrencyRate", method: "GET" },
      { baseUrl: "https://b1s.alphafina.cn/b1s/v1" }
    );
    expect(captured).toBe("https://b1s.alphafina.cn/b1s/v1/SBOBobService_GetCurrencyRate");
  });

  it("POST Orders → method/Content-Type/body 正确", async () => {
    let capturedOpts: RequestInit | null = null;
    mockFetchOnce((_url, opts) => {
      capturedOpts = opts;
      return new Response(JSON.stringify({ DocEntry: 1 }), { status: 201 });
    });
    const body = { CardCode: "C001", DocDate: "2026-08-06", DocumentLines: [{ ItemCode: "I1", Quantity: 2 }] };
    await sapApiCallExecutor(
      { entitySet: "Orders", method: "POST", body },
      { baseUrl: "https://b1s.alphafina.cn/b1s/v1" }
    );
    expect(capturedOpts!.method).toBe("POST");
    expect(JSON.parse(capturedOpts!.body as string)).toEqual(body);
  });

  it("PATCH/DELETE 用主键路径，GET 用 id 时忽略 id", async () => {
    const urls: string[] = [];
    mockFetchOnce((url) => {
      urls.push(url);
      return new Response(JSON.stringify({}), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "Orders", method: "PATCH", id: "123", body: { Comments: "x" } },
      { baseUrl: "https://x" }
    );
    await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET", id: "123", queryOptions: "$top=1" },
      { baseUrl: "https://x" }
    );
    expect(urls[0]).toContain("/Orders('123')");
    expect(urls[1]).toBe(
      "https://x/Orders?$select=DocEntry,DocNum,DocDate,CardCode,CardName,DocTotal,DocCurrency,DocumentStatus&$top=1"
    );
  });

  it("代理链路：带 X-Company-DB / X-Tenant-Id / 原样 Cookie", async () => {
    let capturedHeaders: Record<string, string> = {};
    mockFetchOnce((_url, opts) => {
      capturedHeaders = opts.headers as Record<string, string>;
      return new Response(JSON.stringify({ value: [] }), { status: 200 });
    });
    await sapApiCallExecutor(
      { entitySet: "BusinessPartners", method: "GET", queryOptions: "$top=1" },
      {
        baseUrl: "https://b1s.alphafina.cn/b1s/v1",
        cookie: "B1SESSION=abc; ROUTEID=.node5",
        companyDb: "XSM_ZSK",
        tenantId: "tenant-1",
      }
    );
    expect(capturedHeaders["X-Company-DB"]).toBe("XSM_ZSK");
    expect(capturedHeaders["X-Tenant-Id"]).toBe("tenant-1");
    expect(capturedHeaders["Cookie"]).toBe("B1SESSION=abc; ROUTEID=.node5");
  });

  it("400 响应 → hint 包含 SL 具体错误信息", async () => {
    mockFetchOnce(() =>
      new Response(
        JSON.stringify({
          error: { code: "1", message: { lang: "en", value: "Property 'RoundDif' does not exist" } },
        }),
        { status: 400 }
      )
    );
    const result = await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET", queryOptions: "$top=1" },
      { baseUrl: "https://x" }
    );
    expect(result.ok).toBe(false);
    expect(result.hint).toContain("RoundDif");
  });

  it("返回 $top 满条数 → hasMore=true", async () => {
    mockFetchOnce(() =>
      new Response(JSON.stringify({ value: new Array(20).fill({ DocEntry: 1 }) }), {
        status: 200,
      })
    );
    const result = await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET", queryOptions: "$top=20" },
      { baseUrl: "https://x" }
    );
    expect(result.hasMore).toBe(true);
  });

  it("不足 $top 条数 → hasMore=false", async () => {
    mockFetchOnce(() =>
      new Response(JSON.stringify({ value: new Array(5).fill({ DocEntry: 1 }) }), {
        status: 200,
      })
    );
    const result = await sapApiCallExecutor(
      { entitySet: "Orders", method: "GET", queryOptions: "$top=20" },
      { baseUrl: "https://x" }
    );
    expect(result.hasMore).toBe(false);
  });
});

// ============================================================
// 5. 响应处理：裁剪与噪声清理
// ============================================================

describe("trimNestedCollections / cleanODataNoise", () => {
  it("DocumentLines 保留 Price（与 lineFields 对齐）", () => {
    const data = {
      value: [
        {
          DocEntry: 1,
          DocumentLines: [{ LineNum: 0, ItemCode: "I1", Price: 12.5, UnitPrice: 10, Foo: 1 }],
        },
      ],
    };
    trimNestedCollections(data.value[0] as any);
    const line = (data.value[0] as any).DocumentLines[0];
    expect(line.Price).toBe(12.5);
    expect(line.Foo).toBeUndefined();
  });

  it("清理 odata.* 噪声", () => {
    const data: any = { "odata.metadata": "x", value: [{ DocEntry: 1, "odata.etag": "e" }] };
    cleanODataNoise(data);
    expect(data["odata.metadata"]).toBeUndefined();
    expect(data.value[0]["odata.etag"]).toBeUndefined();
  });
});
