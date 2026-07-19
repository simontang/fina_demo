import z from "zod";
import { registerToolLattice } from "@axiom-lattice/core";

const BASE_URL =
  process.env.SAP_SERVICE_LAYER_URL || "https://b1s.alphafina.cn/b1s/v1";

// ============================================================
// API 元数据
// ============================================================

interface ApiEntry {
  name: string;
  kind: "EntitySet" | "FunctionImport";
  entityType?: string;
  primaryKey?: string;
  domain: string;
  description: string;
  fields: string[];
}

// ============================================================
// 支持 $expand 的文档类实体
// ============================================================

const EXPAND_FIELDS: Record<string, string[]> = {
  "DocumentLines": [
    "LineNum", "ItemCode", "ItemDescription", "Quantity", "UnitPrice",
    "WarehouseCode", "Total", "Currency", "TaxCode", "PriceAfterVAT",
    "UoMEntry", "UoMCode",
  ],
  "DocumentAdditionalExpenses": [
    "LineNum", "ExpenseCode", "LineTotal", "TaxCode", "VatGroup",
  ],
};

const ENTITIES_WITH_LINES = new Set([
  "Orders", "DeliveryNotes", "Invoices", "Quotations", "CreditNotes",
  "Returns", "Drafts", "PurchaseOrders", "PurchaseDeliveryNotes",
  "PurchaseInvoices", "PurchaseReturns", "PurchaseQuotations",
  "DownPayments", "InventoryGenEntries", "InventoryGenExits",
]);

const API_LIST: ApiEntry[] = [
  // ========== Business Partner ==========
  {
    name: "BusinessPartners",
    kind: "EntitySet",
    entityType: "SAPB1.BusinessPartner",
    primaryKey: "CardCode",
    domain: "BusinessPartner",
    description: "客户/供应商主数据",
    fields: [
      "CardCode", "CardName", "CardType", "GroupCode", "Currency",
      "Phone1", "Phone2", "EmailAddress", "Address", "City", "Country",
      "SalesPersonCode", "PriceListNum", "CreditLimit",
      "CurrentAccountBalance", "OpenOrdersBalance", "OpenDeliveryNotesBalance",
      "PayTermsGrpCode", "VatGroup", "VatLiable", "FederalTaxID",
      "Valid", "Frozen", "CompanyPrivate", "CreateDate", "CreateTime", "UpdateDate", "UpdateTime",
      "BPAddresses", "ContactEmployees",
    ],
  },
  {
    name: "BusinessPartnerGroups",
    kind: "EntitySet",
    entityType: "SAPB1.BusinessPartnerGroup",
    primaryKey: "Code",
    domain: "BusinessPartner",
    description: "BP 分组",
    fields: ["Code", "Name", "Type"],
  },
  {
    name: "SalesPersons",
    kind: "EntitySet",
    entityType: "SAPB1.SalesPerson",
    primaryKey: "SalesEmployeeCode",
    domain: "BusinessPartner",
    description: "销售雇员",
    fields: ["SalesEmployeeCode", "SalesEmployeeName", "CommissionGroup", "Valid"],
  },

  // ========== Item / Product ==========
  {
    name: "Items",
    kind: "EntitySet",
    entityType: "SAPB1.Item",
    primaryKey: "ItemCode",
    domain: "Item / Product",
    description: "物料主数据",
    fields: [
      "ItemCode", "ItemName", "ForeignName", "ItemsGroupCode",
      "BarCode", "SalesItem", "PurchaseItem", "InventoryItem",
      "SalesUnit", "PurchaseUnit", "InventoryUOM",
      "QuantityOnStock", "AvgStdPrice", "DefaultWarehouse",
      "ManageSerialNumbers", "ManageBatchNumbers",
      "SalesVATGroup", "PurchaseVATGroup",
      "Valid", "Frozen", "CreateDate", "UpdateDate",
      "ItemPrices", "ItemWarehouseInfoCollection",
    ],
  },
  {
    name: "ItemGroups",
    kind: "EntitySet",
    entityType: "SAPB1.ItemGroups",
    primaryKey: "Number",
    domain: "Item / Product",
    description: "物料组",
    fields: ["Number", "GroupName", "CommissionGroup"],
  },
  {
    name: "PriceLists",
    kind: "EntitySet",
    entityType: "SAPB1.PriceList",
    primaryKey: "PriceListNo",
    domain: "Item / Product",
    description: "价格清单",
    fields: [
      "PriceListNo", "PriceListName", "DefaultPrimeCurrency",
      "DefaultAdditionalCurrency1", "DefaultAdditionalCurrency2",
      "BasePriceList", "Factor", "Active", "IsGrossPrice",
      "RoundingMethod", "GroupNum",
    ],
  },
  {
    name: "BarCodes",
    kind: "EntitySet",
    entityType: "SAPB1.BarCode",
    primaryKey: "AbsEntry",
    domain: "Item / Product",
    description: "条码",
    fields: ["AbsEntry", "ItemCode", "UoMEntry", "BarCode", "FreeText"],
  },

  // ========== Orders — Sales ==========
  {
    name: "Orders",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售订单",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate", "TaxDate",
      "CardCode", "CardName", "Address", "DocTotal", "DocCurrency",
      "SalesPersonCode", "Confirmed", "Cancelled", "DocumentStatus",
      "Comments", "Reference1", "Reference2", "NumAtCard",
      "VatSum", "RoundDif", "DiscountPercent",
      "PaymentGroupCode", "Project",
      "DocumentLines",
    ],
  },
  {
    name: "DeliveryNotes",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "交货单",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate", "TaxDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "SalesPersonCode", "Confirmed", "Cancelled", "DocumentStatus",
      "Comments", "NumAtCard",
      "DocumentLines",
    ],
  },
  {
    name: "Invoices",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售发票 (应收)",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate", "TaxDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "SalesPersonCode", "Confirmed", "Cancelled", "DocumentStatus",
      "Comments", "NumAtCard", "VatSum",
      "DocumentLines",
    ],
  },
  {
    name: "Quotations",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售报价单",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "SalesPersonCode", "Comments", "DocumentStatus",
      "DocumentLines",
    ],
  },
  {
    name: "CreditNotes",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售贷项凭证",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "Comments",
      "DocumentLines",
    ],
  },
  {
    name: "Returns",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售退货",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "Comments",
      "DocumentLines",
    ],
  },
  {
    name: "DownPayments",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "预收款",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "DownPaymentType", "DownPaymentAmount",
      "DocumentLines",
    ],
  },
  {
    name: "Drafts",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "销售草稿/采购草稿（支持创建、查询、更新、删除）",
    fields: [
      "DocEntry", "DocNum", "DocObjectCode", "DocType", "DocDate", "DocDueDate", "TaxDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "Comments", "SalesPersonCode", "Confirmed", "Cancelled", "DocumentStatus",
      "U_YWLX", "U_YWLX2",
      "DocumentLines",
    ],
  },

  // ========== Orders — Purchase ==========
  {
    name: "PurchaseOrders",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "采购订单",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate", "TaxDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "SalesPersonCode", "Confirmed", "Cancelled", "DocumentStatus",
      "Comments", "NumAtCard",
      "DocumentLines",
    ],
  },
  {
    name: "PurchaseDeliveryNotes",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "采购收货单",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "Comments",
      "DocumentLines",
    ],
  },
  {
    name: "PurchaseInvoices",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "采购发票 (应付)",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate", "DocDueDate",
      "CardCode", "CardName", "DocTotal", "DocCurrency",
      "Comments",
      "DocumentLines",
    ],
  },
  {
    name: "PurchaseReturns",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "采购退货",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "Comments",
      "DocumentLines",
    ],
  },
  {
    name: "PurchaseQuotations",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "采购报价",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "CardCode", "CardName", "DocTotal", "Comments",
      "DocumentLines",
    ],
  },

  // ========== Inventory / Warehouse ==========
  {
    name: "InventoryGenEntries",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "库存收货",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "Comments", "JournalMemo", "Reference1", "Reference2",
      "WareHouseUpdateType",
      "DocumentLines",
    ],
  },
  {
    name: "InventoryGenExits",
    kind: "EntitySet",
    entityType: "SAPB1.Document",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "库存发货",
    fields: [
      "DocEntry", "DocNum", "DocType", "DocDate",
      "Comments", "JournalMemo", "Reference1", "Reference2",
      "DocumentLines",
    ],
  },
  {
    name: "StockTransfers",
    kind: "EntitySet",
    entityType: "SAPB1.StockTransfer",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "库存转储",
    fields: [
      "DocEntry", "DocNum", "DocDate", "DueDate",
      "FromWarehouse", "ToWarehouse", "Comments",
      "Reference1", "Reference2",
    ],
  },
  {
    name: "InventoryPostings",
    kind: "EntitySet",
    entityType: "SAPB1.InventoryPosting",
    primaryKey: "DocumentEntry",
    domain: "Inventory / Warehouse",
    description: "库存过账",
    fields: [
      "DocumentEntry", "DocumentNumber", "PostingDate", "CountDate",
      "Remarks", "Reference2", "PriceList",
    ],
  },
  {
    name: "InventoryCountings",
    kind: "EntitySet",
    entityType: "SAPB1.InventoryCounting",
    primaryKey: "DocumentEntry",
    domain: "Inventory / Warehouse",
    description: "库存盘点",
    fields: [
      "DocumentEntry", "DocumentNumber", "CountDate",
      "SingleCounterType", "SingleCounterID", "Remarks",
    ],
  },
  {
    name: "Warehouses",
    kind: "EntitySet",
    entityType: "SAPB1.Warehouse",
    primaryKey: "WarehouseCode",
    domain: "Inventory / Warehouse",
    description: "仓库定义",
    fields: [
      "WarehouseCode", "WarehouseName",
      "Street", "City", "Country", "Location",
      "DropShip", "Nettable", "Inactive",
      "EnableBinLocations", "ManageSerialAndBatchNumbers",
    ],
  },
  {
    name: "BinLocations",
    kind: "EntitySet",
    entityType: "SAPB1.BinLocation",
    primaryKey: "AbsEntry",
    domain: "Inventory / Warehouse",
    description: "库位",
    fields: [
      "AbsEntry", "Warehouse", "BinCode", "Description",
      "Sublevel1", "Sublevel2", "Sublevel3", "Sublevel4",
      "Inactive", "MinimumQty", "MaximumQty",
    ],
  },
  {
    name: "BatchNumberDetails",
    kind: "EntitySet",
    entityType: "SAPB1.BatchNumberDetail",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "批次号明细",
    fields: [
      "DocEntry", "ItemCode", "ItemDescription",
      "Batch", "BatchAttribute1", "BatchAttribute2",
      "AdmissionDate", "ExpirationDate", "ManufacturingDate",
    ],
  },
  {
    name: "SerialNumberDetails",
    kind: "EntitySet",
    entityType: "SAPB1.SerialNumberDetail",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "序列号明细",
    fields: [
      "DocEntry", "ItemCode", "ItemDescription",
      "SerialNumber", "MfrSerialNo",
      "AdmissionDate", "ExpirationDate",
    ],
  },

  // ========== Function Imports (常用) ==========
  {
    name: "ItemsService_InitData",
    kind: "FunctionImport",
    entityType: "SAPB1.Item",
    domain: "Item / Product",
    description: "初始化物料数据",
    fields: [],
  },
  {
    name: "BusinessPartnersService_InitData",
    kind: "FunctionImport",
    entityType: "SAPB1.BusinessPartner",
    domain: "BusinessPartner",
    description: "初始化 BP 数据",
    fields: [],
  },
  {
    name: "OrdersService_InitData",
    kind: "FunctionImport",
    entityType: "SAPB1.Document",
    domain: "Document",
    description: "初始化订单数据",
    fields: [],
  },
  {
    name: "InvoicesService_InitData",
    kind: "FunctionImport",
    entityType: "SAPB1.Document",
    domain: "Document",
    description: "初始化发票数据",
    fields: [],
  },

  // ========== Pricing / Cost / Rate ==========
  {
    name: "SpecialPrices",
    kind: "EntitySet",
    entityType: "SAPB1.SpecialPrice",
    primaryKey: "ItemCode+CardCode",
    domain: "Item / Product",
    description: "特殊价格（BP/客户特定价格）",
    fields: [
      "ItemCode", "CardCode", "Price", "Currency",
      "DiscountPercent", "PriceListNum", "AutoUpdate", "SourcePrice",
    ],
  },
  {
    name: "LandedCosts",
    kind: "EntitySet",
    entityType: "SAPB1.LandedCost",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "到岸成本/附加成本凭证",
    fields: ["DocEntry", "DocNum", "DocDate", "CardCode", "Comments"],
  },
  {
    name: "LandedCostsCodes",
    kind: "EntitySet",
    entityType: "SAPB1.LandedCostsCode",
    primaryKey: "Code",
    domain: "Document",
    description: "到岸成本代码定义",
    fields: ["Code", "Name"],
  },
  {
    name: "CompanyService_GetItemPrice",
    kind: "FunctionImport",
    entityType: "SAPB1.ItemPriceReturnParams",
    domain: "Item / Product",
    description: "根据参数查询物料价格（含历史明细）",
    fields: ["ItemPrices", "ItemUnitOfMeasurementCollection"],
  },
  {
    name: "SBOBobService_GetCurrencyRate",
    kind: "FunctionImport",
    domain: "Finance / Accounting",
    description: "获取货币汇率",
    fields: [],
  },
  {
    name: "SBOBobService_GetIndexRate",
    kind: "FunctionImport",
    domain: "Finance / Accounting",
    description: "获取指数汇率",
    fields: [],
  },
  {
    name: "SBOBobService_SetCurrencyRate",
    kind: "FunctionImport",
    domain: "Finance / Accounting",
    description: "设置货币汇率",
    fields: [],
  },
  {
    name: "Z20_COST",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_COST",
    primaryKey: "DocEntry",
    domain: "Item / Product",
    description: "自定义: 成本记录表（含周期/实例维度的成本历史）",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance", "Series",
      "Status", "RequestStatus", "Creator", "Remark",
      "Canceled", "Object", "CreateDate", "CreateTime",
    ],
  },
  {
    name: "Z20_CPAT",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_CPAT",
    primaryKey: "DocEntry",
    domain: "Item / Product",
    description: "自定义: 成本/价格分摊记录",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance",
      "Status", "Creator", "Remark", "CreateDate", "CreateTime",
    ],
  },
  {
    name: "Z20_OINP",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_OINP",
    primaryKey: "DocEntry",
    domain: "Document",
    description: "自定义: 采购订单输入价格记录",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance",
      "Status", "Creator", "Remark", "CreateDate", "CreateTime",
    ],
  },
  {
    name: "Z20_PWAG",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_PWAG",
    primaryKey: "DocEntry",
    domain: "Item / Product",
    description: "自定义: 工价/工序价格记录",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance",
      "Status", "Creator", "Remark", "CreateDate", "CreateTime",
    ],
  },
  {
    name: "Z20_HOLD",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_HOLD",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "自定义: 暂存/冻结库存记录",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance",
      "Status", "Creator", "Remark", "CreateDate", "CreateTime",
    ],
  },
  {
    name: "Z20_IMIT",
    kind: "EntitySet",
    entityType: "SAPB1.Z20_IMIT",
    primaryKey: "DocEntry",
    domain: "Inventory / Warehouse",
    description: "自定义: 库存初始化记录",
    fields: [
      "DocEntry", "DocNum", "Period", "Instance",
      "Status", "Creator", "Remark", "CreateDate", "CreateTime",
    ],
  },
];

// ============================================================
// Tool 1: sap_api_search
// ============================================================

function mapResult(e: ApiEntry): Record<string, unknown> {
  const r: Record<string, unknown> = {
    name: e.name,
    kind: e.kind,
    domain: e.domain,
    description: e.description,
    primaryKey: e.primaryKey || null,
    fields: e.fields,
    readySelect: e.fields.length > 0 ? `$select=${e.fields.join(",")}` : undefined,
    hint:
      e.kind === "EntitySet"
        ? `${e.name} — ${e.description}。主键: ${e.primaryKey}。调用 sap_api_call 进行 CRUD 操作。`
        : `${e.name} — ${e.description}。调用 sap_api_call 执行此方法。`,
  };
  if (ENTITIES_WITH_LINES.has(e.name)) {
    r.expand = ["DocumentLines", "DocumentAdditionalExpenses"];
    r.lineFields = ["LineNum", "ItemCode", "ItemDescription", "Quantity", "UnitPrice", "Price", "WarehouseCode", "UoMEntry", "UoMCode", "VatGroup", "Currency", "TaxCode", "PriceAfterVAT"];
  }
  if (e.name === "Items") {
    r.expand = ["ItemPrices", "ItemWarehouseInfoCollection"];
  }
  if (e.name === "BusinessPartners") {
    r.expand = ["BPAddresses", "ContactEmployees"];
  }
  return r;
}

registerToolLattice(
  "sap_api_search",
  {
    name: "sap_api_search",
    description:
      "搜索 SAP B1 Service Layer API 接口元数据，覆盖 BP/物料/订单/库存四大领域。" +
      "返回接口名、主键、字段列表、可选 expand 导航。" +
      "可用 discover 模式发现关联接口（如搜 SalesPerson 也能找到 SalesPersons API）。" +
      "搜索结果中的 readySelect 可直接复制到 $select。",
    needUserApprove: false,
    schema: z.object({
      query: z
        .string()
        .optional()
        .describe(
          "搜索关键词（可选，不传则返回全部接口）。" +
            "API 名称如 'BusinessPartners', 'Orders', 'Items', 'PurchaseOrders'，" +
            "或中文描述如 '客户', '订单', '物料', '库存', '采购', '仓库'"
        ),
      domain: z
        .string()
        .optional()
        .describe(
          "按领域过滤: 'BusinessPartner'(BP), 'Item / Product'(物料), 'Document'(订单/发票), 'Inventory / Warehouse'(库存/仓库)"
        ),
      maxResults: z.number().optional().default(20).describe("最大返回条数"),
    }),
  },
  async (input) => {
    const q = input.query?.toLowerCase();
    const max = input.maxResults ?? 20;

    const hasQuery = q && q.trim().length > 0;

    const domainHints: Record<string, string> = {
      "客户": "BusinessPartner",
      "供应商": "BusinessPartner",
      "bp": "BusinessPartner",
      "物料": "Item / Product",
      "产品": "Item / Product",
      "商品": "Item / Product",
      "货品": "Item / Product",
      "订单": "Document",
      "销售": "Document",
      "采购": "Document",
      "发票": "Document",
      "交货": "Document",
      "报价": "Document",
      "草稿": "Document",
      "draft": "Document",
      "暂存": "Document",
      "库存": "Inventory / Warehouse",
      "仓库": "Inventory / Warehouse",
      "库位": "Inventory / Warehouse",
      "批次": "Inventory / Warehouse",
      "序列号": "Inventory / Warehouse",
      "收货": "Inventory / Warehouse",
      "发货": "Inventory / Warehouse",
      "转储": "Inventory / Warehouse",
      "盘点": "Inventory / Warehouse",
      "过账": "Inventory / Warehouse",
      "价格": "Item / Product",
      "成本": "Item / Product",
      "汇率": "Finance / Accounting",
      "到岸": "Document",
    };

    const hintedDomain = q ? (domainHints[q] || undefined) : undefined;
    const effectiveDomain = input.domain || hintedDomain;

    const filtered = API_LIST.filter((e) => {
      if (effectiveDomain && e.domain !== effectiveDomain) return false;
      return true;
    });

    if (!q) {
      const top = filtered.slice(0, max);
      const domainCounts: Record<string, number> = {};
      for (const e of top) domainCounts[e.domain] = (domainCounts[e.domain] || 0) + 1;
      return {
        query: input.query || null,
        domainFilter: effectiveDomain || null,
        totalMatches: filtered.length,
        domainsFound: domainCounts,
        results: top.map(mapResult),
        suggestion: undefined,
      };
    }

    const scored = filtered.map((e) => {
      let score = 0;
      const nameLo = e.name.toLowerCase();
      const descLo = e.description.toLowerCase();
      const typeLo = (e.entityType || "").toLowerCase();

      if (nameLo === q) score += 100;
      else if (nameLo.startsWith(q)) score += 60;
      else if (nameLo.includes(q)) score += 30;
      if (descLo.includes(q)) score += 20;
      if (typeLo.includes(q)) score += 10;

      for (const word of q.split(/[\s_\-/]+/).filter((w: string) => w.length >= 2)) {
        if (nameLo.includes(word)) score += 10;
        if (descLo.includes(word)) score += 5;
      }

      return { ...e, score };
    });

    const top = scored
      .filter((e) => e.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, max);

    const domainCounts: Record<string, number> = {};
    for (const e of top) domainCounts[e.domain] = (domainCounts[e.domain] || 0) + 1;

    return {
      query: input.query!,
      domainFilter: effectiveDomain || null,
      totalMatches: scored.filter((e) => e.score > 0).length,
      domainsFound: domainCounts,
      results: top.map(mapResult),
      suggestion:
        top.length === 0
          ? `未找到匹配 "${input.query}" 的接口。可用领域: BusinessPartner(客户/供应商), Item / Product(物料), Document(订单/发票), Inventory / Warehouse(库存/仓库)。尝试用英文名称搜索。`
          : undefined,
    };
  }
);

// ============================================================
// Tool 2: sap_api_call
// ============================================================

const SAP_COOKIE = process.env.SAP_B1SESSION
  ? `B1SESSION=${process.env.SAP_B1SESSION}; ROUTEID=.node0`
  : "";

registerToolLattice(
  "sap_api_call",
  {
    name: "sap_api_call",
    description:
      "执行 SAP B1 Service Layer 的 OData API 查询/创建/更新/删除。" +
      `Base: ${BASE_URL}。` +
      "⚠️ 先确认是否已通过 sap_api_search 查过字段列表，勿凭记忆编字段名。\n" +
      "⚠️ 草稿/draft/暂存 → 用 Drafts 接口，不是 Orders/Invoices 等正式单据。" +
      "Drafts 通过 DocObjectCode 区分单据类型(17=订单,13=发票,16=交货单,23=采购订单)。\n" +
      "$filter 操作符: eq/ne/gt/lt/ge/le/contains(f,'v')/startswith(f,'v')/endswith(f,'v')，多条件用 and/or。" +
      "字符串值必须单引号包裹。$orderby=Field desc 排序。\n\n" +
      "⚠️ SAP B1 实战经验:\n" +
      "1. 不要用 $expand！$expand 极易触发 400/500。改用 $select 包含嵌套字段，如 $select=DocEntry,DocNum,DocumentLines，\n" +
      "   DocumentLines 会自动作为嵌套 JSON 返回。ItemPrices、BPAddresses 等同理。\n" +
      "2. 不要用主键路径 /Orders('1173')，易 500。用 $filter=DocEntry eq 1173 代替。\n" +
      "3. 查单条记录时优先 $filter，而非传 id 参数。\n" +
      "400 多为特殊字符未编码(引号用 %27)；500 多为字段不存在或用错了 $expand；无结果则放宽 filter。\n" +
      "POST 创建: sap_api_search 返回的 lineFields 是 DocumentLines 子字段。" +
      "body 必含 DocObjectCode(DocType)、CardCode、DocDate；DocumentLines 为数组，每项必含 ItemCode、Quantity。" +
      "PATCH 只传变更字段；DELETE 需传 id。\n" +
      "GET 自动注入 $select+$top=20，手动传入可覆盖。嵌套集合(DocumentLines等)自动裁剪只保留常用字段，防 token 爆炸。" +
      "认证需 SAP_B1SESSION 环境变量。",
    needUserApprove: false,
    schema: z.object({
      entitySet: z
        .string()
        .describe("EntitySet 名称，如 'BusinessPartners', 'Orders', 'Items', 'PurchaseOrders'"),
      method: z.enum(["GET", "POST", "PATCH", "DELETE"]).describe("HTTP 方法"),
      id: z
        .string()
        .optional()
        .describe("主键值，GET/PATCH/DELETE 单个实体时使用。如 /BusinessPartners('C001')"),
      queryOptions: z
        .string()
        .optional()
        .describe(
          "OData 查询参数（不含 `?` 前缀）。" +
            "常用: $top=10, $select=CardCode,CardName, " +
            "$filter=CardName eq 'xxx' or contains(CardName,'xxx'), " +
            "$orderby=DocDate desc, $expand=DocumentLines。字符串值用单引号。" +
            "sap_api_search 返回的 readySelect 可直接复制到 $select。"
        ),
      body: z.record(z.unknown()).optional().describe("POST/PATCH 时的 JSON 请求体"),
    }),
  },
  async (input) => {
    const queryOptions = applyDefaultSelect(input.entitySet, input.method, input.id, input.queryOptions);
    const url = buildUrl(input.entitySet, input.method, input.id, queryOptions);
    const method = input.method;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
      "Accept-Encoding": "identity",
    };
    if (SAP_COOKIE) headers.Cookie = SAP_COOKIE;

    const fetchOptions: RequestInit = { method, headers };
    if ((method === "POST" || method === "PATCH") && input.body) {
      fetchOptions.body = JSON.stringify(input.body);
    }

    try {
      const res = await fetch(url, fetchOptions);
      const buffer = await res.arrayBuffer();
      const text = new TextDecoder("utf-8", { fatal: false }).decode(buffer);

      let data: unknown;
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }

      cleanODataNoise(data);

      const result: Record<string, unknown> = {
        ok: res.ok,
        status: res.status,
        statusText: res.statusText,
        data,
      };

      if (!res.ok) {
        result.error = `HTTP ${res.status} ${res.statusText}`;
        result.hint =
          res.status === 401
            ? "需要有效的 B1SESSION Cookie。请先通过 Login 端点获取，或设置 SAP_B1SESSION 环境变量。"
            : res.status === 404
              ? "接口或实体不存在，请检查 entitySet 名称和 id。"
              : undefined;
      }

      return result;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        ok: false,
        error: `请求失败: ${message}`,
        url,
        hint: "网络连接失败，请检查 SAP_SERVICE_LAYER_URL 是否正确。",
      };
    }
  }
);

// ============================================================
// Helpers
// ============================================================

function applyDefaultSelect(entitySet: string, method: string, id?: string, queryOptions?: string): string | undefined {
  if (method !== "GET") return queryOptions;

  const parts: string[] = [];

  if (!id && (!queryOptions || !/\$top\s*=/.test(queryOptions))) {
    parts.push("$top=20");
  }

  if (!queryOptions || !/\$select\s*=/.test(queryOptions)) {
    const entry = API_LIST.find(
      (e) => e.kind === "EntitySet" && e.name === entitySet
    );
    if (entry && entry.fields.length > 0) {
      parts.push(`$select=${entry.fields.join(",")}`);
    }
  }

  if (queryOptions) {
    parts.push(queryOptions);
  }

  return parts.length > 0 ? parts.join("&").replace(/'/g, "%27") : undefined;
}

function cleanODataNoise(data: unknown): void {
  if (!data || typeof data !== "object") return;
  if (Array.isArray(data)) {
    for (const item of data) cleanODataNoise(item);
    return;
  }
  const obj = data as Record<string, unknown>;

  delete obj["odata.metadata"];
  delete obj["odata.etag"];
  delete obj["odata.nextLink"];

  if (Array.isArray(obj.value)) {
    for (const record of obj.value) {
      if (record && typeof record === "object") {
        trimNestedCollections(record as Record<string, unknown>);
        delete (record as Record<string, unknown>)["odata.etag"];
      }
    }
  }

  for (const v of Object.values(obj)) {
    if (v && typeof v === "object") cleanODataNoise(v);
  }
}

function trimNestedCollections(obj: Record<string, unknown>): void {
  for (const [key, val] of Object.entries(obj)) {
    if (Array.isArray(val) && EXPAND_FIELDS[key]) {
      const keep = new Set(EXPAND_FIELDS[key]);
      for (const item of val) {
        if (item && typeof item === "object") {
          const record = item as Record<string, unknown>;
          for (const k of Object.keys(record)) {
            if (!keep.has(k)) delete record[k];
          }
          cleanODataNoise(item);
        }
      }
    }
  }
}

function buildUrl(
  entitySet: string,
  method: string,
  id?: string,
  queryOptions?: string
): string {
  let url = `${BASE_URL}/${entitySet}`;

  if (id && method !== "POST") {
    url += `('${encodeURIComponent(id)}')`;
  }

  if (queryOptions) {
    url += `?${queryOptions}`;
  }

  return url;
}
