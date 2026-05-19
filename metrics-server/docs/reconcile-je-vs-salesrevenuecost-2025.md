# 口径对齐：JE 毛利 vs SalesRevenueCost 毛利（2025）

**生成日期**：2026-03-04  
**数据源**：SAP B1 HANA（`ZZZZ_KAIMAYMTC_TEST`）  
**视图**：`MTC_VW_AI_JournalEntry` · `MTC_VW_AI_SalesRevenueCost`  
**分析期**：2025-01-01（含）~ 2026-01-01（不含）  

本文件用于解释：为什么 `MTC_VW_AI_JournalEntry`（JE，会计口径）算出来的毛利率 **24.11%**，与 `MTC_VW_AI_SalesRevenueCost`（交易口径）算出来的毛利率 **18.75%** 存在约 **5.36pp** 的差异，以及差异主要来自哪些 **JE 5xx 科目**与**凭证类型（TransType）**。

---

## 1. 口径定义（用于对账）

### 1.1 JournalEntry（JE，会计口径）

- **收入（4xx）**：`SUM(CASE WHEN Account LIKE '4%' THEN Credit - Debit END)`
- **5xx 净额**：`SUM(CASE WHEN Account LIKE '5%' THEN Credit - Debit END)`（通常为负）
- **COGS（正数展示）**：`SUM(CASE WHEN Account LIKE '5%' THEN Debit - Credit END)`
- **毛利**：`收入 + 5xx净额`（等价于 `收入 - COGS`）

> 说明：该客户科目体系里，`502xxxx` 也落在 `5xx` 前缀下（例如促销补贴/返利/运费回收等），在 JE 里体现为 **负的 5xx（成本抵减/对冲）**，会显著抬升账面毛利率；这也是对账的核心。

### 1.2 SalesRevenueCost（交易口径）

- **收入**：`SUM(LineTotal)`
- **销售成本**：`SUM(SalesCost)`（为空按 0 处理）
- **毛利**：`收入 - 销售成本`
- **去重规则**：对 `MTC_VW_AI_SalesRevenueCost` 先做 `SELECT DISTINCT`（字段组合与《毛利率分析 V1.5》一致），避免视图内完全重复行导致汇总放大。

---

## 2. 总体对账（2025）

| 口径 | 收入 | 成本（正数） | 毛利 | 毛利率 |
|------|-----:|-------------:|-----:|------:|
| JE（4xx/5xx） | 32,634,402 | 24,765,547 | 7,868,854 | 24.11% |
| SalesRevenueCost（LineTotal/SalesCost） | 32,127,557 | 26,103,779 | 6,023,777 | 18.75% |
| **差异（JE − SRC）** | **+506,845** | **−1,338,232** | **+1,845,077** | **+5.36pp** |

> 成本差异为负，表示 **JE 口径的“5xx净成本”比 SalesRevenueCost 的 SalesCost 更低**（JE 有更多成本抵减/对冲项）。

---

## 3. 差异对账桥（按 JE 5xx 科目层级拆分）

把 JE 的 5xx 拆成两块：
- `501xxxx`：更接近“商品出库成本 + 传统成本耗用”（偏 COGS 本体）
- `502xxxx`：大量为“补贴/返利/运费回收/佣金激励”等 **成本抵减或其他收入**（但科目仍以 5 开头）

**对账桥（对毛利的影响，SGD）**：

| 项目 | 对毛利差异的贡献 | 解释 |
|------|-----------------:|------|
| 起点：SalesRevenueCost 毛利 | 6,023,777 | 交易口径毛利（不含 JE 5xx 抵减项） |
| + 收入差异（JE 4xx − SRC LineTotal） | +506,845 | JE 的 4xx 收入比 SRC 更高（含部分非销售单据类 4xx、以及口径/去重导致的差异） |
| − `501xxxx` 成本差异（JE 501 − SRC SalesCost） | −636,459 | JE 的 `501xxxx` 成本高于 SRC 的 SalesCost（可能来自非销售单据类成本、或 SRC 覆盖/去重造成的差异） |
| + `502xxxx` 成本抵减（JE 502 合计为负） | +1,974,690 | 促销补贴/返利/佣金激励/运费回收等在 JE 中以 **负 5xx** 体现，抬升账面毛利 |
| **终点：JE 毛利** | **7,868,854** |（与 JE 汇总存在 ±1 的四舍五入差）|

---

## 4. JE 5xx 明细：关键科目（Account）贡献

### 4.1 JE `501xxxx`：主要“正向成本”（Top）

| Account | 科目 | 金额（成本，正数） |
|--------:|------|-------------------:|
| 501030100 | COGS FOOD - CONFECTIONARY | 15,492,840 |
| 501030500 | COGS FOOD - LIQUEUR | 4,977,750 |
| 501050100 | COGS FURNITURE - LOCAL PURCHASE | 2,412,611 |
| 501030400 | COGS FOOD - DRINK | 1,060,374 |
| 501031300 | COGS FOOD - LIQUEUR (BEER) | 642,697 |
| 501031400 | COGS FOOD - TOBACCO | 626,127 |
| 501117000 | TRADING TERM（成本项） | 502,145 |
| 501070000 | CARRIAGE OUTWARDS AND HANDLING CHARGES | 420,798 |
| 501030200 | COGS FOOD - GROCERY | 178,508 |
| 501180000 | RENTAL OF SUPERMARKET/BOOKSHOP SPACE | 159,570 |
| 501130000 | EXPIRED STOCK-COGS FOOD | 122,889 |

> 备注：上表说明 JE 的“5xx”不只是出库成本（COGS），还混入了部分渠道条款、配送、场地费、过期损耗等“成本类”项目；而 SalesRevenueCost 的 SalesCost 更接近“出库成本”。

### 4.2 JE `502xxxx`：主要“成本抵减/对冲”（Top）

| Account | 科目 | 金额（成本抵减，负数） |
|--------:|------|-----------------------:|
| 502100000 | SUBSIDY PROMOTION FEE RECEIVED | -1,713,522 |
| 502010000 | NESTLE COMMISSION/INCENTIVE RECEIVED | -115,510 |
| 502030000 | FREIGHT OUTWARD RECEIVED | -104,574 |
| 502140000 | FD INTEREST RECEIVED | -17,737 |
| 502160000 | HANDLING CHARGE RECEIVED | -16,936 |
| 502180000 | INCOME RELATED GRANTS | -15,015 |
| 502110000 | TRANSPORT CHARGES RECEIVED | -12,596 |

> 核心结论：**`502100000`（促销补贴/返利）单科目就贡献了约 1.71M 的“成本抵减”，是 JE 毛利率显著高于 SalesRevenueCost 的首要原因之一。**

---

## 5. JE 5xx 明细：按凭证类型（TransType）拆分

下面表格按 JE 的 `TransType` 汇总 5xx（用 `Debit−Credit` 展示为正数成本；为负则代表成本抵减/冲回）。同时拆出 `501xxxx` 与 `502xxxx`，便于定位“抵减项主要从哪里来”。

| TransType | 501 成本 | 502 抵减 | 5xx 合计 |
|----------:|---------:|---------:|---------:|
| 15 | 26,756,127 | 0 | 26,756,127 |
| 18 | 510,964 | 415,227 | 926,191 |
| 30 | 260,885 | -41,005 | 219,881 |
| 162 | 0 | 192,495 | 192,495 |
| 60 | 179,995 | 0 | 179,995 |
| 14 | -849,065 | 14,839 | -834,226 |
| 19 | 14,224 | -2,390,932 | -2,376,708 |

**解读要点**：
- `TransType=15` 是 5xx 的主要来源（通常对应销售出库/交货引发的 COGS）。
- `TransType=19`（常见为 A/P Invoice）贡献了 **大量 502 抵减项**（例如 `502100000` 促销补贴/返利），使 JE 的净 5xx 成本显著下降。
- `TransType=14/16` 等为销售退货/贷项凭证相关的冲回（成本为负），会对“净 5xx”产生显著影响。

> 注：TransType 为 SAP B1 对象类型编码；若需在报告中写成“凭证类型名称”，建议后续补一张 TransType→对象名称映射表（或在 view 中直接暴露名称字段）。

---

## 6. SalesRevenueCost 侧的关键异常提示（与对账直接相关）

SalesRevenueCost（去重后）按单据类型汇总如下：

| TransType | 收入 | SalesCost | 毛利 | 毛利率 |
|----------|-----:|---------:|-----:|------:|
| A/R Invoice based on Delivery | 35,266,960 | 27,290,089 | 7,976,871 | 22.62% |
| Direct A/R Invoice | 417,426 | 64,925 | 352,500 | 84.45% |
| A/R Credit Memo based on Invoice | -1,625,048 | -1,189,079 | -435,970 | 26.83% |
| Direct A/R Credit Memo | -1,932,274 | -62,403 | -1,869,871 | 96.77% |

> ⚠ `Direct A/R Credit Memo` 的 **SalesCost 冲回极小**（-62K 对 -1.93M 收入），会把交易口径的毛利显著拉低。  
> 这类差异会导致：当把 SalesRevenueCost 当作“经营毛利”的唯一口径时，退货/贷项凭证的成本冲回不足会造成 GM 偏低或误判。

---

## 7. 建议结论（口径对齐后如何使用）

- **对外/董事会口径**：以 JE（4xx/5xx）为准；`502xxxx` 这类“成本抵减/补贴返利”是否纳入毛利，应与财务口径统一（当前 JE 已纳入）。
- **经营抓手口径（SKU/渠道/客户）**：用 SalesRevenueCost 做“可下钻的产品毛利”，但需补齐/修正：
  - `Direct A/R Credit Memo` 的 SalesCost 冲回逻辑（优先从数据视图层解决：关联 base invoice/delivery 获取原始出库成本；或明确业务上不允许“无基单贷项凭证”）。
  - 明确 Trading Term/补贴返利（JE 502 科目）应如何分摊回 SKU/渠道（否则“经营毛利”无法复刻账面毛利）。

---

## 附：SQL（已执行）

```sql
-- JE 总览
SELECT
  SUM(CASE WHEN "Account" LIKE '4%' THEN "Credit"-"Debit" ELSE 0 END) AS revenue_4xx,
  SUM(CASE WHEN "Account" LIKE '5%' THEN "Credit"-"Debit" ELSE 0 END) AS cogs_5xx_net
FROM "ZZZZ_KAIMAYMTC_TEST"."MTC_VW_AI_JournalEntry"
WHERE "RefDate" >= '2025-01-01' AND "RefDate" < '2026-01-01';

-- SalesRevenueCost 总览（去重口径与报告一致）
WITH src AS (
  SELECT DISTINCT
    "TransType","DocType","DocNum","CardCode","CardName","GroupName","SlpName",
    "BaseType","BaseEntry","BaseLine","SONumber","ItemCode","ItmsGrpNam",
    "LineTotal","SalesCost","OcrCode","OcrCode2","CostCenterName2","Code","Category"
  FROM "ZZZZ_KAIMAYMTC_TEST"."MTC_VW_AI_SalesRevenueCost"
  WHERE "Category"='2025'
)
SELECT
  SUM("LineTotal") AS revenue,
  SUM(COALESCE("SalesCost",0)) AS sales_cost
FROM src;
```

