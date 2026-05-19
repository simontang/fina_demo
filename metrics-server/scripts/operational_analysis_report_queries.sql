-- Local operational analysis report queries
-- Target database: postgresql:///fina_demo_local
-- Target schema: sbodemous

SET search_path TO "sbodemous", public;

-- 1. Core data coverage
SELECT 'OINV' AS object_name, COUNT(*) AS row_count, MIN("DocDate") AS min_date, MAX("DocDate") AS max_date
FROM "OINV"
UNION ALL
SELECT 'ORDR', COUNT(*), MIN("DocDate"), MAX("DocDate")
FROM "ORDR"
UNION ALL
SELECT 'OJDT', COUNT(*), MIN("RefDate"), MAX("RefDate")
FROM "OJDT"
UNION ALL
SELECT 'OITM', COUNT(*), MIN("CreateDate"), MAX("UpdateDate")
FROM "OITM";

-- 2. Management overview for 2025-01 to 2025-06
WITH kpi AS (
    SELECT
        SUM(COALESCE("LineTotal", 0)) AS revenue,
        SUM(COALESCE("SalesCost", 0)) AS cost
    FROM "MTC_VW_AI_SalesRevenueCost"
    WHERE "SODocDate" >= DATE '2025-01-01'
      AND "SODocDate" < DATE '2025-07-01'
)
SELECT
    ROUND(revenue, 2) AS revenue,
    ROUND(cost, 2) AS cost,
    ROUND(revenue - cost, 2) AS gross_profit,
    ROUND((revenue - cost) / NULLIF(revenue, 0) * 100, 2) AS gross_margin_pct
FROM kpi;

-- 3. Monthly revenue / cost / gross profit
WITH m AS (
    SELECT
        DATE_TRUNC('month', "SODocDate")::date AS month_begin,
        SUM(COALESCE("LineTotal", 0)) AS revenue,
        SUM(COALESCE("SalesCost", 0)) AS cost
    FROM "MTC_VW_AI_SalesRevenueCost"
    WHERE "SODocDate" >= DATE '2025-01-01'
      AND "SODocDate" < DATE '2025-07-01'
    GROUP BY 1
)
SELECT
    TO_CHAR(month_begin, 'YYYY-MM') AS month,
    ROUND(revenue, 2) AS revenue,
    ROUND(cost, 2) AS cost,
    ROUND(revenue - cost, 2) AS gross_profit,
    ROUND((revenue - cost) / NULLIF(revenue, 0) * 100, 2) AS gross_margin_pct
FROM m
ORDER BY month_begin;

-- 4. Revenue by customer group
SELECT
    COALESCE("GroupName", '(NoGroup)') AS customer_group,
    ROUND(SUM(COALESCE("LineTotal", 0)), 2) AS revenue,
    ROUND(SUM(COALESCE("SalesCost", 0)), 2) AS cost,
    ROUND(SUM(COALESCE("LineTotal", 0) - COALESCE("SalesCost", 0)), 2) AS gross_profit
FROM "MTC_VW_AI_SalesRevenueCost"
WHERE "SODocDate" >= DATE '2025-01-01'
  AND "SODocDate" < DATE '2025-07-01'
GROUP BY 1
ORDER BY revenue DESC NULLS LAST;

-- 5. Revenue by item group
SELECT
    COALESCE("ItmsGrpNam", '(NoItemGroup)') AS item_group,
    ROUND(SUM(COALESCE("LineTotal", 0)), 2) AS revenue,
    ROUND(SUM(COALESCE("SalesCost", 0)), 2) AS cost,
    ROUND(SUM(COALESCE("LineTotal", 0) - COALESCE("SalesCost", 0)), 2) AS gross_profit
FROM "MTC_VW_AI_SalesRevenueCost"
WHERE "SODocDate" >= DATE '2025-01-01'
  AND "SODocDate" < DATE '2025-07-01'
GROUP BY 1
ORDER BY revenue DESC NULLS LAST;

-- 6. Top customers and concentration
WITH t AS (
    SELECT
        COALESCE("CardCode", '(NoCard)') AS card_code,
        COALESCE("CardName", '(NoName)') AS card_name,
        SUM(COALESCE("LineTotal", 0)) AS revenue
    FROM "MTC_VW_AI_SalesRevenueCost"
    WHERE "SODocDate" >= DATE '2025-01-01'
      AND "SODocDate" < DATE '2025-07-01'
    GROUP BY 1, 2
),
total AS (
    SELECT SUM(revenue) AS total_revenue FROM t
)
SELECT
    t.card_code,
    t.card_name,
    ROUND(t.revenue, 2) AS revenue,
    ROUND(t.revenue / NULLIF(total.total_revenue, 0) * 100, 2) AS revenue_pct
FROM t
CROSS JOIN total
ORDER BY t.revenue DESC
LIMIT 10;

-- 7. Accounts receivable aging
SELECT
    COALESCE("Aging", '(Unknown)') AS aging_bucket,
    ROUND(SUM(COALESCE("BalDue", 0)), 2) AS balance_due,
    COUNT(*) AS line_count
FROM "MTC_VW_AI_CUSTBAL"
GROUP BY 1
ORDER BY balance_due DESC NULLS LAST;

-- 8. Dealer / payable balance aging
SELECT
    COALESCE("Aging", '(Unknown)') AS aging_bucket,
    ROUND(SUM(COALESCE("BalDue", 0)), 2) AS balance_due,
    COUNT(*) AS line_count
FROM "MTC_VW_AI_DEALBAL"
GROUP BY 1
ORDER BY balance_due DESC NULLS LAST;

-- 9. Inventory value snapshot
SELECT
    ROUND(SUM(COALESCE("OnHand", 0) * COALESCE("AvgPrice", 0)), 2) AS stock_value,
    ROUND(SUM(COALESCE("OnHand", 0)), 2) AS onhand_qty,
    COUNT(*) AS stock_rows
FROM "MTC_VW_AI_STOCK";

-- 10. Inventory by warehouse
SELECT
    COALESCE("WhsName", '(NoWarehouse)') AS warehouse_name,
    ROUND(SUM(COALESCE("OnHand", 0) * COALESCE("AvgPrice", 0)), 2) AS stock_value,
    ROUND(SUM(COALESCE("OnHand", 0)), 2) AS onhand_qty
FROM "MTC_VW_AI_STOCK"
GROUP BY 1
ORDER BY stock_value DESC NULLS LAST;

-- 11. Fund balance
SELECT
    COALESCE("BPLName", '(NoBPL)') AS bpl_name,
    ROUND(SUM(COALESCE("CurrTotal", 0)), 2) AS curr_total,
    ROUND(SUM(COALESCE("FcTotal", 0)), 2) AS fc_total
FROM "MTC_VW_AI_FUNDBAL"
GROUP BY 1
ORDER BY curr_total DESC NULLS LAST;

-- 12. Delivery timeliness
SELECT
    'sales' AS metric,
    ROUND(SUM(COALESCE("TimelyDelQty", 0)), 2) AS timely_qty,
    ROUND(SUM(COALESCE("DelayedDelQty", 0)), 2) AS delayed_qty,
    ROUND(
        SUM(COALESCE("TimelyDelQty", 0))
        / NULLIF(SUM(COALESCE("TimelyDelQty", 0)) + SUM(COALESCE("DelayedDelQty", 0)), 0)
        * 100,
        2
    ) AS timely_rate_pct
FROM "MTC_VW_AI_SalesTimelyDelRate"
UNION ALL
SELECT
    'purchase',
    ROUND(SUM(COALESCE("TimelyDelQty", 0)), 2),
    ROUND(SUM(COALESCE("DelayedDelQty", 0)), 2),
    ROUND(
        SUM(COALESCE("TimelyDelQty", 0))
        / NULLIF(SUM(COALESCE("TimelyDelQty", 0)) + SUM(COALESCE("DelayedDelQty", 0)), 0)
        * 100,
        2
    )
FROM "MTC_VW_AI_PurchaseTimelyDelRate";

-- 13. Revenue by salesperson
SELECT
    COALESCE("SlpName", '(NoSalesperson)') AS salesperson,
    ROUND(SUM(COALESCE("LineTotal", 0)), 2) AS revenue,
    ROUND(SUM(COALESCE("SalesCost", 0)), 2) AS cost,
    ROUND(SUM(COALESCE("LineTotal", 0) - COALESCE("SalesCost", 0)), 2) AS gross_profit
FROM "MTC_VW_AI_SalesRevenueCost"
WHERE "SODocDate" >= DATE '2025-01-01'
  AND "SODocDate" < DATE '2025-07-01'
GROUP BY 1
ORDER BY revenue DESC NULLS LAST;

-- 14. Revenue by region / province
SELECT
    COALESCE("Region", '(NoRegion)') AS region,
    COALESCE("Province", '(NoProvince)') AS province,
    ROUND(SUM(COALESCE("LineTotal", 0)), 2) AS revenue
FROM "MTC_VW_AI_SalesRevenueCost"
WHERE "SODocDate" >= DATE '2025-01-01'
  AND "SODocDate" < DATE '2025-07-01'
GROUP BY 1, 2
ORDER BY revenue DESC NULLS LAST;

-- 15. Journal entry by profit center
SELECT
    COALESCE("PrcName", '(NoProfitCenter)') AS profit_center,
    ROUND(SUM(COALESCE("Debit", 0)), 2) AS debit_total,
    ROUND(SUM(COALESCE("Credit", 0)), 2) AS credit_total,
    ROUND(SUM(COALESCE("LineBalance", 0)), 2) AS balance_total
FROM "MTC_VW_AI_JournalEntry"
WHERE "RefDate" >= DATE '2025-01-01'
GROUP BY 1
ORDER BY ABS(SUM(COALESCE("LineBalance", 0))) DESC NULLS LAST;
