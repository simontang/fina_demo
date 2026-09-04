-- Hankel distributor Sell-in and Inventory semantic views.
--
-- Sell-in is a flow dataset. The semantic view preserves every imported row
-- and converts the Excel serial month into a real PostgreSQL date.
--
-- Inventory is a stock snapshot dataset. territory_inventory_* contains the
-- additive sales-team allocation, while inventory_* repeats on allocation
-- rows. Runtime inventory metrics therefore use the latest-snapshot view and
-- never sum raw inventory values across teams or months.

CREATE OR REPLACE VIEW public.hankel_view_distr_sell_in AS
WITH scoped AS (
    SELECT
        s.*,
        (
            LOWER(BTRIM(COALESCE(s.sales_team, ''))) = ANY (ARRAY[
                'gm ta', 'mro', 'north', 'north jiangsu', 'shanghai',
                'south jiangsu', 'south1', 'south2', 'zhejiang&fujian',
                'gm north', 'gm hangzhou', 'gm middle china', 'gm nanjing',
                'gm suzhou', 'gm shenzhen', 'gm ningbo', 'gm shanghai',
                'gm guangzhou', 'gm anhui&shandong', 'gmm ec', 'gm beijing',
                'gm spl', 'ipr team_ipr product', 'ipr team_non ipr product',
                'ipr team_non ipr nes', 'ipr team_ipr nes'
            ])
            AND COALESCE(UPPER(BTRIM(s.gmm_l6_allocation)), '') <> 'Y'
        ) AS is_customer_scope
    FROM public.hankel_distr_sell_in s
)
SELECT
    s.posting_year,
    CASE
        WHEN s.year_month_time BETWEEN 1 AND 100000
        THEN DATE '1899-12-30' + s.year_month_time::integer
        ELSE NULL
    END AS posting_date,
    CASE
        WHEN s.year_month_time BETWEEN 1 AND 100000
        THEN TO_CHAR(DATE '1899-12-30' + s.year_month_time::integer, 'YYYY-MM')
        ELSE NULL
    END AS year_month,
    s.year_month_time AS source_excel_date_serial,
    s.sales_team,
    s.sold_to,
    s.sold_to_name,
    s.product_idh,
    s.product_description,
    s.gmm_l6_allocation,
    s.whether_year_sell_out,
    s.whether_spl,
    CASE WHEN s.is_customer_scope THEN s.sell_in_quantity END AS sell_in_quantity,
    CASE WHEN s.is_customer_scope THEN s.nes END AS nes,
    CASE WHEN s.is_customer_scope THEN s.gross_margin END AS gross_margin,
    CASE WHEN s.is_customer_scope THEN s.product_contribution_15 END AS product_contribution_15,
    s.combined_id,
    s.sell_in_quantity AS raw_sell_in_quantity,
    s.nes AS raw_nes,
    s.gross_margin AS raw_gross_margin,
    s.product_contribution_15 AS raw_product_contribution_15,
    CASE WHEN NOT s.is_customer_scope THEN COALESCE(s.sell_in_quantity, 0) ELSE 0::numeric END AS excluded_sell_in_quantity,
    CASE WHEN NOT s.is_customer_scope THEN COALESCE(s.nes, 0) ELSE 0::numeric END AS excluded_nes,
    s.is_customer_scope,
    CASE
        WHEN s.is_customer_scope THEN NULL
        WHEN LOWER(BTRIM(COALESCE(s.sales_team, ''))) <> ALL (ARRAY[
            'gm ta', 'mro', 'north', 'north jiangsu', 'shanghai',
            'south jiangsu', 'south1', 'south2', 'zhejiang&fujian',
            'gm north', 'gm hangzhou', 'gm middle china', 'gm nanjing',
            'gm suzhou', 'gm shenzhen', 'gm ningbo', 'gm shanghai',
            'gm guangzhou', 'gm anhui&shandong', 'gmm ec', 'gm beijing',
            'gm spl', 'ipr team_ipr product', 'ipr team_non ipr product',
            'ipr team_non ipr nes', 'ipr team_ipr nes'
        ]) THEN 'sales team is outside customer whitelist'
        ELSE 'GMM L6 allocation is Y'
    END AS scope_exclusion_reason
FROM scoped s;

CREATE OR REPLACE VIEW public.hankel_view_distr_inventory_monthly AS
SELECT
    i.customer_idh,
    i.sold_to,
    i.product_idh,
    i.product_idh_2,
    i.product_unit,
    i.year,
    i.year_month,
    CASE
        WHEN i.year_month_time BETWEEN 1 AND 100000
        THEN DATE '1899-12-30' + i.year_month_time::integer
        ELSE NULL
    END AS snapshot_date,
    i.year_month_time AS source_excel_date_serial,
    i.data_source,
    i.sales_team,
    i.whether_spl,
    i.combined_id,
    i.final_price,
    i.final_price_tp,
    i.inventory_coefficient,
    i.sell_in_ppt_idh,
    i.nes_ppt,
    i.inventory_value AS raw_inventory_value,
    i.inventory_quantity AS raw_inventory_quantity,
    i.territory_inventory_value AS inventory_value,
    i.territory_inventory_quantity AS inventory_quantity,
    i.year_month_time = MAX(
        CASE WHEN i.year_month_time BETWEEN 1 AND 100000 THEN i.year_month_time END
    ) OVER () AS is_latest_snapshot
FROM public.hankel_distr_inventory i;

CREATE OR REPLACE VIEW public.hankel_view_distr_inventory_current AS
SELECT
    customer_idh,
    sold_to,
    product_idh,
    product_idh_2,
    product_unit,
    year,
    year_month,
    snapshot_date,
    source_excel_date_serial,
    data_source,
    sales_team,
    whether_spl,
    combined_id,
    final_price,
    final_price_tp,
    inventory_coefficient,
    sell_in_ppt_idh,
    nes_ppt,
    raw_inventory_value,
    raw_inventory_quantity,
    inventory_value,
    inventory_quantity,
    is_latest_snapshot
FROM public.hankel_view_distr_inventory_monthly
WHERE is_latest_snapshot;
