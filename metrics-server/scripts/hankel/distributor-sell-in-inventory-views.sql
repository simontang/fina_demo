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
    s.sell_in_quantity,
    s.nes,
    s.gross_margin,
    s.product_contribution_15,
    s.combined_id
FROM public.hankel_distr_sell_in s;

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
    i.year_month_time = MAX(i.year_month_time) OVER () AS is_latest_snapshot
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
