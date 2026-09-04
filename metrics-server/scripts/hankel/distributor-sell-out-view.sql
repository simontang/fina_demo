-- Hankel distributor Sell-out semantic view.
--
-- The source table is already expanded to one row per sales-team allocation.
-- Raw sales_quantity and sell_out_value therefore repeat across allocation rows;
-- territory_* fields are the additive measures at the stored grain.
--
-- The guardrails below flag, but do not delete, extreme source records found in
-- the demo workbook. Default runtime measures are NULL for excluded rows so SUM
-- remains safe while raw and excluded values stay available for investigation.

CREATE OR REPLACE VIEW public.hankel_view_distr_sell_out AS
WITH parsed AS (
    SELECT
        s.*,
        CASE
            WHEN BTRIM(COALESCE(s.territory_sell_out, ''))
                 ~ '^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
            THEN BTRIM(s.territory_sell_out)::numeric
            ELSE NULL
        END AS parsed_territory_sell_out
    FROM public.hankel_distr_sell_out s
),
quality AS (
    SELECT
        p.*,
        COALESCE(ABS(p.sales_quantity), 0) >= 10000000::numeric AS is_quantity_outlier,
        COALESCE(ABS(p.sell_out_value), 0) >= 1000000000::numeric AS is_amount_outlier,
        p.parsed_territory_sell_out IS NULL AS has_invalid_allocation_value
    FROM parsed p
)
SELECT
    q.customer_code,
    q.customer_name,
    q.product_code,
    q.product_name,
    q.product_unit,
    q.end_customer_name,
    q.whether_nca,
    q.end_cust_num,
    q.province,
    q.city,
    q.district,
    q.national_industry_l1,
    q.national_industry_l2,
    q.combined_national_industry_l2,
    q.national_industry_l3,
    q.combined_national_industry_l3,
    q.henkel_key_mid_ind,
    q.acm_sales_territory,
    q.sales_team,
    q.ta_flag,
    q.data_source,
    q.platform,
    q.year,
    q.year_month,
    q.year_month_time,
    q.combined_id,
    q.whether_spl,
    q.bian_end_cust,
    q.sales_quantity AS raw_sales_quantity,
    q.sell_out_value AS raw_sell_out_value,
    q.territory_sales_quantity AS allocated_sales_quantity,
    q.parsed_territory_sell_out AS allocated_sell_out_value,
    CASE
        WHEN q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        THEN NULL
        ELSE q.territory_sales_quantity
    END AS sell_out_quantity,
    CASE
        WHEN q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        THEN NULL
        ELSE q.parsed_territory_sell_out
    END AS sell_out_value,
    CASE
        WHEN q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        THEN COALESCE(q.territory_sales_quantity, 0)
        ELSE 0::numeric
    END AS excluded_sell_out_quantity,
    CASE
        WHEN q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        THEN COALESCE(q.parsed_territory_sell_out, 0)
        ELSE 0::numeric
    END AS excluded_sell_out_value,
    q.is_quantity_outlier,
    q.is_amount_outlier,
    q.has_invalid_allocation_value,
    q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        AS is_quality_excluded,
    CASE
        WHEN q.is_quantity_outlier OR q.is_amount_outlier OR q.has_invalid_allocation_value
        THEN 1
        ELSE 0
    END AS quality_issue_row_count,
    CONCAT_WS('; ',
        CASE
            WHEN q.is_quantity_outlier
            THEN 'quantity outside demo guardrail (abs >= 10000000)'
        END,
        CASE
            WHEN q.is_amount_outlier
            THEN 'raw sell-out outside demo guardrail (abs >= 1000000000)'
        END,
        CASE
            WHEN q.has_invalid_allocation_value
            THEN 'territory sell-out is missing or non-numeric'
        END
    ) AS quality_issues,
    CASE
        WHEN q.year_month ~ '^[0-9]{4}-[0-9]{2}$'
        THEN TO_DATE(q.year_month || '-01', 'YYYY-MM-DD')
        ELSE NULL
    END AS period_date
FROM quality q;
