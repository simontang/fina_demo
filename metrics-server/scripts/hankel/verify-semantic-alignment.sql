-- Read-only checks for the Hankel semantic views published from datasource 15.
-- Each check returns PASS or FAIL plus the observed values needed to diagnose a
-- mismatch. Run after the three view scripts have completed successfully.

WITH actual AS (
    SELECT
        COALESCE(SUM(nes), 0) AS scoped_nes,
        COALESCE(SUM(raw_nes) FILTER (WHERE is_customer_scope), 0) AS expected_nes,
        COALESCE(SUM(sell_in_quantity), 0) AS scoped_quantity,
        COALESCE(SUM(raw_sell_in_quantity) FILTER (WHERE is_customer_scope), 0) AS expected_quantity,
        COUNT(*) FILTER (
            WHERE NOT is_customer_scope
              AND (nes IS NOT NULL OR sell_in_quantity IS NOT NULL)
        ) AS leaking_rows
    FROM public.hankel_view_distr_sell_in
)
SELECT
    'sell_in_scope' AS check_name,
    CASE
        WHEN scoped_nes = expected_nes
         AND scoped_quantity = expected_quantity
         AND leaking_rows = 0
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    JSONB_BUILD_OBJECT(
        'scopedNes', scoped_nes,
        'expectedNes', expected_nes,
        'scopedQuantity', scoped_quantity,
        'expectedQuantity', expected_quantity,
        'leakingRows', leaking_rows
    ) AS details
FROM actual;

WITH actual AS (
    SELECT
        COUNT(*) FILTER (
            WHERE is_quantity_quality_excluded
              AND NOT is_value_quality_excluded
              AND sell_out_quantity IS NULL
              AND sell_out_value IS NOT NULL
        ) AS quantity_only_rows,
        COUNT(*) FILTER (
            WHERE is_value_quality_excluded
              AND NOT is_quantity_quality_excluded
              AND sell_out_value IS NULL
              AND sell_out_quantity IS NOT NULL
        ) AS value_only_rows,
        COUNT(*) FILTER (
            WHERE is_quantity_quality_excluded AND sell_out_quantity IS NOT NULL
        ) AS quantity_leaks,
        COUNT(*) FILTER (
            WHERE is_value_quality_excluded AND sell_out_value IS NOT NULL
        ) AS value_leaks
    FROM public.hankel_view_distr_sell_out
)
SELECT
    'sell_out_independent_quality_gates' AS check_name,
    CASE WHEN quantity_leaks = 0 AND value_leaks = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    JSONB_BUILD_OBJECT(
        'quantityOnlyRows', quantity_only_rows,
        'valueOnlyRows', value_only_rows,
        'quantityLeaks', quantity_leaks,
        'valueLeaks', value_leaks
    ) AS details
FROM actual;

WITH expected AS (
    SELECT COUNT(DISTINCT opportunity_id)::bigint AS opportunity_count
    FROM public.hankel_view_validated_won_opportunity
    WHERE result = 'Pass'
), actual AS (
    SELECT COALESCE(SUM(validated_won_count), 0)::bigint AS opportunity_count
    FROM public.hankel_view_run_for_gold_sales_summary
)
SELECT
    'validated_won_opportunity_grain' AS check_name,
    CASE WHEN actual.opportunity_count = expected.opportunity_count THEN 'PASS' ELSE 'FAIL' END AS status,
    JSONB_BUILD_OBJECT(
        'summaryCount', actual.opportunity_count,
        'expectedDistinctOpportunityCount', expected.opportunity_count
    ) AS details
FROM actual CROSS JOIN expected;

WITH actual AS (
    SELECT
        (SELECT COUNT(*) FROM public.hankel_view_run_for_gold_leaderboard) AS overall_rows,
        (SELECT COUNT(*) FROM public.hankel_view_run_for_gold_leaderboard WHERE NOT is_qualified) AS overall_unqualified_rows,
        (SELECT COUNT(*) FROM public.hankel_view_run_for_gold_segment_leaderboard) AS segment_rows,
        (SELECT COUNT(*) FROM public.hankel_view_run_for_gold_segment_leaderboard WHERE NOT is_qualified) AS segment_unqualified_rows
)
SELECT
    'leaderboard_qualification_filter' AS check_name,
    CASE
        WHEN overall_unqualified_rows = 0 AND segment_unqualified_rows = 0
        THEN 'PASS' ELSE 'FAIL'
    END AS status,
    JSONB_BUILD_OBJECT(
        'overallRows', overall_rows,
        'overallUnqualifiedRows', overall_unqualified_rows,
        'segmentRows', segment_rows,
        'segmentUnqualifiedRows', segment_unqualified_rows
    ) AS details
FROM actual;

WITH actual AS (
    SELECT
        (SELECT COUNT(*) FROM public.hankel_view_project_opportunity_line
         WHERE sold_to_idh ~ '^[0-9]+[.]0+$' OR product_idh ~ '^[0-9]+[.]0+$') AS project_decimal_ids,
        (SELECT COUNT(*) FROM public.hankel_view_new_order_line
         WHERE sold_to_idh ~ '^[0-9]+[.]0+$' OR product_idh ~ '^[0-9]+[.]0+$') AS order_decimal_ids
)
SELECT
    'normalized_business_identifiers' AS check_name,
    CASE WHEN project_decimal_ids = 0 AND order_decimal_ids = 0 THEN 'PASS' ELSE 'FAIL' END AS status,
    JSONB_BUILD_OBJECT(
        'projectDecimalIds', project_decimal_ids,
        'orderDecimalIds', order_decimal_ids
    ) AS details
FROM actual;

WITH actual AS (
    SELECT
        COUNT(*) AS current_rows,
        COUNT(DISTINCT snapshot_date) AS snapshot_dates,
        MIN(snapshot_date) AS snapshot_date
    FROM public.hankel_view_distr_inventory_current
)
SELECT
    'inventory_latest_snapshot' AS check_name,
    CASE WHEN current_rows > 0 AND snapshot_dates = 1 THEN 'PASS' ELSE 'FAIL' END AS status,
    JSONB_BUILD_OBJECT(
        'currentRows', current_rows,
        'snapshotDates', snapshot_dates,
        'snapshotDate', snapshot_date
    ) AS details
FROM actual;
