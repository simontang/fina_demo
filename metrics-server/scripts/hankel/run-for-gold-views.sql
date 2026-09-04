-- Hankel Run for Gold semantic views.
-- Scope: datasource 15 / tenant hankel / PostgreSQL public schema.
-- These views keep complex competition logic outside Metrics metric meta so
-- runtime metrics can remain SQL-free aggregate or ratio definitions.

CREATE OR REPLACE VIEW hankel_view_sales_name_mapping AS
SELECT
    source_file,
    source_sheet,
    source_row_number,
    NULLIF(BTRIM(sales_team), '') AS sales_team,
    NULLIF(BTRIM(project_sales_name_canonical), '') AS canonical_sales_name,
    LOWER(REGEXP_REPLACE(BTRIM(COALESCE(project_sales_name_canonical, '')), '[[:space:]]+', '', 'g')) AS project_sales_name_key,
    NULLIF(BTRIM(new_order_sales_name), '') AS new_order_sales_name,
    LOWER(REGEXP_REPLACE(BTRIM(COALESCE(new_order_sales_name, '')), '[[:space:]]+', '', 'g')) AS new_order_sales_name_key,
    NULLIF(BTRIM(sales_type), '') AS sales_type,
    imported_at
FROM hankel_sales_name_mapping;

CREATE OR REPLACE VIEW hankel_view_project_opportunity_line AS
WITH params AS (
    SELECT
        DATE '2026-07-01' AS competition_start_date,
        DATE '2026-12-31' AS competition_end_date,
        DATE '2026-08-31' AS report_cutoff_date,
        DATE '2026-01-01' AS target_year_start,
        DATE '2026-12-31' AS target_year_end,
        2026::integer AS target_year,
        0.5::numeric AS validation_threshold
),
sales_mapping AS (
    SELECT DISTINCT ON (project_sales_name_key)
        project_sales_name_key,
        canonical_sales_name,
        sales_team,
        sales_type
    FROM hankel_view_sales_name_mapping
    WHERE project_sales_name_key IS NOT NULL
      AND project_sales_name_key <> ''
    ORDER BY project_sales_name_key, source_row_number NULLS LAST
),
normalized AS (
    SELECT
        p.source_file,
        p.source_sheet,
        p.source_row_number,
        NULLIF(BTRIM(p.sales_name), '') AS sales_name,
        COALESCE(m.canonical_sales_name, NULLIF(BTRIM(p.sales_name), '')) AS canonical_sales_name,
        COALESCE(NULLIF(BTRIM(p.sales_team), ''), m.sales_team) AS sales_team,
        COALESCE(m.sales_team, NULLIF(BTRIM(p.sales_team), '')) AS team,
        m.sales_type,
        NULLIF(BTRIM(p.segment), '') AS segment,
        NULLIF(BTRIM(p.opportunity_id), '') AS opportunity_id,
        NULLIF(BTRIM(p.opportunity_description), '') AS opportunity_description,
        NULLIF(BTRIM(p.account), '') AS account,
        NULLIF(BTRIM(p.account_description), '') AS account_description,
        NULLIF(BTRIM(p.sold_to), '') AS sold_to_idh,
        NULLIF(BTRIM(p.sold_to_description), '') AS sold_to_description,
        NULLIF(BTRIM(p.product_idh), '') AS product_idh,
        NULLIF(BTRIM(p.product_description), '') AS product_description,
        NULLIF(BTRIM(p.item_type), '') AS item_type,
        NULLIF(BTRIM(p.project_source), '') AS project_source,
        NULLIF(BTRIM(p.ssp_type), '') AS ssp_type,
        NULLIF(BTRIM(p.stage), '') AS stage,
        NULLIF(BTRIM(p.status), '') AS status,
        NULLIF(BTRIM(p.reason_for_status), '') AS reason_for_status,
        p.creation_date,
        p.start_date,
        p.close_date,
        COALESCE(p.close_year, EXTRACT(YEAR FROM p.close_date)::integer) AS close_year,
        p.phase_start_date,
        p.project_changed_on,
        NULLIF(BTRIM(p.category), '') AS category,
        p.unit_price,
        COALESCE(p.y1_qty, 0) AS y1_qty,
        COALESCE(p.y2_qty, 0) AS y2_qty,
        COALESCE(p.total_quantity, 0) AS total_quantity,
        COALESCE(p.y1_value, 0) AS y1_value,
        COALESCE(p.y2_value, 0) AS y2_value,
        COALESCE(p.discount_y1, 0) AS discount_y1,
        COALESCE(p.discount_y2, 0) AS discount_y2,
        COALESCE(p.growth_impact_discount_value, 0) AS growth_impact_discount_value,
        p.imported_at,
        m.project_sales_name_key IS NOT NULL AS has_sales_mapping,
        params.competition_start_date,
        params.competition_end_date,
        params.report_cutoff_date,
        params.target_year_start,
        params.target_year_end,
        params.target_year,
        params.validation_threshold
    FROM hankel_project_opportunity_lines p
    CROSS JOIN params
    LEFT JOIN sales_mapping m
      ON LOWER(REGEXP_REPLACE(BTRIM(COALESCE(p.sales_name, '')), '[[:space:]]+', '', 'g')) = m.project_sales_name_key
),
scored AS (
    SELECT
        n.*,
        (
            n.has_sales_mapping
            AND n.canonical_sales_name IS NOT NULL
            AND n.sold_to_idh IS NOT NULL
            AND n.product_idh IS NOT NULL
        ) AS is_valid_match_key,
        (
            n.creation_date BETWEEN n.competition_start_date AND n.report_cutoff_date
        ) AS is_new_project_h2_to_cutoff,
        (
            LOWER(COALESCE(n.status, '')) = 'won'
            AND n.close_date BETWEEN n.target_year_start AND n.report_cutoff_date
        ) AS is_won_2026_ytd,
        CASE
            WHEN n.close_date IS NULL THEN NULL
            ELSE 13 - EXTRACT(MONTH FROM n.close_date)::integer
        END AS close_year_total_months,
        CASE
            WHEN n.close_date IS NULL THEN 0
            ELSE GREATEST(
                0,
                LEAST(
                    13 - EXTRACT(MONTH FROM n.close_date)::integer,
                    (
                        (EXTRACT(YEAR FROM n.report_cutoff_date)::integer - EXTRACT(YEAR FROM n.close_date)::integer) * 12
                        + EXTRACT(MONTH FROM n.report_cutoff_date)::integer
                        - EXTRACT(MONTH FROM n.close_date)::integer
                        + 1
                    )
                )
            )
        END AS report_recognized_months
    FROM normalized n
)
SELECT
    s.*,
    CASE WHEN s.is_new_project_h2_to_cutoff THEN s.opportunity_id END AS new_project_opportunity_id,
    CASE WHEN s.is_new_project_h2_to_cutoff THEN s.y1_value ELSE 0 END AS new_project_y1_value,
    CASE
        WHEN s.is_won_2026_ytd
         AND s.close_year_total_months > 0
        THEN s.y1_value / s.close_year_total_months * s.report_recognized_months
        ELSE 0
    END AS check_period_won_y1,
    ARRAY_TO_STRING(ARRAY_REMOVE(ARRAY[
        CASE WHEN NOT s.has_sales_mapping THEN 'missing sales mapping' END,
        CASE WHEN s.sold_to_idh IS NULL THEN 'missing sold_to_idh' END,
        CASE WHEN s.product_idh IS NULL THEN 'missing product_idh' END,
        CASE WHEN s.opportunity_id IS NULL THEN 'missing opportunity_id' END,
        CASE WHEN s.is_won_2026_ytd AND COALESCE(s.y1_value, 0) = 0 THEN 'won y1 value is zero' END
    ], NULL), '; ') AS quality_issues
FROM scored s;

CREATE OR REPLACE VIEW hankel_view_new_order_line AS
WITH params AS (
    SELECT
        DATE '2026-01-01' AS target_year_start,
        DATE '2026-12-31' AS target_year_end,
        2026::integer AS target_year
),
sales_mapping AS (
    SELECT DISTINCT ON (new_order_sales_name_key)
        new_order_sales_name_key,
        canonical_sales_name,
        sales_team,
        sales_type
    FROM hankel_view_sales_name_mapping
    WHERE new_order_sales_name_key IS NOT NULL
      AND new_order_sales_name_key <> ''
    ORDER BY new_order_sales_name_key, source_row_number NULLS LAST
),
normalized AS (
    SELECT
        n.source_file,
        n.source_sheet,
        n.source_row_number,
        NULLIF(BTRIM(n.sales_name), '') AS sales_name,
        COALESCE(m.canonical_sales_name, NULLIF(BTRIM(n.sales_name), '')) AS canonical_sales_name,
        COALESCE(NULLIF(BTRIM(n.sales_team), ''), NULLIF(BTRIM(n.team), ''), m.sales_team) AS sales_team,
        COALESCE(NULLIF(BTRIM(n.team), ''), m.sales_team, NULLIF(BTRIM(n.sales_team), '')) AS team,
        m.sales_type,
        NULLIF(BTRIM(n.segment), '') AS segment,
        NULLIF(BTRIM(n.ace_id), '') AS ace_id,
        NULLIF(BTRIM(n.sales_rep), '') AS sales_rep,
        NULLIF(BTRIM(n.saty), '') AS saty,
        NULLIF(BTRIM(n.order_type_name), '') AS order_type_name,
        NULLIF(BTRIM(n.order_number), '') AS order_number,
        NULLIF(BTRIM(n.sales_org), '') AS sales_org,
        NULLIF(BTRIM(n.distribution_channel), '') AS distribution_channel,
        n.created_on,
        n.requested_delivery_date,
        NULLIF(BTRIM(n.sold_to), '') AS sold_to_idh,
        NULLIF(BTRIM(n.sold_to_name), '') AS sold_to_name,
        NULLIF(BTRIM(n.material), '') AS product_idh,
        NULLIF(BTRIM(n.mto_mts), '') AS mto_mts,
        NULLIF(BTRIM(n.description), '') AS product_description,
        NULLIF(BTRIM(n.plant), '') AS plant,
        NULLIF(BTRIM(n.open_status), '') AS open_status,
        NULLIF(BTRIM(n.late_status), '') AS late_status,
        NULLIF(BTRIM(n.block_status), '') AS block_status,
        NULLIF(BTRIM(n.credit_status), '') AS credit_status,
        NULLIF(BTRIM(n.incomplete_status), '') AS incomplete_status,
        NULLIF(BTRIM(n.reject_status), '') AS reject_status,
        NULLIF(BTRIM(n.csr), '') AS csr,
        NULLIF(BTRIM(n.csr_name), '') AS csr_name,
        NULLIF(BTRIM(n.sales_document_item), '') AS sales_document_item,
        NULLIF(BTRIM(n.line_status), '') AS line_status,
        COALESCE(n.order_quantity, 0) AS order_quantity,
        NULLIF(BTRIM(n.sales_unit), '') AS sales_unit,
        COALESCE(n.order_value_cny, 0) AS order_value_cny,
        COALESCE(n.unit_price, 0) AS unit_price,
        n.imported_at,
        m.new_order_sales_name_key IS NOT NULL AS has_sales_mapping,
        params.target_year_start,
        params.target_year_end,
        params.target_year
    FROM hankel_new_order_lines n
    CROSS JOIN params
    LEFT JOIN sales_mapping m
      ON LOWER(REGEXP_REPLACE(BTRIM(COALESCE(n.sales_name, '')), '[[:space:]]+', '', 'g')) = m.new_order_sales_name_key
),
scored AS (
    SELECT
        n.*,
        (
            n.has_sales_mapping
            AND n.canonical_sales_name IS NOT NULL
            AND n.sold_to_idh IS NOT NULL
            AND n.product_idh IS NOT NULL
        ) AS is_valid_match_key,
        (
            LOWER(COALESCE(n.reject_status, '')) IN ('x', 'reject', 'rejected')
            OR LOWER(COALESCE(n.line_status, '')) = 'rejected'
        ) AS is_rejected,
        (
            n.created_on BETWEEN n.target_year_start AND n.target_year_end
        ) AS is_target_year_order
    FROM normalized n
)
SELECT
    s.*,
    (
        s.is_target_year_order
        AND NOT s.is_rejected
    ) AS is_eligible_new_order,
    CASE
        WHEN s.is_target_year_order AND NOT s.is_rejected THEN s.order_value_cny
        ELSE 0
    END AS eligible_order_value_cny,
    ARRAY_TO_STRING(ARRAY_REMOVE(ARRAY[
        CASE WHEN NOT s.has_sales_mapping THEN 'missing sales mapping' END,
        CASE WHEN s.sold_to_idh IS NULL THEN 'missing sold_to_idh' END,
        CASE WHEN s.product_idh IS NULL THEN 'missing product_idh' END,
        CASE WHEN s.order_number IS NULL THEN 'missing order_number' END,
        CASE WHEN s.is_rejected THEN 'rejected order line' END
    ], NULL), '; ') AS quality_issues
FROM scored s;

CREATE OR REPLACE VIEW hankel_view_won_validation_match_key AS
WITH project_agg AS (
    SELECT
        canonical_sales_name,
        sold_to_idh,
        product_idh,
        MAX(team) AS team,
        MAX(sales_team) AS sales_team,
        MAX(sales_type) AS sales_type,
        CASE
            WHEN COUNT(DISTINCT segment) FILTER (WHERE segment IS NOT NULL AND segment <> '') = 1
            THEN MAX(segment)
            ELSE 'Multiple'
        END AS primary_segment,
        STRING_AGG(DISTINCT segment, ', ' ORDER BY segment) FILTER (WHERE segment IS NOT NULL AND segment <> '') AS segments,
        MAX(sold_to_description) AS customer,
        MAX(product_description) AS product,
        STRING_AGG(DISTINCT opportunity_id, ', ' ORDER BY opportunity_id) FILTER (WHERE opportunity_id IS NOT NULL) AS opportunity_ids,
        COUNT(DISTINCT opportunity_id) FILTER (WHERE opportunity_id IS NOT NULL) AS opportunity_count,
        COUNT(*) AS won_line_count,
        SUM(y1_value) AS raw_won_y1,
        SUM(check_period_won_y1) AS check_period_won_y1,
        SUM(y1_qty) AS won_y1_qty,
        STRING_AGG(DISTINCT quality_issues, '; ' ORDER BY quality_issues) FILTER (WHERE quality_issues IS NOT NULL AND quality_issues <> '') AS quality_issues
    FROM hankel_view_project_opportunity_line
    WHERE is_won_2026_ytd
      AND is_valid_match_key
    GROUP BY canonical_sales_name, sold_to_idh, product_idh
),
order_agg AS (
    SELECT
        canonical_sales_name,
        sold_to_idh,
        product_idh,
        SUM(eligible_order_value_cny) AS matched_new_order_value,
        COUNT(*) AS order_line_count,
        STRING_AGG(DISTINCT order_number, ', ' ORDER BY order_number) FILTER (WHERE order_number IS NOT NULL) AS order_numbers,
        STRING_AGG(DISTINCT CONCAT_WS('-', order_number, sales_document_item), ', ' ORDER BY CONCAT_WS('-', order_number, sales_document_item))
            FILTER (WHERE order_number IS NOT NULL) AS order_line_refs,
        STRING_AGG(DISTINCT quality_issues, '; ' ORDER BY quality_issues) FILTER (WHERE quality_issues IS NOT NULL AND quality_issues <> '') AS quality_issues
    FROM hankel_view_new_order_line
    WHERE is_eligible_new_order
      AND is_valid_match_key
    GROUP BY canonical_sales_name, sold_to_idh, product_idh
),
matched AS (
    SELECT
        CONCAT_WS('|', p.canonical_sales_name, p.sold_to_idh, p.product_idh) AS match_key,
        p.canonical_sales_name,
        p.team,
        p.sales_team,
        p.sales_type,
        p.primary_segment,
        p.primary_segment AS segment,
        p.segments,
        p.sold_to_idh,
        p.customer,
        p.product_idh,
        p.product,
        p.opportunity_ids,
        p.opportunity_count,
        p.won_line_count,
        p.raw_won_y1,
        p.check_period_won_y1,
        p.check_period_won_y1 * 0.5 AS required_new_order_value,
        COALESCE(o.matched_new_order_value, 0) AS matched_new_order_value,
        COALESCE(o.order_line_count, 0) AS order_line_count,
        o.order_numbers,
        o.order_line_refs,
        p.won_y1_qty,
        ARRAY_TO_STRING(ARRAY_REMOVE(ARRAY[
            p.quality_issues,
            o.quality_issues,
            CASE WHEN o.matched_new_order_value IS NULL THEN 'no matching new order line' END
        ], NULL), '; ') AS quality_issues
    FROM project_agg p
    LEFT JOIN order_agg o
      ON p.canonical_sales_name = o.canonical_sales_name
     AND p.sold_to_idh = o.sold_to_idh
     AND p.product_idh = o.product_idh
)
SELECT
    m.*,
    m.matched_new_order_value / NULLIF(m.check_period_won_y1, 0) AS coverage,
    CASE
        WHEN m.matched_new_order_value >= m.required_new_order_value THEN 'Pass'
        ELSE 'Below 50%'
    END AS result,
    CASE
        WHEN m.matched_new_order_value >= m.required_new_order_value THEN m.match_key
    END AS validated_match_key,
    CASE
        WHEN m.matched_new_order_value >= m.required_new_order_value THEN m.raw_won_y1
        ELSE 0
    END AS counted_won_y1,
    CASE
        WHEN m.matched_new_order_value >= m.required_new_order_value THEN 0
        ELSE m.raw_won_y1
    END AS held_won_y1,
    GREATEST(m.required_new_order_value - m.matched_new_order_value, 0) AS new_order_gap,
    m.required_new_order_value - m.matched_new_order_value AS new_order_gap_raw,
    CASE
        WHEN m.matched_new_order_value >= m.required_new_order_value THEN 'Validated'
        ELSE 'Confirm or add New Order gap CNY ' || TO_CHAR(ROUND(GREATEST(m.required_new_order_value - m.matched_new_order_value, 0), 0), 'FM999999999999990')
    END AS status_action,
    DATE '2026-08-31' AS report_cutoff_date
FROM matched m;

CREATE OR REPLACE VIEW hankel_view_run_for_gold_sales_summary AS
WITH new_project AS (
    SELECT
        canonical_sales_name,
        MAX(team) AS team,
        MAX(sales_team) AS sales_team,
        MAX(sales_type) AS sales_type,
        COUNT(DISTINCT new_project_opportunity_id) AS new_project_count,
        SUM(new_project_y1_value) AS new_y1
    FROM hankel_view_project_opportunity_line
    WHERE is_new_project_h2_to_cutoff
      AND has_sales_mapping
    GROUP BY canonical_sales_name
),
won AS (
    SELECT
        canonical_sales_name,
        MAX(team) AS team,
        MAX(sales_team) AS sales_team,
        MAX(sales_type) AS sales_type,
        COUNT(*) AS won_lines,
        COUNT(validated_match_key) AS validated_won_count,
        COUNT(*) FILTER (WHERE result = 'Below 50%') AS below_50_count,
        SUM(raw_won_y1) AS raw_won_y1,
        SUM(check_period_won_y1) AS validation_won_y1,
        SUM(required_new_order_value) AS required_new_order_value,
        SUM(matched_new_order_value) AS matched_new_order_value,
        SUM(counted_won_y1) AS competition_won_y1,
        SUM(held_won_y1) AS held_won_y1,
        SUM(new_order_gap) AS new_order_gap
    FROM hankel_view_won_validation_match_key
    GROUP BY canonical_sales_name
)
SELECT
    COALESCE(n.canonical_sales_name, w.canonical_sales_name) AS canonical_sales_name,
    COALESCE(n.canonical_sales_name, w.canonical_sales_name) AS sales_name,
    COALESCE(n.team, w.team) AS team,
    COALESCE(n.sales_team, w.sales_team) AS sales_team,
    COALESCE(n.sales_type, w.sales_type, 'Unmapped') AS sales_type,
    COALESCE(n.new_project_count, 0) AS new_project_count,
    COALESCE(n.new_y1, 0) AS new_y1,
    COALESCE(w.won_lines, 0) AS won_lines,
    COALESCE(w.validated_won_count, 0) AS validated_won_count,
    COALESCE(w.below_50_count, 0) AS below_50_count,
    COALESCE(w.raw_won_y1, 0) AS raw_won_y1,
    COALESCE(w.validation_won_y1, 0) AS validation_won_y1,
    COALESCE(w.required_new_order_value, 0) AS required_new_order_value,
    COALESCE(w.matched_new_order_value, 0) AS matched_new_order_value,
    COALESCE(w.competition_won_y1, 0) AS competition_won_y1,
    COALESCE(w.held_won_y1, 0) AS held_won_y1,
    COALESCE(w.new_order_gap, 0) AS new_order_gap,
    COALESCE(w.matched_new_order_value, 0) / NULLIF(COALESCE(w.validation_won_y1, 0), 0) AS order_coverage_rate,
    DATE '2026-08-31' AS report_cutoff_date
FROM new_project n
FULL OUTER JOIN won w
  ON n.canonical_sales_name = w.canonical_sales_name;

CREATE OR REPLACE VIEW hankel_view_run_for_gold_leaderboard AS
WITH thresholds AS (
    SELECT 'New'::text AS sales_type, 12::integer AS new_project_required, 5::integer AS validated_won_required
    UNION ALL
    SELECT 'Experienced'::text AS sales_type, 25::integer AS new_project_required, 12::integer AS validated_won_required
),
base AS (
    SELECT
        CASE
            WHEN s.sales_type = 'New' THEN 'Overall · New 新人组'
            WHEN s.sales_type = 'Experienced' THEN 'Overall · Experienced 资深组'
            ELSE 'Overall · ' || s.sales_type
        END AS award_pool,
        s.team,
        s.canonical_sales_name AS leader,
        s.canonical_sales_name,
        s.report_cutoff_date,
        s.sales_type,
        s.new_project_count,
        s.validated_won_count,
        s.new_y1,
        s.competition_won_y1 AS won_y1,
        s.validation_won_y1,
        s.required_new_order_value,
        s.matched_new_order_value,
        s.new_order_gap,
        COALESCE(t.new_project_required, 0) AS new_project_required,
        COALESCE(t.validated_won_required, 0) AS validated_won_required,
        (
            s.new_project_count >= COALESCE(t.new_project_required, 0)
            AND s.validated_won_count >= COALESCE(t.validated_won_required, 0)
        ) AS is_qualified
    FROM hankel_view_run_for_gold_sales_summary s
    LEFT JOIN thresholds t
      ON s.sales_type = t.sales_type
    WHERE s.sales_type IN ('New', 'Experienced')
),
scored AS (
    SELECT
        b.*,
        COUNT(*) OVER (PARTITION BY award_pool) AS pool_size,
        RANK() OVER (PARTITION BY award_pool ORDER BY new_y1 DESC, new_project_count DESC, leader ASC) AS new_y1_rank,
        RANK() OVER (PARTITION BY award_pool ORDER BY won_y1 DESC, validated_won_count DESC, leader ASC) AS won_y1_rank
    FROM base b
),
ranked AS (
    SELECT
        s.*,
        (
            0.3 * ((s.pool_size - s.new_y1_rank + 1)::numeric / NULLIF(s.pool_size, 0) * 100)
            + 0.7 * ((s.pool_size - s.won_y1_rank + 1)::numeric / NULLIF(s.pool_size, 0) * 100)
        ) AS score
    FROM scored s
)
SELECT
    award_pool,
    RANK() OVER (
        PARTITION BY award_pool
        ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
    ) AS rank,
    team,
    leader,
    canonical_sales_name,
    sales_type,
    new_project_count,
    validated_won_count,
    new_y1,
    won_y1,
    validation_won_y1,
    required_new_order_value,
    matched_new_order_value,
    new_order_gap,
    ROUND(score, 6) AS score,
    CASE
        WHEN RANK() OVER (
            PARTITION BY award_pool
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) = 1 THEN '暂列金位 Gold'
        WHEN RANK() OVER (
            PARTITION BY award_pool
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) IN (2, 3) THEN '暂列银位 Silver'
        WHEN RANK() OVER (
            PARTITION BY award_pool
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) IN (4, 5) THEN '暂列铜位 Bronze'
        ELSE NULL
    END AS position,
    is_qualified,
    new_project_required,
    validated_won_required,
    report_cutoff_date
FROM ranked;

CREATE OR REPLACE VIEW hankel_view_run_for_gold_segment_leaderboard AS
WITH thresholds AS (
    SELECT 'Emotor'::text AS segment, 5::integer AS new_project_required, 2::integer AS validated_won_required
    UNION ALL
    SELECT 'Fluid'::text AS segment, 3::integer AS new_project_required, 1::integer AS validated_won_required
    UNION ALL
    SELECT 'Medical'::text AS segment, 5::integer AS new_project_required, 2::integer AS validated_won_required
),
new_project AS (
    SELECT
        canonical_sales_name,
        segment,
        MAX(team) AS team,
        MAX(sales_type) AS sales_type,
        COUNT(DISTINCT new_project_opportunity_id) AS new_project_count,
        SUM(new_project_y1_value) AS new_y1
    FROM hankel_view_project_opportunity_line
    WHERE is_new_project_h2_to_cutoff
      AND has_sales_mapping
      AND segment IN ('Emotor', 'Fluid', 'Medical')
    GROUP BY canonical_sales_name, segment
),
won AS (
    SELECT
        canonical_sales_name,
        primary_segment AS segment,
        MAX(team) AS team,
        MAX(sales_type) AS sales_type,
        COUNT(validated_match_key) AS validated_won_count,
        SUM(counted_won_y1) AS won_y1,
        SUM(check_period_won_y1) AS validation_won_y1,
        SUM(required_new_order_value) AS required_new_order_value,
        SUM(matched_new_order_value) AS matched_new_order_value,
        SUM(new_order_gap) AS new_order_gap
    FROM hankel_view_won_validation_match_key
    WHERE primary_segment IN ('Emotor', 'Fluid', 'Medical')
    GROUP BY canonical_sales_name, primary_segment
),
base AS (
    SELECT
        COALESCE(n.segment, w.segment) AS segment,
        'Key Segment · ' || COALESCE(n.segment, w.segment) AS award_pool,
        COALESCE(n.canonical_sales_name, w.canonical_sales_name) AS leader,
        COALESCE(n.canonical_sales_name, w.canonical_sales_name) AS canonical_sales_name,
        COALESCE(n.team, w.team) AS team,
        DATE '2026-08-31' AS report_cutoff_date,
        COALESCE(n.sales_type, w.sales_type, 'Unmapped') AS sales_type,
        COALESCE(n.new_project_count, 0) AS new_project_count,
        COALESCE(w.validated_won_count, 0) AS validated_won_count,
        COALESCE(n.new_y1, 0) AS new_y1,
        COALESCE(w.won_y1, 0) AS won_y1,
        COALESCE(w.validation_won_y1, 0) AS validation_won_y1,
        COALESCE(w.required_new_order_value, 0) AS required_new_order_value,
        COALESCE(w.matched_new_order_value, 0) AS matched_new_order_value,
        COALESCE(w.new_order_gap, 0) AS new_order_gap
    FROM new_project n
    FULL OUTER JOIN won w
      ON n.canonical_sales_name = w.canonical_sales_name
     AND n.segment = w.segment
),
thresholded AS (
    SELECT
        b.*,
        COALESCE(t.new_project_required, 0) AS new_project_required,
        COALESCE(t.validated_won_required, 0) AS validated_won_required,
        (
            b.new_project_count >= COALESCE(t.new_project_required, 0)
            AND b.validated_won_count >= COALESCE(t.validated_won_required, 0)
        ) AS is_qualified
    FROM base b
    LEFT JOIN thresholds t
      ON b.segment = t.segment
),
scored AS (
    SELECT
        t.*,
        COUNT(*) OVER (PARTITION BY segment) AS pool_size,
        RANK() OVER (PARTITION BY segment ORDER BY new_y1 DESC, new_project_count DESC, leader ASC) AS new_y1_rank,
        RANK() OVER (PARTITION BY segment ORDER BY won_y1 DESC, validated_won_count DESC, leader ASC) AS won_y1_rank
    FROM thresholded t
),
ranked AS (
    SELECT
        s.*,
        (
            0.3 * ((s.pool_size - s.new_y1_rank + 1)::numeric / NULLIF(s.pool_size, 0) * 100)
            + 0.7 * ((s.pool_size - s.won_y1_rank + 1)::numeric / NULLIF(s.pool_size, 0) * 100)
        ) AS score
    FROM scored s
)
SELECT
    segment,
    award_pool,
    RANK() OVER (
        PARTITION BY segment
        ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
    ) AS rank,
    team,
    leader,
    canonical_sales_name,
    sales_type,
    new_project_count,
    validated_won_count,
    new_y1,
    won_y1,
    validation_won_y1,
    required_new_order_value,
    matched_new_order_value,
    new_order_gap,
    ROUND(score, 6) AS score,
    CASE
        WHEN RANK() OVER (
            PARTITION BY segment
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) = 1 THEN '暂列金位 Gold'
        WHEN RANK() OVER (
            PARTITION BY segment
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) IN (2, 3) THEN '暂列银位 Silver'
        WHEN RANK() OVER (
            PARTITION BY segment
            ORDER BY score DESC, won_y1 DESC, validated_won_count DESC, new_y1 DESC, new_project_count DESC, leader ASC
        ) IN (4, 5) THEN '暂列铜位 Bronze'
        ELSE NULL
    END AS position,
    is_qualified,
    new_project_required,
    validated_won_required,
    report_cutoff_date
FROM ranked;

CREATE OR REPLACE VIEW hankel_view_run_for_gold_qualification_gap AS
SELECT
    'overall'::text AS scope_type,
    award_pool,
    NULL::text AS segment,
    team,
    leader AS sales_name,
    sales_type,
    new_project_count::integer AS new_project_current,
    new_project_required::integer AS new_project_required,
    GREATEST(new_project_required - new_project_count, 0)::integer AS new_project_gap,
    validated_won_count::integer AS validated_won_current,
    validated_won_required::integer AS validated_won_required,
    GREATEST(validated_won_required - validated_won_count, 0)::integer AS validated_won_gap,
    CASE
        WHEN is_qualified THEN 'Qualified'
        WHEN GREATEST(new_project_required - new_project_count, 0) > 0
          AND GREATEST(validated_won_required - validated_won_count, 0) > 0
        THEN CONCAT(GREATEST(new_project_required - new_project_count, 0), ' new projects and ',
                    GREATEST(validated_won_required - validated_won_count, 0), ' validated wins short')
        WHEN GREATEST(new_project_required - new_project_count, 0) > 0
        THEN CONCAT(GREATEST(new_project_required - new_project_count, 0), ' new projects short')
        ELSE CONCAT(GREATEST(validated_won_required - validated_won_count, 0), ' validated wins short')
    END AS status
FROM hankel_view_run_for_gold_leaderboard
UNION ALL
SELECT
    'segment'::text AS scope_type,
    award_pool,
    segment,
    team,
    leader AS sales_name,
    sales_type,
    new_project_count::integer AS new_project_current,
    new_project_required::integer AS new_project_required,
    GREATEST(new_project_required - new_project_count, 0)::integer AS new_project_gap,
    validated_won_count::integer AS validated_won_current,
    validated_won_required::integer AS validated_won_required,
    GREATEST(validated_won_required - validated_won_count, 0)::integer AS validated_won_gap,
    CASE
        WHEN is_qualified THEN 'Qualified'
        WHEN GREATEST(new_project_required - new_project_count, 0) > 0
          AND GREATEST(validated_won_required - validated_won_count, 0) > 0
        THEN CONCAT(GREATEST(new_project_required - new_project_count, 0), ' new projects and ',
                    GREATEST(validated_won_required - validated_won_count, 0), ' validated wins short')
        WHEN GREATEST(new_project_required - new_project_count, 0) > 0
        THEN CONCAT(GREATEST(new_project_required - new_project_count, 0), ' new projects short')
        ELSE CONCAT(GREATEST(validated_won_required - validated_won_count, 0), ' validated wins short')
    END AS status
FROM hankel_view_run_for_gold_segment_leaderboard;

CREATE OR REPLACE VIEW hankel_view_run_for_gold_report_reconciliation AS
WITH calc_won_by_sales AS (
    SELECT
        canonical_sales_name AS sales_name,
        MAX(team) AS team,
        MAX(sales_type) AS sales_type,
        SUM(won_lines) AS won_lines,
        SUM(validated_won_count) AS pass_count,
        SUM(below_50_count) AS below_50_count,
        SUM(raw_won_y1) AS raw_won_y1,
        SUM(competition_won_y1) AS counted_won_y1,
        SUM(held_won_y1) AS held_won_y1,
        SUM(new_order_gap) AS total_new_order_gap
    FROM hankel_view_run_for_gold_sales_summary
    GROUP BY canonical_sales_name
),
report_won_by_sales AS (
    SELECT
        sales_name,
        MAX(team) AS team,
        MAX(sales_type) AS sales_type,
        SUM(won_lines) AS won_lines,
        SUM(pass_count) AS pass_count,
        SUM(below_50_count) AS below_50_count,
        SUM(raw_won_y1) AS raw_won_y1,
        SUM(counted_won_y1) AS counted_won_y1,
        SUM(held_won_y1) AS held_won_y1,
        SUM(total_new_order_gap) AS total_new_order_gap
    FROM hankel_report_won_reconciliation_by_sales
    GROUP BY sales_name
),
won_by_sales_recon AS (
    SELECT
        'won_by_sales_counted_won_y1'::text AS check_type,
        COALESCE(c.sales_name, r.sales_name) AS business_key,
        CASE WHEN c.sales_name IS NULL THEN 0 ELSE 1 END AS calc_row_count,
        CASE WHEN r.sales_name IS NULL THEN 0 ELSE 1 END AS report_row_count,
        COALESCE(c.counted_won_y1, 0) AS calc_amount,
        COALESCE(r.counted_won_y1, 0) AS report_amount,
        COALESCE(c.counted_won_y1, 0) - COALESCE(r.counted_won_y1, 0) AS amount_diff,
        CASE
            WHEN c.sales_name IS NULL OR r.sales_name IS NULL THEN 'KEY_MISMATCH'
            WHEN ABS(COALESCE(c.counted_won_y1, 0) - COALESCE(r.counted_won_y1, 0)) <= 0.01 THEN 'PASS'
            ELSE 'AMOUNT_MISMATCH'
        END AS status,
        JSONB_BUILD_OBJECT(
            'calc', TO_JSONB(c),
            'report', TO_JSONB(r)
        )::text AS detail
    FROM calc_won_by_sales c
    FULL OUTER JOIN report_won_by_sales r
      ON LOWER(REGEXP_REPLACE(BTRIM(COALESCE(c.sales_name, '')), '[[:space:]]+', '', 'g'))
       = LOWER(REGEXP_REPLACE(BTRIM(COALESCE(r.sales_name, '')), '[[:space:]]+', '', 'g'))
),
calc_detail AS (
    SELECT
        LOWER(REGEXP_REPLACE(BTRIM(COALESCE(canonical_sales_name, '')), '[[:space:]]+', '', 'g'))
          || '|' || COALESCE(sold_to_idh, '')
          || '|' || COALESCE(product_idh, '') AS business_key,
        COUNT(*) AS row_count,
        SUM(raw_won_y1) AS raw_won_y1,
        SUM(check_period_won_y1) AS check_won_y1,
        SUM(required_new_order_value) AS required_new_order,
        SUM(matched_new_order_value) AS matched_new_order,
        SUM(counted_won_y1) AS counted_won_y1,
        SUM(new_order_gap) AS new_order_gap
    FROM hankel_view_won_validation_match_key
    GROUP BY business_key
),
report_detail AS (
    SELECT
        LOWER(REGEXP_REPLACE(BTRIM(COALESCE(sales_name, '')), '[[:space:]]+', '', 'g'))
          || '|' || COALESCE(sold_to_idh, '')
          || '|' || COALESCE(product_idh, '') AS business_key,
        COUNT(*) AS row_count,
        SUM(raw_won_y1) AS raw_won_y1,
        SUM(check_won_y1) AS check_won_y1,
        SUM(required_new_order) AS required_new_order,
        SUM(matched_new_order) AS matched_new_order,
        SUM(counted_won_y1) AS counted_won_y1,
        SUM(new_order_gap) AS new_order_gap
    FROM hankel_report_won_reconciliation_detail
    GROUP BY business_key
),
detail_recon AS (
    SELECT
        'won_detail_counted_won_y1'::text AS check_type,
        COALESCE(c.business_key, r.business_key) AS business_key,
        COALESCE(c.row_count, 0)::integer AS calc_row_count,
        COALESCE(r.row_count, 0)::integer AS report_row_count,
        COALESCE(c.counted_won_y1, 0) AS calc_amount,
        COALESCE(r.counted_won_y1, 0) AS report_amount,
        COALESCE(c.counted_won_y1, 0) - COALESCE(r.counted_won_y1, 0) AS amount_diff,
        CASE
            WHEN c.business_key IS NULL OR r.business_key IS NULL THEN 'KEY_MISMATCH'
            WHEN COALESCE(c.row_count, 0) <> COALESCE(r.row_count, 0) THEN 'ROW_COUNT_MISMATCH'
            WHEN ABS(COALESCE(c.counted_won_y1, 0) - COALESCE(r.counted_won_y1, 0)) <= 0.01 THEN 'PASS'
            ELSE 'AMOUNT_MISMATCH'
        END AS status,
        JSONB_BUILD_OBJECT(
            'calc', TO_JSONB(c),
            'report', TO_JSONB(r)
        )::text AS detail
    FROM calc_detail c
    FULL OUTER JOIN report_detail r
      ON c.business_key = r.business_key
),
calc_leaders AS (
    SELECT
        award_pool,
        leader,
        rank,
        team,
        sales_type,
        new_project_count,
        validated_won_count,
        new_y1,
        won_y1,
        score,
        position
    FROM hankel_view_run_for_gold_leaderboard
    UNION ALL
    SELECT
        award_pool,
        leader,
        rank,
        team,
        sales_type,
        new_project_count,
        validated_won_count,
        new_y1,
        won_y1,
        score,
        position
    FROM hankel_view_run_for_gold_segment_leaderboard
),
report_leaders AS (
    SELECT
        award_pool,
        leader,
        rank,
        team,
        sales_type,
        new_project_count,
        validated_won_count,
        new_y1,
        won_y1,
        score,
        position
    FROM hankel_report_current_leaders
),
leader_recon AS (
    SELECT
        'current_leaders_score'::text AS check_type,
        COALESCE(c.award_pool, r.award_pool, '')
          || '|'
          || COALESCE(c.leader, r.leader, '') AS business_key,
        CASE WHEN c.leader IS NULL THEN 0 ELSE 1 END AS calc_row_count,
        CASE WHEN r.leader IS NULL THEN 0 ELSE 1 END AS report_row_count,
        COALESCE(c.score, 0) AS calc_amount,
        COALESCE(r.score, 0) AS report_amount,
        COALESCE(c.score, 0) - COALESCE(r.score, 0) AS amount_diff,
        CASE
            WHEN c.leader IS NULL OR r.leader IS NULL THEN 'KEY_MISMATCH'
            WHEN COALESCE(c.rank, -1) <> COALESCE(r.rank, -1) THEN 'RANK_MISMATCH'
            WHEN ABS(COALESCE(c.score, 0) - COALESCE(r.score, 0)) <= 0.01 THEN 'PASS'
            ELSE 'AMOUNT_MISMATCH'
        END AS status,
        JSONB_BUILD_OBJECT(
            'calc', TO_JSONB(c),
            'report', TO_JSONB(r)
        )::text AS detail
    FROM calc_leaders c
    FULL OUTER JOIN report_leaders r
      ON c.award_pool = r.award_pool
     AND LOWER(REGEXP_REPLACE(BTRIM(COALESCE(c.leader, '')), '[[:space:]]+', '', 'g'))
       = LOWER(REGEXP_REPLACE(BTRIM(COALESCE(r.leader, '')), '[[:space:]]+', '', 'g'))
)
SELECT * FROM won_by_sales_recon
UNION ALL
SELECT * FROM detail_recon
UNION ALL
SELECT * FROM leader_recon;
