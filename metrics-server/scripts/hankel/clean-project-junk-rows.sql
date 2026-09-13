-- Clean trailing junk rows from hankel_project_opportunity_lines.
--
-- Background: "project raw data_1st Aug.xlsx" is a Power BI table export. The
-- sheet tail carries two non-business rows that the import previously loaded:
--   * a fully blank separator row, and
--   * an "Applied filters: ..." footer whose text lands in the sales_team column.
-- Both rows have empty Sales Name / Opportunity ID / Sold-to / Product IDH, so
-- they never enter a Run for Gold match key, but they do pollute missing-value
-- quality counts (e.g. they made Opportunity ID / Product IDH look like each
-- had 1 missing row among "valid" rows).
--
-- Idempotent: safe to run after every re-import of the project raw file.
-- Scope: only rows where ALL four match-key fields are blank, or where the
-- Power BI footer marker is present. Business rows are never touched.

-- 1) Remove junk rows.
DELETE FROM hankel_project_opportunity_lines
WHERE (
        BTRIM(COALESCE(sales_name, '')) = ''
    AND BTRIM(COALESCE(opportunity_id, '')) = ''
    AND BTRIM(COALESCE(sold_to, '')) = ''
    AND BTRIM(COALESCE(product_idh, '')) = ''
  )
  OR BTRIM(COALESCE(sales_name, '')) LIKE '%Applied filters%'
  OR BTRIM(COALESCE(sales_team, '')) LIKE '%Applied filters%';

-- 2) Keep the import manifest aligned with the cleaned table.
UPDATE hankel_import_manifest
SET row_count = (SELECT count(*) FROM hankel_project_opportunity_lines)
WHERE table_name = 'hankel_project_opportunity_lines';

-- 3) Post-check: should report 0 junk rows and 14,624 business rows.
SELECT
    count(*)                                                          AS total_rows,
    count(*) FILTER (
        WHERE BTRIM(COALESCE(sales_name, '')) <> ''
    )                                                                 AS business_rows,
    count(*) FILTER (
        WHERE BTRIM(COALESCE(sales_name, '')) = ''
          AND BTRIM(COALESCE(opportunity_id, '')) = ''
          AND BTRIM(COALESCE(sold_to, '')) = ''
          AND BTRIM(COALESCE(product_idh, '')) = ''
    )                                                                 AS remaining_junk_rows
FROM hankel_project_opportunity_lines;

-- 4) Post-check: missing-field counts on business rows.
-- Expected on the current snapshot: missing_sold_to = 163, others = 0.
SELECT
    count(*)                                                        AS business_rows,
    count(*) FILTER (WHERE BTRIM(COALESCE(opportunity_id, '')) = '') AS missing_opportunity_id,
    count(*) FILTER (WHERE BTRIM(COALESCE(sold_to, '')) = '')        AS missing_sold_to,
    count(*) FILTER (WHERE BTRIM(COALESCE(product_idh, '')) = '')    AS missing_product_idh
FROM hankel_project_opportunity_lines
WHERE BTRIM(COALESCE(sales_name, '')) <> '';
