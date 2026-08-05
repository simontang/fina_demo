-- Caterpillar datasource, metric registrations, and tenant metrics configuration.
-- The encrypted datasource password is copied in-database from datasource 11 and is never exposed.
\set ON_ERROR_STOP on

BEGIN;

SELECT pg_advisory_xact_lock(hashtext('caterpillar-metadata-first-config'));

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM t_datasource_config
        WHERE id = 11
          AND source_type = 'cdp_postgres'
          AND status = 1
          AND deleted = 0) <> 1 THEN
        RAISE EXCEPTION 'Active Retail CDP datasource 11 is required';
    END IF;

    IF (SELECT COUNT(*) FROM lattice_tenants
        WHERE id = 'caterpillar'
          AND name = 'caterpillar'
          AND status = 'active') <> 1 THEN
        RAISE EXCEPTION 'Active caterpillar tenant is required';
    END IF;

    IF (SELECT COUNT(*)
        FROM lattice_users users
        JOIN lattice_user_tenant_links links ON links.user_id = users.id
        WHERE users.email = 'sharon@fina.com'
          AND users.status = 'active'
          AND links.tenant_id = 'caterpillar'
          AND links.role = 'owner') <> 1 THEN
        RAISE EXCEPTION 'Sharon owner link for caterpillar tenant is required';
    END IF;
END
$$;

INSERT INTO t_datasource_config (
    name,
    url,
    username,
    password,
    schema_name,
    description,
    status,
    deleted,
    instance_type,
    source_type
)
SELECT
    'Caterpillar PostgreSQL',
    source.url,
    source.username,
    source.password,
    source.schema_name,
    'Caterpillar marketing and customer operations dataset in caterpillar_* tables',
    1,
    0,
    source.instance_type,
    'cdp_postgres'
FROM t_datasource_config source
WHERE source.id = 11
  AND source.source_type = 'cdp_postgres'
  AND source.status = 1
  AND source.deleted = 0
  AND NOT EXISTS (
      SELECT 1
      FROM t_datasource_config existing
      WHERE existing.name = 'Caterpillar PostgreSQL'
        AND existing.deleted = 0
  );

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM t_datasource_config
        WHERE name = 'Caterpillar PostgreSQL'
          AND source_type = 'cdp_postgres'
          AND status = 1
          AND deleted = 0) <> 1 THEN
        RAISE EXCEPTION 'Expected exactly one active Caterpillar PostgreSQL datasource';
    END IF;
END
$$;

WITH datasource AS (
    SELECT id
    FROM t_datasource_config
    WHERE name = 'Caterpillar PostgreSQL'
      AND source_type = 'cdp_postgres'
      AND status = 1
      AND deleted = 0
), definitions(metric_code, metric_name, description, query_sql) AS (
    VALUES
        (
            'caterpillar_leads_received',
            'Caterpillar Leads Received',
            'Total number of Caterpillar marketing leads received.',
            'SELECT COUNT(*) AS value FROM public.caterpillar_lead'
        ),
        (
            'caterpillar_qualified_lead_rate',
            'Caterpillar Qualified Lead Rate',
            'Share of Caterpillar leads in VALID, ASSIGNED, or CONVERTED status.',
            'SELECT (COUNT(*) FILTER (WHERE status IN (''VALID'',''ASSIGNED'',''CONVERTED'')))::numeric / NULLIF(COUNT(*), 0) AS value FROM public.caterpillar_lead'
        ),
        (
            'caterpillar_call_answer_rate',
            'Caterpillar Call Answer Rate',
            'Share of Caterpillar call records in ANSWERED status.',
            'SELECT (COUNT(*) FILTER (WHERE call_status = ''ANSWERED''))::numeric / NULLIF(COUNT(*), 0) AS value FROM public.caterpillar_call_record'
        ),
        (
            'caterpillar_assignment_acceptance_rate',
            'Caterpillar Assignment Acceptance Rate',
            'Share of Caterpillar lead assignments in ACCEPTED delivery status.',
            'SELECT (COUNT(*) FILTER (WHERE delivery_status = ''ACCEPTED''))::numeric / NULLIF(COUNT(*), 0) AS value FROM public.caterpillar_lead_assignment'
        ),
        (
            'caterpillar_assignment_order_rate',
            'Caterpillar Assignment Order Rate',
            'Share of Caterpillar lead assignments in ORDERED conversion status.',
            'SELECT (COUNT(*) FILTER (WHERE conversion_status = ''ORDERED''))::numeric / NULLIF(COUNT(*), 0) AS value FROM public.caterpillar_lead_assignment'
        ),
        (
            'caterpillar_successful_orders',
            'Caterpillar Successful Orders',
            'Count of Caterpillar orders in PAID or COMPLETED status.',
            'SELECT COUNT(*) FILTER (WHERE order_status IN (''PAID'',''COMPLETED'')) AS value FROM public.caterpillar_order_record'
        ),
        (
            'caterpillar_paid_order_revenue',
            'Caterpillar Paid Order Revenue',
            'Paid amount from Caterpillar orders in PAID or COMPLETED status.',
            'SELECT COALESCE(SUM(paid_amount) FILTER (WHERE order_status IN (''PAID'',''COMPLETED'')), 0) AS value FROM public.caterpillar_order_record'
        ),
        (
            'caterpillar_survey_completion_rate',
            'Caterpillar Survey Completion Rate',
            'Share of Caterpillar survey responses in SUBMITTED status.',
            'SELECT (COUNT(*) FILTER (WHERE status = ''SUBMITTED''))::numeric / NULLIF(COUNT(*), 0) AS value FROM public.caterpillar_survey_response'
        )
)
INSERT INTO t_metrics_meta (
    datasource_id,
    metric_code,
    metric_name,
    description,
    query_sql,
    parameters,
    value_column,
    status,
    deleted
)
SELECT
    datasource.id,
    definitions.metric_code,
    definitions.metric_name,
    definitions.description,
    definitions.query_sql,
    '[]',
    'value',
    1,
    0
FROM datasource
CROSS JOIN definitions
ON CONFLICT (datasource_id, metric_code) DO UPDATE
SET metric_name = EXCLUDED.metric_name,
    description = EXCLUDED.description,
    query_sql = EXCLUDED.query_sql,
    parameters = EXCLUDED.parameters,
    value_column = EXCLUDED.value_column,
    status = 1,
    deleted = 0,
    updated_at = CURRENT_TIMESTAMP;

WITH datasource AS (
    SELECT id
    FROM t_datasource_config
    WHERE name = 'Caterpillar PostgreSQL'
      AND source_type = 'cdp_postgres'
      AND status = 1
      AND deleted = 0
)
UPDATE lattice_tenants tenant
SET metadata = COALESCE(tenant.metadata, '{}'::jsonb) || jsonb_build_object(
        'source', 'codex',
        'tenantPrefix', 'caterpillar_',
        'metricsDatasourceId', datasource.id
    ),
    updated_at = CURRENT_TIMESTAMP
FROM datasource
WHERE tenant.id = 'caterpillar';

WITH datasource AS (
    SELECT id
    FROM t_datasource_config
    WHERE name = 'Caterpillar PostgreSQL'
      AND source_type = 'cdp_postgres'
      AND status = 1
      AND deleted = 0
)
INSERT INTO lattice_metrics_configs (
    id,
    tenant_id,
    key,
    name,
    description,
    config
)
SELECT
    gen_random_uuid()::text,
    'caterpillar',
    'caterpillar-cdp',
    'Caterpillar CDP',
    'Metrics server config for the Caterpillar PostgreSQL datasource',
    jsonb_build_object(
        'type', 'semantic',
        'serverUrl', 'https://ada.alphafina.cn/api/metrics/api/v1/',
        'timeout', 30000,
        'selectedDataSources', jsonb_build_array(datasource.id::text)
    )
FROM datasource
ON CONFLICT (tenant_id, key) DO UPDATE
SET name = EXCLUDED.name,
    description = EXCLUDED.description,
    config = EXCLUDED.config,
    updated_at = CURRENT_TIMESTAMP;

DO $$
DECLARE
    target_datasource_id BIGINT;
BEGIN
    SELECT id INTO STRICT target_datasource_id
    FROM t_datasource_config
    WHERE name = 'Caterpillar PostgreSQL'
      AND source_type = 'cdp_postgres'
      AND status = 1
      AND deleted = 0;

    IF (SELECT COUNT(*) FROM t_metrics_meta
        WHERE datasource_id = target_datasource_id
          AND metric_code LIKE 'caterpillar\_%' ESCAPE '\'
          AND status = 1
          AND deleted = 0) <> 8 THEN
        RAISE EXCEPTION 'Expected exactly eight active Caterpillar metric registrations';
    END IF;

    IF (SELECT metadata ->> 'metricsDatasourceId'
        FROM lattice_tenants
        WHERE id = 'caterpillar') <> target_datasource_id::text THEN
        RAISE EXCEPTION 'Caterpillar tenant metadata datasource id mismatch';
    END IF;

    IF (SELECT config -> 'selectedDataSources'
        FROM lattice_metrics_configs
        WHERE tenant_id = 'caterpillar'
          AND key = 'caterpillar-cdp') <> jsonb_build_array(target_datasource_id::text) THEN
        RAISE EXCEPTION 'Caterpillar metrics config datasource selection mismatch';
    END IF;
END
$$;

COMMIT;
