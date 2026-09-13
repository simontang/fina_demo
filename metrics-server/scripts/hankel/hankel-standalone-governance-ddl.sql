-- Hankel standalone governance, semantic, quality, and KB metadata schema.
--
-- Purpose:
--   Store non-secret datasource access metadata, semantic-layer publication
--   metadata, and Hankel data-quality insights inside a customer-specific
--   PostgreSQL database.
--
-- This script does not create or modify raw fact tables. Load the raw/report
-- tables and create hankel_view_* views separately before publishing runtime
-- meta to Metrics Server.

CREATE SCHEMA IF NOT EXISTS hankel_governance;
CREATE SCHEMA IF NOT EXISTS hankel_quality;
CREATE SCHEMA IF NOT EXISTS hankel_kb;

CREATE TABLE IF NOT EXISTS hankel_governance.datasource_profile (
    id                    BIGSERIAL PRIMARY KEY,
    tenant_id             TEXT NOT NULL DEFAULT 'hankel',
    datasource_key         TEXT NOT NULL,
    metrics_datasource_id  BIGINT,
    display_name           TEXT NOT NULL,
    engine                 TEXT NOT NULL DEFAULT 'postgresql',
    default_schema         TEXT NOT NULL DEFAULT 'public',
    jdbc_url_ref           TEXT,
    credential_ref         TEXT,
    purpose                TEXT,
    status                 TEXT NOT NULL DEFAULT 'active',
    payload                JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, datasource_key)
);

CREATE TABLE IF NOT EXISTS hankel_governance.datasource_access_grant (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    datasource_key TEXT NOT NULL,
    grant_kind     TEXT NOT NULL,
    schema_name    TEXT NOT NULL DEFAULT 'public',
    table_pattern  TEXT NOT NULL,
    pattern_type   TEXT NOT NULL,
    case_sensitive BOOLEAN NOT NULL DEFAULT false,
    status         TEXT NOT NULL DEFAULT 'active',
    purpose        TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (grant_kind IN ('builder_scope', 'runtime_table')),
    CHECK (pattern_type IN ('PREFIX', 'EXACT')),
    UNIQUE (tenant_id, datasource_key, schema_name, table_pattern, pattern_type)
);

CREATE TABLE IF NOT EXISTS hankel_governance.semantic_table_meta (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    datasource_key TEXT NOT NULL,
    object_key     TEXT NOT NULL,
    schema_name    TEXT NOT NULL DEFAULT 'public',
    table_name     TEXT NOT NULL,
    domain         TEXT NOT NULL,
    display_name   TEXT NOT NULL,
    grain          TEXT,
    business_status TEXT NOT NULL,
    source_tables  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    purpose        TEXT,
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, datasource_key, object_key)
);

CREATE TABLE IF NOT EXISTS hankel_governance.semantic_metric_meta (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    datasource_key TEXT NOT NULL,
    metric_key     TEXT NOT NULL,
    display_name   TEXT NOT NULL,
    domain         TEXT NOT NULL,
    source_view    TEXT NOT NULL,
    calculation    JSONB NOT NULL,
    supported_dimensions JSONB NOT NULL DEFAULT '[]'::jsonb,
    format         TEXT NOT NULL DEFAULT 'number',
    business_status TEXT NOT NULL,
    business_note  TEXT,
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, datasource_key, metric_key)
);

CREATE TABLE IF NOT EXISTS hankel_governance.runtime_endpoint_contract (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    tool_group     TEXT NOT NULL,
    method         TEXT NOT NULL,
    path_template  TEXT NOT NULL,
    audience       TEXT NOT NULL,
    purpose        TEXT NOT NULL,
    constraints    JSONB NOT NULL DEFAULT '{}'::jsonb,
    status         TEXT NOT NULL DEFAULT 'active',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, tool_group, method, path_template)
);

CREATE TABLE IF NOT EXISTS hankel_quality.quality_report (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    report_key     TEXT NOT NULL,
    title          TEXT NOT NULL,
    generated_at   TIMESTAMPTZ,
    datasource_key TEXT NOT NULL,
    summary        TEXT NOT NULL,
    source_files   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, report_key)
);

CREATE TABLE IF NOT EXISTS hankel_quality.quality_issue (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    report_key     TEXT NOT NULL,
    issue_key      TEXT NOT NULL,
    dataset_group  TEXT NOT NULL,
    priority       TEXT NOT NULL,
    title          TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    impact         TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    business_status TEXT NOT NULL DEFAULT 'pending_business_confirmation',
    examples       JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics_refs   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    view_refs      TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, report_key, issue_key)
);

CREATE TABLE IF NOT EXISTS hankel_kb.knowledge_document (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT NOT NULL DEFAULT 'hankel',
    kb_id          TEXT NOT NULL,
    domain         TEXT NOT NULL,
    title          TEXT NOT NULL,
    status         TEXT NOT NULL,
    source_path    TEXT,
    tags           TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    content_md     TEXT NOT NULL,
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, kb_id)
);

INSERT INTO hankel_governance.datasource_profile (
    tenant_id, datasource_key, metrics_datasource_id, display_name, engine,
    default_schema, jdbc_url_ref, credential_ref, purpose, payload
)
VALUES (
    'hankel',
    'hankel_postgres',
    15,
    'Hankel PostgreSQL Analytics Datasource',
    'postgresql',
    'public',
    'Metrics Server datasource 15 URL reference',
    'secret-manager-or-metrics-server-encrypted-password',
    'Hankel ACM analytics datasource for River Distributor Review and Caren Run for Gold.',
    '{"source_type":"cdp_postgres","do_not_store_plain_credentials":true}'::jsonb
)
ON CONFLICT (tenant_id, datasource_key) DO UPDATE SET
    metrics_datasource_id = EXCLUDED.metrics_datasource_id,
    display_name = EXCLUDED.display_name,
    engine = EXCLUDED.engine,
    default_schema = EXCLUDED.default_schema,
    jdbc_url_ref = EXCLUDED.jdbc_url_ref,
    credential_ref = EXCLUDED.credential_ref,
    purpose = EXCLUDED.purpose,
    payload = EXCLUDED.payload,
    updated_at = now();

INSERT INTO hankel_governance.datasource_access_grant (
    tenant_id, datasource_key, grant_kind, schema_name, table_pattern,
    pattern_type, case_sensitive, purpose
)
VALUES
    ('hankel','hankel_postgres','builder_scope','public','hankel_','PREFIX',false,'Builder/Admin read-only exploration scope.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_sales_name_mapping','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_project_opportunity_line','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_new_order_line','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_won_validation_match_key','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_sales_summary','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_leaderboard','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_segment_leaderboard','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_qualification_gap','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_report_reconciliation','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_distr_sell_out','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_distr_sell_in','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_distr_inventory_monthly','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_distr_inventory_current','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_parameters','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_new_project_opportunity','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_validated_won_opportunity','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_qualification_status','EXACT',false,'Runtime semantic view.'),
    ('hankel','hankel_postgres','runtime_table','public','hankel_view_run_for_gold_segment_qualification_status','EXACT',false,'Runtime semantic view.')
ON CONFLICT (tenant_id, datasource_key, schema_name, table_pattern, pattern_type) DO UPDATE SET
    grant_kind = EXCLUDED.grant_kind,
    case_sensitive = EXCLUDED.case_sensitive,
    purpose = EXCLUDED.purpose,
    status = 'active',
    updated_at = now();

INSERT INTO hankel_governance.semantic_table_meta (
    tenant_id, datasource_key, object_key, schema_name, table_name, domain,
    display_name, grain, business_status, source_tables, purpose
)
VALUES
    ('hankel','hankel_postgres','hankel_view_distr_sell_in','public','hankel_view_distr_sell_in','River Distributor Performance','Distributor Sell-in','imported sell-in row','customer_confirmed_semantic_foundation',ARRAY['hankel_distr_sell_in'],'Sell-in view with customer scope flags and excluded-value audit fields.'),
    ('hankel','hankel_postgres','hankel_view_distr_sell_out','public','hankel_view_distr_sell_out','River Distributor Performance','Distributor Sell-out','customer-product-month-sales-team allocation row','customer_confirmed_semantic_foundation',ARRAY['hankel_distr_sell_out'],'Sell-out view using territory allocation and demo quality guardrails.'),
    ('hankel','hankel_postgres','hankel_view_distr_inventory_monthly','public','hankel_view_distr_inventory_monthly','River Distributor Performance','Distributor Inventory Monthly','customer-product-month-sales-team snapshot','customer_confirmed_semantic_foundation',ARRAY['hankel_distr_inventory'],'Monthly inventory snapshot view.'),
    ('hankel','hankel_postgres','hankel_view_distr_inventory_current','public','hankel_view_distr_inventory_current','River Distributor Performance','Distributor Inventory Current','customer-product-sales-team latest snapshot','customer_confirmed_semantic_foundation',ARRAY['hankel_view_distr_inventory_monthly'],'Latest inventory snapshot view.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_parameters','public','hankel_view_run_for_gold_parameters','Caren Run for Gold','Run for Gold Parameters','one parameter row','demo_fixed_parameters',ARRAY[]::TEXT[],'Fixed demo parameter view.'),
    ('hankel','hankel_postgres','hankel_view_sales_name_mapping','public','hankel_view_sales_name_mapping','Caren Run for Gold','Sales Name Mapping','mapping row','customer_confirmed_semantic_foundation',ARRAY['hankel_sales_name_mapping'],'Canonical sales name mapping view.'),
    ('hankel','hankel_postgres','hankel_view_project_opportunity_line','public','hankel_view_project_opportunity_line','Caren Run for Gold','Project Opportunity Line','opportunity-product line','customer_confirmed_semantic_foundation',ARRAY['hankel_project_opportunity_lines','hankel_view_sales_name_mapping'],'Normalized project opportunity line view.'),
    ('hankel','hankel_postgres','hankel_view_new_project_opportunity','public','hankel_view_new_project_opportunity','Caren Run for Gold','New Project Opportunity','sales-opportunity','written_spec_reference',ARRAY['hankel_view_project_opportunity_line'],'Opportunity-level new project aggregation.'),
    ('hankel','hankel_postgres','hankel_view_new_order_line','public','hankel_view_new_order_line','Caren Run for Gold','New Order Line','order-item line','customer_confirmed_semantic_foundation',ARRAY['hankel_new_order_lines','hankel_view_sales_name_mapping'],'Normalized new order line view.'),
    ('hankel','hankel_postgres','hankel_view_won_validation_match_key','public','hankel_view_won_validation_match_key','Caren Run for Gold','Won Validation Match Key','canonical sales + sold-to + product','customer_confirmed_semantic_foundation',ARRAY['hankel_view_project_opportunity_line','hankel_view_new_order_line'],'Won to New Order 50 percent validation view.'),
    ('hankel','hankel_postgres','hankel_view_validated_won_opportunity','public','hankel_view_validated_won_opportunity','Caren Run for Gold','Validated Won Opportunity','sales-opportunity','written_spec_reference',ARRAY['hankel_view_project_opportunity_line','hankel_view_won_validation_match_key'],'Distinct opportunity validation view.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_sales_summary','public','hankel_view_run_for_gold_sales_summary','Caren Run for Gold','Run for Gold Sales Summary','canonical sales name','written_spec_reference',ARRAY['hankel_view_new_project_opportunity','hankel_view_won_validation_match_key','hankel_view_validated_won_opportunity'],'Sales-level Run for Gold summary.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_qualification_status','public','hankel_view_run_for_gold_qualification_status','Caren Run for Gold','Run for Gold Qualification','award pool + sales','written_spec_reference',ARRAY['hankel_view_run_for_gold_sales_summary'],'Overall award qualification status.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_leaderboard','public','hankel_view_run_for_gold_leaderboard','Caren Run for Gold','Run for Gold Overall Leaderboard','award pool + qualified sales','written_spec_reference',ARRAY['hankel_view_run_for_gold_qualification_status'],'Overall leaderboard view.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_segment_qualification_status','public','hankel_view_run_for_gold_segment_qualification_status','Caren Run for Gold','Run for Gold Segment Qualification','segment + sales','written_spec_reference',ARRAY['hankel_view_project_opportunity_line','hankel_view_won_validation_match_key','hankel_view_validated_won_opportunity'],'Segment award qualification status.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_segment_leaderboard','public','hankel_view_run_for_gold_segment_leaderboard','Caren Run for Gold','Run for Gold Segment Leaderboard','segment + qualified sales','written_spec_reference',ARRAY['hankel_view_run_for_gold_segment_qualification_status'],'Segment leaderboard view.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_qualification_gap','public','hankel_view_run_for_gold_qualification_gap','Caren Run for Gold','Run for Gold Qualification Gap','scope + pool/segment + sales','written_spec_reference',ARRAY['hankel_view_run_for_gold_qualification_status','hankel_view_run_for_gold_segment_qualification_status'],'Qualification gap view.'),
    ('hankel','hankel_postgres','hankel_view_run_for_gold_report_reconciliation','public','hankel_view_run_for_gold_report_reconciliation','Caren Run for Gold','Run for Gold Report Reconciliation','qa check + business key','demo_quality',ARRAY['hankel_view_run_for_gold_sales_summary','hankel_view_run_for_gold_leaderboard','hankel_view_run_for_gold_segment_leaderboard','hankel_report_*'],'Golden report QA reconciliation view.')
ON CONFLICT (tenant_id, datasource_key, object_key) DO UPDATE SET
    domain = EXCLUDED.domain,
    display_name = EXCLUDED.display_name,
    grain = EXCLUDED.grain,
    business_status = EXCLUDED.business_status,
    source_tables = EXCLUDED.source_tables,
    purpose = EXCLUDED.purpose,
    updated_at = now();

INSERT INTO hankel_governance.semantic_metric_meta (
    tenant_id, datasource_key, metric_key, display_name, domain, source_view,
    calculation, format, business_status, business_note
)
VALUES
    ('hankel','hankel_postgres','hankel_sell_in_nes','Sell-in NES 净外部销售额','River Distributor Performance','public.hankel_view_distr_sell_in','{"type":"aggregate","aggregation":"sum","measure":"nes"}','currency','customer_confirmed','Customer-confirmed Sell-in scope is enforced in the semantic view.'),
    ('hankel','hankel_postgres','hankel_sell_in_quantity','Sell-in 进货数量','River Distributor Performance','public.hankel_view_distr_sell_in','{"type":"aggregate","aggregation":"sum","measure":"sell_in_quantity"}','number','customer_confirmed','Current River POC quantity metric.'),
    ('hankel','hankel_postgres','hankel_sell_in_gross_margin','Sell-in 毛利额','River Distributor Performance','public.hankel_view_distr_sell_in','{"type":"aggregate","aggregation":"sum","measure":"gross_margin"}','currency','pending_business_confirmation','Gross Margin field source and production filter semantics require confirmation.'),
    ('hankel','hankel_postgres','hankel_sell_in_contribution','Sell-in 产品贡献额','River Distributor Performance','public.hankel_view_distr_sell_in','{"type":"aggregate","aggregation":"sum","measure":"product_contribution_15"}','currency','pending_business_confirmation','The business meaning of the 15* contribution field requires confirmation.'),
    ('hankel','hankel_postgres','hankel_sell_in_gross_margin_rate','Sell-in 毛利率','River Distributor Performance','public.hankel_view_distr_sell_in','{"type":"derived","operator":"ratio","numerator":"hankel_sell_in_gross_margin","denominator":"hankel_sell_in_nes"}','percent','pending_business_confirmation','Depends on pending Gross Margin definition.'),
    ('hankel','hankel_postgres','hankel_sell_out_value','Sell-out 出货金额','River Distributor Performance','public.hankel_view_distr_sell_out','{"type":"aggregate","aggregation":"sum","measure":"sell_out_value"}','currency','customer_confirmed','Territory amount with demo quality guardrails.'),
    ('hankel','hankel_postgres','hankel_sell_out_quantity','Sell-out 出货数量','River Distributor Performance','public.hankel_view_distr_sell_out','{"type":"aggregate","aggregation":"sum","measure":"sell_out_quantity"}','number','customer_confirmed','Territory quantity with quantity quality guardrail.'),
    ('hankel','hankel_postgres','hankel_sell_out_excluded_value','Sell-out 质量规则排除金额','River Distributor Performance','public.hankel_view_distr_sell_out','{"type":"aggregate","aggregation":"sum","measure":"excluded_sell_out_value"}','currency','demo_quality','Quality metric; not a business KPI.'),
    ('hankel','hankel_postgres','hankel_sell_out_quality_issue_count','Sell-out 异常行数','River Distributor Performance','public.hankel_view_distr_sell_out','{"type":"aggregate","aggregation":"sum","measure":"quality_issue_row_count"}','number','demo_quality','Quality metric; not a business KPI.'),
    ('hankel','hankel_postgres','hankel_inventory_value','当前库存金额','River Distributor Performance','public.hankel_view_distr_inventory_current','{"type":"aggregate","aggregation":"sum","measure":"inventory_value"}','currency','written_spec_reference','Inventory amount is the default meaning but not selected as current River POC KPI.'),
    ('hankel','hankel_postgres','hankel_inventory_quantity','当前库存数量','River Distributor Performance','public.hankel_view_distr_inventory_current','{"type":"aggregate","aggregation":"sum","measure":"inventory_quantity"}','number','customer_confirmed','Current River POC quantity metric; latest snapshot only.'),
    ('hankel','hankel_postgres','hankel_new_projects_count','Hankel New Projects Count','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"new_project_count"}','number','written_spec_reference','Competition metric from written specification.'),
    ('hankel','hankel_postgres','hankel_new_projects_y1','Hankel New Projects Y1','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"new_y1"}','currency','written_spec_reference','Opportunity-level SUM is applied before salesperson totals.'),
    ('hankel','hankel_postgres','hankel_validated_won_count','Hankel Validated Won Count','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"validated_won_count"}','number','written_spec_reference','Counts distinct Opportunity IDs.'),
    ('hankel','hankel_postgres','hankel_validation_won_y1','Hankel Validation Won Y1','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"validation_won_y1"}','currency','written_spec_reference','Check-period Won Y1.'),
    ('hankel','hankel_postgres','hankel_required_new_order_value','Hankel Required New Order Value','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"required_new_order_value"}','currency','customer_confirmed','Exact 50 percent threshold; rounding is display-only.'),
    ('hankel','hankel_postgres','hankel_matched_new_order_value','Hankel Matched New Order Value','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"matched_new_order_value"}','currency','customer_confirmed','Exact Sales Name + Sold-to IDH + Product IDH match keys.'),
    ('hankel','hankel_postgres','hankel_order_coverage_rate','Hankel Order Coverage Rate','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"derived","operator":"ratio","numerator":"hankel_matched_new_order_value","denominator":"hankel_validation_won_y1"}','percent','customer_confirmed','Uncapped Matched New Order divided by Check-period Won Y1.'),
    ('hankel','hankel_postgres','hankel_new_order_gap','Hankel New Order Action Gap','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"new_order_gap"}','currency','pending_business_confirmation','Non-negative Action Gap.'),
    ('hankel','hankel_postgres','hankel_new_order_signed_gap','Hankel New Order Signed Gap','Caren Run for Gold','public.hankel_view_won_validation_match_key','{"type":"aggregate","aggregation":"sum","measure":"new_order_gap_raw"}','currency','pending_business_confirmation','Signed Gap: negative values represent over-coverage.'),
    ('hankel','hankel_postgres','hankel_competition_won_y1','Hankel Competition Won Y1','Caren Run for Gold','public.hankel_view_run_for_gold_sales_summary','{"type":"aggregate","aggregation":"sum","measure":"competition_won_y1"}','currency','written_spec_reference','Ranking input from written specification.'),
    ('hankel','hankel_postgres','hankel_final_score','Hankel Final Score','Caren Run for Gold','public.hankel_view_run_for_gold_leaderboard','{"type":"aggregate","aggregation":"avg","measure":"score"}','number','written_spec_reference','Non-additive; must group by canonical sales name.'),
    ('hankel','hankel_postgres','hankel_segment_final_score','Hankel Segment Final Score','Caren Run for Gold','public.hankel_view_run_for_gold_segment_leaderboard','{"type":"aggregate","aggregation":"avg","measure":"score"}','number','written_spec_reference','Non-additive; must group by segment and canonical sales name.'),
    ('hankel','hankel_postgres','hankel_match_key_count','Hankel Won Match Key Count','Caren Run for Gold','public.hankel_view_won_validation_match_key','{"type":"aggregate","aggregation":"count_distinct","measure":"match_key"}','number','technical_diagnostic','Diagnostic match key count; not Validated Won Opportunity count.')
ON CONFLICT (tenant_id, datasource_key, metric_key) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    domain = EXCLUDED.domain,
    source_view = EXCLUDED.source_view,
    calculation = EXCLUDED.calculation,
    format = EXCLUDED.format,
    business_status = EXCLUDED.business_status,
    business_note = EXCLUDED.business_note,
    updated_at = now();

INSERT INTO hankel_governance.runtime_endpoint_contract (
    tenant_id, tool_group, method, path_template, audience, purpose, constraints
)
VALUES
    ('hankel','direct_db_analysis','CONNECT','postgresql://{host}:{port}/{database}','analyst_admin','Direct read-only PostgreSQL connection for analyst or DBA validation.','{"ordinary_agent_allowed":false,"credential_storage":"secret_ref_only","recommended_role":"hankel_analytics_ro","sslmode":"require","allowed_objects":"public.hankel_* and public.hankel_view_*"}'::jsonb),
    ('hankel','direct_db_analysis','CONNECT','ssh-tunnel://{bastion-host}:{local-port}->{pg-host}:{pg-port}','analyst_admin','SSH tunnel pattern when the Hankel PostgreSQL database is private-network only.','{"ordinary_agent_allowed":false,"credential_storage":"secret_ref_only","example_local_port":15432}'::jsonb),
    ('hankel','datasource_tool','GET','/api/v1/datasources/{dsId}/table-grants','builder_admin','Read governed exploration scope.','{"ordinary_agent_allowed":false}'::jsonb),
    ('hankel','datasource_tool','POST','/api/v1/datasources/{dsId}/query','builder_admin','Read-only datasource exploration within grant scope.','{"ordinary_agent_allowed":false,"allowed_sql":["SELECT","WITH"],"forbidden":["DDL","DML","multi_statement"]}'::jsonb),
    ('hankel','meta_tool','GET','/api/v1/datasources/{dsId}/meta/tables','builder_admin','List published table meta.','{"ordinary_agent_allowed":false}'::jsonb),
    ('hankel','meta_tool','POST','/api/v1/datasources/{dsId}/meta/tables','builder_admin','Publish table meta.','{"ordinary_agent_allowed":false}'::jsonb),
    ('hankel','meta_tool','GET','/api/v1/datasources/{dsId}/meta/metrics','builder_admin','List published metric meta.','{"ordinary_agent_allowed":false}'::jsonb),
    ('hankel','meta_tool','POST','/api/v1/datasources/{dsId}/meta/metrics','builder_admin','Publish metric meta.','{"ordinary_agent_allowed":false}'::jsonb),
    ('hankel','runtime_tool','GET','/api/v1/datasources/{dsId}/meta','ordinary_agent','Read runtime semantic catalog.','{"ordinary_agent_allowed":true}'::jsonb),
    ('hankel','runtime_tool','POST','/api/v1/metrics/query','ordinary_agent','Execute semantic metric query.','{"ordinary_agent_allowed":true,"requires_published_metric":true,"requires_published_dimension":true}'::jsonb)
ON CONFLICT (tenant_id, tool_group, method, path_template) DO UPDATE SET
    audience = EXCLUDED.audience,
    purpose = EXCLUDED.purpose,
    constraints = EXCLUDED.constraints,
    updated_at = now();

INSERT INTO hankel_quality.quality_report (
    tenant_id, report_key, title, generated_at, datasource_key, summary, source_files, payload
)
VALUES (
    'hankel',
    'hankel-quality-insights-20260905',
    'Hankel 数据质量问题报告',
    '2026-09-05T00:00:00+08:00',
    'hankel_postgres',
    '按 River Distributor Review 与 Caren Run for Gold 拆分数据质量、主数据、业务口径和 golden report 对账问题。',
    ARRAY[
        '2060903_dist_review_template.xlsx',
        'Data of New Order.xlsx',
        'Sales_Name_Mapping 1.xlsx',
        'project raw data_1st Aug.xlsx',
        'Run_for_Gold_A_Sales_Edition_YTD_Aug_2026 3.xlsx'
    ],
    '{"scope":{"tenant":"hankel","datasourceId":15,"database":"Aliyun PostgreSQL"}}'::jsonb
)
ON CONFLICT (tenant_id, report_key) DO UPDATE SET
    title = EXCLUDED.title,
    generated_at = EXCLUDED.generated_at,
    datasource_key = EXCLUDED.datasource_key,
    summary = EXCLUDED.summary,
    source_files = EXCLUDED.source_files,
    payload = EXCLUDED.payload,
    updated_at = now();

INSERT INTO hankel_quality.quality_issue (
    tenant_id, report_key, issue_key, dataset_group, priority, title,
    evidence, impact, recommended_action, business_status, examples,
    metrics_refs, view_refs
)
VALUES
    ('hankel','hankel-quality-insights-20260905','SELL_OUT_EXTREME_OUTLIER','River','P0','Sell-out 极端数量/金额','7 个分摊行、3 笔基础交易触发 demo guardrail；示例数量 2,147,483,647，金额 343,929,169,743.46。','污染 Sell-out 排名、趋势和金额规模。','请客户确认是真实交易、单位错误、系统溢出还是测试数据。','pending_business_confirmation','[{"combined_id":"1579020","period":"2025-05","customer":"沈阳赛福化工材料有限公司","product":"1800574 · LOCTITE 680 RC BO250ML EN/CH/J","raw_quantity":2147483647,"raw_amount":343929169743.46,"allocation_teams":["GM Anhui&Shandong","GM Beijing","GM Ningbo"]}]'::jsonb,ARRAY['hankel_sell_out_excluded_value','hankel_sell_out_quality_issue_count'],ARRAY['hankel_view_distr_sell_out']),
    ('hankel','hankel-quality-insights-20260905','PRODUCT_UNIT_INCONSISTENT','River','P0','产品单位不统一','666 个 Product Code 存在多个非空单位，例如 CON / PC / PCS / 支 / KG。','数量、Sell-through、库存周转不能直接跨单位汇总。','提供产品基础单位和换算表。','pending_business_confirmation','[{"product_code":"2982626","product":"LOCTITE 243 BO50ML CH","units":["CON","PC","PCS","支"],"affected_rows":19208},{"product_code":"1311320","product":"LOCTITE 243 BO50MLEN/CH/JP","units":["CON","PC","PCS","支"],"affected_rows":13082}]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_distr_sell_out']),
    ('hankel','hankel-quality-insights-20260905','END_CUSTOMER_ID_COLLISION','River','P0','End Customer 身份碰撞','规范化后 25 组名称对应多个非空 End Customer Number，影响 5,248 行。','按名称聚合会合并不同编号；按编号聚合可能拆分同一客户。','确认终端客户唯一键和主数据治理规则。','pending_business_confirmation','[{"normalized_name":"SEW传动设备（苏州）有限公司","end_customer_numbers":["endCust_40406","endCust_324","endCust_6","endCust_25","endCust_18"],"affected_rows":798}]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_distr_sell_out']),
    ('hankel','hankel-quality-insights-20260905','RIVER_SALES_TYPE_MAPPING_MISSING','River','P1','Sales Type 跨事实映射缺失','尚无 Sell-in/Sell-out/Inventory Team 到统一 Sales Type 的受治理映射。','不能正式做跨事实 Sales Type 对比。','提供维护表和冲突处理规则。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_distr_sell_in','hankel_view_distr_sell_out','hankel_view_distr_inventory_current']),
    ('hankel','hankel-quality-insights-20260905','NEGATIVE_TRANSACTION_MEANING','River','P1','负数交易含义未闭环','Sell-in 有效 NES 负数 5,974 行，Sell-out 金额负数 709 行，数量负数 740 行。','趋势分析需要决定按原月冲减还是回冲业务发生月。','确认退货、贷项、冲销和跨月回补规则。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_distr_sell_in','hankel_view_distr_sell_out']),
    ('hankel','hankel-quality-insights-20260905','DATA_FRESHNESS_2026_07','River','P1','数据截至时点','Sell-in、Sell-out、Inventory 当前最新数据月为 2026-07。','查询 2026-08 不能返回 0 或暗示完整。','对外回答必须展示数据截至月。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_distr_sell_in','hankel_view_distr_sell_out','hankel_view_distr_inventory_current']),
    ('hankel','hankel-quality-insights-20260905','SALES_NAME_MAPPING_GAP','Caren','P0','Sales Name Mapping 覆盖不足','2026 年有效 New Order 有 7,803 行未映射，占 38.67%；金额 179,054,447.57，占 44.22%。','未映射订单不能进入自动 Won 验证，coverage 可能偏低。','补齐 Project/New Order sales name 到 canonical name 的维护表。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_sales_name_mapping','hankel_view_new_order_line','hankel_view_project_opportunity_line']),
    ('hankel','hankel-quality-insights-20260905','GOLDEN_REPORT_RAW_MISMATCH','Caren','P1','Golden Report 与 raw-derived 结果未完全对齐','统一竞赛口径后，Won detail 仍有 356 个 key 只存在一侧，119 个双方 key 金额不一致。','可能来自不同快照版本、报表过滤差异或映射版本差异。','获取 golden report 生成版本、数据时间、映射表版本和取整规则。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_run_for_gold_report_reconciliation']),
    ('hankel','hankel-quality-insights-20260905','PROJECT_MISSING_SOLD_TO','Caren','P1','Project 关键字段缺失','有效业务行中 Sold-to 缺失 163 行；缺 Sold-to 的行不能形成完整 match key。','不能自动参与 Won→New Order 匹配。','补齐 Sold-to 或明确排除规则。','pending_business_confirmation','[]'::jsonb,ARRAY[]::TEXT[],ARRAY['hankel_view_project_opportunity_line']),
    ('hankel','hankel-quality-insights-20260905','GAP_DEFINITION_PENDING','Caren','P1','Gap 展示口径待固化','底层同时支持 Action Gap 与 Signed Gap。','报表、告警、Agent 回答如果混用会产生差异。','客户确认默认展示口径。','pending_business_confirmation','[]'::jsonb,ARRAY['hankel_new_order_gap','hankel_new_order_signed_gap'],ARRAY['hankel_view_run_for_gold_sales_summary','hankel_view_won_validation_match_key'])
ON CONFLICT (tenant_id, report_key, issue_key) DO UPDATE SET
    dataset_group = EXCLUDED.dataset_group,
    priority = EXCLUDED.priority,
    title = EXCLUDED.title,
    evidence = EXCLUDED.evidence,
    impact = EXCLUDED.impact,
    recommended_action = EXCLUDED.recommended_action,
    business_status = EXCLUDED.business_status,
    examples = EXCLUDED.examples,
    metrics_refs = EXCLUDED.metrics_refs,
    view_refs = EXCLUDED.view_refs,
    updated_at = now();
