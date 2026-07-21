-- ============================================================
-- CDP Service — Master Schema DDL (PostgreSQL)
-- Stores segment definitions and materialized segment snapshots.
-- Datasource credentials/configs are reused from t_datasource_config.
-- ============================================================

CREATE TABLE IF NOT EXISTS t_datasource_config (
    id          BIGSERIAL       PRIMARY KEY,
    name        VARCHAR(200)    NOT NULL,
    url         VARCHAR(500)    NOT NULL,
    username    VARCHAR(200)    NOT NULL,
    password    VARCHAR(500)    NOT NULL,
    schema_name VARCHAR(128),
    source_type VARCHAR(64)     NOT NULL DEFAULT 'cdp_postgres',
    description VARCHAR(1000),
    status      SMALLINT        NOT NULL DEFAULT 1,
    created_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     SMALLINT        NOT NULL DEFAULT 0
);

ALTER TABLE t_datasource_config
    ADD COLUMN IF NOT EXISTS source_type VARCHAR(64) NOT NULL DEFAULT 'cdp_postgres';

CREATE INDEX IF NOT EXISTS idx_cdp_ds_status ON t_datasource_config (status, deleted);
CREATE INDEX IF NOT EXISTS idx_cdp_ds_source_type ON t_datasource_config (source_type, status, deleted);

CREATE TABLE IF NOT EXISTS t_segment_definition (
    id            BIGSERIAL       PRIMARY KEY,
    tenant_id     VARCHAR(128)    NOT NULL DEFAULT 'default',
    name          VARCHAR(200)    NOT NULL,
    description   VARCHAR(1000),
    datasource_id BIGINT          NOT NULL,
    query_sql     TEXT            NOT NULL,
    status        SMALLINT        NOT NULL DEFAULT 1,
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       SMALLINT        NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_segment_definition_tenant
    ON t_segment_definition (tenant_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_segment_definition_datasource
    ON t_segment_definition (datasource_id);
CREATE INDEX IF NOT EXISTS idx_segment_definition_name
    ON t_segment_definition (tenant_id, name);

ALTER TABLE t_segment_definition
    ADD COLUMN IF NOT EXISTS thread_id VARCHAR(128);

CREATE TABLE IF NOT EXISTS t_segment_data (
    id            BIGSERIAL       PRIMARY KEY,
    tenant_id     VARCHAR(128)    NOT NULL DEFAULT 'default',
    definition_id BIGINT          NOT NULL,
    run_id        VARCHAR(128)    NOT NULL,
    data_json     TEXT            NOT NULL,
    row_count     INTEGER         NOT NULL DEFAULT 0,
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       SMALLINT        NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_segment_data_tenant_definition
    ON t_segment_data (tenant_id, definition_id, deleted, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_segment_data_run_id
    ON t_segment_data (tenant_id, run_id);

CREATE TABLE IF NOT EXISTS t_marketing_campaign (
    id                            BIGSERIAL       PRIMARY KEY,
    tenant_id                     VARCHAR(128)    NOT NULL DEFAULT 'default',
    name                          VARCHAR(200)    NOT NULL,
    description                   VARCHAR(1000),
    type                          VARCHAR(100)    NOT NULL,
    status                        VARCHAR(32)     NOT NULL DEFAULT 'draft',
    goal                          TEXT            NOT NULL,
    start_time                    TIMESTAMP       NOT NULL,
    end_time                      TIMESTAMP       NOT NULL,
    main_segment_data_id          BIGINT,
    segmentation_strategy_json    TEXT            NOT NULL DEFAULT '{}',
    control_group_strategy_json   TEXT            NOT NULL DEFAULT '{}',
    content_channel_strategy_json TEXT            NOT NULL DEFAULT '{}',
    offer_strategy_json           TEXT            NOT NULL DEFAULT '{}',
    wave_strategy_json            TEXT            NOT NULL DEFAULT '{}',
    ab_test_strategy_json         TEXT            NOT NULL DEFAULT '{}',
    statistics_json               TEXT            NOT NULL DEFAULT '{}',
    actual_started_at             TIMESTAMP,
    actual_stopped_at             TIMESTAMP,
    created_at                    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at                    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted                       SMALLINT        NOT NULL DEFAULT 0
);

ALTER TABLE t_marketing_campaign
    ADD COLUMN IF NOT EXISTS thread_id VARCHAR(128);

CREATE INDEX IF NOT EXISTS idx_marketing_campaign_tenant_status
    ON t_marketing_campaign (tenant_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_marketing_campaign_tenant_type
    ON t_marketing_campaign (tenant_id, type, deleted);
CREATE INDEX IF NOT EXISTS idx_marketing_campaign_tenant_time
    ON t_marketing_campaign (tenant_id, start_time, end_time);
