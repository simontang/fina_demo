-- ============================================================
-- Metrics Server — Master Schema DDL (PostgreSQL)
-- Master DB stores datasource configs and metric definitions.
-- and metric definitions. Run once on a fresh PostgreSQL database.
-- ============================================================

-- Dynamic datasource configurations.
-- Each row represents one external connection available for metric queries.
CREATE TABLE IF NOT EXISTS t_datasource_config (
    id          BIGSERIAL       PRIMARY KEY,
    name        VARCHAR(200)    NOT NULL,
    url         VARCHAR(500)    NOT NULL,       -- jdbc:sap://..., jdbc:postgresql://...
    username    VARCHAR(200)    NOT NULL,
    password    VARCHAR(500)    NOT NULL,       -- AES-encrypted datasource password
    schema_name VARCHAR(128),                  -- optional default schema/search_path
    source_type VARCHAR(64)     NOT NULL DEFAULT 'sap_b1_hana',
    description VARCHAR(1000),
    status      SMALLINT        NOT NULL DEFAULT 1,  -- 1=active 0=inactive
    created_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     SMALLINT        NOT NULL DEFAULT 0   -- 1=deleted (soft delete)
);

CREATE INDEX IF NOT EXISTS idx_ds_status ON t_datasource_config (status, deleted);
CREATE INDEX IF NOT EXISTS idx_ds_name   ON t_datasource_config (name);
CREATE INDEX IF NOT EXISTS idx_ds_source_type ON t_datasource_config (source_type, status, deleted);

-- Metric definitions bound to a SAP B1 HANA datasource
-- Each row describes a named, parameterised SQL query executed on the target HANA.
CREATE TABLE IF NOT EXISTS t_metrics_meta (
    id            BIGSERIAL       PRIMARY KEY,
    datasource_id BIGINT          NOT NULL,
    metric_code   VARCHAR(100)    NOT NULL,    -- unique within a datasource
    metric_name   VARCHAR(200)    NOT NULL,
    description   VARCHAR(1000),
    query_sql     TEXT            NOT NULL,    -- SQL with :paramName placeholders
    parameters    TEXT,                        -- JSON array of param descriptors
    value_column  VARCHAR(100),               -- column that holds the primary value
    status        SMALLINT        NOT NULL DEFAULT 1,
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted       SMALLINT        NOT NULL DEFAULT 0,
    CONSTRAINT uq_metrics_code UNIQUE (datasource_id, metric_code)
);

CREATE INDEX IF NOT EXISTS idx_mm_ds_id ON t_metrics_meta (datasource_id, status, deleted);
CREATE INDEX IF NOT EXISTS idx_mm_code  ON t_metrics_meta (metric_code);

-- ============================================================
-- Sample seed data (adjust to your environment)
-- ============================================================

-- Example: add a SAP B1 HANA datasource
-- The password field must be AES-encrypted (use POST /api/v1/datasources/test
-- or EncryptUtil directly to generate the encrypted value).

-- INSERT INTO t_datasource_config (name, url, username, password, schema_name, description, status)
-- VALUES (
--     'SAP B1 Production',
--     'jdbc:sap://hana-host:39015?currentSchema=SBO_DEMO',
--     'B1_USER',
--     '<aes-encrypted-password>',
--     'SBO_DEMO',
--     'Main SAP Business One HANA instance',
--     1
-- );

-- Example: register a metric definition for the above datasource
-- INSERT INTO t_metrics_meta (datasource_id, metric_code, metric_name, description, query_sql, parameters, value_column, status)
-- VALUES (
--     1,
--     'delivery_qty',
--     '交货数量',
--     'Returns total delivered quantity within a date range',
--     'SELECT SUM("Quantity") AS value FROM "MTC_VW_AI_ODLN" WHERE "DocDate" BETWEEN :startDate AND :endDate',
--     '[{"name":"startDate","type":"STRING","required":true,"description":"Start date yyyy-MM-dd"},{"name":"endDate","type":"STRING","required":true,"description":"End date yyyy-MM-dd"}]',
--     'value',
--     1
-- );
