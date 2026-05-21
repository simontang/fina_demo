-- ============================================================
-- B1S — Master Schema DDL (PostgreSQL)
-- Master DB stores datasource configs (pointing to SAP B1 SQL Server)
-- and metric definitions. Run once on a fresh PostgreSQL database.
-- ============================================================

CREATE TABLE IF NOT EXISTS t_datasource_config (
    id          BIGSERIAL       PRIMARY KEY,
    name        VARCHAR(200)    NOT NULL,
    url         VARCHAR(500)    NOT NULL,
    username    VARCHAR(200)    NOT NULL,
    password    VARCHAR(500)    NOT NULL,
    schema_name VARCHAR(128),
    instance_type VARCHAR(50)   NOT NULL DEFAULT 'SQLSERVER',
    description VARCHAR(1000),
    status      SMALLINT        NOT NULL DEFAULT 1,  -- 1=active 0=inactive
    created_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted     SMALLINT        NOT NULL DEFAULT 0   -- 1=deleted (soft delete)
);

-- Existing deployments may already have t_datasource_config without instance_type.
ALTER TABLE t_datasource_config
    ADD COLUMN IF NOT EXISTS instance_type VARCHAR(50) NOT NULL DEFAULT 'SQLSERVER';

CREATE INDEX IF NOT EXISTS idx_ds_status ON t_datasource_config (status, deleted);
CREATE INDEX IF NOT EXISTS idx_ds_name   ON t_datasource_config (name);
CREATE INDEX IF NOT EXISTS idx_ds_instance_type ON t_datasource_config (instance_type, status, deleted);

-- Metric definitions bound to a SAP B1 SQL Server datasource
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

-- Inbound email messages fetched through IMAP.
CREATE TABLE IF NOT EXISTS t_mail_message (
    id               BIGSERIAL     PRIMARY KEY,
    provider         VARCHAR(50)   NOT NULL DEFAULT 'larksuite',
    mailbox          VARCHAR(200)  NOT NULL,
    folder_name      VARCHAR(200)  NOT NULL DEFAULT 'INBOX',
    uid              BIGINT,
    message_id       VARCHAR(500),
    subject          TEXT,
    from_address     TEXT,
    to_addresses     TEXT,
    cc_addresses     TEXT,
    sent_at          VARCHAR(50),
    received_at      VARCHAR(50),
    has_attachments  BOOLEAN       NOT NULL DEFAULT FALSE,
    attachment_count INTEGER       NOT NULL DEFAULT 0,
    snippet          TEXT,
    created_at       TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_mail_message_uid
    ON t_mail_message (mailbox, folder_name, uid)
    WHERE uid IS NOT NULL AND uid > 0;
CREATE UNIQUE INDEX IF NOT EXISTS uq_mail_message_message_id
    ON t_mail_message (mailbox, message_id)
    WHERE message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mail_message_created_at ON t_mail_message (created_at DESC);

-- Inbound email attachments. The binary file is stored in Volcano Engine TOS;
-- this table stores only metadata and object location.
CREATE TABLE IF NOT EXISTS t_mail_attachment (
    id               BIGSERIAL     PRIMARY KEY,
    mail_message_id  BIGINT        NOT NULL REFERENCES t_mail_message(id) ON DELETE CASCADE,
    file_name        TEXT,
    content_type     TEXT,
    size_bytes       BIGINT,
    tos_bucket       VARCHAR(200),
    tos_key          TEXT,
    tos_url          TEXT,
    upload_status    VARCHAR(50)   NOT NULL DEFAULT 'PENDING',
    error_message    TEXT,
    created_at       TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mail_attachment_message_id ON t_mail_attachment (mail_message_id);
CREATE INDEX IF NOT EXISTS idx_mail_attachment_upload_status ON t_mail_attachment (upload_status);

-- ============================================================
-- Sample seed data (adjust to your environment)
-- ============================================================

-- Example: add a SAP B1 SQL Server datasource
-- The password field must be AES-encrypted (use POST /api/v1/datasources/test
-- or EncryptUtil directly to generate the encrypted value).

-- INSERT INTO t_datasource_config (name, url, username, password, schema_name, instance_type, description, status)
-- VALUES (
--     'SAP B1 SQL Server',
--     'jdbc:sqlserver://tengnat.yiyuntong.net:14333;databaseName=XSM_ZSK;encrypt=true;trustServerCertificate=true',
--     'sa',
--     '<aes-encrypted-password>',
--     'XSM_ZSK',
--     'SQLSERVER',
--     'Main SAP Business One SQL Server instance',
--     1
-- );

-- Example: register a metric definition for the above datasource
-- INSERT INTO t_metrics_meta (datasource_id, metric_code, metric_name, description, query_sql, parameters, value_column, status)
-- VALUES (
--     1,
--     'delivery_qty',
--     '交货数量',
--     'Returns total delivered quantity within a date range',
--     'SELECT SUM("Quantity") AS value FROM "VW_ODLN" WHERE "DocDate" BETWEEN :startDate AND :endDate',
--     '[{"name":"startDate","type":"STRING","required":true,"description":"Start date yyyy-MM-dd"},{"name":"endDate","type":"STRING","required":true,"description":"End date yyyy-MM-dd"}]',
--     'value',
--     1
-- );
