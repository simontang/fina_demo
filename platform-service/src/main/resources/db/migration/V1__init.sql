-- file_service metadata schema
-- V1 baseline schema. Applied automatically by Flyway on startup.
-- Target: SPRING_DATASOURCE_URL database (document-postgres/file_service in dev, RDS in prod)
-- Apply once: psql "$FILE_SERVICE_PG_URL" -f ddl/file_service.sql

CREATE TABLE IF NOT EXISTS file_objects (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     VARCHAR(64)  NOT NULL,
    path          VARCHAR(512) NOT NULL DEFAULT '',
    filename      VARCHAR(255) NOT NULL,
    version       INT          NOT NULL DEFAULT 1,
    sha256        CHAR(64),
    md5           CHAR(32),
    size          BIGINT       NOT NULL DEFAULT 0,
    mime          VARCHAR(128),
    file_category VARCHAR(64),
    usage         VARCHAR(32),
    uuid          CHAR(32),
    meta          JSONB,
    storage_key   VARCHAR(900) NOT NULL,
    status        VARCHAR(16)  NOT NULL DEFAULT 'active',
    created_by    VARCHAR(64),
    created_at    TIMESTAMP     NOT NULL DEFAULT now(),
    updated_at    TIMESTAMP     NOT NULL DEFAULT now(),
    CONSTRAINT uq_file_objects_tenant_path_version
        UNIQUE (tenant_id, path, filename, version)
);

CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_status
    ON file_objects (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_path
    ON file_objects (tenant_id, path);
CREATE UNIQUE INDEX IF NOT EXISTS uq_file_objects_uuid
    ON file_objects (uuid);
