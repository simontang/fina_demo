-- Business Object Runtime metadata. These tables live in the platform
-- metadata database; each BO store points at a PostgreSQL database where
-- object tables are created dynamically.

CREATE TABLE IF NOT EXISTS bo_stores (
    id          BIGSERIAL PRIMARY KEY,
    store_key   VARCHAR(128) NOT NULL,
    name        VARCHAR(255) NOT NULL,
    description TEXT,
    jdbc_url    TEXT NOT NULL,
    schema_name VARCHAR(128) NOT NULL DEFAULT 'public',
    username    VARCHAR(255) NOT NULL,
    password    TEXT NOT NULL,
    status      INT NOT NULL DEFAULT 1,
    deleted     INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bo_stores_key_active
    ON bo_stores (store_key)
    WHERE deleted = 0;

CREATE INDEX IF NOT EXISTS idx_bo_stores_status
    ON bo_stores (status, deleted);

CREATE TABLE IF NOT EXISTS bo_store_grants (
    id          BIGSERIAL PRIMARY KEY,
    store_id    BIGINT NOT NULL REFERENCES bo_stores(id),
    grantee_key VARCHAR(128) NOT NULL DEFAULT 'tenant',
    can_read    BOOLEAN NOT NULL DEFAULT true,
    can_write   BOOLEAN NOT NULL DEFAULT false,
    can_manage  BOOLEAN NOT NULL DEFAULT false,
    status      INT NOT NULL DEFAULT 1,
    deleted     INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bo_store_grants_scope_active
    ON bo_store_grants (store_id, grantee_key)
    WHERE deleted = 0;

CREATE INDEX IF NOT EXISTS idx_bo_store_grants_store
    ON bo_store_grants (store_id, status, deleted);

CREATE INDEX IF NOT EXISTS idx_bo_store_grants_grantee
    ON bo_store_grants (grantee_key, status, deleted);

CREATE TABLE IF NOT EXISTS bo_object_definitions (
    id            BIGSERIAL PRIMARY KEY,
    store_id      BIGINT NOT NULL REFERENCES bo_stores(id),
    object_key    VARCHAR(128) NOT NULL,
    table_name    VARCHAR(128) NOT NULL,
    display_name  VARCHAR(255) NOT NULL,
    description   TEXT,
    schema_json   TEXT NOT NULL,
    status        INT NOT NULL DEFAULT 1,
    deleted       INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    updated_at    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bo_object_definitions_tenant_object_active
    ON bo_object_definitions (store_id, object_key)
    WHERE deleted = 0;

CREATE INDEX IF NOT EXISTS idx_bo_object_definitions_store
    ON bo_object_definitions (store_id, status, deleted);

CREATE TABLE IF NOT EXISTS bo_object_ddl_history (
    id          BIGSERIAL PRIMARY KEY,
    store_id    BIGINT NOT NULL REFERENCES bo_stores(id),
    object_key  VARCHAR(128) NOT NULL,
    ddl_sql     TEXT NOT NULL,
    status      VARCHAR(32) NOT NULL,
    message     TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bo_object_ddl_history_object
    ON bo_object_ddl_history (store_id, object_key, created_at DESC);
