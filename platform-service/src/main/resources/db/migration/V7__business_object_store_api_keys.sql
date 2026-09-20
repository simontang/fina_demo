-- Store-bound API keys for Business Object runtime access.
-- A store can have many keys, but each key authorizes exactly one store.

CREATE TABLE IF NOT EXISTS bo_store_api_keys (
    id               BIGSERIAL PRIMARY KEY,
    store_id         BIGINT NOT NULL REFERENCES bo_stores(id),
    key_name         VARCHAR(128) NOT NULL,
    key_hash         VARCHAR(128) NOT NULL,
    permissions_json TEXT NOT NULL,
    status           INT NOT NULL DEFAULT 1,
    deleted          INT NOT NULL DEFAULT 0,
    created_at       TIMESTAMP NOT NULL DEFAULT now(),
    updated_at       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_bo_store_api_keys_store_name_active
    ON bo_store_api_keys (store_id, key_name)
    WHERE deleted = 0;

CREATE UNIQUE INDEX IF NOT EXISTS uq_bo_store_api_keys_hash_active
    ON bo_store_api_keys (key_hash)
    WHERE deleted = 0;

CREATE INDEX IF NOT EXISTS idx_bo_store_api_keys_store
    ON bo_store_api_keys (store_id, status, deleted);
