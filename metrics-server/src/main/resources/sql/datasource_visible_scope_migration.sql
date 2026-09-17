-- Run after init.sql. Preserve every historical grant, including legacy tenants.
BEGIN;

-- Add without a default first so legacy rows can be distinguished from new rows.
ALTER TABLE t_datasource_config
    ADD COLUMN IF NOT EXISTS visible_scope_mode VARCHAR(16);

-- Disabled and soft-deleted grants both prove that visibility was restricted.
-- Only null modes are initialized: reruns must preserve explicit user choices.
UPDATE t_datasource_config ds
SET visible_scope_mode = CASE
    WHEN EXISTS (
        SELECT 1 FROM t_datasource_table_grant grant_row
        WHERE grant_row.datasource_id = ds.id
    ) THEN 'RESTRICTED'
    ELSE 'ALL'
END
WHERE ds.visible_scope_mode IS NULL;

ALTER TABLE t_datasource_config
    ALTER COLUMN visible_scope_mode SET DEFAULT 'RESTRICTED',
    ALTER COLUMN visible_scope_mode SET NOT NULL,
    DROP CONSTRAINT IF EXISTS ck_ds_visible_scope_mode,
    ADD CONSTRAINT ck_ds_visible_scope_mode
        CHECK (visible_scope_mode IN ('ALL', 'RESTRICTED'));

ALTER TABLE t_datasource_table_grant
    ALTER COLUMN tenant_id SET DEFAULT '__datasource__';

CREATE INDEX IF NOT EXISTS idx_dstg_ds_status_deleted
    ON t_datasource_table_grant (datasource_id, status, deleted);

COMMIT;
