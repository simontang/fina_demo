-- CDP datasource migration/seed template for metrics-server.
-- Replace <aes-encrypted-password> before running directly, or create the datasource
-- through POST /api/v1/datasources so the service encrypts the plain password.

ALTER TABLE t_datasource_config
    ADD COLUMN IF NOT EXISTS source_type VARCHAR(64) NOT NULL DEFAULT 'sap_b1_hana';

UPDATE t_datasource_config
SET source_type = CASE
    WHEN lower(url) LIKE 'jdbc:postgresql:%' THEN 'cdp_postgres'
    WHEN lower(url) LIKE 'jdbc:sap:%' THEN 'sap_b1_hana'
    ELSE source_type
END
WHERE source_type IS NULL OR source_type = '' OR source_type = 'sap_b1_hana';

CREATE INDEX IF NOT EXISTS idx_ds_source_type
    ON t_datasource_config (source_type, status, deleted);

-- Idempotent seed for the imported CRM Agent CDP demo data.
WITH updated AS (
    UPDATE t_datasource_config
    SET
        url = 'jdbc:postgresql://pgm-uf615169n98t95tflo.pg.rds.aliyuncs.com:5432/postgres',
        username = 'postgres_fuli',
        password = '<aes-encrypted-password>',
        schema_name = NULL,
        source_type = 'cdp_postgres',
        description = 'CRM Agent CDP demo dataset in Aliyun PostgreSQL (demo_* tables and views)',
        status = 1,
        deleted = 0,
        updated_at = CURRENT_TIMESTAMP
    WHERE name = 'CDP Demo PostgreSQL'
    RETURNING id
)
INSERT INTO t_datasource_config (
    name,
    url,
    username,
    password,
    schema_name,
    source_type,
    description,
    status,
    deleted
)
SELECT
    'CDP Demo PostgreSQL',
    'jdbc:postgresql://pgm-uf615169n98t95tflo.pg.rds.aliyuncs.com:5432/postgres',
    'postgres_fuli',
    '<aes-encrypted-password>',
    NULL,
    'cdp_postgres',
    'CRM Agent CDP demo dataset in Aliyun PostgreSQL (demo_* tables and views)',
    1,
    0
WHERE NOT EXISTS (SELECT 1 FROM updated)
  AND NOT EXISTS (
      SELECT 1 FROM t_datasource_config
      WHERE name = 'CDP Demo PostgreSQL' AND deleted = 0
  );
