-- Set datasource 3 URL host to the same as datasource 1 (port and query string of ds3 unchanged).
-- Run against the metrics-server PostgreSQL (master DB). Then call POST /api/v1/datasources/3/reload.

UPDATE t_datasource_config c3
SET url = regexp_replace(
  c3.url,
  '^jdbc:sap://[^:/]+',
  'jdbc:sap://' || (SELECT substring(d1.url FROM 'jdbc:sap://([^:/]+)') FROM t_datasource_config d1 WHERE d1.id = 1)
),
updated_at = CURRENT_TIMESTAMP
WHERE c3.id = 3
  AND EXISTS (SELECT 1 FROM t_datasource_config WHERE id = 1);
