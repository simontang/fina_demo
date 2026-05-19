-- Rename datasource display names (run against metrics-server master PostgreSQL).
-- id=1 → SAP Business One Demo
-- id=3 → SAP Cloud ERP Demo

UPDATE t_datasource_config
SET name = 'SAP Business One Demo', updated_at = CURRENT_TIMESTAMP
WHERE id = 1;

UPDATE t_datasource_config
SET name = 'SAP Cloud ERP Demo', updated_at = CURRENT_TIMESTAMP
WHERE id = 3;
