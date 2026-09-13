-- Search support. Two kinds of queries, two kinds of index:
--   * browse (path prefix)  → btree (tenant_id, path, filename)  [V2]
--   * search (substring)    → GIN trigram on filename / path
-- The trigram part is best-effort: creating an EXTENSION needs privileges that
-- a managed database (e.g. Aliyun RDS) may not grant. If it fails we keep the
-- always-available btree filters and search still works (sequential scan),
-- instead of blocking startup.
DO $$
BEGIN
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pg_trgm';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_file_objects_filename_trgm
             ON file_objects USING gin (filename gin_trgm_ops)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_file_objects_path_trgm
             ON file_objects USING gin (path gin_trgm_ops)';
    RAISE NOTICE 'trigram indexes ready';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'trigram indexes unavailable (%), substring search will scan', SQLERRM;
END $$;

-- Filters that always work (tenant_id leads: the tenant interceptor injects it).
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_status_id
    ON file_objects (tenant_id, status, id DESC);
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_category
    ON file_objects (tenant_id, file_category);
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_usage
    ON file_objects (tenant_id, usage);
