-- Prefix matching on btree needs *_pattern_ops in a non-C collation
-- (our clusters run en_US.utf8). Without these, `path LIKE 'folder/%'` — the
-- directory browse — cannot use the plain btree and falls back to a scan.
-- This is the ordinary "path in a btree" behaviour, no extension required.
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_path_pattern
    ON file_objects (tenant_id, path text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_name_pattern
    ON file_objects (tenant_id, filename text_pattern_ops);
