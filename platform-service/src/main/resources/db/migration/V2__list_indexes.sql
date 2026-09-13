-- Listing support: path-prefix scans and keyset pagination.
-- (Tenant filtering is applied by the tenant interceptor, so tenant_id leads.)
CREATE INDEX IF NOT EXISTS idx_file_objects_tenant_path_name
    ON file_objects (tenant_id, path, filename);
