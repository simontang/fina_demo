-- GIN index for metadata containment queries used by the list endpoint
-- (e.g. meta @> '{"baId":"u_1001","customerId":"cus_8899"}').
CREATE INDEX IF NOT EXISTS idx_file_objects_meta_path_ops
    ON file_objects USING gin (meta jsonb_path_ops);
