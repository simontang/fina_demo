CREATE TABLE IF NOT EXISTS document_engine_jobs (
  id VARCHAR(64) NOT NULL,
  run_id VARCHAR(64) NOT NULL,
  engine VARCHAR(64) NOT NULL,
  local_status VARCHAR(32) NOT NULL DEFAULT 'selected',
  remote_status VARCHAR(64) NULL,
  remote_job_id VARCHAR(255) NULL,
  remote_poll_url VARCHAR(2048) NULL,
  submit_payload JSON NULL,
  submit_response JSON NULL,
  last_poll_response JSON NULL,
  attempt_count INT NOT NULL DEFAULT 0,
  error_code VARCHAR(128) NULL,
  error_message TEXT NULL,
  last_polled_at DATETIME(6) NULL,
  completed_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id),
  KEY ix_document_engine_jobs_run_id (run_id),
  KEY ix_document_engine_jobs_remote_job (engine, remote_job_id(191)),
  KEY ix_document_engine_jobs_local_status (local_status),
  CONSTRAINT fk_document_engine_jobs_run
    FOREIGN KEY (run_id) REFERENCES document_runs(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
