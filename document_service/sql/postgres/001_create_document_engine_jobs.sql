CREATE TABLE IF NOT EXISTS public.document_engine_jobs (
  id VARCHAR(64) PRIMARY KEY,
  run_id VARCHAR(64) NOT NULL,
  engine VARCHAR(64) NOT NULL,
  local_status VARCHAR(32) NOT NULL DEFAULT 'selected',
  remote_status VARCHAR(64),
  remote_job_id VARCHAR(255),
  remote_poll_url VARCHAR(2048),
  submit_payload JSONB,
  submit_response JSONB,
  last_poll_response JSONB,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  error_code VARCHAR(128),
  error_message TEXT,
  last_polled_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_document_engine_jobs_run_id
  ON public.document_engine_jobs (run_id);

CREATE INDEX IF NOT EXISTS ix_document_engine_jobs_remote_job
  ON public.document_engine_jobs (engine, remote_job_id);

CREATE INDEX IF NOT EXISTS ix_document_engine_jobs_local_status
  ON public.document_engine_jobs (local_status);

CREATE OR REPLACE FUNCTION public.document_engine_jobs_set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_document_engine_jobs_updated_at
  ON public.document_engine_jobs;

CREATE TRIGGER trg_document_engine_jobs_updated_at
BEFORE UPDATE ON public.document_engine_jobs
FOR EACH ROW
EXECUTE FUNCTION public.document_engine_jobs_set_updated_at();

DO $$
BEGIN
  IF to_regclass('public.document_runs') IS NOT NULL
     AND NOT EXISTS (
       SELECT 1
       FROM pg_constraint
       WHERE conname = 'fk_document_engine_jobs_run'
     )
  THEN
    ALTER TABLE public.document_engine_jobs
      ADD CONSTRAINT fk_document_engine_jobs_run
      FOREIGN KEY (run_id)
      REFERENCES public.document_runs(id)
      ON DELETE CASCADE;
  END IF;
END $$;
