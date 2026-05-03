-- Remove dev/seed jobs with placeholder `source` values (optional cleanup).
-- Salaries and job_skills rows for these jobs are removed via ON DELETE CASCADE.
USE job_market_db;

DELETE FROM jobs
WHERE LOWER(TRIM(IFNULL(source, ''))) IN ('seed', 'rozee.pk (fallback)');
