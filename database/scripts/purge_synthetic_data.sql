-- =============================================================================
-- Remove all rows generated as synthetic load (source = 'Synthetic')
-- Database: job_market_db (operational schema for this project)
--
-- Effects:
--   * Deletes matching jobs (salaries + job_skills CASCADE from jobs FK)
--   * Deletes companies that no longer have any job after the purge
--
-- Other MySQL schemas (bronze / silver / gold): not present in a default
-- install. If you add them later, run database/scripts/purge_synthetic_medallion.sql
-- or re-apply medallion refresh after this script.
-- =============================================================================

USE job_market_db;

START TRANSACTION;

-- Child tables use ON DELETE CASCADE from jobs; explicit deletes are optional.
DELETE FROM jobs
WHERE TRIM(IFNULL(source, '')) = 'Synthetic';

-- Companies only referenced by deleted synthetic jobs (or already orphaned)
DELETE c
FROM companies c
WHERE NOT EXISTS (
    SELECT 1 FROM jobs j WHERE j.company_id = c.company_id
);

COMMIT;

-- Sanity check (should all be zero / no Synthetic source)
SELECT 'remaining_jobs_with_source_synthetic' AS chk, COUNT(*) AS n
FROM jobs
WHERE TRIM(IFNULL(source, '')) = 'Synthetic';
