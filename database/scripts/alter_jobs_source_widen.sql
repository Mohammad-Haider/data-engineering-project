-- Widen jobs.source for Kaggle / multi-source labels (existing deployments).
USE job_market_db;
ALTER TABLE jobs MODIFY COLUMN source VARCHAR(191) NULL;
