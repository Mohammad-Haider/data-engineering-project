#!/usr/bin/env bash
# Purge synthetic rows in job_market_db, then in bronze/silver/gold only if those DBs exist.
# Usage:
#   ./database/scripts/purge_synthetic_all.sh
#   MYSQL="mysql -h127.0.0.1 -P3307 -uroot -proot" ./database/scripts/purge_synthetic_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MYSQL_CMD="${MYSQL:-mysql -h127.0.0.1 -P3307 -uroot -proot}"

echo "==> job_market_db (purge_synthetic_data.sql)"
$MYSQL_CMD job_market_db <"$ROOT/database/scripts/purge_synthetic_data.sql"

schema_exists() {
  local s="$1"
  [[ "$($MYSQL_CMD -N -e "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name='${s}'")" == "1" ]]
}

if schema_exists bronze; then
  echo "==> bronze.raw_job_ingestion"
  $MYSQL_CMD bronze -e "
    DELETE FROM raw_job_ingestion
    WHERE TRIM(IFNULL(source_system, '')) = 'Synthetic'
       OR JSON_UNQUOTE(JSON_EXTRACT(payload_json, '$.source')) = 'Synthetic';
  "
fi

if schema_exists silver; then
  echo "==> silver.job_posting_curated"
  $MYSQL_CMD silver -e "
    DELETE FROM job_posting_curated WHERE TRIM(IFNULL(source, '')) = 'Synthetic';
  "
fi

if schema_exists gold; then
  echo "==> gold (source-keyed tables)"
  $MYSQL_CMD gold -e "
    DELETE FROM jobs_by_location_source WHERE TRIM(IFNULL(source, '')) = 'Synthetic';
    DELETE FROM salary_summary_by_source WHERE TRIM(IFNULL(source, '')) = 'Synthetic';
  "
fi

echo "Done."
