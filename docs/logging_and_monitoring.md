# Logging and monitoring

In the Streamlit app, open **OPS → Logging & monitoring** for live **health metrics**, the **runbook** (this page), and **recent Prefect runs** when the API is reachable.

## What is logged today

| Area | Where it appears | What is captured |
|------|------------------|------------------|
| Prefect pipeline | Prefect UI (run + task logs), `docker logs job_market_prefect_pipeline` | Task `logging.*` messages, training output via `log_prints=True` on the flow |
| API ingestion | Same + `data_ingestion/api_clients/*.py` | Missing keys, HTTP retries, 429s, fetch counts |
| ML training | Prefect task logs, container stdout | DB connect, row counts, correlation, MSE/R², save path; DB errors include stack trace |
| Dashboard inference | Terminal / Docker stdout for `streamlit` | Each predict: city, role, years, skill; RF base and adjusted PKR; failures with stack (`LOG_LEVEL` default INFO) |

Set **`LOG_LEVEL=DEBUG`** for the dashboard container or your shell when you need more detail.

## If something goes wrong

1. **Prefect run failed** — Open **http://localhost:4200** → **Deployments** → **daily-job-market** → **Runs** → failed run → expand **task** logs. Check the last successful task (extract vs load vs train).
2. **Pipeline container** — `docker logs -f job_market_prefect_pipeline` (or `docker compose logs -f prefect-pipeline`). Look for SQL errors, missing `DATABASE_URL`, or API auth warnings.
3. **Empty or dummy data** — Search logs for `No jobs fetched`, `No training rows`, or `dummy training data` to see whether APIs or the DB salary join is empty.
4. **Predictor errors** — Streamlit console: `salary_inference failed` includes a full traceback. Confirm `analytics_ml/salary_prediction/salary_model.pkl` exists after a training run.
5. **Database** — `docker logs job_market_db` for MySQL errors; verify `DATABASE_URL` matches users in `database/schemas/init.sql`.

## Optional hardening (later)

- Ship logs to **OpenSearch / CloudWatch / Datadog** from Docker with a log driver or sidecar.
- Add **Sentry** (or similar) for the dashboard and pipeline processes for error grouping and alerts.
