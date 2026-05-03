#!/usr/bin/env python3
"""
Run the daily job-market flow as a Prefect 3 deployment (local process / hybrid).

Prefect 3 uses ``flow.serve()`` instead of the removed Prefect 2 ``Deployment`` API. This
script starts a long-lived runner that registers the deployment with the API
(if ``PREFECT_API_URL`` is set) and executes scheduled or ad-hoc runs in
subprocesses. Run history and state appear in the Prefect UI (self-hosted server
or Cloud) and in this process logs.

Prerequisites
-------------
* Running Prefect API (e.g. ``docker compose up -d prefect-server``).
* ``export PREFECT_API_URL=http://127.0.0.1:4200/api`` (from the host) or
  ``http://prefect-server:4200/api`` inside Compose.

From project root::

  PYTHONPATH=. python orchestration/deployments/register_deployment.py

Scheduling (env)
----------------
* ``PREFECT_DEPLOYMENT_INTERVAL_SECONDS`` — if set, use interval-based schedule
  (seconds). Otherwise cron is used.
* ``PREFECT_DEPLOYMENT_CRON`` — default ``0 2 * * *`` (02:00 daily).
* ``PREFECT_DEPLOYMENT_TIMEZONE`` — IANA zone for cron (uses ``prefect.schedules.Cron`` when not UTC).
* ``PREFECT_DEPLOYMENT_NAME`` — deployment name (default ``daily-job-market``).
* ``PREFECT_DEPLOYMENT_TAGS`` — comma-separated tags.
* ``PREFECT_PAUSE_ON_SHUTDOWN`` — default ``false`` (keep schedule active across restarts in Docker).

Observability
-------------
* UI: ``http://<host>:4200`` when using the bundled ``prefect-server`` service.
* Logs: container/process stdout (flow and task logs); failed runs in UI under Deployments → Runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    try:
        from prefect.schedules import Cron
    except ImportError as e:
        print("Prefect is required: pip install 'prefect>=3.6,<3.7'", e, file=sys.stderr)
        return 1

    from orchestration.flows.main_pipeline import daily_job_pipeline

    deployment_name = os.environ.get("PREFECT_DEPLOYMENT_NAME", "daily-job-market")
    cron = os.environ.get("PREFECT_DEPLOYMENT_CRON", "0 2 * * *")
    tz = os.environ.get("PREFECT_DEPLOYMENT_TIMEZONE", "UTC")
    interval_raw = os.environ.get("PREFECT_DEPLOYMENT_INTERVAL_SECONDS", "").strip()
    pause_on_shutdown = _truthy(os.environ.get("PREFECT_PAUSE_ON_SHUTDOWN"), False)

    tags_raw = os.environ.get("PREFECT_DEPLOYMENT_TAGS", "job-market,ingestion,mysql")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    serve_kwargs: dict = {
        "name": deployment_name,
        "tags": tags,
        "pause_on_shutdown": pause_on_shutdown,
        "print_starting_message": True,
    }

    if interval_raw:
        serve_kwargs["interval"] = int(interval_raw)
    else:
        if tz.strip().upper() != "UTC":
            serve_kwargs["schedule"] = Cron(cron, timezone=tz.strip())
        else:
            serve_kwargs["cron"] = cron

    api_url = os.environ.get("PREFECT_API_URL", "")
    print(
        "Starting Prefect flow.serve (blocks until stopped).",
        f"deployment={deployment_name!r}",
        f"PREFECT_API_URL={api_url or '(ephemeral — set for UI/history)'}",
        file=sys.stderr,
    )

    daily_job_pipeline.serve(**serve_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
