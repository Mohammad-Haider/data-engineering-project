import logging
import os
import time

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AdzunaClient:
    def __init__(self, app_id=None, app_key=None, country=None):
        self.app_id = app_id or os.environ.get("ADZUNA_APP_ID")
        self.app_key = app_key or os.environ.get("ADZUNA_APP_KEY")
        self.country = (country or os.environ.get("ADZUNA_COUNTRY", "gb")).lower()
        self.base_url = f"http://api.adzuna.com/v1/api/jobs/{self.country}/search"

        if not self.app_id or not self.app_key:
            logging.warning("ADZUNA_APP_ID / ADZUNA_APP_KEY not found in environment variables.")

    def fetch_jobs(self, query="javascript developer", pages=1, results_per_page=50):
        if not self.app_id or not self.app_key:
            logging.error("Cannot fetch Adzuna jobs without app credentials.")
            return []

        connect_timeout = float(os.environ.get("ADZUNA_CONNECT_TIMEOUT", "15"))
        read_timeout = float(os.environ.get("ADZUNA_READ_TIMEOUT", "60"))
        pause_s = float(os.environ.get("ADZUNA_REQUEST_PAUSE_SEC", "0.5"))

        all_jobs = []
        pages = max(1, int(pages))
        results_per_page = max(1, min(int(results_per_page), 50))

        for page in range(1, pages + 1):
            url = f"{self.base_url}/{page}"
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": str(results_per_page),
                "what": query,
                "content-type": "application/json",
            }
            try:
                response = requests.get(url, params=params, timeout=(connect_timeout, read_timeout))
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("results", [])
                if not rows:
                    break

                for job in rows:
                    location = ""
                    loc = job.get("location") or {}
                    area = loc.get("area") or []
                    if area:
                        location = ", ".join(str(x) for x in area if x)
                    if not location:
                        location = loc.get("display_name") or ""

                    all_jobs.append(
                        {
                            "title": job.get("title"),
                            "company": (job.get("company") or {}).get("display_name"),
                            "location": location,
                            "source": "Adzuna",
                            "description": job.get("description"),
                            "adzuna_job_id": job.get("id"),
                            "salary_raw": None,
                        }
                    )
            except requests.exceptions.RequestException as e:
                logging.error("Adzuna fetch failed for query '%s' page %s: %s", query, page, e)
                continue

            if pause_s > 0:
                time.sleep(pause_s)

        logging.info("Fetched %s Adzuna jobs for query '%s'.", len(all_jobs), query)
        return all_jobs

