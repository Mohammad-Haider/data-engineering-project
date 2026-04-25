import logging
import os
import time

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# JSearch allows up to 20 pages per request; fewer pages = smaller payloads, fewer timeouts.
_MAX_PAGES_PER_REQUEST = 20


class JSearchClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("JSEARCH_API_KEY")
        if not self.api_key:
            logging.warning("JSEARCH_API_KEY not found in environment variables.")
        self.base_url = "https://jsearch.p.rapidapi.com/search"
        self.headers = {
            "Content-Type": "application/json",
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "jsearch.p.rapidapi.com",
        }

    def _request_with_retries(self, params):
        connect_timeout = float(os.environ.get("JSEARCH_CONNECT_TIMEOUT", "20"))
        read_timeout = float(os.environ.get("JSEARCH_READ_TIMEOUT", "120"))
        max_retries = int(os.environ.get("JSEARCH_MAX_RETRIES", "4"))
        backoff_base = float(os.environ.get("JSEARCH_RETRY_BACKOFF", "2.0"))

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(
                    "JSearch GET attempt %s/%s (page=%s num_pages=%s)",
                    attempt,
                    max_retries,
                    params.get("page"),
                    params.get("num_pages"),
                )
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=(connect_timeout, read_timeout),
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                resp = getattr(e.response, "status_code", None)
                if resp == 429:
                    retry_after = None
                    try:
                        retry_after = float(e.response.headers.get("Retry-After", "0") or 0)
                    except (TypeError, ValueError):
                        retry_after = None
                    wait = retry_after if retry_after and retry_after > 0 else min(120.0, backoff_base * (3 ** attempt))
                    logging.warning("JSearch rate limited (429); sleeping %.1fs before retry %s/%s", wait, attempt, max_retries)
                    time.sleep(wait)
                else:
                    logging.warning("JSearch HTTP error (%s/%s): %s", attempt, max_retries, e)
                    if attempt < max_retries:
                        time.sleep(backoff_base * (2 ** (attempt - 1)))
            except requests.exceptions.RequestException as e:
                last_error = e
                logging.warning("JSearch request failed (%s/%s): %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(backoff_base * (2 ** (attempt - 1)))
        logging.error("JSearch request exhausted retries: %s", last_error)
        return None

    def fetch_jobs(
        self,
        query="Software Engineer",
        location="Pakistan",
        num_pages=1,
        country="pk",
        date_posted="all",
    ):
        """
        Fetch jobs from JSearch. ``num_pages`` is the total number of *API pages* to try
        for this query (paginated using ``page`` + ``num_pages`` per request, up to 20
        pages per HTTP call to minimize round-trips).
        """
        if not self.api_key:
            logging.error("Cannot fetch jobs without an API key.")
            return []

        pages_per_request = int(
            os.environ.get("JSEARCH_PAGES_PER_REQUEST", str(_MAX_PAGES_PER_REQUEST))
        )
        pages_per_request = max(1, min(pages_per_request, _MAX_PAGES_PER_REQUEST))

        pause_s = float(os.environ.get("JSEARCH_REQUEST_PAUSE_SEC", "1.0"))

        all_jobs = []
        start_page = 1
        remaining = int(num_pages)

        while remaining > 0:
            chunk = min(pages_per_request, remaining)
            querystring = {
                "query": f"{query} in {location}",
                "page": str(start_page),
                "num_pages": str(chunk),
                "country": country,
                "date_posted": date_posted,
            }

            data = self._request_with_retries(querystring)
            if not data:
                break

            jobs = data.get("data") or []
            if not jobs:
                logging.info(
                    "JSearch returned no rows for query=%r start_page=%s chunk=%s; stopping.",
                    query,
                    start_page,
                    chunk,
                )
                break

            for job in jobs:
                job_id = job.get("job_id")
                city = job.get("job_city")
                country_code = job.get("job_country")
                loc = f"{city}, {country_code}" if city or country_code else ""

                all_jobs.append(
                    {
                        "title": job.get("job_title"),
                        "company": job.get("employer_name"),
                        "location": loc,
                        "source": "JSearch/LinkedIn",
                        "description": job.get("job_description"),
                        "jsearch_job_id": job_id,
                    }
                )

            remaining -= chunk
            start_page += chunk
            if pause_s > 0:
                time.sleep(pause_s)

        logging.info("Successfully fetched %s jobs from JSearch for query=%r.", len(all_jobs), query)
        return all_jobs


if __name__ == "__main__":
    pass
