from prefect import flow, task
import logging
import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

# Add parent directory to path so we can import modules when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_ingestion.api_clients.adzuna_client import AdzunaClient
from data_ingestion.api_clients.jsearch_client import JSearchClient
from data_transformation.cleaners import clean_jobs_dataframe
from analytics_ml.salary_prediction.train_model import train_salary_model

# Default to localhost if not in docker
DB_URL = os.environ.get("DATABASE_URL", "mysql+pymysql://appuser:apppassword@localhost:3307/job_market_db")

@task(retries=3, retry_delay_seconds=60)
def extract_adzuna_jobs():
    """Extracts job postings from Adzuna API."""
    logging.info("Extracting data from Adzuna API...")
    client = AdzunaClient()
    pages = int(os.environ.get("ADZUNA_PAGES", "20"))
    results_per_page = int(os.environ.get("ADZUNA_RESULTS_PER_PAGE", "50"))
    role_queries = [
        "software engineer",
        "backend developer",
        "frontend developer",
        "full stack developer",
        "data engineer",
        "data scientist",
        "devops engineer",
        "qa engineer",
        "machine learning engineer",
        "product manager",
    ]

    jobs = []
    for role in role_queries:
        role_jobs = client.fetch_jobs(query=role, pages=pages, results_per_page=results_per_page)
        jobs.extend(role_jobs)

    if not jobs:
        logging.warning("No jobs fetched from Adzuna API. Returning fallback row.")
        return [{
            "title": "Data Engineer",
            "company": "Tech Corp",
            "location": "Lahore",
            "salary_raw": "100k-150k",
            "source": "Adzuna (fallback)"
        }]
    return jobs

@task(retries=2, retry_delay_seconds=30)
def extract_jsearch_jobs():
    """Extracts job postings via JSearch API"""
    logging.info("Extracting data from JSearch API...")
    client = JSearchClient()
    # Total API pages to fetch per role (chunked into up to 20 pages per HTTP request in client).
    max_pages = int(os.environ.get("JSEARCH_MAX_PAGES", "60"))
    location = os.environ.get("JSEARCH_LOCATION", "Pakistan")
    country = os.environ.get("JSEARCH_COUNTRY", "pk")

    # Multiple queries maximize coverage within API / plan limits.
    role_queries = [
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Data Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "QA Engineer",
        "Machine Learning Engineer",
        "Product Manager",
        "Mobile Developer",
        "Cloud Engineer",
        "Cyber Security",
        "Business Analyst",
        "Project Manager",
    ]

    jobs = []
    for role in role_queries:
        try:
            role_jobs = client.fetch_jobs(
                query=role,
                location=location,
                num_pages=max_pages,
                country=country,
                date_posted="all"
            )
            logging.info("Fetched %s jobs for query '%s'.", len(role_jobs), role)
            jobs.extend(role_jobs)
        except Exception as e:
            logging.error("JSearch fetch failed for query '%s': %s", role, e)

    if not jobs:
        logging.warning("No jobs fetched from JSearch API. Returning dummy data.")
        return [{"title": "Software Engineer", "company": "Global Inc", "location": "Karachi", "salary_raw": None}]
    return jobs

@task
def transform_jobs(adzuna_data, jsearch_data):
    """Cleans and deduplicates job data"""
    logging.info("Transforming combined data...")
    combined_data = adzuna_data + jsearch_data
    df = pd.DataFrame(combined_data)
    clean_df = clean_jobs_dataframe(df)
    return clean_df

@task
def load_to_mysql(cleaned_df):
    """Loads cleaned data into MySQL database"""
    logging.info(f"Loading {len(cleaned_df)} records to MySQL...")
    engine = create_engine(DB_URL)

    def safe_text(value, default=""):
        return default if pd.isna(value) else value
    
    with engine.begin() as conn:
        for _, row in cleaned_df.iterrows():
            # Insert or get company
            company_name = row.get("company")
            if pd.isna(company_name) or not str(company_name).strip():
                continue
            conn.execute(
                text("INSERT IGNORE INTO companies (name) VALUES (:name)"),
                {"name": company_name}
            )
            company_res = conn.execute(
                text("SELECT company_id FROM companies WHERE name = :name"),
                {"name": company_name}
            ).fetchone()
            
            if not company_res:
                continue
                
            company_id = company_res[0]
            
            # Insert job
            try:
                conn.execute(
                    text("""
                    INSERT INTO jobs (title, company_id, location, source, description, fingerprint)
                    VALUES (:title, :company_id, :location, :source, :description, :fingerprint)
                    ON DUPLICATE KEY UPDATE title=title
                    """),
                    {
                        "title": safe_text(row.get('title', '')),
                        "company_id": company_id,
                        "location": safe_text(row.get('location_clean', row.get('location', ''))),
                        "source": safe_text(row.get('source', '')),
                        "description": safe_text(row.get('description', '')),
                        "fingerprint": safe_text(row.get('fingerprint', ''))
                    }
                )
            except Exception as e:
                logging.error(f"Error inserting job: {e}")
                continue
                
            job_res = conn.execute(
                text("SELECT job_id FROM jobs WHERE fingerprint = :fingerprint"),
                {"fingerprint": row.get('fingerprint', '')}
            ).fetchone()
            
            if not job_res:
                continue
                
            job_id = job_res[0]
            
            # Insert salary if available
            min_salary = row.get('min_salary')
            max_salary = row.get('max_salary')
            
            if pd.notna(min_salary) and pd.notna(max_salary):
                conn.execute(
                    text("""
                    INSERT INTO salaries (job_id, min_salary, max_salary)
                    VALUES (:job_id, :min_salary, :max_salary)
                    """),
                    {
                        "job_id": job_id,
                        "min_salary": min_salary,
                        "max_salary": max_salary
                    }
                )

@task
def trigger_model_training():
    """Trains the ML model using data from the database"""
    logging.info("Triggering ML model training...")
    try:
        train_salary_model(db_url=DB_URL)
    except Exception as e:
        logging.error(f"Model training failed: {e}")

@flow(name="Daily Job Market Ingestion Pipeline")
def daily_job_pipeline():
    """Main execution flow"""
    # 1. Extraction
    adzuna_data = extract_adzuna_jobs()
    jsearch_data = extract_jsearch_jobs()
    
    # 2. Transformation
    cleaned_data = transform_jobs(adzuna_data, jsearch_data)
    
    # 3. Loading
    load_to_mysql(cleaned_data)
    
    # 4. Trigger Analytics Update
    trigger_model_training()

if __name__ == "__main__":
    daily_job_pipeline()
