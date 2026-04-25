import pandas as pd
import hashlib
import re

def generate_fingerprint(title, company, location, external_id=None):
    """Generates a unique hash for a job posting to handle deduplication."""
    if external_id is not None and pd.notna(external_id) and str(external_id).strip():
        raw = f"jsearch:{str(external_id).strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if not all([title, company, location]):
        return None
    raw_string = f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}"
    return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()

def normalize_city(city_name):
    """Normalizes common Pakistani city names."""
    if not city_name:
        return "Unknown"
        
    city_name = str(city_name).lower()
    
    mapping = {
        'lhr': 'Lahore',
        'isb': 'Islamabad',
        'khi': 'Karachi',
        'pindi': 'Rawalpindi',
        'rwp': 'Rawalpindi',
        'fsd': 'Faisalabad'
    }
    
    for key, value in mapping.items():
        if key in city_name:
            return value
            
    return city_name.title()

def extract_salary(salary_str):
    """Extracts min and max salary from a string like 'PKR 50K - 80K'."""
    if pd.isna(salary_str) or not isinstance(salary_str, str):
        return None, None
        
    # Simplified regex for extracting numbers (assuming thousands 'K')
    numbers = re.findall(r'\d+', salary_str)
    
    if not numbers:
        return None, None
        
    if len(numbers) == 1:
        val = float(numbers[0]) * 1000 if 'k' in salary_str.lower() else float(numbers[0])
        return val, val
        
    if len(numbers) >= 2:
        min_val = float(numbers[0]) * 1000 if 'k' in salary_str.lower() else float(numbers[0])
        max_val = float(numbers[1]) * 1000 if 'k' in salary_str.lower() else float(numbers[1])
        return min_val, max_val

def clean_jobs_dataframe(df):
    """Main transformation function."""
    if df.empty:
        return df
        
    def _fingerprint_row(row):
        ext = row.get("jsearch_job_id")
        return generate_fingerprint(row.get("title"), row.get("company"), row.get("location"), ext)

    df["fingerprint"] = df.apply(_fingerprint_row, axis=1)
    
    # Drop duplicates based on fingerprint
    df = df.drop_duplicates(subset=['fingerprint'])
    
    # Normalize locations
    df['location_clean'] = df['location'].apply(normalize_city)
    
    # Extract salaries if the column exists
    if 'salary_raw' in df.columns:
        df[['min_salary', 'max_salary']] = df.apply(lambda row: pd.Series(extract_salary(row['salary_raw'])), axis=1)
        
    return df
