import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import os

# Resolve project root regardless of where streamlit is started from
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Page config
st.set_page_config(
    page_title="Pakistan Job Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_db_engine():
    db_url = os.environ.get("DATABASE_URL", "mysql+pymysql://appuser:apppassword@localhost:3307/job_market_db")
    return create_engine(db_url)

def fetch_data(query):
    engine = get_db_engine()
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

st.title("Pakistan Job Market Intelligence Platform")
st.markdown("Insights and predictions for the Pakistani job market.")

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Market Overview", "Salary Predictor", "Skills Gap"])

with tab1:
    st.subheader("Market Overview")
    
    # Fetch metrics from DB
    jobs_df = fetch_data("SELECT COUNT(*) as total FROM jobs")
    total_jobs = jobs_df.iloc[0]['total'] if not jobs_df.empty else 0
    
    salaries_df = fetch_data("SELECT AVG((min_salary + max_salary) / 2) as avg_sal FROM salaries")
    avg_salary = salaries_df.iloc[0]['avg_sal'] if not salaries_df.empty and pd.notna(salaries_df.iloc[0]['avg_sal']) else 180000
    
    cities_df = fetch_data("SELECT location, COUNT(*) as cnt FROM jobs GROUP BY location ORDER BY cnt DESC LIMIT 1")
    top_city = cities_df.iloc[0]['location'] if not cities_df.empty else "Lahore"
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Jobs Analyzed", value=f"{total_jobs:,}")
    col2.metric(label="Average Salary (Tech)", value=f"{avg_salary:,.0f} PKR")
    col3.metric(label="Top Hiring City", value=top_city)
    
    st.markdown("### Job Postings by City")
    city_dist_df = fetch_data("SELECT location, COUNT(*) as count FROM jobs GROUP BY location")
    if not city_dist_df.empty:
        city_dist_df.set_index('location', inplace=True)
        st.bar_chart(city_dist_df)
    else:
        st.info("No location data available yet.")

with tab2:
    st.subheader("Salary Predictor")
    st.markdown("Predict expected salary based on our ML model.")
    
    p_city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Rawalpindi"])
    p_role = st.selectbox("Role", ["Software Engineer", "Data Engineer", "Data Scientist", "Product Manager", "Backend Developer", "Frontend Developer"])
    p_exp = st.slider("Years of Experience", 0, 20, 2)
    
    if st.button("Predict Salary"):
        model_path = os.path.join(BASE_DIR, 'analytics_ml', 'salary_prediction', 'salary_model.pkl')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # Create a dataframe for prediction
                input_df = pd.DataFrame([[p_city, p_role, p_exp]], columns=['city', 'job_role', 'experience_years'])
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Average Salary: PKR {prediction:,.2f} / month")
            except Exception as e:
                st.error(f"Error predicting salary: {e}")
        else:
            st.warning("Model not trained yet. Run the pipeline first!")
            # Fallback
            st.info(f"Fallback Estimate: PKR {120000 + (p_exp * 20000):,.2f} / month")

with tab3:
    st.subheader("Skills Demand")
    st.markdown("Most in-demand skills in the selected region.")

    # Use live DB counts from job title + description text.
    skills_query = """
    SELECT 'Python' AS Skill, SUM(CASE WHEN LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(description, ''))) LIKE '%%python%%' THEN 1 ELSE 0 END) AS demand
    FROM jobs
    UNION ALL
    SELECT 'SQL', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(description, ''))) LIKE '%%sql%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL
    SELECT 'AWS', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(description, ''))) LIKE '%%aws%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL
    SELECT 'React', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(description, ''))) LIKE '%%react%%' THEN 1 ELSE 0 END) FROM jobs
    UNION ALL
    SELECT 'Docker', SUM(CASE WHEN LOWER(CONCAT(IFNULL(title, ''), ' ', IFNULL(description, ''))) LIKE '%%docker%%' THEN 1 ELSE 0 END) FROM jobs
    """

    skills_df = fetch_data(skills_query)
    if not skills_df.empty:
        skills_df['demand'] = skills_df['demand'].fillna(0).astype(int)
        skills_df = skills_df.sort_values('demand', ascending=False)
        skills_data = skills_df.set_index('Skill').rename(columns={'demand': 'Demand Score'})
        st.bar_chart(skills_data)
    else:
        st.info("No skill demand data available yet.")
