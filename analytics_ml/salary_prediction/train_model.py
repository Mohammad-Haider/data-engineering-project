import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from sqlalchemy import create_engine

def train_salary_model(db_url=None, data_path="sample_data.csv", model_output_path="analytics_ml/salary_prediction/salary_model.pkl"):
    """
    Trains a RandomForest model to predict average salary based on city and job title/industry.
    """
    df = pd.DataFrame()
    
    if db_url:
        print("Connecting to database for training data...")
        try:
            engine = create_engine(db_url)
            query = """
            SELECT j.location as city, j.title as job_role, 
                   s.min_salary, s.max_salary 
            FROM jobs j
            JOIN salaries s ON j.job_id = s.job_id
            WHERE s.min_salary IS NOT NULL AND s.max_salary IS NOT NULL
            """
            df = pd.read_sql(query, engine)
            if not df.empty:
                df['average_salary'] = (df['min_salary'] + df['max_salary']) / 2
                # Simulated experience since we didn't scrape it
                import numpy as np
                df['experience_years'] = np.random.randint(1, 10, size=len(df))
        except Exception as e:
            print(f"Error reading from database: {e}")
            
    if df.empty:
        print("Using dummy data for model training.")
        # Dummy data fallback
        df = pd.DataFrame({
            'city': ['Lahore', 'Karachi', 'Islamabad', 'Lahore', 'Karachi'],
            'job_role': ['Software Engineer', 'Data Scientist', 'Data Engineer', 'Product Manager', 'Software Engineer'],
            'experience_years': [2, 3, 1, 5, 4],
            'average_salary': [150000, 200000, 120000, 300000, 250000]
        })

    # Features and Target
    X = df[['city', 'job_role', 'experience_years']]
    y = df['average_salary']

    # Preprocessing pipeline
    categorical_features = ['city', 'job_role']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Full modeling pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train-test split
    # If we have very little data (like the dummy data), we might get an error splitting
    if len(df) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Train model
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model MSE: {mse}")
    print(f"Model R2 Score: {r2}")

    # Save model
    os.makedirs(os.path.dirname(model_output_path) or '.', exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_salary_model()
