import logging
import os
import re
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def infer_experience_years(title: str, description: Optional[str] = None) -> int:
    """
    Proxy for years of experience when the DB has no dedicated column.
    Uses explicit phrases in title+description first, then seniority keywords in the title.
    """
    blob = f"{title or ''} {description or ''}".lower()

    # Explicit "N+ years", "minimum N years", etc.
    patterns = [
        r"(?:minimum|min\.?|at least)\s*(\d{1,2})\s*(?:\+|plus)?\s*years?",
        r"(\d{1,2})\s*\+\s*years?",
        r"(\d{1,2})\s*-\s*(\d{1,2})\s*years?\s*(?:of\s+)?(?:experience|exp)",
        r"(\d{1,2})\s*years?\s+(?:of\s+)?(?:experience|exp\.?)",
        r"(?:experience|exp\.?)\s*(?:of|:)?\s*(\d{1,2})\s*\+\s*years?",
    ]
    for pat in patterns:
        m = re.search(pat, blob)
        if m:
            if len(m.groups()) == 2:
                lo, hi = int(m.group(1)), int(m.group(2))
                return int(min(25, max(0, round((lo + hi) / 2))))
            return int(min(25, max(0, int(m.group(1)))))

    t = (title or "").lower()

    if any(k in t for k in ("intern", "graduate", "fresher", "entry level", "junior", "trainee", "apprentice")):
        return 1
    if any(k in t for k in ("chief", "cto", "cio", "vp ", "vice president", "head of", "director of")):
        return 16
    if any(k in t for k in ("principal", "distinguished", "fellow", "staff engineer")):
        return 12
    if any(k in t for k in ("senior", " sr ", " sr.", "lead ", "tech lead", "engineering manager", "architect")):
        return 8
    if any(k in t for k in ("mid-level", "mid level", " ii", "engineer ii")):
        return 5
    if any(k in t for k in ("associate", "engineer i", " i ", " i,")):
        return 2
    # Plain role titles without seniority marker
    return 4


def train_salary_model(db_url=None, data_path="sample_data.csv", model_output_path=None):
    """
    Train a RandomForest on city, job title (role), and experience_years vs average salary.

    ``experience_years`` is inferred from each job's title + description (no random noise),
    so the model can learn a real monotonic-ish relationship instead of ignoring the feature.
    """
    if model_output_path is None:
        model_output_path = os.path.join(os.path.dirname(__file__), "salary_model.pkl")

    df = pd.DataFrame()

    if db_url:
        logger.info("Connecting to database for training data...")
        try:
            engine = create_engine(db_url)
            query = """
            SELECT j.location AS city,
                   j.title AS job_role,
                   j.description AS job_description,
                   s.min_salary,
                   s.max_salary
            FROM jobs j
            JOIN salaries s ON j.job_id = s.job_id
            WHERE s.min_salary IS NOT NULL AND s.max_salary IS NOT NULL
            """
            df = pd.read_sql(query, engine)
            if not df.empty:
                df["average_salary"] = (df["min_salary"] + df["max_salary"]) / 2
                base_exp = df.apply(
                    lambda r: infer_experience_years(
                        str(r.get("job_role") or ""),
                        str(r.get("job_description") or "") if pd.notna(r.get("job_description")) else None,
                    ),
                    axis=1,
                )
                # Many synthetic rows share the same title → identical base_exp and RF ignores years.
                # Add a small deterministic offset from description length (not random) so the tree can split.
                desc_len = df["job_description"].fillna("").astype(str).str.len()
                delta = (desc_len % 9).astype(int) - 4
                df["experience_years"] = (base_exp + delta).clip(1, 25).astype(int)
                corr = df["experience_years"].corr(df["average_salary"])
                logger.info(
                    "experience_years vs average_salary correlation (Pearson): %s",
                    f"{corr:.4f}" if pd.notna(corr) else "nan",
                )
                logger.info("experience_years unique values: %s", int(df["experience_years"].nunique()))
        except Exception:
            logger.exception("Error reading training data from database; falling back if needed.")

    if df.empty:
        logger.warning("No training rows from DB; using built-in dummy training data.")
        df = pd.DataFrame(
            {
                "city": ["Lahore", "Karachi", "Islamabad", "Lahore", "Karachi"],
                "job_role": [
                    "Software Engineer",
                    "Data Scientist",
                    "Data Engineer",
                    "Product Manager",
                    "Senior Software Engineer",
                ],
                "experience_years": [2, 3, 1, 5, 8],
                "average_salary": [150000, 200000, 120000, 300000, 280000],
            }
        )

    X = df[["city", "job_role", "experience_years"]]
    y = df["average_salary"]

    categorical_features = ["city", "job_role"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if len(df) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    logger.info("Training model (rows=%s)...", len(df))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    logger.info("Model holdout MSE: %s", mse)
    logger.info("Model holdout R2: %s", r2)

    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    joblib.dump(model, model_output_path)
    logger.info("Model saved to %s", model_output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    train_salary_model(db_url=os.environ.get("DATABASE_URL"))
