"""
Shared data loading, cleaning, and feature engineering for the churn project.
Used by every level so results are comparable apples-to-apples.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_PATH = "data/telco_churn.csv"
RANDOM_STATE = 42


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.copy()

    # TotalCharges is loaded as string; blanks correspond to tenure=0 (brand new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    df = df.drop(columns=["customerID"])
    return df


def engineer_features(df):
    """Level 2+: engineered features beyond the raw columns."""
    df = df.copy()

    # Tenure buckets — captures nonlinear "new customer risk" that raw tenure
    # (used linearly) can miss in a plain logistic regression.
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )

    # "Monetary" signal — total value extracted from the customer relative to tenure.
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Service adoption count — how "embedded" the customer is in the ecosystem.
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # Contract commitment as an ordinal signal (month-to-month = lowest lock-in).
    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_commitment"] = df["Contract"].map(contract_order)

    # High-risk flag: month-to-month + electronic check is a well-known churn combo
    # in this dataset — worth surfacing as an explicit interaction feature.
    df["high_risk_combo"] = (
        (df["Contract"] == "Month-to-month")
        & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    return df


def get_feature_target(df, use_engineered=True):
    df = clean(df)
    if use_engineered:
        df = engineer_features(df)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def split(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(X):
    """Returns column lists so each level's script can build its own
    ColumnTransformer (kept explicit rather than hidden in a pipeline object,
    since different levels need slightly different preprocessing)."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return cat_cols, num_cols
