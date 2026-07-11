"""
LEVEL 2 — Proper evaluation + feature engineering + gradient boosting.
Swaps accuracy for precision/recall/F1/ROC-AUC/PR-AUC, adds engineered
features, adds cross-validation, and benchmarks XGBoost against Level 1's
baselines.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, make_scorer
)
from xgboost import XGBClassifier

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def evaluate_cv(pipe, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    print(f"\n{name} (5-fold CV):")
    for metric in SCORING:
        scores = results[f"test_{metric}"]
        print(f"  {metric:10s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    return {m: results[f"test_{m}"].mean() for m in SCORING}


def main():
    df = load_raw()
    X, y = get_feature_target(df, use_engineered=True)  # NOW with engineered features
    cat_cols, num_cols = build_preprocessor(X)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    print("=" * 60)
    print("LEVEL 2: Proper metrics + engineered features + XGBoost")
    print("=" * 60)
    print(f"New engineered features added: tenure_bucket, avg_monthly_spend,")
    print(f"num_services, contract_commitment, high_risk_combo\n")

    all_results = {}

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("XGBoost", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        )),
    ]

    for name, model in models:
        pipe = Pipeline([("pre", pre), ("clf", model)])
        all_results[name] = evaluate_cv(pipe, X, y, name)

    print("\n" + "=" * 60)
    print("SUMMARY (ROC-AUC, higher is better, 0.5 = random)")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name:22s}: ROC-AUC={res['roc_auc']:.3f}  PR-AUC={res['pr_auc']:.3f}  F1={res['f1']:.3f}")

    print("\nNote: PR-AUC (precision-recall AUC) matters more than ROC-AUC here")
    print("since we care specifically about catching the minority 'churn' class.")
    print("-> See Level 3 for SHAP explainability on the best model.")


if __name__ == "__main__":
    main()
