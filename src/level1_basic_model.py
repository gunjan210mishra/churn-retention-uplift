"""
LEVEL 1 — Basic churn model.
This mirrors the likely original resume project: LogReg + Random Forest,
accuracy as the headline metric. Kept deliberately simple as a baseline —
Level 2 will show why accuracy alone is misleading here.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor


def main():
    df = load_raw()
    X, y = get_feature_target(df, use_engineered=False)  # raw features only, no engineering yet
    X_train, X_test, y_train, y_test = split(X, y)

    cat_cols, num_cols = build_preprocessor(X)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    print("=" * 60)
    print("LEVEL 1: Basic models (accuracy only)")
    print("=" * 60)
    print(f"Churn rate in test set: {y_test.mean():.1%}  <- naive baseline: always predict 'No churn'\n")

    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]:
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: accuracy = {acc:.3f}")

    # The naive "always predict No" baseline for comparison
    naive_acc = (y_test == 0).mean()
    print(f"\nNaive baseline (always predict 'No churn'): accuracy = {naive_acc:.3f}")
    print("^ Notice how close this is to the model accuracy above.")
    print("  This is exactly why accuracy alone is a weak metric for imbalanced churn data.")
    print("  -> See Level 2 for precision/recall/ROC-AUC and why they tell a different story.")


if __name__ == "__main__":
    main()
