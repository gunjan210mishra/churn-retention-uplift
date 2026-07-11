"""
LEVEL 4 — Cost-sensitive decision making.
Moves from "predict churn" to "decide who's worth a retention offer" by
optimizing the decision threshold against a cost matrix, instead of using
the default 0.5 cutoff or optimizing F1.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor

# Business assumptions (documented so they're easy to defend/adjust in an interview):
#   - Average customer monthly value ~ dataset's mean MonthlyCharges
#   - Losing a churning customer costs ~ 12 months of their charges (LTV proxy)
#   - A retention offer costs a flat amount regardless of whether it works
#   - If we offer a discount to someone who wasn't going to churn anyway, we
#     still lose the offer cost (that's the false-positive cost)
RETENTION_OFFER_COST = 50.0       # cost of the incentive/outreach
AVG_MONTHS_RETAINED_IF_SUCCESSFUL = 12  # assumed LTV horizon


def expected_value(y_true, y_proba, threshold, monthly_charges):
    """
    For each customer, decide to intervene if predicted churn prob > threshold.
    Compute net expected value of using this policy vs. doing nothing.
    """
    decide_intervene = y_proba >= threshold

    # True churner + we intervene -> assume intervention saves them with some
    # success probability. We use a conservative fixed save-rate assumption.
    SAVE_RATE = 0.35  # documented assumption: 35% of offered churners are retained

    saved_value = (
        decide_intervene & (y_true == 1)
    ) * SAVE_RATE * (monthly_charges * AVG_MONTHS_RETAINED_IF_SUCCESSFUL)

    offer_cost = decide_intervene * RETENTION_OFFER_COST

    net = saved_value.sum() - offer_cost.sum()
    return net


def main():
    df = load_raw()
    X, y = get_feature_target(df, use_engineered=True)
    X_train, X_test, y_train, y_test = split(X, y)
    cat_cols, num_cols = build_preprocessor(X)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    monthly_charges = X_test["MonthlyCharges"].values
    y_true = y_test.values

    print("=" * 60)
    print("LEVEL 4: Cost-sensitive threshold optimization")
    print("=" * 60)
    print(f"Assumptions: offer cost=${RETENTION_OFFER_COST}, save rate=35%,")
    print(f"LTV horizon={AVG_MONTHS_RETAINED_IF_SUCCESSFUL} months\n")

    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    for t in thresholds:
        ev = expected_value(y_true, y_proba, t, monthly_charges)
        n_targeted = (y_proba >= t).sum()
        results.append({"threshold": t, "expected_value": ev, "n_targeted": n_targeted})

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["expected_value"].idxmax()]

    print("Expected value by threshold:")
    for _, row in results_df.iterrows():
        marker = "  <-- best" if row["threshold"] == best_row["threshold"] else ""
        print(f"  threshold={row['threshold']:.2f}  targeted={int(row['n_targeted']):4d}  "
              f"expected_value=${row['expected_value']:8.0f}{marker}")

    # Compare to naive default threshold of 0.5
    default_ev = expected_value(y_true, y_proba, 0.5, monthly_charges)
    print(f"\nDefault threshold (0.5): expected value = ${default_ev:.0f}")
    print(f"Optimized threshold ({best_row['threshold']:.2f}): expected value = ${best_row['expected_value']:.0f}")
    print(f"Improvement: ${best_row['expected_value'] - default_ev:.0f} "
          f"({(best_row['expected_value'] / default_ev - 1) * 100:.1f}% more value captured)")

    results_df.to_csv("outputs/threshold_analysis.csv", index=False)
    print("\nSaved -> outputs/threshold_analysis.csv")
    print("\n-> This is the business-facing story: not 'our model has 84% AUC' but")
    print("   'this policy captures $X more in retained revenue than the naive approach.'")
    print("-> See Level 5 for uplift modeling: targeting customers the offer would ACTUALLY change.")


if __name__ == "__main__":
    main()
