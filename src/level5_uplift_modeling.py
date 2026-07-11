"""
LEVEL 5 — Uplift modeling.

IMPORTANT / HONEST NOTE (keep this in mind when discussing this project):
The Telco dataset has no real "did this customer receive a retention offer"
field -- no public churn dataset does, because that requires an actual
randomized experiment a company ran internally. To demonstrate uplift
modeling methodology, we SIMULATE a retention-offer experiment on top of the
real customer features:
  - Treatment (received offer) is randomly assigned 50/50, like an RCT.
  - The treatment effect (how much the offer reduces churn probability) is a
    known function we define, varying by contract type and tenure, plus noise.
  - This lets us both fit an uplift model AND check whether it recovers the
    true effect we planted -- a validation step real-world uplift projects
    usually can't do (ground truth is normally unobservable).
This is a common, accepted technique for demonstrating uplift methodology
without access to real experimental data. Be upfront about this if asked in
an interview -- it's a legitimate way to show you understand the method
end-to-end, not a way to fake real business results.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor

RNG = np.random.default_rng(42)


def simulate_experiment(X, baseline_prob):
    """Simulate a random retention-offer experiment with known heterogeneous effect."""
    n = len(X)
    treatment = RNG.binomial(1, 0.5, size=n)

    contract = X["Contract"].values
    internet = X["InternetService"].values
    tenure = X["tenure"].values

    # True (planted) uplift function.
    # Deliberately designed so treatment effect is NOT just a copy of
    # baseline risk -- both Fiber and DSL month-to-month customers have high
    # baseline churn risk, but only DSL customers are genuinely price-
    # sensitive (a discount works). Fiber customers are assumed to be
    # churning over service-quality issues a discount won't fix -- these
    # are the "lost causes" a naive risk-based policy can't distinguish
    # from true persuadables.
    true_uplift = np.zeros(n)
    mtm = contract == "Month-to-month"
    true_uplift += np.where(mtm & (internet == "DSL"), -0.28, 0.0)          # persuadable
    true_uplift += np.where(mtm & (internet == "Fiber optic"), -0.03, 0.0)  # lost cause (high risk, won't respond)
    true_uplift += np.where(mtm & (internet == "No"), -0.15, 0.0)
    true_uplift += np.where(contract == "One year", -0.06, 0.0)
    true_uplift += np.where(contract == "Two year", -0.01, 0.0)             # sure thing (low risk already)
    true_uplift += RNG.normal(0, 0.015, size=n)  # noise
    true_uplift = np.clip(true_uplift, -0.35, 0.0)  # offer never increases churn here

    treated_prob = np.clip(baseline_prob + true_uplift, 0.01, 0.99)
    observed_prob = np.where(treatment == 1, treated_prob, baseline_prob)
    observed_churn = RNG.binomial(1, observed_prob)

    return treatment, observed_churn, true_uplift


def main():
    df = load_raw()
    X, y = get_feature_target(df, use_engineered=True)
    X_train, X_test, y_train, y_test = split(X, y)
    cat_cols, num_cols = build_preprocessor(X)

    # Use the Level 2/3 model's predicted probabilities as our "baseline
    # churn probability" (i.e. probability of churn with NO offer) --
    # this is what we perturb with the simulated treatment effect.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    base_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    base_pipe = Pipeline([("pre", pre), ("clf", base_model)])
    base_pipe.fit(X_train, y_train)
    baseline_prob_test = base_pipe.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("LEVEL 5: Uplift modeling (T-learner) on simulated experiment")
    print("=" * 60)

    treatment, observed_churn, true_uplift = simulate_experiment(X_test, baseline_prob_test)
    X_test = X_test.reset_index(drop=True)
    exp_df = X_test.copy()
    exp_df["treatment"] = treatment
    exp_df["observed_churn"] = observed_churn
    exp_df["true_uplift"] = true_uplift

    print(f"\nSimulated experiment: {len(exp_df)} customers, "
          f"{treatment.sum()} treated, {(1-treatment).sum()} control")

    # --- T-learner: fit two separate models on treated vs control groups ---
    pre_t = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    pre_c = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])

    treated_mask = exp_df["treatment"] == 1
    control_mask = exp_df["treatment"] == 0

    model_treated = Pipeline([("pre", pre_t), ("clf", LogisticRegression(max_iter=1000, C=0.5))])
    model_control = Pipeline([("pre", pre_c), ("clf", LogisticRegression(max_iter=1000, C=0.5))])

    model_treated.fit(X_test[treated_mask], exp_df.loc[treated_mask, "observed_churn"])
    model_control.fit(X_test[control_mask], exp_df.loc[control_mask, "observed_churn"])

    # Estimated uplift for EVERY customer = P(churn | control) - P(churn | treated)
    # (positive number = offer helps, i.e. reduces churn)
    p_control_all = model_control.predict_proba(X_test)[:, 1]
    p_treated_all = model_treated.predict_proba(X_test)[:, 1]
    estimated_uplift = p_treated_all - p_control_all  # negative = offer reduces churn, matching true_uplift sign

    exp_df["estimated_uplift"] = estimated_uplift

    # --- Validate: does estimated uplift track the true planted uplift? ---
    exp_df["uplift_quintile"] = pd.qcut(exp_df["estimated_uplift"], 5, labels=False, duplicates="drop")
    calibration = exp_df.groupby("uplift_quintile")[["estimated_uplift", "true_uplift"]].mean()
    print("\nCalibration check -- estimated vs true (planted) uplift by quintile:")
    print("(quintile 0 = model thinks offer helps MOST)")
    print(calibration.round(4).to_string())

    corr = exp_df["estimated_uplift"].corr(exp_df["true_uplift"])
    print(f"\nCorrelation(estimated uplift, true uplift) = {corr:.3f}")
    print("(Higher = the T-learner is successfully recovering the planted effect)")

    # --- Segment customers into the 4 classic uplift buckets ---
    med_risk = exp_df["treatment"].map({0: p_control_all.mean(), 1: p_treated_all.mean()}).mean()
    exp_df["segment"] = np.select(
        [
            (exp_df["estimated_uplift"] < -0.05),                                    # offer meaningfully helps
            (exp_df["estimated_uplift"] >= -0.05) & (p_control_all < 0.3),            # low risk regardless
            (exp_df["estimated_uplift"] >= -0.05) & (p_control_all >= 0.3),           # high risk, offer won't help (lost cause)
        ],
        ["Persuadable (target these)", "Sure Thing (skip)", "Lost Cause (skip)"],
        default="Unclear",
    )
    print("\nCustomer segments based on uplift model:")
    print(exp_df["segment"].value_counts().to_string())

    # --- Compare targeting strategies: uplift-based vs risk-based (Level 4) ---
    RETENTION_OFFER_COST = 50.0
    LTV_MONTHS = 12
    SAVE_RATE = 0.35  # kept consistent with Level 4 assumption

    def policy_value(target_mask):
        # crude expected value using true_uplift as the "real" effect for evaluation
        true_saves = target_mask & (exp_df["true_uplift"] < -0.05)
        saved_value = true_saves.sum() * SAVE_RATE * (X_test["MonthlyCharges"][true_saves].mean() if true_saves.sum() else 0) * LTV_MONTHS
        cost = target_mask.sum() * RETENTION_OFFER_COST
        return saved_value - cost

    uplift_policy = exp_df["segment"] == "Persuadable (target these)"
    risk_policy = p_control_all >= np.quantile(p_control_all, 1 - uplift_policy.mean())  # same targeting rate, but by risk only

    uplift_value = policy_value(uplift_policy)
    risk_value = policy_value(risk_policy)

    print(f"\nTargeting {uplift_policy.sum()} customers either way (same budget):")
    print(f"  Risk-based targeting (Level 4 style):    expected value = ${risk_value:,.0f}")
    print(f"  Uplift-based targeting (persuadables):   expected value = ${uplift_value:,.0f}")
    print(f"  Improvement from uplift targeting: ${uplift_value - risk_value:,.0f}")

    exp_df.to_csv("outputs/uplift_results.csv", index=False)
    print("\nSaved -> outputs/uplift_results.csv")
    print("\n-> The key insight: targeting by CHURN RISK alone wastes offers on")
    print("   'lost causes' (churning regardless) and 'sure things' (staying regardless).")
    print("   Targeting by UPLIFT focuses budget only on customers the offer will actually flip.")


if __name__ == "__main__":
    main()
