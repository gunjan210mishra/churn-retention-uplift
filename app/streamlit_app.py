"""
LEVEL 6 — Ship it.
Interactive Streamlit app: explore customer churn risk, see SHAP-based
explanations for individual customers, and get a retention-offer
recommendation driven by the Level 4/5 cost-sensitive + uplift logic.

Run with:  streamlit run app/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor

st.set_page_config(page_title="Churn & Retention Targeting", layout="wide")

RETENTION_OFFER_COST = 50.0
LTV_MONTHS = 12
SAVE_RATE = 0.35
BEST_THRESHOLD = 0.15  # from Level 4 analysis


@st.cache_resource
def train_model():
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

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    return pipe, explainer, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def main():
    st.title("Customer Churn & Retention Targeting")
    st.caption(
        "XGBoost churn model + SHAP explainability + cost-sensitive targeting, "
        "built on the IBM Telco Customer Churn dataset."
    )

    pipe, explainer, X_test, y_test = train_model()
    proba = pipe.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results["churn_probability"] = proba
    results["actual_churn"] = y_test.values
    results["recommend_offer"] = results["churn_probability"] >= BEST_THRESHOLD

    tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "Customer Explorer", "Business Impact"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers scored", f"{len(results):,}")
        col2.metric("Avg. churn probability", f"{results['churn_probability'].mean():.1%}")
        col3.metric("Recommended for offer", f"{results['recommend_offer'].sum():,}")
        col4.metric("Model ROC-AUC", "0.84")

        st.subheader("Churn probability distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(results["churn_probability"], bins=30, color="#1F2A44")
        ax.axvline(BEST_THRESHOLD, color="red", linestyle="--", label=f"Decision threshold ({BEST_THRESHOLD})")
        ax.set_xlabel("Predicted churn probability")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("Look up an individual customer")
        idx = st.number_input("Customer row index", min_value=0, max_value=len(results) - 1, value=0, step=1)
        row = X_test.iloc[[idx]]
        prob = results.loc[idx, "churn_probability"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn probability", f"{prob:.1%}")
            st.write("**Recommendation:**",
                     "🟠 Offer retention deal" if prob >= BEST_THRESHOLD else "🟢 No action needed")
            st.write("**Key attributes:**")
            st.write(row[["tenure", "Contract", "MonthlyCharges", "InternetService"]].T)

        with c2:
            X_transformed = pipe.named_steps["pre"].transform(row)
            feature_names = pipe.named_steps["pre"].get_feature_names_out()
            shap_vals = explainer.shap_values(pd.DataFrame(X_transformed, columns=feature_names))

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            shap.bar_plot(shap_vals[0], feature_names=feature_names, max_display=8, show=False)
            st.pyplot(fig2)
            plt.close(fig2)

    with tab3:
        st.subheader("Expected value by targeting policy")
        thresholds = np.arange(0.05, 0.95, 0.05)
        rows = []
        for t in thresholds:
            mask = results["churn_probability"] >= t
            saved = (mask & (results["actual_churn"] == 1)).sum() * SAVE_RATE * results.loc[mask, "MonthlyCharges"].mean() * LTV_MONTHS if mask.sum() else 0
            cost = mask.sum() * RETENTION_OFFER_COST
            rows.append({"threshold": t, "expected_value": (saved or 0) - cost, "n_targeted": mask.sum()})
        ev_df = pd.DataFrame(rows)

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.plot(ev_df["threshold"], ev_df["expected_value"], marker="o", color="#1F2A44")
        best = ev_df.loc[ev_df["expected_value"].idxmax()]
        ax3.axvline(best["threshold"], color="red", linestyle="--", label=f"Best threshold ({best['threshold']:.2f})")
        ax3.set_xlabel("Decision threshold")
        ax3.set_ylabel("Expected value ($)")
        ax3.legend()
        st.pyplot(fig3)

        st.info(
            f"Optimal threshold ≈ **{best['threshold']:.2f}**, targeting **{int(best['n_targeted'])}** "
            f"customers for an expected value of **${best['expected_value']:,.0f}** — "
            "see the project README for the uplift-modeling layer on top of this."
        )


if __name__ == "__main__":
    main()
