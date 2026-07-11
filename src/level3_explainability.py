"""
LEVEL 3 — Explainability with SHAP.
Moves beyond feature_importances_ (which can be misleading, especially for
correlated features) to SHAP values, which show both the direction and
magnitude of each feature's effect on individual predictions.
Uses XGBoost since SHAP's TreeExplainer is fast and exact for tree models.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_prep import load_raw, get_feature_target, split, build_preprocessor


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

    # Transform test set to get feature names post-encoding
    X_test_transformed = pipe.named_steps["pre"].transform(X_test)
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    print("=" * 60)
    print("LEVEL 3: SHAP explainability")
    print("=" * 60)

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_df)

    # Global importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 10 churn drivers (by mean |SHAP value|):")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['mean_abs_shap']:.4f}")

    # Save summary plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=140, bbox_inches="tight")
    print("\nSaved SHAP summary plot -> outputs/shap_summary.png")

    # Directional insight for top 3 features
    print("\nDirectional interpretation of top drivers:")
    top3 = importance_df.head(3)["feature"].tolist()
    for feat in top3:
        idx = list(feature_names).index(feat)
        vals = X_test_df[feat].values
        shap_col = shap_values[:, idx]
        corr = np.corrcoef(vals, shap_col)[0, 1]
        direction = "increases" if corr > 0 else "decreases"
        print(f"  Higher '{feat}' -> {direction} churn risk (corr={corr:.2f})")

    print("\n-> This replaces the old bullet 'identify churn drivers' with actual")
    print("   evidence: which features, in which direction, by how much.")
    print("-> See Level 4 for turning this into a cost-sensitive business decision.")


if __name__ == "__main__":
    main()
